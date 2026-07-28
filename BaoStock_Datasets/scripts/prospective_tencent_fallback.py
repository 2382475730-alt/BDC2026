from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from BaoStock_Datasets.scripts.download_akshare_expanded_data import (
    call_with_retry,
    download_universe as download_akshare_universe,
)
from BaoStock_Datasets.scripts.download_baostock_expanded_data import (
    TRAIN_COLUMNS,
    normalize_stock_code,
    normalize_train_frame,
)


TENCENT_SOURCE = "Tencent"
TENCENT_SOURCE_HOST = "web.ifzq.gtimg.cn:443"
TENCENT_STOCK_ENDPOINT = "akshare.stock_zh_a_hist_tx"
TENCENT_ADJUST = "qfq"
TENCENT_SSE_SYMBOL = "sh000001"
OVERLAP_CALENDAR_DAYS = 60
MINIMUM_OVERLAP_ROWS = 5
MAXIMUM_RETURN_DIFF_P95 = 0.0025
MAXIMUM_SCALE_CV_P95 = 0.01
MINIMUM_CALIBRATION_COVERAGE = 0.95


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["股票代码"] = working["股票代码"].map(normalize_stock_code)
    working["日期"] = pd.to_datetime(
        working["日期"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    if working.duplicated(["股票代码", "日期"]).any():
        raise ValueError("Tencent fallback contains duplicate stock/date rows")
    return normalize_train_frame(working)


def _positive_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(
        denominator,
        errors="coerce",
    ).replace(0.0, np.nan)
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("quality metric has no valid observations")
    return float(np.nanpercentile(np.asarray(values, dtype=float), percentile))


def calibrate_tencent_causal_increment(
    base: pd.DataFrame,
    downloaded_qfq: pd.DataFrame,
    *,
    cutoff: str,
    minimum_calibrated_stocks: int,
    minimum_overlap_rows: int = MINIMUM_OVERLAP_ROWS,
    maximum_return_diff_p95: float = MAXIMUM_RETURN_DIFF_P95,
    maximum_scale_cv_p95: float = MAXIMUM_SCALE_CV_P95,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Calibrate one all-Tencent qfq increment against frozen causal history.

    Historical overlap is used only to estimate per-stock scales. Only rows after
    the frozen base end are returned, so the caller cannot replace the reference
    history or mix providers inside an increment.
    """

    normalized_base = _normalized(base)
    normalized_download = _normalized(downloaded_qfq)
    if normalized_base.empty or normalized_download.empty:
        raise ValueError("Tencent fallback base/download is empty")

    cutoff_date = pd.Timestamp(cutoff).normalize()
    base_end = pd.to_datetime(normalized_base["日期"]).max().normalize()
    download_dates = pd.to_datetime(normalized_download["日期"]).dt.normalize()
    future_rows = int((download_dates > cutoff_date).sum())
    if future_rows:
        raise ValueError(
            f"Tencent fallback contains {future_rows} future rows after {cutoff}"
        )
    if cutoff_date <= base_end:
        raise ValueError("Tencent fallback cutoff must be after frozen base end")

    overlap = normalized_base.merge(
        normalized_download.loc[download_dates <= base_end],
        on=["股票代码", "日期"],
        suffixes=("_base", "_tx"),
        validate="one_to_one",
    )
    calibration_rows: list[dict[str, Any]] = []
    return_differences: list[float] = []
    scale_cvs: list[float] = []
    for code, group in overlap.groupby("股票代码", sort=True):
        ordered = group.sort_values("日期").tail(20)
        if len(ordered) < int(minimum_overlap_rows):
            continue
        price_ratios = _positive_ratio(
            ordered["收盘_base"],
            ordered["收盘_tx"],
        )
        volume_ratios = _positive_ratio(
            ordered["成交量_base"],
            ordered["成交量_tx"],
        )
        amount_price_ratios = _positive_ratio(
            ordered["成交额_base"],
            ordered["成交量_base"] * ordered["收盘_base"],
        )
        free_float_shares = _positive_ratio(
            ordered["成交量_base"],
            ordered["换手率_base"] / 100.0,
        )
        if (
            len(price_ratios) < int(minimum_overlap_rows)
            or len(volume_ratios) < int(minimum_overlap_rows)
            or amount_price_ratios.empty
            or free_float_shares.empty
        ):
            continue

        price_scale = float(price_ratios.median())
        volume_scale = float(volume_ratios.median())
        ratio_mean = float(price_ratios.mean())
        scale_cv = (
            float(price_ratios.std(ddof=0)) / abs(ratio_mean)
            if ratio_mean
            else math.inf
        )
        base_return = ordered["收盘_base"].astype(float).pct_change(
            fill_method=None
        )
        tx_return = ordered["收盘_tx"].astype(float).pct_change(
            fill_method=None
        )
        stock_return_differences = (
            (base_return - tx_return)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .abs()
        )
        return_differences.extend(
            float(value) for value in stock_return_differences
        )
        scale_cvs.append(scale_cv)
        calibration_rows.append(
            {
                "股票代码": code,
                "overlap_rows": int(len(ordered)),
                "price_scale": price_scale,
                "volume_scale": volume_scale,
                "amount_price_ratio": float(amount_price_ratios.median()),
                "free_float_shares": float(free_float_shares.median()),
                "price_scale_cv": scale_cv,
                "abs_return_diff_p95": (
                    float(np.nanpercentile(stock_return_differences, 95))
                    if not stock_return_differences.empty
                    else 0.0
                ),
            }
        )

    calibration = pd.DataFrame(calibration_rows)
    calibrated_stocks = int(len(calibration))
    if calibrated_stocks < int(minimum_calibrated_stocks):
        raise ValueError(
            "Tencent calibration coverage too low: "
            f"{calibrated_stocks} < {minimum_calibrated_stocks}"
        )
    return_diff_p95 = _percentile(return_differences, 95)
    if return_diff_p95 > float(maximum_return_diff_p95):
        raise ValueError(
            "Tencent overlap return-difference gate failed: "
            f"{return_diff_p95:.8f} > {maximum_return_diff_p95:.8f}"
        )
    scale_cv_p95 = _percentile(scale_cvs, 95)
    if scale_cv_p95 > float(maximum_scale_cv_p95):
        raise ValueError(
            "Tencent overlap price-scale gate failed: "
            f"{scale_cv_p95:.8f} > {maximum_scale_cv_p95:.8f}"
        )

    calibration_by_code = calibration.set_index("股票代码")
    increment_source = normalized_download.loc[
        (download_dates > base_end) & (download_dates <= cutoff_date)
    ].copy()
    if increment_source.empty:
        raise ValueError("Tencent fallback increment is empty")
    uncalibrated_increment = sorted(
        set(increment_source["股票代码"]) - set(calibration_by_code.index)
    )
    if uncalibrated_increment:
        raise ValueError(
            "Tencent increment contains uncalibrated stocks: "
            f"{len(uncalibrated_increment)}"
        )

    rebuilt_groups: list[pd.DataFrame] = []
    for code, group in increment_source.groupby("股票代码", sort=True):
        parameters = calibration_by_code.loc[code]
        rebuilt = group.sort_values("日期").copy()
        for column in ["开盘", "收盘", "最高", "最低"]:
            rebuilt[column] = (
                pd.to_numeric(rebuilt[column], errors="raise")
                * float(parameters["price_scale"])
            )
        rebuilt["成交量"] = (
            pd.to_numeric(rebuilt["成交量"], errors="raise")
            * float(parameters["volume_scale"])
        )
        rebuilt["成交额"] = (
            rebuilt["成交量"]
            * rebuilt["收盘"]
            * float(parameters["amount_price_ratio"])
        )
        rebuilt["换手率"] = (
            rebuilt["成交量"]
            / float(parameters["free_float_shares"])
            * 100.0
        )
        previous_close = pd.concat(
            [
                pd.Series(
                    [
                        float(
                            normalized_base.loc[
                                normalized_base["股票代码"] == code,
                                "收盘",
                            ].iloc[-1]
                        )
                    ]
                ),
                rebuilt["收盘"].iloc[:-1].reset_index(drop=True),
            ],
            ignore_index=True,
        )
        previous_close.index = rebuilt.index
        rebuilt["振幅"] = (
            (rebuilt["最高"] - rebuilt["最低"])
            / previous_close.replace(0.0, np.nan)
            * 100.0
        )
        rebuilt["涨跌额"] = rebuilt["收盘"] - previous_close
        rebuilt["涨跌幅"] = (
            rebuilt["涨跌额"]
            / previous_close.replace(0.0, np.nan)
            * 100.0
        )
        rebuilt_groups.append(rebuilt[TRAIN_COLUMNS])

    increment = normalize_train_frame(
        pd.concat(rebuilt_groups, ignore_index=True)
    )
    increment_dates = pd.to_datetime(increment["日期"]).dt.normalize()
    if (increment_dates <= base_end).any():
        raise ValueError("Tencent fallback attempted to replace historical rows")
    if (increment_dates > cutoff_date).any():
        raise ValueError("Tencent fallback emitted future rows")
    report = {
        "status": "PASS",
        "provider": TENCENT_SOURCE,
        "endpoint": TENCENT_STOCK_ENDPOINT,
        "adjust": TENCENT_ADJUST,
        "base_end": base_end.date().isoformat(),
        "cutoff": cutoff_date.date().isoformat(),
        "overlap_rows": int(len(overlap)),
        "calibrated_stocks": calibrated_stocks,
        "minimum_calibrated_stocks": int(minimum_calibrated_stocks),
        "minimum_overlap_rows": int(minimum_overlap_rows),
        "abs_return_diff_p95": return_diff_p95,
        "maximum_return_diff_p95": float(maximum_return_diff_p95),
        "price_scale_cv_p95": scale_cv_p95,
        "maximum_scale_cv_p95": float(maximum_scale_cv_p95),
        "increment_rows": int(len(increment)),
        "increment_stocks": int(increment["股票代码"].nunique()),
        "history_overlap_rows_written": 0,
        "future_rows": 0,
        "source_mix": False,
    }
    return increment[TRAIN_COLUMNS], calibration.reset_index(drop=True), report


def build_tencent_causal_increment(
    base: pd.DataFrame,
    *,
    cutoff: str,
    cache_dir: Path,
    retries: int,
    retry_sec: float,
    sleep_sec: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    normalized_base = _normalized(base)
    universe = sorted(set(normalized_base["股票代码"]))
    base_end = pd.to_datetime(normalized_base["日期"]).max().normalize()
    overlap_start = max(
        pd.to_datetime(normalized_base["日期"]).min().normalize(),
        base_end - pd.Timedelta(days=OVERLAP_CALENDAR_DAYS),
    )
    downloaded, reports = download_akshare_universe(
        universe,
        overlap_start.date().isoformat(),
        pd.Timestamp(cutoff).date().isoformat(),
        TENCENT_ADJUST,
        retries,
        retry_sec,
        sleep_sec,
        "tencent",
        cache_dir,
    )
    report_payloads = [asdict(report) for report in reports]
    failed = [report for report in report_payloads if not bool(report["ok"])]
    if failed:
        raise RuntimeError(
            f"Tencent causal download failed for {len(failed)} symbols"
        )
    increment, calibration, validation = calibrate_tencent_causal_increment(
        normalized_base,
        downloaded,
        cutoff=cutoff,
        minimum_calibrated_stocks=math.ceil(
            len(universe) * MINIMUM_CALIBRATION_COVERAGE
        ),
    )
    validation["query_bounds"] = {
        "start_date": overlap_start.date().isoformat(),
        "end_date": pd.Timestamp(cutoff).date().isoformat(),
    }
    validation["api_request_successes"] = len(report_payloads)
    validation["api_request_failures"] = 0
    return increment, report_payloads, calibration, validation


def download_tencent_outcome_market(
    universe: list[str],
    start_date: str,
    end_date: str,
    cache_dir: Path,
    sleep_sec: float,
    retries: int,
    retry_sec: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    market, reports = download_akshare_universe(
        universe,
        start_date,
        end_date,
        TENCENT_ADJUST,
        retries,
        retry_sec,
        sleep_sec,
        "tencent",
        cache_dir,
    )
    payloads = [asdict(report) for report in reports]
    failed = [report for report in payloads if not bool(report["ok"])]
    if failed:
        raise RuntimeError(
            f"Tencent outcome download failed for {len(failed)} symbols"
        )
    return market, payloads


def download_tencent_sse_open(
    start_date: str,
    end_date: str,
    retries: int,
    retry_sec: float,
) -> pd.DataFrame:
    import akshare as ak

    raw = call_with_retry(
        lambda: ak.stock_zh_a_hist_tx(
            symbol=TENCENT_SSE_SYMBOL,
            start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
            end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
            adjust="",
        ),
        retries,
        retry_sec,
    )
    if raw.empty:
        return pd.DataFrame(columns=["日期", "开盘"])
    output = pd.DataFrame(
        {
            "日期": pd.to_datetime(raw["date"], errors="raise").dt.strftime(
                "%Y-%m-%d"
            ),
            "开盘": pd.to_numeric(raw["open"], errors="coerce"),
        }
    )
    return (
        output.dropna(subset=["开盘"])
        .sort_values("日期")
        .reset_index(drop=True)
    )

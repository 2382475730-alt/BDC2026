from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRAIN_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]

NUMERIC_COLUMNS = [column for column in TRAIN_COLUMNS if column not in {"股票代码", "日期"}]


@dataclass
class DownloadReport:
    symbol: str
    source: str
    rows: int
    start_date: str | None
    end_date: str | None
    ok: bool
    error: str = ""


def normalize_stock_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"invalid stock code: {value!r}")
    return digits[-6:].zfill(6)


def tx_symbol(symbol: str) -> str:
    code = normalize_stock_code(symbol)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def yyyymmdd(value: str) -> str:
    return pd.to_datetime(value).strftime("%Y%m%d")


def yyyy_mm_dd(value: str) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def normalize_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    missing = [column for column in TRAIN_COLUMNS if column not in output.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    output = output[TRAIN_COLUMNS].copy()
    output["股票代码"] = output["股票代码"].map(normalize_stock_code)
    output["日期"] = pd.to_datetime(output["日期"]).dt.strftime("%Y-%m-%d")
    for column in NUMERIC_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["开盘", "收盘", "最高", "最低"])
    output[NUMERIC_COLUMNS] = output[NUMERIC_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output[TRAIN_COLUMNS].sort_values(["日期", "股票代码"]).reset_index(drop=True)


def convert_eastmoney_hist(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TRAIN_COLUMNS)
    output = pd.DataFrame(
        {
            "股票代码": normalize_stock_code(symbol),
            "日期": df["日期"],
            "开盘": df["开盘"],
            "收盘": df["收盘"],
            "最高": df["最高"],
            "最低": df["最低"],
            "成交量": pd.to_numeric(df["成交量"], errors="coerce") * 100.0,
            "成交额": df["成交额"],
            "振幅": df["振幅"],
            "涨跌额": df["涨跌额"],
            "换手率": df["换手率"],
            "涨跌幅": df["涨跌幅"],
        }
    )
    return normalize_train_frame(output)


def convert_tencent_hist(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TRAIN_COLUMNS)
    numeric = df.copy()
    for column in ["open", "close", "high", "low", "amount"]:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    preclose = numeric["close"].shift(1).replace(0.0, np.nan)
    pct_chg = (numeric["close"] / preclose - 1.0) * 100.0
    price_chg = numeric["close"] - preclose
    amplitude = (numeric["high"] - numeric["low"]) / preclose * 100.0
    output = pd.DataFrame(
        {
            "股票代码": normalize_stock_code(symbol),
            "日期": numeric["date"],
            "开盘": numeric["open"],
            "收盘": numeric["close"],
            "最高": numeric["high"],
            "最低": numeric["low"],
            "成交量": numeric["amount"] * 100.0,
            "成交额": 0.0,
            "振幅": amplitude,
            "涨跌额": price_chg,
            "换手率": 0.0,
            "涨跌幅": pct_chg,
        }
    )
    return normalize_train_frame(output)


def call_with_retry(fn, retries: int, sleep_sec: float):
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - downloader should keep going per symbol
            last_error = exc
            if attempt < retries and sleep_sec > 0:
                time.sleep(sleep_sec)
    if last_error is not None:
        raise last_error
    raise RuntimeError("unreachable retry state")


def download_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
    retries: int,
    retry_sleep: float,
    source: str,
) -> tuple[pd.DataFrame, DownloadReport]:
    import akshare as ak

    code = normalize_stock_code(symbol)
    start8 = yyyymmdd(start_date)
    end8 = yyyymmdd(end_date)
    if source == "tencent":
        try:
            raw = call_with_retry(
                lambda: ak.stock_zh_a_hist_tx(symbol=tx_symbol(code), start_date=start8, end_date=end8, adjust=adjust),
                retries,
                retry_sleep,
            )
            frame = convert_tencent_hist(raw, code)
            return frame, DownloadReport(
                symbol=code,
                source=f"akshare.stock_zh_a_hist_tx:{adjust or 'none'}",
                rows=len(frame),
                start_date=frame["日期"].min() if not frame.empty else None,
                end_date=frame["日期"].max() if not frame.empty else None,
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001
            return pd.DataFrame(columns=TRAIN_COLUMNS), DownloadReport(
                symbol=code,
                source="failed",
                rows=0,
                start_date=None,
                end_date=None,
                ok=False,
                error=f"tencent {type(exc).__name__}: {exc}",
            )

    if source == "eastmoney":
        try:
            raw = call_with_retry(
                lambda: ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start8, end_date=end8, adjust=adjust),
                retries,
                retry_sleep,
            )
            frame = convert_eastmoney_hist(raw, code)
            return frame, DownloadReport(
                symbol=code,
                source=f"akshare.stock_zh_a_hist:{adjust or 'none'}",
                rows=len(frame),
                start_date=frame["日期"].min() if not frame.empty else None,
                end_date=frame["日期"].max() if not frame.empty else None,
                ok=True,
            )
        except Exception as exc:  # noqa: BLE001
            return pd.DataFrame(columns=TRAIN_COLUMNS), DownloadReport(
                symbol=code,
                source="failed",
                rows=0,
                start_date=None,
                end_date=None,
                ok=False,
                error=f"eastmoney {type(exc).__name__}: {exc}",
            )

    try:
        raw = call_with_retry(
            lambda: ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start8, end_date=end8, adjust=adjust),
            retries,
            retry_sleep,
        )
        frame = convert_eastmoney_hist(raw, code)
        return frame, DownloadReport(
            symbol=code,
            source=f"akshare.stock_zh_a_hist:{adjust or 'none'}",
            rows=len(frame),
            start_date=frame["日期"].min() if not frame.empty else None,
            end_date=frame["日期"].max() if not frame.empty else None,
            ok=True,
        )
    except Exception as primary_error:  # noqa: BLE001
        try:
            raw = call_with_retry(
                lambda: ak.stock_zh_a_hist_tx(symbol=tx_symbol(code), start_date=start8, end_date=end8, adjust=adjust),
                retries,
                retry_sleep,
            )
            frame = convert_tencent_hist(raw, code)
            return frame, DownloadReport(
                symbol=code,
                source=f"akshare.stock_zh_a_hist_tx:{adjust or 'none'}",
                rows=len(frame),
                start_date=frame["日期"].min() if not frame.empty else None,
                end_date=frame["日期"].max() if not frame.empty else None,
                ok=True,
                error=f"primary failed: {type(primary_error).__name__}: {primary_error}",
            )
        except Exception as fallback_error:  # noqa: BLE001
            return pd.DataFrame(columns=TRAIN_COLUMNS), DownloadReport(
                symbol=code,
                source="failed",
                rows=0,
                start_date=None,
                end_date=None,
                ok=False,
                error=(
                    f"primary {type(primary_error).__name__}: {primary_error}; "
                    f"fallback {type(fallback_error).__name__}: {fallback_error}"
                ),
            )


def merge_append_new(base_df: pd.DataFrame, downloaded_df: pd.DataFrame) -> pd.DataFrame:
    base = normalize_train_frame(base_df)
    downloaded = normalize_train_frame(downloaded_df)
    combined = pd.concat([downloaded, base], ignore_index=True)
    combined = combined.drop_duplicates(subset=["股票代码", "日期"], keep="last")
    return normalize_train_frame(combined)


def merge_replace_overlap(base_df: pd.DataFrame, downloaded_df: pd.DataFrame) -> pd.DataFrame:
    base = normalize_train_frame(base_df)
    downloaded = normalize_train_frame(downloaded_df)
    combined = pd.concat([base, downloaded], ignore_index=True)
    combined = combined.drop_duplicates(subset=["股票代码", "日期"], keep="last")
    return normalize_train_frame(combined)


def stock_universe(base_df: pd.DataFrame, max_stocks: int | None) -> list[str]:
    codes = sorted(base_df["股票代码"].map(normalize_stock_code).unique())
    return codes[:max_stocks] if max_stocks else codes


def summarize_frame(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {"rows": 0, "stocks": 0, "date_min": None, "date_max": None}
    return {
        "rows": int(len(df)),
        "stocks": int(df["股票代码"].nunique()),
        "date_min": str(df["日期"].min()),
        "date_max": str(df["日期"].max()),
        "rows_per_stock_min": int(df.groupby("股票代码")["日期"].count().min()),
        "rows_per_stock_max": int(df.groupby("股票代码")["日期"].count().max()),
    }


def overlap_quality(base_df: pd.DataFrame, downloaded_df: pd.DataFrame) -> dict[str, object]:
    base = normalize_train_frame(base_df)
    downloaded = normalize_train_frame(downloaded_df)
    merged = base.merge(downloaded, on=["股票代码", "日期"], suffixes=("_base", "_download"))
    if merged.empty:
        return {"overlap_rows": 0}
    price_ratios = []
    return_diffs = []
    for _, group in merged.sort_values(["股票代码", "日期"]).groupby("股票代码"):
        if len(group) < 2:
            continue
        ratio = group["收盘_base"].astype(float) / group["收盘_download"].astype(float).replace(0.0, np.nan)
        price_ratios.extend(ratio.replace([np.inf, -np.inf], np.nan).dropna().tolist())
        base_ret = group["收盘_base"].astype(float).pct_change(fill_method=None)
        dl_ret = group["收盘_download"].astype(float).pct_change(fill_method=None)
        return_diffs.extend((base_ret - dl_ret).replace([np.inf, -np.inf], np.nan).dropna().abs().tolist())
    return {
        "overlap_rows": int(len(merged)),
        "overlap_stocks": int(merged["股票代码"].nunique()),
        "close_ratio_median": float(np.nanmedian(price_ratios)) if price_ratios else None,
        "close_ratio_p05": float(np.nanpercentile(price_ratios, 5)) if price_ratios else None,
        "close_ratio_p95": float(np.nanpercentile(price_ratios, 95)) if price_ratios else None,
        "abs_return_diff_median": float(np.nanmedian(return_diffs)) if return_diffs else None,
        "abs_return_diff_p95": float(np.nanpercentile(return_diffs, 95)) if return_diffs else None,
    }


def download_universe(
    codes: Iterable[str],
    start_date: str,
    end_date: str,
    adjust: str,
    retries: int,
    retry_sleep: float,
    sleep_sec: float,
    source: str,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[DownloadReport]]:
    frames: list[pd.DataFrame] = []
    reports: list[DownloadReport] = []
    codes = list(codes)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for index, code in enumerate(codes, start=1):
        cache_path = cache_dir / f"{normalize_stock_code(code)}.csv" if cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            frame = normalize_train_frame(pd.read_csv(cache_path, dtype={0: str}))
            report = DownloadReport(
                symbol=normalize_stock_code(code),
                source=f"cache:{source}:{adjust or 'none'}",
                rows=len(frame),
                start_date=frame["日期"].min() if not frame.empty else None,
                end_date=frame["日期"].max() if not frame.empty else None,
                ok=True,
            )
        else:
            frame, report = download_symbol(code, start_date, end_date, adjust, retries, retry_sleep, source)
            if cache_path is not None and report.ok:
                frame.to_csv(cache_path, index=False, encoding="utf-8")
        reports.append(report)
        if not frame.empty:
            frames.append(frame)
        status = "ok" if report.ok else "failed"
        print(f"[{index:03d}/{len(codes):03d}] {code} {status} {report.rows} rows via {report.source}")
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    if not frames:
        return pd.DataFrame(columns=TRAIN_COLUMNS), reports
    return normalize_train_frame(pd.concat(frames, ignore_index=True)), reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download expanded A-share history for the current competition universe.")
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--history-start", default="2021-01-01")
    parser.add_argument("--train-end", default=None, help="Defaults to max date in base train.")
    parser.add_argument("--future-start", default=None, help="Defaults to one day after train-end.")
    parser.add_argument("--future-end", default="2026-06-30")
    parser.add_argument("--adjust", default="hfq", choices=["", "qfq", "hfq"])
    parser.add_argument("--source", default="auto", choices=["auto", "eastmoney", "tencent"])
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--sleep-sec", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = normalize_train_frame(pd.read_csv(args.base_train, dtype={0: str}))
    train_end = yyyy_mm_dd(args.train_end or base["日期"].max())
    future_start = yyyy_mm_dd(args.future_start or (pd.to_datetime(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    future_end = yyyy_mm_dd(args.future_end)
    codes = stock_universe(base, args.max_stocks)

    print(f"Universe: {len(codes)} stocks")
    print(f"History window: {args.history_start} to {train_end}, adjust={args.adjust or 'none'}")
    history, history_reports = download_universe(
        codes,
        args.history_start,
        train_end,
        args.adjust,
        args.retries,
        args.retry_sleep,
        args.sleep_sec,
        args.source,
        out_dir / "history_by_symbol",
    )
    history_path = out_dir / f"akshare_history_{args.adjust or 'none'}_{args.history_start}_{train_end}.csv"
    history.to_csv(history_path, index=False, encoding="utf-8")

    append_new = merge_append_new(base, history)
    replace_overlap = merge_replace_overlap(base, history)
    append_path = out_dir / "train_extended_append_new.csv"
    replace_path = out_dir / "train_extended_replace_overlap.csv"
    append_new.to_csv(append_path, index=False, encoding="utf-8")
    replace_overlap.to_csv(replace_path, index=False, encoding="utf-8")

    print(f"Future test window: {future_start} to {future_end}, adjust={args.adjust or 'none'}")
    future, future_reports = download_universe(
        codes,
        future_start,
        future_end,
        args.adjust,
        args.retries,
        args.retry_sleep,
        args.sleep_sec,
        args.source,
        out_dir / "future_by_symbol",
    )
    future_path = out_dir / f"akshare_future_{args.adjust or 'none'}_{future_start}_{future_end}.csv"
    future.to_csv(future_path, index=False, encoding="utf-8")

    report = {
        "base_train": summarize_frame(base),
        "history": summarize_frame(history),
        "future": summarize_frame(future),
        "append_new_train": summarize_frame(append_new),
        "replace_overlap_train": summarize_frame(replace_overlap),
        "overlap_quality": overlap_quality(base, history),
        "paths": {
            "history": str(history_path),
            "future": str(future_path),
            "append_new_train": str(append_path),
            "replace_overlap_train": str(replace_path),
        },
        "history_reports": [asdict(item) for item in history_reports],
        "future_reports": [asdict(item) for item in future_reports],
    }
    (out_dir / "download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["base_train", "history", "future", "append_new_train", "replace_overlap_train", "overlap_quality", "paths"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

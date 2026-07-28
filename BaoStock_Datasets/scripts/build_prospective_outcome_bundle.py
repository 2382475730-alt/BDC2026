from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from BaoStock_Datasets.scripts.download_baostock_expanded_data import (
    TRAIN_COLUMNS,
    download_universe,
    login_baostock,
    normalize_stock_code,
    normalize_train_frame,
)
from BaoStock_Datasets.scripts.prospective_tencent_fallback import (
    TENCENT_ADJUST,
    TENCENT_SOURCE,
    TENCENT_SOURCE_HOST,
    TENCENT_SSE_SYMBOL,
    download_tencent_outcome_market,
    download_tencent_sse_open,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
SOURCE = "BaoStock"
SOURCE_HOST = "public-api.baostock.com:10030"
STOCK_ADJUSTFLAG = "1"
SSE_CODE = "sh.000001"
SSE_ADJUSTFLAG = "3"
MINIMUM_ENDPOINT_COVERAGE = 0.95
SOURCE_POLICY = "baostock_then_tencent_qfq"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - competition provenance requires MD5.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def assert_scoring_time(sell_date: str, now: datetime) -> dict[str, Any]:
    if now.tzinfo is None:
        raise ValueError("scoring time must be timezone-aware")
    local = now.astimezone(TIMEZONE)
    unlock = datetime.combine(
        pd.Timestamp(sell_date).date(),
        datetime.min.time(),
        tzinfo=TIMEZONE,
    ).replace(hour=9, minute=31)
    if local < unlock:
        raise ValueError(f"scoring time gate opens at {unlock.isoformat()}")
    return {
        "status": "PASS",
        "sell_date": sell_date,
        "checked_at": local.isoformat(),
        "unlock_time": unlock.isoformat(),
    }


def _normalize_market_with_duplicate_check(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in TRAIN_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"outcome market missing required columns: {missing}")
    working = frame[TRAIN_COLUMNS].copy()
    working["股票代码"] = working["股票代码"].map(normalize_stock_code)
    working["日期"] = pd.to_datetime(
        working["日期"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    if working.duplicated(["股票代码", "日期"]).any():
        raise ValueError("outcome market contains duplicate stock/date rows")
    return normalize_train_frame(working)


def validate_outcome_market(
    frame: pd.DataFrame,
    *,
    buy_date: str,
    sell_date: str,
    universe: set[str],
    minimum_endpoint_stocks: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized = _normalize_market_with_duplicate_check(frame)
    if normalized.empty:
        raise ValueError("outcome market is empty")

    normalized_universe = {normalize_stock_code(code) for code in universe}
    observed_universe = set(normalized["股票代码"])
    unexpected = sorted(observed_universe - normalized_universe)
    if unexpected:
        raise ValueError(
            f"outcome universe drift contains {len(unexpected)} unexpected stocks"
        )

    buy = pd.Timestamp(buy_date).normalize()
    sell = pd.Timestamp(sell_date).normalize()
    if sell < buy:
        raise ValueError("sell_date must not precede buy_date")
    dates = pd.to_datetime(normalized["日期"]).dt.normalize()
    later_rows = int((dates > sell).sum())
    if later_rows:
        raise ValueError(
            f"outcome market contains {later_rows} later-window rows after {sell_date}"
        )
    earlier_rows = int((dates < buy).sum())
    if earlier_rows:
        raise ValueError(
            f"outcome market contains {earlier_rows} rows before buy endpoint {buy_date}"
        )

    buy_codes = set(
        normalized.loc[normalized["日期"] == buy_date, "股票代码"]
    )
    sell_codes = set(
        normalized.loc[normalized["日期"] == sell_date, "股票代码"]
    )
    if not buy_codes:
        raise ValueError(f"buy endpoint {buy_date} is missing")
    if not sell_codes:
        raise ValueError(f"sell endpoint {sell_date} is missing")
    if len(buy_codes) < int(minimum_endpoint_stocks):
        raise ValueError(
            "buy endpoint coverage too low: "
            f"{len(buy_codes)} < {minimum_endpoint_stocks}"
        )
    if len(sell_codes) < int(minimum_endpoint_stocks):
        raise ValueError(
            "sell endpoint coverage too low: "
            f"{len(sell_codes)} < {minimum_endpoint_stocks}"
        )

    validation = {
        "status": "PASS",
        "buy_date": buy_date,
        "sell_date": sell_date,
        "rows": int(len(normalized)),
        "stocks": int(normalized["股票代码"].nunique()),
        "date_start": str(normalized["日期"].min()),
        "date_end": str(normalized["日期"].max()),
        "buy_endpoint_stocks": len(buy_codes),
        "sell_endpoint_stocks": len(sell_codes),
        "minimum_endpoint_stocks": int(minimum_endpoint_stocks),
        "later_window_rows": 0,
        "earlier_window_rows": 0,
        "unexpected_universe_stocks": 0,
    }
    return normalized[TRAIN_COLUMNS], validation


def download_sse_open(
    start_date: str,
    end_date: str,
    retries: int,
    retry_sec: float,
) -> pd.DataFrame:
    import baostock as bs

    login = login_baostock(bs, retries, retry_sec)
    if login.error_code != "0":
        raise RuntimeError(
            f"baostock login failed: {login.error_code} {login.error_msg}"
        )
    try:
        result = bs.query_history_k_data_plus(
            SSE_CODE,
            "date,code,open",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=SSE_ADJUSTFLAG,
        )
        if result.error_code != "0":
            raise RuntimeError(
                "SSE query failed: "
                f"{result.error_code} {result.error_msg}"
            )
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        if not rows:
            return pd.DataFrame(columns=["日期", "开盘"])
        raw = pd.DataFrame(rows, columns=result.fields)
        output = pd.DataFrame(
            {
                "日期": pd.to_datetime(
                    raw["date"],
                    errors="raise",
                ).dt.strftime("%Y-%m-%d"),
                "开盘": pd.to_numeric(raw["open"], errors="coerce"),
            }
        )
        return output.dropna(subset=["开盘"]).sort_values("日期").reset_index(
            drop=True
        )
    finally:
        bs.logout()


def _validate_sse_open(
    frame: pd.DataFrame,
    *,
    buy_date: str,
    sell_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = ["日期", "开盘"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"SSE outcome missing required columns: {missing}")
    normalized = frame[required].copy()
    normalized["日期"] = pd.to_datetime(
        normalized["日期"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    normalized["开盘"] = pd.to_numeric(normalized["开盘"], errors="coerce")
    normalized = normalized.dropna(subset=["开盘"])
    if normalized.duplicated(["日期"]).any():
        raise ValueError("SSE outcome contains duplicate dates")
    if normalized.empty:
        raise ValueError("SSE outcome is empty")

    buy = pd.Timestamp(buy_date).normalize()
    sell = pd.Timestamp(sell_date).normalize()
    dates = pd.to_datetime(normalized["日期"]).dt.normalize()
    if (dates > sell).any():
        raise ValueError("SSE outcome contains later-window rows")
    if (dates < buy).any():
        raise ValueError("SSE outcome contains rows before buy endpoint")
    available_dates = set(normalized["日期"])
    if buy_date not in available_dates:
        raise ValueError(f"SSE buy endpoint {buy_date} is missing")
    if sell_date not in available_dates:
        raise ValueError(f"SSE sell endpoint {sell_date} is missing")
    normalized = normalized.sort_values("日期").reset_index(drop=True)
    return normalized, {
        "status": "PASS",
        "series": "上证指数",
        "code": SSE_CODE,
        "buy_date": buy_date,
        "sell_date": sell_date,
        "rows": int(len(normalized)),
        "date_start": str(normalized["日期"].min()),
        "date_end": str(normalized["日期"].max()),
        "later_window_rows": 0,
        "earlier_window_rows": 0,
    }


def _report_payload(report: Any) -> dict[str, Any]:
    if is_dataclass(report) and not isinstance(report, type):
        return asdict(report)
    if isinstance(report, dict):
        return dict(report)
    raise TypeError(f"unsupported download report type: {type(report).__name__}")


def build_outcome_bundle(
    *,
    window_id: str,
    training_prefix: Path,
    buy_date: str,
    sell_date: str,
    out_dir: Path,
    now: datetime,
    sleep_sec: float,
    login_retries: int,
    retry_sec: float,
) -> dict[str, Any]:
    time_gate = assert_scoring_time(sell_date, now)
    if not training_prefix.is_file():
        raise FileNotFoundError(training_prefix)

    market_path = out_dir / "outcome_market.csv"
    sse_path = out_dir / "sse_open.csv"
    manifest_path = out_dir / "outcome_bundle_manifest.json"
    if market_path.exists() or sse_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite completed outcome artifacts in {out_dir}"
        )

    prefix_raw = pd.read_csv(
        training_prefix,
        dtype={"股票代码": str},
    )
    prefix = _normalize_market_with_duplicate_check(prefix_raw)
    universe = sorted(set(prefix["股票代码"]))
    if len(universe) != 300:
        raise ValueError(
            f"frozen training universe must contain 300 stocks: {len(universe)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    minimum_endpoint_stocks = math.ceil(
        len(universe) * MINIMUM_ENDPOINT_COVERAGE
    )
    attempts: list[dict[str, Any]] = []
    try:
        market_raw, reports = download_universe(
            universe,
            buy_date,
            sell_date,
            STOCK_ADJUSTFLAG,
            out_dir / "source_attempts" / "baostock",
            sleep_sec,
            login_retries,
            retry_sec,
        )
        report_payloads = [_report_payload(report) for report in reports]
        failed = [
            report for report in report_payloads if not bool(report.get("ok"))
        ]
        if failed:
            raise RuntimeError(
                f"BaoStock outcome download failed for {len(failed)} symbols"
            )
        market, market_validation = validate_outcome_market(
            market_raw,
            buy_date=buy_date,
            sell_date=sell_date,
            universe=set(universe),
            minimum_endpoint_stocks=minimum_endpoint_stocks,
        )
        sse_raw = download_sse_open(
            buy_date,
            sell_date,
            login_retries,
            retry_sec,
        )
        sse, sse_validation = _validate_sse_open(
            sse_raw,
            buy_date=buy_date,
            sell_date=sell_date,
        )
        source = SOURCE
        source_host = SOURCE_HOST
        stock_adjustflag = STOCK_ADJUSTFLAG
        sse_code = SSE_CODE
        sse_adjustflag = SSE_ADJUSTFLAG
        attempts.append(
            {
                "provider": SOURCE,
                "status": "PASS",
                "selected": True,
            }
        )
    except Exception as baostock_error:  # noqa: BLE001 - frozen failover boundary
        attempts.append(
            {
                "provider": SOURCE,
                "status": "FAILED",
                "selected": False,
                "error": (
                    f"{type(baostock_error).__name__}: {baostock_error}"
                ),
            }
        )
        try:
            market_raw, report_payloads = download_tencent_outcome_market(
                universe,
                buy_date,
                sell_date,
                out_dir / "source_attempts" / "tencent_qfq",
                sleep_sec,
                login_retries,
                retry_sec,
            )
            market, market_validation = validate_outcome_market(
                market_raw,
                buy_date=buy_date,
                sell_date=sell_date,
                universe=set(universe),
                minimum_endpoint_stocks=minimum_endpoint_stocks,
            )
            sse_raw = download_tencent_sse_open(
                buy_date,
                sell_date,
                login_retries,
                retry_sec,
            )
            sse, sse_validation = _validate_sse_open(
                sse_raw,
                buy_date=buy_date,
                sell_date=sell_date,
            )
        except Exception as tencent_error:  # noqa: BLE001 - report both failures
            raise RuntimeError(
                "both frozen outcome providers failed; "
                f"BaoStock={type(baostock_error).__name__}: {baostock_error}; "
                f"Tencent={type(tencent_error).__name__}: {tencent_error}"
            ) from tencent_error
        source = TENCENT_SOURCE
        source_host = TENCENT_SOURCE_HOST
        stock_adjustflag = TENCENT_ADJUST
        sse_code = TENCENT_SSE_SYMBOL
        sse_adjustflag = "none"
        attempts.append(
            {
                "provider": TENCENT_SOURCE,
                "status": "PASS",
                "selected": True,
            }
        )

    market.to_csv(market_path, index=False, encoding="utf-8")
    sse.to_csv(sse_path, index=False, encoding="utf-8")
    manifest = {
        "schema_version": "test_d_prospective_outcome_bundle_v1",
        "classification": "SEALED_TEST_OUTCOME_INPUT",
        "window_id": window_id,
        "created_at": now.astimezone(TIMEZONE).isoformat(),
        "source_policy": SOURCE_POLICY,
        "source": source,
        "source_host": source_host,
        "source_mix": False,
        "attempts": attempts,
        "query_bounds": {
            "start_date": buy_date,
            "end_date": sell_date,
        },
        "stock_adjustflag": stock_adjustflag,
        "sse": {
            "name": "上证指数",
            "code": sse_code,
            "adjustflag": sse_adjustflag,
            "file": {
                "path": str(sse_path.resolve()),
                "sha256": sha256_file(sse_path),
                "md5": md5_file(sse_path),
            },
            "validation": sse_validation,
        },
        "training_prefix": {
            "path": str(training_prefix.resolve()),
            "sha256": sha256_file(training_prefix),
            "md5": md5_file(training_prefix),
            "stocks": len(universe),
            "date_end": str(prefix["日期"].max()),
        },
        "outcome_market": {
            "path": str(market_path.resolve()),
            "sha256": sha256_file(market_path),
            "md5": md5_file(market_path),
            "download_reports": report_payloads,
        },
        "validation": market_validation,
        "time_gate": time_gate,
        "later_window_query_performed": False,
        "later_window_rows": 0,
        "training_workspace_written": False,
        "candidate_package_written": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one sell-date-bounded outcome bundle using the frozen "
            "BaoStock-then-Tencent source policy."
        )
    )
    parser.add_argument("--window", required=True)
    parser.add_argument("--training-prefix", required=True, type=Path)
    parser.add_argument("--buy-date", required=True)
    parser.add_argument("--sell-date", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--sleep-sec", type=float, default=0.02)
    parser.add_argument("--login-retries", type=int, default=3)
    parser.add_argument("--retry-sec", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = (
        datetime.fromisoformat(args.as_of)
        if args.as_of
        else datetime.now(TIMEZONE)
    )
    manifest = build_outcome_bundle(
        window_id=args.window,
        training_prefix=args.training_prefix.resolve(),
        buy_date=args.buy_date,
        sell_date=args.sell_date,
        out_dir=args.out_dir.resolve(),
        now=now,
        sleep_sec=max(0.0, float(args.sleep_sec)),
        login_retries=max(1, int(args.login_retries)),
        retry_sec=max(0.0, float(args.retry_sec)),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from BaoStock_Datasets.scripts.download_baostock_expanded_data import (
    TRAIN_COLUMNS,
    download_universe,
    normalize_stock_code,
    normalize_train_frame,
)
from BaoStock_Datasets.scripts.prospective_tencent_fallback import (
    TENCENT_ADJUST,
    TENCENT_SOURCE,
    TENCENT_SOURCE_HOST,
    build_tencent_causal_increment,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
SOURCE = "BaoStock"
SOURCE_HOST = "public-api.baostock.com:10030"
ADJUSTFLAG = "1"
MINIMUM_CUTOFF_COVERAGE = 0.95
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


def causal_query_bounds(base: pd.DataFrame, cutoff: str) -> dict[str, str]:
    if "日期" not in base.columns or base.empty:
        raise ValueError("base prefix is empty or missing 日期")
    dates = pd.to_datetime(base["日期"], errors="raise").dt.normalize()
    base_end = dates.max()
    cutoff_date = pd.Timestamp(cutoff).normalize()
    if cutoff_date <= base_end:
        raise ValueError(
            f"cutoff {cutoff} must be after base prefix end {base_end.date()}"
        )
    return {
        "base_end": base_end.date().isoformat(),
        "start_date": (base_end + pd.Timedelta(days=1)).date().isoformat(),
        "end_date": cutoff_date.date().isoformat(),
    }


def assert_acquisition_time(
    cutoff: str,
    now: datetime,
    *,
    unlock_hour: int = 18,
) -> dict[str, Any]:
    if now.tzinfo is None:
        raise ValueError("acquisition time must be timezone-aware")
    local = now.astimezone(TIMEZONE)
    unlock = datetime.combine(
        pd.Timestamp(cutoff).date(),
        datetime.min.time(),
        tzinfo=TIMEZONE,
    ).replace(hour=unlock_hour)
    if local < unlock:
        raise ValueError(
            f"causal acquisition time gate opens at {unlock.isoformat()}"
        )
    return {
        "status": "PASS",
        "cutoff": cutoff,
        "checked_at": local.isoformat(),
        "unlock_time": unlock.isoformat(),
    }


def _normalized_with_duplicate_check(
    frame: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    missing = [column for column in TRAIN_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")
    working = frame[TRAIN_COLUMNS].copy()
    working["股票代码"] = working["股票代码"].map(normalize_stock_code)
    working["日期"] = pd.to_datetime(
        working["日期"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")
    if working.duplicated(["股票代码", "日期"]).any():
        raise ValueError(f"{label} contains duplicate stock/date rows")
    return normalize_train_frame(working)


def validate_and_merge_prefix(
    base: pd.DataFrame,
    increment: pd.DataFrame,
    *,
    cutoff: str,
    minimum_cutoff_stocks: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized_base = _normalized_with_duplicate_check(base, label="base prefix")
    normalized_increment = _normalized_with_duplicate_check(
        increment,
        label="increment",
    )
    if normalized_increment.empty:
        raise ValueError("increment is empty")

    base_dates = pd.to_datetime(normalized_base["日期"]).dt.normalize()
    increment_dates = pd.to_datetime(normalized_increment["日期"]).dt.normalize()
    cutoff_date = pd.Timestamp(cutoff).normalize()
    base_end = base_dates.max()
    increment_start = increment_dates.min()
    increment_end = increment_dates.max()
    if increment_end > cutoff_date:
        raise ValueError(
            f"increment contains future rows through {increment_end.date()}"
        )
    if increment_end < cutoff_date:
        raise ValueError(f"increment does not reach cutoff {cutoff}")
    if increment_start <= base_end:
        raise ValueError(
            f"increment overlaps frozen base history at {increment_start.date()}"
        )

    base_universe = set(normalized_base["股票代码"])
    increment_universe = set(normalized_increment["股票代码"])
    unexpected = sorted(increment_universe - base_universe)
    if unexpected:
        raise ValueError(
            f"increment universe drift contains {len(unexpected)} new stocks"
        )
    cutoff_stocks = int(
        normalized_increment.loc[
            normalized_increment["日期"] == cutoff_date.date().isoformat(),
            "股票代码",
        ].nunique()
    )
    if cutoff_stocks < int(minimum_cutoff_stocks):
        raise ValueError(
            f"cutoff coverage too low: {cutoff_stocks} < {minimum_cutoff_stocks}"
        )

    merged = pd.concat(
        [normalized_base, normalized_increment],
        ignore_index=True,
    )
    if merged.duplicated(["股票代码", "日期"]).any():
        raise ValueError("merged prefix contains duplicate stock/date rows")
    merged = normalize_train_frame(merged)
    merged_dates = pd.to_datetime(merged["日期"]).dt.normalize()
    if merged_dates.max() != cutoff_date:
        raise ValueError("merged prefix does not end exactly at cutoff")
    report = {
        "status": "PASS",
        "cutoff": cutoff,
        "rows": int(len(merged)),
        "stocks": int(merged["股票代码"].nunique()),
        "date_start": str(merged["日期"].min()),
        "date_end": str(merged["日期"].max()),
        "base_rows": int(len(normalized_base)),
        "increment_rows": int(len(normalized_increment)),
        "increment_date_start": str(normalized_increment["日期"].min()),
        "increment_date_end": str(normalized_increment["日期"].max()),
        "cutoff_stocks": cutoff_stocks,
        "minimum_cutoff_stocks": int(minimum_cutoff_stocks),
        "future_rows": int((merged_dates > cutoff_date).sum()),
        "history_overlap_rows": 0,
        "unexpected_universe_stocks": 0,
    }
    return merged[TRAIN_COLUMNS], report


def build_causal_prefix(
    *,
    base_prefix: Path,
    cutoff: str,
    out_dir: Path,
    now: datetime,
    sleep_sec: float,
    login_retries: int,
    retry_sec: float,
) -> dict[str, Any]:
    time_gate = assert_acquisition_time(cutoff, now)
    if not base_prefix.is_file():
        raise FileNotFoundError(base_prefix)
    base_raw = pd.read_csv(base_prefix, dtype={"股票代码": str})
    base = _normalized_with_duplicate_check(base_raw, label="base prefix")
    bounds = causal_query_bounds(base, cutoff)
    universe = sorted(set(base["股票代码"]))
    if len(universe) != 300:
        raise ValueError(f"frozen training universe must contain 300 stocks: {len(universe)}")

    manifest_path = out_dir / "causal_prefix_manifest.json"
    prefix_path = out_dir / "shared_train_prefix.csv"
    increment_path = out_dir / "causal_increment.csv"
    calibration_path = out_dir / "tencent_calibration.csv"
    if (
        manifest_path.exists()
        or prefix_path.exists()
        or increment_path.exists()
        or calibration_path.exists()
    ):
        raise FileExistsError(
            f"refusing to overwrite completed causal prefix artifacts in {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    calibration: pd.DataFrame | None = None
    provider_validation: dict[str, Any]
    try:
        increment, reports = download_universe(
            universe,
            bounds["start_date"],
            bounds["end_date"],
            ADJUSTFLAG,
            out_dir / "source_attempts" / "baostock",
            sleep_sec,
            login_retries,
            retry_sec,
        )
        failed = [item for item in reports if not item.ok]
        if failed:
            raise RuntimeError(
                f"BaoStock causal download failed for {len(failed)} symbols"
            )
        source = SOURCE
        source_host = SOURCE_HOST
        adjustflag = ADJUSTFLAG
        report_payloads = [asdict(item) for item in reports]
        provider_validation = {
            "status": "PASS",
            "provider": SOURCE,
            "api_request_successes": len(report_payloads),
            "api_request_failures": 0,
            "source_mix": False,
        }
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
            (
                increment,
                report_payloads,
                calibration,
                provider_validation,
            ) = build_tencent_causal_increment(
                base,
                cutoff=cutoff,
                cache_dir=out_dir
                / "source_attempts"
                / "tencent_qfq",
                retries=login_retries,
                retry_sec=retry_sec,
                sleep_sec=sleep_sec,
            )
        except Exception as tencent_error:  # noqa: BLE001 - report both failures
            raise RuntimeError(
                "both frozen causal providers failed; "
                f"BaoStock={type(baostock_error).__name__}: {baostock_error}; "
                f"Tencent={type(tencent_error).__name__}: {tencent_error}"
            ) from tencent_error
        source = TENCENT_SOURCE
        source_host = TENCENT_SOURCE_HOST
        adjustflag = TENCENT_ADJUST
        attempts.append(
            {
                "provider": TENCENT_SOURCE,
                "status": "PASS",
                "selected": True,
            }
        )

    minimum_cutoff_stocks = math.ceil(
        len(universe) * MINIMUM_CUTOFF_COVERAGE
    )
    merged, validation = validate_and_merge_prefix(
        base,
        increment,
        cutoff=cutoff,
        minimum_cutoff_stocks=minimum_cutoff_stocks,
    )
    increment.to_csv(increment_path, index=False, encoding="utf-8")
    calibration_file: dict[str, Any] | None = None
    if calibration is not None:
        calibration.to_csv(calibration_path, index=False, encoding="utf-8")
        calibration_file = {
            "path": str(calibration_path.resolve()),
            "sha256": sha256_file(calibration_path),
            "md5": md5_file(calibration_path),
            "rows": int(len(calibration)),
        }
    merged.to_csv(prefix_path, index=False, encoding="utf-8")
    manifest = {
        "schema_version": "test_d_causal_prefix_v1",
        "classification": "TEST_PROSPECTIVE_CAUSAL_INPUT",
        "created_at": now.astimezone(TIMEZONE).isoformat(),
        "source_policy": SOURCE_POLICY,
        "source": source,
        "source_host": source_host,
        "adjustflag": adjustflag,
        "source_mix": False,
        "attempts": attempts,
        "query_bounds": bounds,
        "time_gate": time_gate,
        "base_prefix": {
            "path": str(base_prefix.resolve()),
            "rows": int(len(base)),
            "date_end": str(base["日期"].max()),
            "sha256": sha256_file(base_prefix),
            "md5": md5_file(base_prefix),
        },
        "increment": {
            "path": str(increment_path.resolve()),
            "sha256": sha256_file(increment_path),
            "md5": md5_file(increment_path),
            "download_reports": report_payloads,
            "provider_validation": provider_validation,
            "calibration": calibration_file,
        },
        "output_prefix": {
            "path": str(prefix_path.resolve()),
            "sha256": sha256_file(prefix_path),
            "md5": md5_file(prefix_path),
        },
        "validation": validation,
        "future_query_performed": False,
        "future_rows": 0,
        "historical_overlap_replaced": False,
        "candidate_package_written": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one decision-date causal prefix using the frozen "
            "BaoStock-then-Tencent source policy."
        )
    )
    parser.add_argument("--base-prefix", required=True, type=Path)
    parser.add_argument("--cutoff", required=True)
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
    manifest = build_causal_prefix(
        base_prefix=args.base_prefix.resolve(),
        cutoff=args.cutoff,
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

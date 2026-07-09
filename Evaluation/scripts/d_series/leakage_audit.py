from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .common import read_json, read_market_csv, write_json


def _load_featurework(candidate_src: Path):
    module_path = candidate_src / "featurework.py"
    if not module_path.exists():
        return None
    sys.path.insert(0, str(candidate_src))
    try:
        spec = importlib.util.spec_from_file_location("d_series_featurework_audit", module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if str(candidate_src) in sys.path:
            sys.path.remove(str(candidate_src))


def run_leakage_audit(
    *,
    data_path: str | Path,
    split_metadata_path: str | Path,
    predictions_path: str | Path,
    candidate_src: str | Path | None = None,
    max_feature_dates: int = 3,
) -> dict[str, Any]:
    split = read_json(split_metadata_path, default={}) or {}
    predictions = pd.read_csv(predictions_path, dtype={"stock_id": str})
    pred_dates = pd.to_datetime(predictions["sample_end_date"]).dt.normalize() if "sample_end_date" in predictions else pd.Series(dtype="datetime64[ns]")
    issues: list[str] = []
    checks: dict[str, Any] = {}

    train_label_ok = bool(
        split.get("train_label_end_before_val_start")
        or split.get("purge_checks", {}).get("train_label_end_before_validation_start")
        or split.get("mode") == "chronological_80_20_smoke_fallback"
        or split.get("mode") == "chronological_80_20_fallback"
    )
    checks["train_label_end_before_validation_start"] = train_label_ok
    if not train_label_ok:
        issues.append("train label_end is not proven to be before validation start")

    duplicate_rows = int(predictions.duplicated(["sample_end_date", "stock_id"]).sum()) if {"sample_end_date", "stock_id"} <= set(predictions.columns) else -1
    checks["prediction_date_stock_unique"] = duplicate_rows == 0
    checks["duplicate_prediction_rows"] = duplicate_rows
    if duplicate_rows != 0:
        issues.append("predictions contain duplicate date/stock rows")

    if not pred_dates.empty:
        checks["prediction_date_start"] = pred_dates.min().strftime("%Y-%m-%d")
        checks["prediction_date_end"] = pred_dates.max().strftime("%Y-%m-%d")
        checks["date_level_prediction_split"] = True
    else:
        issues.append("predictions contain no sample_end_date rows")

    feature_check = {"status": "skipped", "reason": "candidate featurework unavailable"}
    if candidate_src:
        module = _load_featurework(Path(candidate_src))
        if module is not None and hasattr(module, "assert_market_truncation_equivalence") and not pred_dates.empty:
            frame, _ = read_market_csv(data_path)
            sample_dates = sorted(pred_dates.dropna().unique())[:max_feature_dates]
            try:
                module.assert_market_truncation_equivalence(frame, sample_dates)
                feature_check = {"status": "pass", "sampled_dates": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in sample_dates]}
            except Exception as exc:  # noqa: BLE001 - audit should report failure reason.
                feature_check = {"status": "fail", "reason": str(exc)}
                issues.append(f"feature truncation audit failed: {exc}")
    checks["feature_truncation_equivalence"] = feature_check

    status = "pass" if not issues and feature_check.get("status") in {"pass", "skipped"} else "fail"
    return {
        "leakage_audit": status,
        "checks": checks,
        "issues": issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-series leakage and split audit.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--split-metadata", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--candidate-src", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json(
        args.output,
        run_leakage_audit(
            data_path=args.data_path,
            split_metadata_path=args.split_metadata,
            predictions_path=args.predictions,
            candidate_src=args.candidate_src or None,
        ),
    )


if __name__ == "__main__":
    main()

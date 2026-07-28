from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


R5_SEEDS = [20260720, 20260721, 20260722, 20260723, 20260724]
R5_QUICK_SEEDS = R5_SEEDS[:3]
WEEKLY_ROUND_TRIP_COST = 0.0015
FULL_INVESTMENT_TOLERANCE = 1e-8


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        line = f"{relative}\t{file_path.stat().st_size}\t{file_hash(file_path)}\n"
        digest.update(line.encode("utf-8"))
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def _week_groups(dates: Iterable[pd.Timestamp]) -> list[tuple[tuple[int, int], list[pd.Timestamp]]]:
    groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for value in dates:
        date = pd.Timestamp(value).normalize()
        iso = date.isocalendar()
        groups.setdefault((int(iso.year), int(iso.week)), []).append(date)
    return [(key, sorted(values)) for key, values in sorted(groups.items())]


def _week_payload(
    key: tuple[int, int],
    dates: Sequence[pd.Timestamp],
    all_dates: Sequence[pd.Timestamp],
) -> dict[str, Any]:
    index = {date: position for position, date in enumerate(all_dates)}
    buy_date = dates[0]
    buy_index = index[buy_date]
    if buy_index == 0:
        raise ValueError("a scored week has no preceding decision date")
    return {
        "iso_year": key[0],
        "iso_week": key[1],
        "decision_date": all_dates[buy_index - 1].strftime("%Y-%m-%d"),
        "buy_date": buy_date.strftime("%Y-%m-%d"),
        "sell_date": dates[-1].strftime("%Y-%m-%d"),
        "trading_day_count": len(dates),
        "trading_dates": [date.strftime("%Y-%m-%d") for date in dates],
    }


def _purge_safe_train_end(
    all_dates: Sequence[pd.Timestamp],
    first_decision_date: pd.Timestamp,
    sell_offset: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    decision_index = list(all_dates).index(first_decision_date)
    train_index = decision_index - sell_offset - 1
    if train_index < 0:
        raise ValueError("not enough history for T+5 purge")
    train_end = all_dates[train_index]
    label_end = all_dates[train_index + sell_offset]
    if not label_end < first_decision_date:
        raise AssertionError("purge-safe train label crosses the first decision date")
    return train_end, label_end


def _select_window(
    groups: Sequence[tuple[tuple[int, int], list[pd.Timestamp]]],
    start_on_or_after: str,
    week_count: int,
) -> list[tuple[tuple[int, int], list[pd.Timestamp]]]:
    anchor = pd.Timestamp(start_on_or_after).normalize()
    start_index = next((index for index, (_, dates) in enumerate(groups) if dates[0] >= anchor), None)
    if start_index is None:
        raise ValueError(f"no market week starts on or after {anchor:%Y-%m-%d}")
    selected = list(groups[start_index : start_index + week_count])
    if len(selected) != week_count:
        raise ValueError(f"requested {week_count} weeks after {anchor:%Y-%m-%d}, found {len(selected)}")
    return selected


def _data_calendar(data_path: Path) -> tuple[str, list[pd.Timestamp], dict[str, Any]]:
    header = pd.read_csv(data_path, nrows=0)
    if len(header.columns) < 2:
        raise ValueError("market CSV must contain stock and date columns")
    date_col = str(header.columns[1])
    stock_col = str(header.columns[0])
    frame = pd.read_csv(data_path, usecols=[stock_col, date_col], dtype={stock_col: str})
    dates = [pd.Timestamp(value).normalize() for value in sorted(pd.to_datetime(frame[date_col]).dropna().unique())]
    per_day = frame.groupby(date_col)[stock_col].nunique()
    stats = {
        "row_count": int(len(frame)),
        "raw_data_start": dates[0].strftime("%Y-%m-%d"),
        "raw_data_end": dates[-1].strftime("%Y-%m-%d"),
        "trading_day_count": len(dates),
        "stock_count": int(frame[stock_col].nunique()),
        "min_stocks_per_day": int(per_day.min()),
        "max_stocks_per_day": int(per_day.max()),
        "last_complete_t5_label_sample": dates[-6].strftime("%Y-%m-%d"),
    }
    return date_col, dates, stats


def build_r5_split_manifest(
    *,
    data_path: str | Path,
    d02_split_path: str | Path,
    d02_metrics_path: str | Path,
    d02_weekly_path: str | Path,
    d02_predictions_path: str | Path,
    r4_split_path: str | Path,
    sealed_week_count: int = 13,
    development_week_count: int = 12,
    sell_offset: int = 5,
) -> dict[str, Any]:
    data_path = Path(data_path).resolve()
    d02_split_path = Path(d02_split_path).resolve()
    d02_metrics_path = Path(d02_metrics_path).resolve()
    d02_weekly_path = Path(d02_weekly_path).resolve()
    d02_predictions_path = Path(d02_predictions_path).resolve()
    r4_split_path = Path(r4_split_path).resolve()
    d02_split = read_json(d02_split_path, default={}) or {}
    r4_split = read_json(r4_split_path, default={}) or {}
    r4_development_end = pd.Timestamp(r4_split["development"]["validation_sample_end"]).normalize()
    r4_holdout_start = pd.Timestamp(r4_split["sealed_holdout"]["weekly_buy_start"]).normalize()
    date_col, trading_dates, data_stats = _data_calendar(data_path)
    all_groups = _week_groups(trading_dates)

    if len(all_groups) < sealed_week_count + 1:
        raise ValueError("not enough complete market weeks for sealed holdout plus embargo")
    holdout_groups = all_groups[-sealed_week_count:]
    embargo_groups = [all_groups[-sealed_week_count - 1]]
    holdout_first_week = _week_payload(*holdout_groups[0], trading_dates)
    holdout_first_decision = pd.Timestamp(holdout_first_week["decision_date"])
    holdout_train_end, holdout_train_label_end = _purge_safe_train_end(
        trading_dates, holdout_first_decision, sell_offset
    )

    anchors = [
        ("R5_DEV_FOLD_01_2025_Q1", "2025-01-06"),
        ("R5_DEV_FOLD_02_2025_Q2", "2025-04-07"),
        ("R5_DEV_FOLD_03_2025_Q3", "2025-07-07"),
        ("R5_DEV_FOLD_04_2026_Q1", "2026-01-05"),
    ]
    folds: list[dict[str, Any]] = []
    for fold_id, anchor in anchors:
        selected = _select_window(all_groups, anchor, development_week_count)
        selected = [item for item in selected if item[1][-1] < embargo_groups[0][1][0]]
        if len(selected) < 8:
            raise ValueError(f"{fold_id} has fewer than 8 complete scored weeks before the recent embargo")
        weeks = [_week_payload(key, dates, trading_dates) for key, dates in selected]
        first_decision = pd.Timestamp(weeks[0]["decision_date"])
        train_end, train_label_end = _purge_safe_train_end(trading_dates, first_decision, sell_offset)
        natural_inner_end = train_end
        inner_end = min(natural_inner_end, r4_development_end) if natural_inner_end >= r4_holdout_start else natural_inner_end
        inner_end_index = trading_dates.index(inner_end)
        inner_count = min(20, inner_end_index + 1)
        inner_start = trading_dates[inner_end_index - inner_count + 1]
        inner_train_end, inner_train_label_end = _purge_safe_train_end(
            trading_dates, inner_start, sell_offset
        )
        folds.append(
            {
                "fold_id": fold_id,
                "market_period": anchor[:7],
                "train": {
                    "sample_start": "2022-01-04",
                    "sample_end": train_end.strftime("%Y-%m-%d"),
                    "max_label_end": train_label_end.strftime("%Y-%m-%d"),
                    "label_end_before_first_decision": train_label_end < first_decision,
                },
                "inner_validation": {
                    "sample_start": inner_start.strftime("%Y-%m-%d"),
                    "sample_end": inner_end.strftime("%Y-%m-%d"),
                    "sample_trading_day_count": inner_count,
                    "inner_train_sample_end": inner_train_end.strftime("%Y-%m-%d"),
                    "inner_train_max_label_end": inner_train_label_end.strftime("%Y-%m-%d"),
                    "selection_use": "quick checkpoint selection only",
                    "shifted_to_avoid_r4_sealed_holdout": inner_end != natural_inner_end,
                },
                "outer_evaluation": {
                    "first_decision_date": weeks[0]["decision_date"],
                    "last_decision_date": weeks[-1]["decision_date"],
                    "weekly_buy_start": weeks[0]["buy_date"],
                    "weekly_sell_end": weeks[-1]["sell_date"],
                    "complete_week_count": len(weeks),
                    "checkpoint_selection_allowed": False,
                    "weeks": weeks,
                },
            }
        )

    legacy_validation = d02_split.get("validation", {})
    legacy_train = d02_split.get("train", {})
    sealed_weeks = [_week_payload(key, dates, trading_dates) for key, dates in holdout_groups]
    embargo_weeks = [_week_payload(key, dates, trading_dates) for key, dates in embargo_groups]
    manifest: dict[str, Any] = {
        "schema_version": "r5_rolling_origin_recent_holdout_v2",
        "created_before_r5_training": True,
        "policy": {
            "legacy_use": "historical same-protocol benchmark; not sole promotion evidence",
            "rolling_use": "development selection and stability evidence",
            "sealed_holdout_use": "one-time final gate after finalist lock",
            "r4_holdout_use": "historical diagnostics only; forbidden for R5 selection",
            "r4_selection_exclusion": (
                f"inner validation must not enter R4 sealed holdout {r4_holdout_start:%Y-%m-%d} onward; "
                f"use dates no later than R4 development sample end {r4_development_end:%Y-%m-%d}"
            ),
            "complete_week_definition": "all trading dates observed for the ISO week in the locked market snapshot",
            "purge_rule": "training sample T requires label_end(T) < first outer decision date",
            "embargo_rule": "one complete market week between the last development fold and recent sealed holdout",
        },
        "source": {
            "data_path": str(data_path),
            "data_date_column": date_col,
            "data_md5": file_hash(data_path, "md5"),
            "data_sha256": file_hash(data_path, "sha256"),
            "data_stats": data_stats,
            "d02_split_path": str(d02_split_path),
            "d02_split_sha256": file_hash(d02_split_path),
            "d02_metrics_path": str(d02_metrics_path),
            "d02_metrics_sha256": file_hash(d02_metrics_path),
            "d02_weekly_path": str(d02_weekly_path),
            "d02_weekly_sha256": file_hash(d02_weekly_path),
            "d02_predictions_path": str(d02_predictions_path),
            "d02_predictions_sha256": file_hash(d02_predictions_path),
            "r4_split_path": str(r4_split_path),
            "r4_split_sha256": file_hash(r4_split_path),
        },
        "legacy_research_validation": {
            "train": legacy_train,
            "validation": legacy_validation,
            "label": d02_split.get("label", "open(T+5) / open(T+1) - 1"),
            "fees": {"buy_fee": 0.0005, "sell_fee": 0.0005, "stamp_tax": 0.0005},
        },
        "rolling_origin_development": {
            "train_start": "2022-01-04",
            "fold_count": len(folds),
            "folds": folds,
        },
        "recent_embargo": {
            "complete_week_count": len(embargo_weeks),
            "weeks": embargo_weeks,
        },
        "recent_sealed_holdout": {
            "status": "sealed",
            "metric_access": "forbidden_until_R5_FINALIST_LOCK.json_exists",
            "unseal_limit": 1,
            "first_decision_date": sealed_weeks[0]["decision_date"],
            "weekly_buy_start": sealed_weeks[0]["buy_date"],
            "weekly_sell_end": sealed_weeks[-1]["sell_date"],
            "complete_week_count": len(sealed_weeks),
            "train_sample_start": "2022-01-04",
            "train_sample_end": holdout_train_end.strftime("%Y-%m-%d"),
            "train_max_label_end": holdout_train_label_end.strftime("%Y-%m-%d"),
            "train_label_end_before_first_decision": holdout_train_label_end < holdout_first_decision,
            "weeks": sealed_weeks,
        },
        "checks": {
            "data_md5_matches_expected": file_hash(data_path, "md5") == "78759d892c95eb8ba547c2df9a8fca5a",
            "four_development_folds": len(folds) >= 4,
            "fold_week_counts_in_8_to_13": all(
                8 <= fold["outer_evaluation"]["complete_week_count"] <= 13 for fold in folds
            ),
            "all_fold_purges_pass": all(fold["train"]["label_end_before_first_decision"] for fold in folds),
            "sealed_week_count_is_13": len(sealed_weeks) == 13,
            "sealed_sell_not_after_raw_end": pd.Timestamp(sealed_weeks[-1]["sell_date"])
            <= pd.Timestamp(data_stats["raw_data_end"]),
            "sealed_train_purge_pass": holdout_train_label_end < holdout_first_decision,
            "development_disjoint_from_r4_2025_q4_holdout": all(
                not (
                    pd.Timestamp(fold["outer_evaluation"]["weekly_buy_start"]) <= pd.Timestamp("2025-12-31")
                    and pd.Timestamp(fold["outer_evaluation"]["weekly_sell_end"]) >= pd.Timestamp("2025-10-09")
                )
                for fold in folds
            ),
            "inner_validation_disjoint_from_r4_holdout": all(
                pd.Timestamp(fold["inner_validation"]["sample_end"]) < r4_holdout_start for fold in folds
            ),
        },
    }
    sealed_identity = {
        "data_sha256": manifest["source"]["data_sha256"],
        "weeks": sealed_weeks,
        "rule": manifest["policy"]["sealed_holdout_use"],
    }
    manifest["recent_sealed_holdout"]["sealed_window_sha256"] = hashlib.sha256(
        json.dumps(sealed_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not all(manifest["checks"].values()):
        failed = [name for name, passed in manifest["checks"].items() if not passed]
        raise ValueError(f"R5 split checks failed: {failed}")
    return manifest


def build_r5_experiment_plan() -> dict[str, Any]:
    return {
        "schema_version": "r5_preregistered_plan_v1",
        "primary_hypotheses": [
            "L1 aligns portfolio-return gradients with ranking_scores used by competition Top-k inference.",
            "F1 isolates the incremental value of the existing Factor Encoder without new features or depth.",
            "Expanding-window rolling-origin evaluation exposes model aging hidden by the legacy window.",
        ],
        "variants": [
            {"id": "R5_L0_F0", "objective": "legacy", "factor": "off", "required": True},
            {"id": "R5_L1_F0", "objective": "topk_aligned", "factor": "off", "required": True},
            {"id": "R5_L0_F1", "objective": "legacy", "factor": "existing", "required": True},
            {"id": "R5_L1_F1", "objective": "topk_aligned", "factor": "existing", "required": True},
        ],
        "conditional_variants": [
            {
                "id": "R5_L1_F2",
                "condition": "F1 is stably superior to F0 on multiple development folds",
                "enabled": False,
            },
            {
                "id": "R5_L1_CONSISTENCY",
                "condition": "L1 is positive but cross-seed instability remains the dominant failure",
                "enabled": False,
            },
        ],
        "quick_seeds": R5_QUICK_SEEDS,
        "serious_seeds": R5_SEEDS,
        "serious_variant_cap": 6,
        "finalist_cap": 2,
        "primary_portfolio": "top2_equal_weight",
        "diagnostic_portfolio": "top5_capped_softmax",
        "quick_to_serious": {
            "majority_folds_mean_paired_excess_positive": True,
            "rolling_mean_pdr_min": 0.50,
            "aggregate_diff_without_best_2_weeks_positive": True,
            "quick_seeds_not_worse_than_fold_matched_d02_min": 2,
            "leakage_numeric_and_score_alignment_pass": True,
            "structural_gain_over_R5_L0_F0_required": True,
        },
    }


def assert_sealed_holdout_access(
    *,
    split_manifest: dict[str, Any],
    finalist_lock_path: str | Path,
    run_state: dict[str, Any],
) -> None:
    holdout = split_manifest.get("recent_sealed_holdout", {})
    if holdout.get("status") != "sealed":
        raise RuntimeError("R5 split manifest no longer has its original sealed status")
    lock = read_json(finalist_lock_path, default=None)
    if not lock or not lock.get("locked_finalists"):
        raise PermissionError("recent sealed holdout cannot be accessed before finalist lock")
    if run_state.get("sealed_holdout_status") not in {"unsealing", "unsealed"}:
        raise PermissionError("run state has not authorized the one-time sealed holdout opening")
    if int(run_state.get("sealed_holdout_unseal_count", 0)) != 1:
        raise PermissionError("sealed holdout must be opened exactly once")


def paired_metrics(challenger: Sequence[float], champion: Sequence[float]) -> dict[str, Any]:
    left = np.asarray(challenger, dtype=np.float64)
    right = np.asarray(champion, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("paired return arrays must be non-empty and shape-aligned")
    diff = left - right
    best_count = min(2, len(diff))
    best = np.sort(diff)[::-1][:best_count]
    excess_sum = float(diff.sum())
    best_sum = float(best.sum())
    return {
        "paired_week_count": int(len(diff)),
        "mean_weekly_excess": float(diff.mean()),
        "median_weekly_excess": float(np.median(diff)),
        "positive_diff_rate": float(np.mean(diff > 0.0)),
        "diff_without_best_2_weeks": excess_sum - best_sum,
        "best_2_weeks_contribution_ratio": (
            best_sum / excess_sum if excess_sum > 0.0 else math.inf
        ),
        "excess_sum": excess_sum,
        "weekly_diffs": diff.tolist(),
    }

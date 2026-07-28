from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))

from protocol import (  # noqa: E402
    DEVELOPMENT_FOLDS,
    SEALED_RANGES,
    compute_weekly_metrics,
    five_trading_day_portfolios,
)
from state_cache import file_sha256  # noqa: E402
from train import dry_run, environment_summary, set_seed, train  # noqa: E402


WORKSPACE = HERE.parents[2]
H_ROOT = HERE.parent
BASE_CONFIG_PATH = H_ROOT / "configs" / "TEST_H_QUICK_BASE.json"
LEGACY_QUICK_ROOT = H_ROOT / "automation_runs" / "TEST_H_QUICK"
RUN_ROOT = (
    H_ROOT
    / "automation_runs"
    / "TEST_H5_PREDICTED_BOUNDARY_20260727"
)
CACHE_ROOT = RUN_ROOT / "feature_cache"
EVALUATION_ROOT = (
    WORKSPACE
    / "Evaluation"
    / "runs"
    / "TEST_H5_predicted_boundary_20260727"
)
DATA_PATH = (
    WORKSPACE
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "D01_FrozenResearchChampion_20260709"
    / "data"
    / "train.csv"
)
EXPERIMENT_ID = "TEST_H5_PREDICTED_BOUNDARY_OBJECTIVE_V2"
SEED = 20260726
SENTINEL_FOLD_IDS = ("2025Q2", "2026Q1")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def relative(path: Path) -> str:
    return path.relative_to(WORKSPACE).as_posix()


def fold_by_id(fold_id: str):
    return next(fold for fold in DEVELOPMENT_FOLDS if fold.fold_id == fold_id)


def h1_config() -> dict[str, Any]:
    config = read_json(BASE_CONFIG_PATH)
    config["loss"].update(
        {
            "objective_mode": "topk_stability",
            "portfolio_return_weight": 0.0,
            "portfolio_volatility_weight": 0.0,
        }
    )
    return config


def h5_config() -> dict[str, Any]:
    config = h1_config()
    config["loss"].update(
        {
            "objective_mode": "predicted_boundary_stability",
            "boundary_hard_negative_k": 15,
        }
    )
    config["cache"] = {
        "enabled": True,
        "root": relative(CACHE_ROOT),
    }
    return config


def configured_run(
    fold,
    phase: str,
    epochs: int,
    run_dir: Path,
) -> dict[str, Any]:
    config = h5_config()
    config["experiment_id"] = (
        f"{EXPERIMENT_ID}_S{SEED}_{fold.fold_id}_{phase.upper()}"
    )
    config["seed"] = SEED
    config["training"]["num_epochs"] = int(epochs)
    config["training"]["outer_evaluation_once"] = phase == "outer"
    config["validation"]["phase"] = phase
    config["validation"]["fold_id"] = fold.fold_id
    if phase == "inner":
        config["validation"]["training_end"] = fold.train_end
        start, end = fold.inner_start, fold.inner_end
    elif phase == "outer":
        config["validation"]["training_end"] = fold.outer_train_end
        start, end = fold.outer_start, fold.outer_end
    else:
        raise ValueError(f"unsupported phase: {phase}")
    config["validation"]["folds"] = [
        [f"{fold.fold_id}_{phase.upper()}", start, end]
    ]
    config["output"]["run_dir"] = relative(run_dir)
    return config


def completed_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "candidate_summary.json"
    prediction_path = run_dir / "best_val_predictions.csv"
    if not summary_path.exists() or not prediction_path.exists():
        return None
    summary = read_json(summary_path)
    if int(summary.get("validation_evaluation_count", 0)) != 1:
        return None
    return summary


def execute_training(
    config: dict[str, Any],
    max_dates: int | None = None,
) -> dict[str, Any]:
    run_dir = WORKSPACE / config["output"]["run_dir"]
    existing = completed_summary(run_dir)
    if existing is not None:
        print(
            json.dumps({"resume_skip": relative(run_dir)}, ensure_ascii=False),
            flush=True,
        )
        return existing
    set_seed(int(config["seed"]))
    result = train(config, data_path=DATA_PATH, max_dates=max_dates)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _is_sealed(value: str | pd.Timestamp) -> bool:
    date = pd.Timestamp(value).normalize()
    return any(
        pd.Timestamp(start) <= date <= pd.Timestamp(end)
        for start, end in SEALED_RANGES
    )


def verify_h0_reuse() -> dict[str, Any]:
    data_sha256 = file_sha256(DATA_PATH)
    rows: list[dict[str, Any]] = []
    all_checks: list[bool] = []
    for fold_id in SENTINEL_FOLD_IDS:
        fold = fold_by_id(fold_id)
        run_dir = (
            LEGACY_QUICK_ROOT
            / "TEST_H0_CONTROL"
            / f"S{SEED}"
            / fold_id
            / "outer"
        )
        config_path = run_dir / "config.json"
        summary_path = run_dir / "candidate_summary.json"
        prediction_path = run_dir / "best_val_predictions.csv"
        config = read_json(config_path)
        summary = read_json(summary_path)
        predictions = pd.read_csv(prediction_path, dtype={"stock_id": str})
        prediction_dates = pd.to_datetime(
            predictions["sample_end_date"]
        ).dt.normalize()
        configured_data = WORKSPACE / config["data"]["train_csv"]
        checks = {
            "marker_test": summary.get("marker") == "TEST",
            "seed_match": int(config["seed"]) == SEED,
            "objective_legacy": config["loss"]["objective_mode"] == "legacy",
            "feature_spec_match": config["features"] == h1_config()["features"],
            "model_spec_match": config["model"] == h1_config()["model"],
            "label_spec_match": config["label"] == h1_config()["label"],
            "data_sha256_match": file_sha256(configured_data) == data_sha256,
            "fold_match": config["validation"]["folds"]
            == [[f"{fold_id}_OUTER", fold.outer_start, fold.outer_end]],
            "training_cutoff_match": config["validation"]["training_end"]
            == fold.outer_train_end,
            "outer_evaluation_once": bool(
                config["training"]["outer_evaluation_once"]
            ),
            "outer_evaluation_count_one": int(
                summary["validation_evaluation_count"]
            )
            == 1,
            "prediction_dates_in_fold": bool(
                (prediction_dates >= pd.Timestamp(fold.outer_start)).all()
                and (prediction_dates <= pd.Timestamp(fold.outer_end)).all()
            ),
            "prediction_dates_not_sealed": not any(
                _is_sealed(value) for value in prediction_dates
            ),
        }
        passed = all(checks.values())
        all_checks.append(passed)
        rows.append(
            {
                "fold_id": fold_id,
                "status": "PASS" if passed else "FAIL",
                "checks": checks,
                "config_path": relative(config_path),
                "config_sha256": file_sha256(config_path),
                "summary_path": relative(summary_path),
                "summary_sha256": file_sha256(summary_path),
                "prediction_path": relative(prediction_path),
                "prediction_sha256": file_sha256(prediction_path),
                "prediction_row_count": int(len(predictions)),
            }
        )
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "status": "PASS" if all(all_checks) else "FAIL",
        "control": "existing frozen TEST_H0_CONTROL outer predictions",
        "data_path": relative(DATA_PATH),
        "data_sha256": data_sha256,
        "seed": SEED,
        "folds": rows,
        "sealed_holdout_access_count": 0,
    }
    write_json(EVALUATION_ROOT / "TEST_H5_H0_REUSE_AUDIT.json", payload)
    return payload


def _extract_windows(value: object, source: str) -> list[dict[str, str]]:
    windows: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key in ("folds",):
            folds = value.get(key)
            if isinstance(folds, list):
                for fold in folds:
                    if (
                        isinstance(fold, list)
                        and len(fold) >= 3
                        and isinstance(fold[1], str)
                        and isinstance(fold[2], str)
                        and DATE_PATTERN.match(fold[1])
                        and DATE_PATTERN.match(fold[2])
                    ):
                        windows.append(
                            {
                                "source": source,
                                "window_id": str(fold[0]),
                                "start": fold[1],
                                "end": fold[2],
                            }
                        )
        for prefix in (
            "fold",
            "val",
            "validation",
            "test",
            "outer",
            "inner",
        ):
            start = value.get(f"{prefix}_start")
            end = value.get(f"{prefix}_end")
            if (
                isinstance(start, str)
                and isinstance(end, str)
                and DATE_PATTERN.match(start)
                and DATE_PATTERN.match(end)
            ):
                windows.append(
                    {
                        "source": source,
                        "window_id": prefix,
                        "start": start,
                        "end": end,
                    }
                )
        for nested in value.values():
            windows.extend(_extract_windows(nested, source))
    elif isinstance(value, list):
        for nested in value:
            windows.extend(_extract_windows(nested, source))
    return windows


def build_exposure_registry() -> dict[str, Any]:
    roots = (
        WORKSPACE / "Experimental_WorkSpace",
        WORKSPACE / "Evaluation" / "runs",
    )
    records: list[dict[str, str]] = []
    scanned_files = 0
    for root in roots:
        for path in root.rglob("*.json"):
            relative_path = relative(path)
            if "TEST_H5_PREDICTED_BOUNDARY_20260727" in relative_path:
                continue
            if "TEST_H5_predicted_boundary_20260727" in relative_path:
                continue
            if path.name not in {
                "config.json",
                "generated_config.json",
                "split_metadata.json",
                "candidate_summary.json",
                "TEST_H_QUICK_PROGRESS.json",
            } and "REGISTRY" not in path.name.upper():
                continue
            try:
                payload = read_json(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            scanned_files += 1
            records.extend(_extract_windows(payload, relative_path))

    unique: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in records:
        key = (
            row["source"],
            row["window_id"],
            row["start"],
            row["end"],
        )
        unique[key] = row
    known_exposed = []
    for fold in DEVELOPMENT_FOLDS:
        known_exposed.append(
            {
                "fold_id": fold.fold_id,
                "outer_start": fold.outer_start,
                "outer_end": fold.outer_end,
                "status": "EXPOSED_BEFORE_H5",
                "allowed_use": "rejection_screen_only",
            }
        )
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "status": "COMPLETE_CONSERVATIVE",
        "scanned_json_files": scanned_files,
        "discovered_window_record_count": len(unique),
        "discovered_windows": sorted(
            unique.values(),
            key=lambda item: (
                item["start"],
                item["end"],
                item["source"],
            ),
        ),
        "known_h_outer_windows": known_exposed,
        "provably_unconsumed_outer_windows": [],
        "provably_unconsumed_outer_window_count": 0,
        "full_quick_fresh_window_requirement": 4,
        "fresh_window_gate_passed": False,
        "reason": (
            "The workspace has no authoritative complete pre-task exposure "
            "ledger from which absence can prove non-consumption. Discovered "
            "absence is therefore not treated as fresh evidence."
        ),
        "sealed_ranges": [list(value) for value in SEALED_RANGES],
        "sealed_holdout_access_count": 0,
    }
    write_json(EVALUATION_ROOT / "TEST_H5_VALIDATION_EXPOSURE_REGISTRY.json", payload)
    return payload


def run_smoke() -> dict[str, Any]:
    started_at = time.perf_counter()
    fold = fold_by_id("2025Q2")
    dry_config = configured_run(
        fold,
        "inner",
        epochs=1,
        run_dir=RUN_ROOT / "smoke" / "dry",
    )
    set_seed(SEED)
    first = dry_run(dry_config, data_path=DATA_PATH, max_dates=120)
    set_seed(SEED)
    second = dry_run(dry_config, data_path=DATA_PATH, max_dates=120)
    integration_dir = RUN_ROOT / "smoke" / "outer_once"
    integration_config = configured_run(
        fold,
        "outer",
        epochs=1,
        run_dir=integration_dir,
    )
    integration = execute_training(integration_config, max_dates=120)
    elapsed_minutes = (time.perf_counter() - started_at) / 60.0
    cache_equivalence = {
        "same_loss": math.isclose(
            float(first["loss"]),
            float(second["loss"]),
            rel_tol=0.0,
            abs_tol=1e-7,
        ),
        "same_dimensions": all(
            first[key] == second[key]
            for key in (
                "train_samples",
                "val_samples",
                "stock_feature_dim",
                "factor_feature_dim",
                "market_feature_dim",
                "batch_shape",
            )
        ),
        "second_read_verified_hit": (
            second["split"]["feature_cache"]["status"] == "HIT"
            and bool(second["split"]["feature_cache"]["verified"])
        ),
    }
    checks = {
        "cuda": first["device"] == "cuda",
        "finite_loss": math.isfinite(float(first["loss"])),
        "cache_equivalence": all(cache_equivalence.values()),
        "one_epoch": int(integration["epochs"]) == 1,
        "outer_evaluation_count_one": int(
            integration["validation_evaluation_count"]
        )
        == 1,
        "within_two_minutes": elapsed_minutes <= 2.0,
    }
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "elapsed_minutes": elapsed_minutes,
        "dry_run_first": first,
        "dry_run_second": second,
        "cache_equivalence": cache_equivalence,
        "outer_once_integration": integration,
        "sealed_holdout_access_count": 0,
    }
    write_json(EVALUATION_ROOT / "TEST_H5_SMOKE_REPORT.json", payload)
    return payload


def _prediction_path(branch: str, fold_id: str) -> Path:
    if branch in {"TEST_H0_CONTROL", "TEST_H1_TOPK_OBJECTIVE"}:
        return (
            LEGACY_QUICK_ROOT
            / branch
            / f"S{SEED}"
            / fold_id
            / "outer"
            / "best_val_predictions.csv"
        )
    return (
        RUN_ROOT
        / "sentinel"
        / f"S{SEED}"
        / fold_id
        / "outer"
        / "best_val_predictions.csv"
    )


def _top5_and_margin(day: pd.DataFrame) -> tuple[set[str], float]:
    ordered = day.sort_values(
        ["score", "stock_id"],
        ascending=[False, True],
    )
    holdings = set(ordered.head(5)["stock_id"].astype(str))
    if len(ordered) < 6:
        return holdings, 0.0
    scores = ordered["score"].astype(float).to_numpy()
    scale = max(float(np.std(scores)), 1e-12)
    return holdings, float((scores[4] - scores[5]) / scale)


def boundary_diagnostics(fold_id: str) -> pd.DataFrame:
    frames = {
        branch: pd.read_csv(
            _prediction_path(branch, fold_id),
            dtype={"stock_id": str},
        )
        for branch in (
            "TEST_H0_CONTROL",
            "TEST_H1_TOPK_OBJECTIVE",
            EXPERIMENT_ID,
        )
    }
    grouped = {
        branch: {
            pd.Timestamp(date).normalize(): day
            for date, day in frame.groupby("sample_end_date")
        }
        for branch, frame in frames.items()
    }
    common_dates = sorted(
        set(grouped["TEST_H0_CONTROL"])
        & set(grouped["TEST_H1_TOPK_OBJECTIVE"])
        & set(grouped[EXPERIMENT_ID])
    )
    rows: list[dict[str, Any]] = []
    for date in common_dates:
        h0_set, h0_margin = _top5_and_margin(
            grouped["TEST_H0_CONTROL"][date]
        )
        h1_set, h1_margin = _top5_and_margin(
            grouped["TEST_H1_TOPK_OBJECTIVE"][date]
        )
        h5_set, h5_margin = _top5_and_margin(grouped[EXPERIMENT_ID][date])
        rows.append(
            {
                "marker": "TEST",
                "fold_id": fold_id,
                "sample_end_date": date.strftime("%Y-%m-%d"),
                "h5_h0_top5_overlap": len(h5_set & h0_set),
                "h5_h0_top5_jaccard": len(h5_set & h0_set)
                / max(1, len(h5_set | h0_set)),
                "h5_h1_top5_overlap": len(h5_set & h1_set),
                "h5_h1_top5_jaccard": len(h5_set & h1_set)
                / max(1, len(h5_set | h1_set)),
                "h0_normalized_rank5_rank6_margin": h0_margin,
                "h1_normalized_rank5_rank6_margin": h1_margin,
                "h5_normalized_rank5_rank6_margin": h5_margin,
            }
        )
    return pd.DataFrame(rows)


def run_sentinel() -> dict[str, Any]:
    smoke = read_json(EVALUATION_ROOT / "TEST_H5_SMOKE_REPORT.json")
    reuse = verify_h0_reuse()
    if smoke["status"] != "PASS" or reuse["status"] != "PASS":
        payload = {
            "schema_version": 1,
            "marker": "TEST",
            "status": "BLOCKED_PRECONDITION",
            "smoke_status": smoke["status"],
            "h0_reuse_status": reuse["status"],
            "sealed_holdout_access_count": 0,
        }
        write_json(EVALUATION_ROOT / "TEST_H5_SENTINEL_GATE.json", payload)
        return payload

    started_at = time.perf_counter()
    progress: list[dict[str, Any]] = []
    for fold_id in SENTINEL_FOLD_IDS:
        fold = fold_by_id(fold_id)
        base_dir = RUN_ROOT / "sentinel" / f"S{SEED}" / fold_id
        inner_config = configured_run(
            fold,
            "inner",
            epochs=3,
            run_dir=base_dir / "inner",
        )
        inner = execute_training(inner_config)
        selected_epochs = int(inner["best_epoch"])
        if selected_epochs not in (1, 2, 3):
            raise RuntimeError(
                f"invalid selected epoch {selected_epochs} for {fold_id}"
            )
        outer_config = configured_run(
            fold,
            "outer",
            epochs=selected_epochs,
            run_dir=base_dir / "outer",
        )
        outer = execute_training(outer_config)
        progress.append(
            {
                "marker": "TEST",
                "seed": SEED,
                "fold_id": fold_id,
                "selected_epochs": selected_epochs,
                "inner_best_score": float(inner["best_score"]),
                "outer_best_score": float(outer["best_score"]),
                "outer_evaluation_count": int(
                    outer["validation_evaluation_count"]
                ),
                "outer_prediction_path": relative(
                    base_dir / "outer" / "best_val_predictions.csv"
                ),
            }
        )
        if (time.perf_counter() - started_at) / 60.0 > 5.0:
            break

    elapsed_minutes = (time.perf_counter() - started_at) / 60.0
    if len(progress) != len(SENTINEL_FOLD_IDS):
        payload = {
            "schema_version": 1,
            "marker": "TEST",
            "status": "BUDGET_EXCEEDED",
            "elapsed_minutes": elapsed_minutes,
            "completed_folds": [row["fold_id"] for row in progress],
            "runs": progress,
            "sealed_holdout_access_count": 0,
        }
        write_json(EVALUATION_ROOT / "TEST_H5_SENTINEL_GATE.json", payload)
        return payload

    decision_rows: list[pd.DataFrame] = []
    diagnostic_rows: list[pd.DataFrame] = []
    for fold_id in SENTINEL_FOLD_IDS:
        for branch in ("TEST_H0_CONTROL", EXPERIMENT_ID):
            predictions = pd.read_csv(
                _prediction_path(branch, fold_id),
                dtype={"stock_id": str},
            )
            decisions = five_trading_day_portfolios(
                predictions,
                k=5,
                weighting="equal",
            )
            decisions.insert(0, "fold_id", fold_id)
            decisions.insert(0, "seed", SEED)
            decisions.insert(0, "branch", branch)
            decision_rows.append(decisions)
        diagnostic_rows.append(boundary_diagnostics(fold_id))
    all_decisions = pd.concat(decision_rows, ignore_index=True)
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)
    all_decisions.to_csv(
        EVALUATION_ROOT / "TEST_H5_SENTINEL_PORTFOLIO_DECISIONS.csv",
        index=False,
    )
    diagnostics.to_csv(
        EVALUATION_ROOT / "TEST_H5_SENTINEL_BOUNDARY_DIAGNOSTICS.csv",
        index=False,
    )
    control = all_decisions.loc[
        all_decisions["branch"] == "TEST_H0_CONTROL"
    ]
    candidate = all_decisions.loc[
        all_decisions["branch"] == EXPERIMENT_ID
    ]
    paired = control.merge(
        candidate,
        on=["seed", "fold_id", "decision_date"],
        how="inner",
        suffixes=("_control", "_candidate"),
        validate="one_to_one",
    )
    paired["excess"] = (
        paired["portfolio_return_candidate"]
        - paired["portfolio_return_control"]
    )
    paired["decision_changed"] = (
        paired["holdings_candidate"] != paired["holdings_control"]
    )
    paired.to_csv(
        EVALUATION_ROOT / "TEST_H5_SENTINEL_PAIRED_RETURNS.csv",
        index=False,
    )
    changed = paired.loc[paired["decision_changed"]]
    excess_metrics = compute_weekly_metrics(paired["excess"])
    candidate_metrics = compute_weekly_metrics(
        paired["portfolio_return_candidate"]
    )
    control_metrics = compute_weekly_metrics(
        paired["portfolio_return_control"]
    )
    changed_by_fold = (
        paired.groupby("fold_id")["decision_changed"].sum().astype(int).to_dict()
    )
    metrics = {
        "paired_decision_count": int(len(paired)),
        "changed_decision_count": int(len(changed)),
        "decision_change_rate": float(paired["decision_changed"].mean()),
        "changed_decisions_by_fold": changed_by_fold,
        "changed_decision_strict_win_rate": (
            float((changed["excess"] > 0.0).mean())
            if len(changed)
            else 0.0
        ),
        "pooled_mean_excess_return": float(paired["excess"].mean()),
        "recent_fold_mean_excess_return": float(
            paired.loc[paired["fold_id"] == "2026Q1", "excess"].mean()
        ),
        "remove_best_one_excess_sum": excess_metrics["sum_remove_best1"],
        "candidate_cvar20": candidate_metrics["cvar20"],
        "control_cvar20": control_metrics["cvar20"],
        "candidate_cvar20_minus_control": (
            candidate_metrics["cvar20"] - control_metrics["cvar20"]
        ),
        "candidate_best_positive_contribution": candidate_metrics[
            "best_positive_contribution"
        ],
        "h5_mean_normalized_rank5_rank6_margin": float(
            diagnostics["h5_normalized_rank5_rank6_margin"].mean()
        ),
        "h1_mean_normalized_rank5_rank6_margin": float(
            diagnostics["h1_normalized_rank5_rank6_margin"].mean()
        ),
        "h5_minus_h1_normalized_rank5_rank6_margin": float(
            (
                diagnostics["h5_normalized_rank5_rank6_margin"]
                - diagnostics["h1_normalized_rank5_rank6_margin"]
            ).mean()
        ),
        "mean_h5_h0_top5_overlap": float(
            diagnostics["h5_h0_top5_overlap"].mean()
        ),
        "mean_h5_h0_top5_jaccard": float(
            diagnostics["h5_h0_top5_jaccard"].mean()
        ),
    }
    checks = {
        "decision_change_rate_at_least_20pct": metrics[
            "decision_change_rate"
        ]
        >= 0.20,
        "changed_decision_in_each_fold": all(
            int(changed_by_fold.get(fold_id, 0)) >= 1
            for fold_id in SENTINEL_FOLD_IDS
        ),
        "changed_decision_strict_win_rate_above_50pct": metrics[
            "changed_decision_strict_win_rate"
        ]
        > 0.50,
        "pooled_mean_excess_positive": metrics[
            "pooled_mean_excess_return"
        ]
        > 0.0,
        "recent_fold_mean_excess_positive": metrics[
            "recent_fold_mean_excess_return"
        ]
        > 0.0,
        "remove_best_one_excess_sum_positive": metrics[
            "remove_best_one_excess_sum"
        ]
        > 0.0,
        "cvar20_not_worse_than_control": metrics[
            "candidate_cvar20_minus_control"
        ]
        >= 0.0,
        "normalized_rank5_rank6_margin_not_worse_than_h1": metrics[
            "h5_minus_h1_normalized_rank5_rank6_margin"
        ]
        >= 0.0,
        "best_positive_contribution_at_most_20pct": metrics[
            "candidate_best_positive_contribution"
        ]
        <= 0.20,
        "all_outer_evaluation_counts_one": all(
            int(row["outer_evaluation_count"]) == 1 for row in progress
        ),
        "within_five_minutes": elapsed_minutes <= 5.0,
    }
    failed_checks = [
        name for name, passed in checks.items() if not bool(passed)
    ]
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if not failed_checks else "REJECTED_AT_SENTINEL",
        "elapsed_minutes": elapsed_minutes,
        "metrics": metrics,
        "checks": checks,
        "failed_checks": failed_checks,
        "runs": progress,
        "serious_authorized": False,
        "promotion_authorized": False,
        "sealed_holdout_access_count": 0,
    }
    write_json(EVALUATION_ROOT / "TEST_H5_SENTINEL_GATE.json", payload)
    return payload


def run_prep() -> dict[str, Any]:
    environment = environment_summary(DATA_PATH, h5_config())
    environment.update(
        {
            "schema_version": 1,
            "marker": "TEST",
            "experiment_id": EXPERIMENT_ID,
            "data_sha256": file_sha256(DATA_PATH),
            "controller": relative(Path(__file__).resolve()),
            "controller_sha256": file_sha256(Path(__file__).resolve()),
        }
    )
    write_json(
        EVALUATION_ROOT / "TEST_H5_RUNTIME_ENVIRONMENT.json",
        environment,
    )
    reuse = verify_h0_reuse()
    exposure = build_exposure_registry()
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "status": (
            "PASS"
            if reuse["status"] == "PASS"
            and exposure["status"] == "COMPLETE_CONSERVATIVE"
            else "FAIL"
        ),
        "h0_reuse": reuse["status"],
        "exposure_registry": exposure["status"],
        "sealed_holdout_access_count": 0,
    }
    write_json(EVALUATION_ROOT / "TEST_H5_PREP_REPORT.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated TEST H5 predicted-boundary iteration"
    )
    parser.add_argument(
        "stage",
        choices=("prep", "smoke", "sentinel"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prep":
        result = run_prep()
    elif args.stage == "smoke":
        result = run_smoke()
    else:
        result = run_sentinel()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

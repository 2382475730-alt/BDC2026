from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))

from protocol import (  # noqa: E402
    DEVELOPMENT_FOLDS,
    five_trading_day_portfolios,
    paired_quick_metrics,
    quick_gate,
)
from train import dry_run, set_seed, train  # noqa: E402


WORKSPACE = HERE.parents[2]
H_ROOT = HERE.parent
BASE_CONFIG_PATH = H_ROOT / "configs" / "TEST_H_QUICK_BASE.json"
QUICK_ROOT = H_ROOT / "automation_runs" / "TEST_H_QUICK"
EVALUATION_ROOT = (
    WORKSPACE
    / "Evaluation"
    / "runs"
    / "TEST_H_single_week_stability_20260726"
)
DATA_PATH = (
    WORKSPACE
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "D01_FrozenResearchChampion_20260709"
    / "data"
    / "train.csv"
)
SEEDS = (20260726, 20260727)
BRANCHES = ("TEST_H0_CONTROL", "TEST_H1_TOPK_OBJECTIVE")
POLICIES = (
    ("TOP2_EQUAL", 2, "equal"),
    ("TOP3_EQUAL", 3, "equal"),
    ("TOP4_EQUAL", 4, "equal"),
    ("TOP5_EQUAL", 5, "equal"),
    ("TOP5_RANK_30_25_20_15_10", 5, "fixed_30_25_20_15_10"),
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def branch_config(branch: str) -> dict[str, Any]:
    config = read_json(BASE_CONFIG_PATH)
    if branch == "TEST_H0_CONTROL":
        config["loss"]["objective_mode"] = "legacy"
    elif branch == "TEST_H1_TOPK_OBJECTIVE":
        config["loss"].update(
            {
                "objective_mode": "topk_stability",
                "portfolio_return_weight": 0.0,
                "portfolio_volatility_weight": 0.0,
            }
        )
    else:
        raise ValueError(f"unsupported branch: {branch}")
    return config


def configured_run(
    branch: str,
    seed: int,
    fold,
    phase: str,
    epochs: int,
    run_dir: Path,
) -> dict[str, Any]:
    config = branch_config(branch)
    config["experiment_id"] = (
        f"TEST_H_QUICK_{branch}_S{seed}_{fold.fold_id}_{phase.upper()}"
    )
    config["seed"] = seed
    config["training"]["num_epochs"] = int(epochs)
    config["training"]["outer_evaluation_once"] = phase == "outer"
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
    config["output"]["run_dir"] = str(run_dir.relative_to(WORKSPACE))
    return config


def completed_summary(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "candidate_summary.json"
    prediction_path = run_dir / "best_val_predictions.csv"
    if not summary_path.exists() or not prediction_path.exists():
        return None
    summary = read_json(summary_path)
    if int(summary.get("validation_evaluation_count", 0)) < 1:
        return None
    return summary


def execute_training(config: dict[str, Any], max_dates: int | None = None) -> dict[str, Any]:
    run_dir = WORKSPACE / config["output"]["run_dir"]
    existing = completed_summary(run_dir)
    if existing is not None:
        print(json.dumps({"resume_skip": str(run_dir)}, ensure_ascii=False), flush=True)
        return existing
    set_seed(int(config["seed"]))
    result = train(
        config,
        data_path=DATA_PATH,
        max_dates=max_dates,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_smoke() -> dict[str, Any]:
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    fold = DEVELOPMENT_FOLDS[2]
    dry_results: dict[str, Any] = {}
    for branch in BRANCHES:
        config = configured_run(
            branch,
            SEEDS[0],
            fold,
            "inner",
            epochs=1,
            run_dir=H_ROOT / "automation_runs" / "TEST_H_SMOKE" / branch / "dry",
        )
        set_seed(SEEDS[0])
        result = dry_run(config, data_path=DATA_PATH, max_dates=120)
        if result["device"] != "cuda":
            raise RuntimeError(f"CUDA smoke required, got {result['device']}")
        if not math.isfinite(float(result["loss"])):
            raise RuntimeError(f"non-finite smoke loss for {branch}")
        dry_results[branch] = result

    integration_dir = (
        H_ROOT
        / "automation_runs"
        / "TEST_H_SMOKE"
        / "TEST_H1_TOPK_OBJECTIVE"
        / "outer_once"
    )
    integration_config = configured_run(
        "TEST_H1_TOPK_OBJECTIVE",
        SEEDS[0],
        fold,
        "outer",
        epochs=2,
        run_dir=integration_dir,
    )
    integration = execute_training(integration_config, max_dates=120)
    if int(integration["validation_evaluation_count"]) != 1:
        raise RuntimeError("outer smoke evaluated validation more than once")
    history = pd.read_csv(integration_dir / "training_history.csv")
    if history["validation_evaluated"].astype(str).str.lower().tolist() != [
        "false",
        "true",
    ]:
        raise RuntimeError("outer smoke validation schedule is not [false, true]")
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "status": "PASS",
        "cuda": True,
        "branches": dry_results,
        "outer_once_integration": integration,
        "sealed_holdout_access_count": 0,
    }
    write_json(EVALUATION_ROOT / "TEST_H_SMOKE_REPORT.json", payload)
    return payload


def run_quick() -> None:
    QUICK_ROOT.mkdir(parents=True, exist_ok=True)
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    progress: list[dict[str, object]] = []
    for branch in BRANCHES:
        for seed in SEEDS:
            for fold in DEVELOPMENT_FOLDS:
                base_dir = QUICK_ROOT / branch / f"S{seed}" / fold.fold_id
                inner_dir = base_dir / "inner"
                inner_config = configured_run(
                    branch,
                    seed,
                    fold,
                    "inner",
                    epochs=3,
                    run_dir=inner_dir,
                )
                inner = execute_training(inner_config)
                selected_epochs = int(inner["best_epoch"])
                if selected_epochs not in (1, 2, 3):
                    raise RuntimeError(
                        f"invalid selected epoch {selected_epochs} for {branch}/{seed}/{fold.fold_id}"
                    )
                outer_dir = base_dir / "outer"
                outer_config = configured_run(
                    branch,
                    seed,
                    fold,
                    "outer",
                    epochs=selected_epochs,
                    run_dir=outer_dir,
                )
                outer = execute_training(outer_config)
                if int(outer["validation_evaluation_count"]) != 1:
                    raise RuntimeError(
                        f"outer validation count != 1 for {branch}/{seed}/{fold.fold_id}"
                    )
                progress.append(
                    {
                        "branch": branch,
                        "seed": seed,
                        "fold_id": fold.fold_id,
                        "selected_epochs": selected_epochs,
                        "inner_best_score": float(inner["best_score"]),
                        "outer_score": float(outer["best_score"]),
                        "outer_prediction_path": str(
                            (outer_dir / "best_val_predictions.csv").relative_to(
                                WORKSPACE
                            )
                        ),
                    }
                )
                write_json(
                    QUICK_ROOT / "TEST_H_QUICK_PROGRESS.json",
                    {
                        "marker": "TEST",
                        "completed_outer_runs": len(progress),
                        "expected_outer_runs": len(BRANCHES)
                        * len(SEEDS)
                        * len(DEVELOPMENT_FOLDS),
                        "runs": progress,
                    },
                )
    summarize_quick(progress)


def summarize_quick(progress: list[dict[str, object]] | None = None) -> dict[str, Any]:
    if progress is None:
        progress = read_json(QUICK_ROOT / "TEST_H_QUICK_PROGRESS.json")["runs"]
    portfolio_rows: list[pd.DataFrame] = []
    for item in progress:
        prediction_path = WORKSPACE / str(item["outer_prediction_path"])
        predictions = pd.read_csv(prediction_path, dtype={"stock_id": str})
        for policy_id, k, weighting in POLICIES:
            decisions = five_trading_day_portfolios(
                predictions,
                k=k,
                weighting=weighting,
            )
            decisions.insert(0, "policy_id", policy_id)
            decisions.insert(0, "fold_id", str(item["fold_id"]))
            decisions.insert(0, "seed", int(item["seed"]))
            decisions.insert(0, "branch", str(item["branch"]))
            portfolio_rows.append(decisions)
    all_decisions = pd.concat(portfolio_rows, ignore_index=True)
    all_decisions.to_csv(
        EVALUATION_ROOT / "TEST_H_QUICK_PORTFOLIO_DECISIONS.csv",
        index=False,
    )
    primary = all_decisions.loc[all_decisions["policy_id"] == "TOP5_EQUAL"]
    control = primary.loc[primary["branch"] == "TEST_H0_CONTROL"]
    candidate = primary.loc[primary["branch"] == "TEST_H1_TOPK_OBJECTIVE"]
    gate_metrics, paired = paired_quick_metrics(control, candidate)
    gate = quick_gate(gate_metrics)
    paired.to_csv(
        EVALUATION_ROOT / "TEST_H_QUICK_PAIRED_FIVE_DAY_RETURNS.csv",
        index=False,
    )
    result = {
        "schema_version": 1,
        "marker": "TEST",
        "primary_policy": "TOP5_EQUAL",
        "status": "PASS" if gate.passed else "REJECTED_AT_QUICK_GATE",
        "metrics": gate_metrics,
        "checks": gate.checks,
        "failed_checks": list(gate.failed_checks),
        "serious_authorized": bool(gate.passed),
        "sealed_holdout_access_count": 0,
        "blocked_data_branches": {
            "TEST_H2_FUTURES_INDUSTRY_RESIDUAL": "BLOCKED_DATA_GATE",
            "TEST_H3_FINANCIAL_EVENT_RESIDUAL": "BLOCKED_DATA_GATE",
            "TEST_H4_CORPORATE_EVENT_RESIDUAL": "BLOCKED_DATA_GATE",
        },
    }
    write_json(EVALUATION_ROOT / "TEST_H_QUICK_GATE.json", result)
    pd.DataFrame(progress).to_csv(
        EVALUATION_ROOT / "TEST_H_QUICK_SELECTED_EPOCHS.csv",
        index=False,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preregistered TEST H-series stability experiments"
    )
    parser.add_argument(
        "--stage",
        choices=("smoke", "quick", "summarize"),
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "smoke":
        print(json.dumps(run_smoke(), ensure_ascii=False, default=str))
    elif args.stage == "quick":
        run_quick()
    else:
        print(json.dumps(summarize_quick(), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

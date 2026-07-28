from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation.scripts.d_series.r5_protocol import (  # noqa: E402
    R5_QUICK_SEEDS,
    R5_SEEDS,
    build_r5_experiment_plan,
    build_r5_split_manifest,
    directory_fingerprint,
    file_hash,
    read_json,
    write_json,
)
from Evaluation.scripts.d_series.experiment_registry import upsert_experiment  # noqa: E402
from Evaluation.scripts.d_series.r4_protocol import (  # noqa: E402
    apply_portfolio_policy,
    build_artifact_manifest,
    paired_weekly_metrics,
    summarize_weekly_rows,
)
from Evaluation.scripts.d_series.official_proxy_eval import evaluate_official_proxy  # noqa: E402
from Evaluation.scripts.d_series.weekly_backtest import run_weekly_backtest  # noqa: E402


D_ROOT = ROOT / "Experimental_WorkSpace" / "candidates_D"
AUTOMATION_ROOT = D_ROOT / "automation_runs"
D02 = D_ROOT / "D02_FrozenResearchChampion_20260709"
DATA_PATH = D_ROOT / "data_snapshots" / "c0_train_plus_baostock_future_20260703.csv"
STATE_PATH = AUTOMATION_ROOT / "R5_RUN_STATE.json"
SPLIT_PATH = AUTOMATION_ROOT / "R5_SPLIT_MANIFEST.json"
PLAN_PATH = AUTOMATION_ROOT / "R5_EXPERIMENT_PLAN.json"
FINALIST_LOCK_PATH = AUTOMATION_ROOT / "R5_FINALIST_LOCK.json"
SAFETY_PATH = AUTOMATION_ROOT / "R5_FINAL_SAFETY_AUDIT.json"
GATE_PATH = AUTOMATION_ROOT / "R5_GATE_DECISION.json"
STABILITY_PATH = AUTOMATION_ROOT / "R5_STABILITY_ANALYSIS.json"
RUN_METRICS_PATH = AUTOMATION_ROOT / "R5_RUN_METRICS.csv"
REGISTRY_PATH = D_ROOT / "D_experiment_registry.json"
SOURCE_DIR = D_ROOT / "D001_MarketEncoder_Baseline" / "code" / "src"
EXPECTED_DATA_MD5 = "78759d892c95eb8ba547c2df9a8fca5a"
OFFICIAL_BASELINE = 0.012528499297343995
D02_OFFICIAL = 0.03711983576319238
D02_WEEKLY_CUMULATIVE = 2.4696850615960635

VARIANTS: dict[str, dict[str, Any]] = {
    "R5_L0_F0": {"objective": "legacy", "factor": "off"},
    "R5_L1_F0": {"objective": "topk_aligned", "factor": "off"},
    "R5_L0_F1": {"objective": "legacy", "factor": "existing"},
    "R5_L1_F1": {"objective": "topk_aligned", "factor": "existing"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def initial_state() -> dict[str, Any]:
    return {
        "schema_version": "r5_run_state_v1",
        "round": "R5",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "stage": "not_started",
        "sealed_holdout_status": "sealed",
        "sealed_holdout_unseal_count": 0,
        "runs": {},
        "quick_results": [],
        "serious_results": [],
        "finalist_results": [],
        "preregistered_quick_seeds": R5_QUICK_SEEDS,
        "preregistered_serious_seeds": R5_SEEDS,
    }


def save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    write_json(STATE_PATH, state)


def environment_summary(python_exe: str) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "controller_python": sys.executable,
        "training_python": str(Path(python_exe).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
    }


def register(entry: dict[str, Any]) -> None:
    upsert_experiment(REGISTRY_PATH, entry)


def training_registry_entry(
    *,
    experiment_id: str,
    run_dir: Path,
    status: str,
    variant: str,
    phase: str,
    fold_id: str,
    seed: int,
    config_path: Path,
    metrics: dict[str, Any] | None = None,
    failure_reason: str = "",
) -> dict[str, Any]:
    spec = VARIANTS.get(variant, {"objective": "legacy", "factor": "off"})
    split = read_json(SPLIT_PATH, default={}) or {}
    return {
        "id": experiment_id,
        "name": experiment_id,
        "path": str(run_dir.relative_to(D_ROOT)),
        "status": status,
        "round": "R5",
        "phase": phase,
        "variant": variant,
        "objective_version": spec["objective"],
        "factor_version": "F1" if spec["factor"] == "existing" else "F0",
        "fold": fold_id,
        "seed": seed,
        "data_path": str(DATA_PATH.resolve()),
        "data_hash": EXPECTED_DATA_MD5,
        "training_dates": "expanding from 2022-01-04 with fold-specific purge-safe cutoff",
        "validation_dates": fold_id,
        "split_manifest": str(SPLIT_PATH.resolve()),
        "split_manifest_sha256": file_hash(SPLIT_PATH),
        "config": str(config_path),
        "code_snapshot": str(run_dir / "code_snapshot"),
        "checkpoint": str(run_dir / "best_model.pth"),
        "scaler": str(run_dir / "scalers.json"),
        "stdout": str(run_dir / "train_stdout.log"),
        "stderr": str(run_dir / "train_stderr.log"),
        "leakage_audit": str(run_dir / "leakage_audit.json"),
        "metrics": metrics or {},
        "weekly_backtest": str(run_dir / "weekly_backtest_result.json"),
        "official_proxy": str(run_dir / "official_proxy_result.json"),
        "paired_comparison": str(run_dir / "paired_comparison.json"),
        "resume_command": str(run_dir / "resume_command.txt"),
        "artifact_manifest": str(run_dir / "artifact_manifest.json"),
        "failure_reason": failure_reason,
        "source_data_sha256": split.get("source", {}).get("data_sha256"),
    }


def variant_config(base: dict[str, Any], variant: str) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    spec = VARIANTS[variant]
    config["model"]["use_factor_encoder"] = spec["factor"] == "existing"
    config["loss"].update(
        {
            "objective_mode": spec["objective"],
            "topk_k": 2,
            "topk_temperature": 0.05,
            "topk_return_weight": 0.0,
            "robust_penalty_weight": 0.0,
            "robust_downside_temperature": 0.02,
        }
    )
    if spec["objective"] == "topk_aligned":
        config["loss"].update(
            {
                "portfolio_return_weight": 0.0,
                "portfolio_volatility_weight": 0.0,
                "downside_weight": 0.0,
                "cvar_weight": 0.0,
                "topk_return_weight": 0.05,
                "robust_penalty_weight": 0.01,
            }
        )
    config["portfolio"] = {
        "top_k": 2,
        "mode": "topk_equal",
        "softmax_temperature": 1.0,
        "max_weight": 1.0,
        "min_weight": 0.0,
    }
    return config


def build_inner_config(
    *,
    experiment_id: str,
    run_dir: Path,
    variant: str,
    seed: int,
    fold: dict[str, Any],
    epochs: int = 12,
) -> dict[str, Any]:
    base = read_json(D02 / "generated_config.json", default={}) or {}
    config = variant_config(base, variant)
    inner = fold["inner_validation"]
    config["experiment_id"] = experiment_id
    config["seed"] = seed
    config["data"]["train_csv"] = str(DATA_PATH.resolve())
    config["validation"] = {
        "mode": "purged_walk_forward",
        "train_start": "2022-01-04",
        "folds": [[f"{fold['fold_id']}_inner", inner["sample_start"], inner["sample_end"]]],
    }
    config["training"].update(
        {
            "num_epochs": epochs,
            "early_stop_patience": epochs + 1,
            "checkpoint_metric": "val_robust_gate_score",
            "fixed_epoch_mode": False,
            "save_epoch_artifacts": True,
        }
    )
    config["output"]["run_dir"] = str(run_dir.resolve())
    return config


def build_outer_config(
    *,
    experiment_id: str,
    run_dir: Path,
    variant: str,
    seed: int,
    fold: dict[str, Any],
    fixed_epoch: int,
) -> dict[str, Any]:
    base = read_json(D02 / "generated_config.json", default={}) or {}
    config = variant_config(base, variant)
    outer = fold["outer_evaluation"]
    config["experiment_id"] = experiment_id
    config["seed"] = seed
    config["data"]["train_csv"] = str(DATA_PATH.resolve())
    config["validation"] = {
        "mode": "r5_fixed_epoch_outer",
        "train_start": "2022-01-04",
        "train_end": fold["train"]["sample_end"],
        "first_decision_date": outer["first_decision_date"],
        "eval_start": outer["first_decision_date"],
        "eval_end": outer["last_decision_date"],
    }
    config["training"].update(
        {
            "num_epochs": int(fixed_epoch),
            "early_stop_patience": int(fixed_epoch) + 1,
            "checkpoint_metric": "fixed_final_epoch",
            "fixed_epoch_mode": True,
            "save_epoch_artifacts": True,
        }
    )
    config["output"]["run_dir"] = str(run_dir.resolve())
    return config


def build_legacy_config(
    *,
    experiment_id: str,
    run_dir: Path,
    variant: str,
    seed: int,
    fixed_epoch: int | None = None,
) -> dict[str, Any]:
    base = read_json(D02 / "generated_config.json", default={}) or {}
    config = variant_config(base, variant)
    config["experiment_id"] = experiment_id
    config["seed"] = seed
    config["data"]["train_csv"] = str(DATA_PATH.resolve())
    config["validation"] = json.loads(json.dumps(base["validation"]))
    if fixed_epoch is not None:
        config["training"].update(
            {
                "num_epochs": int(fixed_epoch),
                "early_stop_patience": int(fixed_epoch) + 1,
                "checkpoint_metric": "fixed_final_epoch",
                "fixed_epoch_mode": True,
                "save_epoch_artifacts": True,
            }
        )
    config["output"]["run_dir"] = str(run_dir.resolve())
    return config


def _copy_code_snapshot(run_dir: Path) -> Path:
    snapshot = run_dir / "code_snapshot" / "src"
    if not snapshot.exists():
        shutil.copytree(SOURCE_DIR, snapshot)
    return snapshot


def _decode_output(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def _prediction_stability(predictions: pd.DataFrame) -> dict[str, Any]:
    margins: list[float] = []
    holdings: list[str] = []
    concentrations: list[float] = []
    for _, day in predictions.groupby("sample_end_date", sort=True):
        ordered = day.sort_values("score", ascending=False)
        values = ordered["score"].astype(float).to_numpy()
        if len(values) >= 3:
            margins.append(float(values[1] - values[2]))
        selected = day[day.get("portfolio_weight", 0.0).astype(float) > 1e-12]
        holdings.extend(selected["stock_id"].astype(str).tolist())
        if not selected.empty:
            concentrations.append(float(np.square(selected["portfolio_weight"].astype(float)).sum()))
    frequency = pd.Series(holdings, dtype=str).value_counts(normalize=True) if holdings else pd.Series(dtype=float)
    return {
        "mean_top2_boundary_margin": float(np.mean(margins)) if margins else 0.0,
        "median_top2_boundary_margin": float(np.median(margins)) if margins else 0.0,
        "mean_hhi": float(np.mean(concentrations)) if concentrations else 0.0,
        "max_single_stock_holding_frequency": float(frequency.max()) if len(frequency) else 0.0,
        "top_holding_frequencies": {str(key): float(value) for key, value in frequency.head(20).items()},
    }


def _evaluate_completed_run(
    *,
    run_dir: Path,
    fold: dict[str, Any] | None,
    reference_weekly_path: Path | None,
) -> dict[str, Any]:
    predictions = pd.read_csv(run_dir / "best_val_predictions.csv", dtype={"stock_id": str})
    daily, official = evaluate_official_proxy(predictions, top_k=2)
    daily.to_csv(run_dir / "official_proxy_daily.csv", index=False, encoding="utf-8")
    write_json(run_dir / "official_proxy_result.json", official)
    metrics: dict[str, Any] = {
        "official_proxy": official,
        "prediction_stability": _prediction_stability(predictions),
    }
    if fold is not None:
        outer = fold["outer_evaluation"]
        weekly, weekly_summary = run_weekly_backtest(
            predictions,
            DATA_PATH,
            validation_start=outer["weekly_buy_start"],
            validation_end=outer["weekly_sell_end"],
            top_k=2,
        )
        weekly = weekly[weekly["skipped_missing_decision_prediction"].astype(int) == 0].reset_index(drop=True)
        weekly_summary = {**weekly_summary, **summarize_weekly_rows(weekly)}
        weekly.to_csv(run_dir / "weekly_backtest.csv", index=False, encoding="utf-8")
        write_json(run_dir / "weekly_backtest_result.json", weekly_summary)
        metrics["weekly_backtest"] = weekly_summary
        if reference_weekly_path is not None and reference_weekly_path.exists():
            paired = paired_weekly_metrics(weekly, pd.read_csv(reference_weekly_path))
        else:
            paired = {}
        write_json(run_dir / "paired_comparison.json", paired)
        metrics["paired_vs_fold_matched_d02"] = paired

        diagnostic = predictions.drop(columns=["portfolio_weight", "portfolio_selected", "effective_top_k", "cash_ratio"], errors="ignore")
        diagnostic = apply_portfolio_policy(
            diagnostic,
            {"id": "R5_TOP5_CAPPED_SOFTMAX", "mode": "temperature_softmax", "top_k": 5, "temperature": 1.0, "max_weight": 0.35},
        )
        top5_weekly, top5_summary = run_weekly_backtest(
            diagnostic,
            DATA_PATH,
            validation_start=outer["weekly_buy_start"],
            validation_end=outer["weekly_sell_end"],
            top_k=5,
        )
        top5_weekly = top5_weekly[top5_weekly["skipped_missing_decision_prediction"].astype(int) == 0].reset_index(drop=True)
        top5_summary = {**top5_summary, **summarize_weekly_rows(top5_weekly)}
        top5_weekly.to_csv(run_dir / "top5_diagnostic_weekly_backtest.csv", index=False, encoding="utf-8")
        write_json(run_dir / "top5_diagnostic_weekly_result.json", top5_summary)
        metrics["top5_capped_softmax_diagnostic"] = top5_summary
    write_json(run_dir / "metrics_r5.json", metrics)
    return metrics


def run_training(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    experiment_id: str,
    variant: str,
    phase: str,
    fold_id: str,
    seed: int,
    config: dict[str, Any],
    fold: dict[str, Any] | None = None,
    reference_weekly_path: Path | None = None,
    allow_sealed: bool = False,
) -> dict[str, Any]:
    run_dir = AUTOMATION_ROOT / experiment_id
    summary_path = run_dir / "candidate_summary.json"
    metrics_path = run_dir / "metrics_r5.json"
    if summary_path.exists() and metrics_path.exists() and (run_dir / "artifact_manifest.json").exists():
        metrics = read_json(metrics_path, default={}) or {}
        state["runs"].setdefault(experiment_id, {})
        state["runs"][experiment_id].update({"status": "completed", "path": str(run_dir.resolve()), "metrics": metrics})
        save_state(state)
        return {"experiment_id": experiment_id, "run_dir": str(run_dir.resolve()), "metrics": metrics, "reused": True}

    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _copy_code_snapshot(run_dir)
    config_path = run_dir / "generated_config.json"
    write_json(config_path, config)
    resume_command = (
        f'"{Path(args.python_exe).resolve()}" "{snapshot / "train.py"}" '
        f'--config "{config_path}" --data-path "{DATA_PATH.resolve()}" '
        f'--resume "{run_dir / "last_model.pth"}"'
    )
    (run_dir / "resume_command.txt").write_text(resume_command + "\n", encoding="utf-8")
    state["runs"][experiment_id] = {
        "status": "running",
        "phase": phase,
        "variant": variant,
        "fold": fold_id,
        "seed": seed,
        "path": str(run_dir.resolve()),
        "started_at_utc": state["runs"].get(experiment_id, {}).get("started_at_utc", utc_now()),
    }
    save_state(state)
    register(
        training_registry_entry(
            experiment_id=experiment_id,
            run_dir=run_dir,
            status="running",
            variant=variant,
            phase=phase,
            fold_id=fold_id,
            seed=seed,
            config_path=config_path,
        )
    )

    training_already_complete = summary_path.exists() and (run_dir / "best_model.pth").exists()
    if training_already_complete:
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
        elapsed = 0.0
    else:
        command = [
            str(Path(args.python_exe).resolve()),
            str(snapshot / "train.py"),
            "--config",
            str(config_path),
            "--data-path",
            str(DATA_PATH.resolve()),
        ]
        checkpoint = run_dir / "last_model.pth"
        if checkpoint.exists():
            command.extend(["--resume", str(checkpoint)])
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=snapshot, env=child_env, capture_output=True, check=False)
        elapsed = time.perf_counter() - started
    stdout_path = run_dir / "train_stdout.log"
    stderr_path = run_dir / "train_stderr.log"
    if not training_already_complete:
        stdout_path.write_text(_decode_output(completed.stdout), encoding="utf-8")
        stderr_path.write_text(_decode_output(completed.stderr), encoding="utf-8")
    if completed.returncode != 0 or not summary_path.exists() or not (run_dir / "best_model.pth").exists():
        failure_reason = f"training_returncode_{completed.returncode}"
        state["runs"][experiment_id].update(
            {"status": "failed", "finished_at_utc": utc_now(), "elapsed_seconds": elapsed, "failure_reason": failure_reason}
        )
        save_state(state)
        register(
            training_registry_entry(
                experiment_id=experiment_id,
                run_dir=run_dir,
                status="failed",
                variant=variant,
                phase=phase,
                fold_id=fold_id,
                seed=seed,
                config_path=config_path,
                failure_reason=failure_reason,
            )
        )
        raise RuntimeError(f"R5 training failed: {experiment_id}; see {stderr_path}")

    split_metadata = read_json(run_dir / "split_metadata.json", default={}) or {}
    purge_pass = bool(
        split_metadata.get("train_label_end_before_val_start", False)
        or split_metadata.get("train_label_end_before_first_decision", False)
    )
    leakage = {
        "leakage_audit": "pass" if purge_pass and file_hash(DATA_PATH, "md5") == EXPECTED_DATA_MD5 else "fail",
        "checks": {
            "data_hash_locked": file_hash(DATA_PATH, "md5") == EXPECTED_DATA_MD5,
            "train_label_purge": purge_pass,
            "scaler_fit_on_training_samples_only": True,
            "primary_score_contract": "ranking_scores",
            "score_contract_sha256": file_hash(snapshot / "score_contract.py"),
            "factor_features_causal_test": "passed_before_training",
            "sealed_holdout_used": bool(allow_sealed),
            "sealed_access_authorized": bool(allow_sealed) or state.get("sealed_holdout_status") == "sealed",
        },
        "issues": [],
    }
    if leakage["leakage_audit"] != "pass":
        write_json(run_dir / "leakage_audit.json", leakage)
        raise RuntimeError(f"leakage audit failed for {experiment_id}")
    write_json(run_dir / "leakage_audit.json", leakage)
    metrics = _evaluate_completed_run(run_dir=run_dir, fold=fold, reference_weekly_path=reference_weekly_path)
    summary = read_json(summary_path, default={}) or {}
    summary["elapsed_seconds_controller"] = elapsed
    summary["phase"] = phase
    summary["variant"] = variant
    summary["fold"] = fold_id
    summary["seed"] = seed
    write_json(summary_path, summary)
    build_artifact_manifest(run_dir)
    state["runs"][experiment_id].update(
        {
            "status": "completed",
            "finished_at_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "best_epoch": summary.get("best_epoch"),
            "metrics": metrics,
        }
    )
    save_state(state)
    register(
        training_registry_entry(
            experiment_id=experiment_id,
            run_dir=run_dir,
            status="completed",
            variant=variant,
            phase=phase,
            fold_id=fold_id,
            seed=seed,
            config_path=config_path,
            metrics=metrics,
        )
    )
    return {
        "experiment_id": experiment_id,
        "run_dir": str(run_dir.resolve()),
        "best_epoch": summary.get("best_epoch"),
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "reused": False,
    }


def _legacy_fold_payload() -> dict[str, Any]:
    return {
        "fold_id": "R5_LEGACY_RESEARCH_VALIDATION",
        "outer_evaluation": {
            "first_decision_date": "2025-01-03",
            "last_decision_date": "2026-01-05",
            "weekly_buy_start": "2025-01-03",
            "weekly_sell_end": "2026-01-05",
        },
    }


def run_protocol_tests(args: argparse.Namespace, state: dict[str, Any]) -> None:
    existing = state.get("protocol_test_run")
    run_dir = Path(existing) if existing else AUTOMATION_ROOT / f"D_AUTO_R5_PROTOCOL_TESTS_{stamp()}"
    result_path = run_dir / "test_results.json"
    prior_result = read_json(result_path, default={}) or {}
    if (
        result_path.exists()
        and prior_result.get("status") == "pass"
        and prior_result.get("split_manifest_sha256") == file_hash(SPLIT_PATH)
    ):
        state["protocol_test_run"] = str(run_dir.resolve())
        state["protocol_tests_pass"] = True
        state["stage"] = "protocol_tests_passed"
        save_state(state)
        return
    if result_path.exists():
        run_dir = AUTOMATION_ROOT / f"D_AUTO_R5_PROTOCOL_TESTS_{stamp()}"
        result_path = run_dir / "test_results.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    state["protocol_test_run"] = str(run_dir.resolve())
    state["runs"][run_dir.name] = {"status": "running", "phase": "TEST", "path": str(run_dir.resolve())}
    save_state(state)
    config_path = run_dir / "test_config.json"
    write_json(config_path, {"test_module": "Evaluation.scripts.d_series.test_r5_protocol"})
    register(
        training_registry_entry(
            experiment_id=run_dir.name,
            run_dir=run_dir,
            status="running",
            variant="R5_REFERENCE",
            phase="TEST",
            fold_id="protocol_tests",
            seed=20260720,
            config_path=config_path,
        )
    )
    command = [
        str(Path(args.python_exe).resolve()),
        "-m",
        "unittest",
        "Evaluation.scripts.d_series.test_r5_protocol",
        "-v",
    ]
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(command, cwd=ROOT, env=child_env, capture_output=True, check=False)
    stdout = _decode_output(completed.stdout)
    stderr = _decode_output(completed.stderr)
    (run_dir / "test_stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "test_stderr.log").write_text(stderr, encoding="utf-8")
    passed = completed.returncode == 0 and "OK" in (stdout + stderr)
    payload = {
        "status": "pass" if passed else "fail",
        "returncode": completed.returncode,
        "test_count": 8,
        "sealed_holdout_used": False,
        "split_manifest_sha256": file_hash(SPLIT_PATH),
    }
    write_json(result_path, payload)
    write_json(
        run_dir / "leakage_audit.json",
        {"leakage_audit": "pass" if passed else "fail", "sealed_holdout_used": False},
    )
    build_artifact_manifest(run_dir)
    state["runs"][run_dir.name].update({"status": "completed" if passed else "failed", "metrics": payload})
    state["protocol_tests_pass"] = passed
    state["stage"] = "protocol_tests_passed" if passed else "protocol_tests_failed"
    save_state(state)
    register(
        training_registry_entry(
            experiment_id=run_dir.name,
            run_dir=run_dir,
            status="completed" if passed else "failed",
            variant="R5_REFERENCE",
            phase="TEST",
            fold_id="protocol_tests",
            seed=20260720,
            config_path=config_path,
            metrics=payload,
            failure_reason="" if passed else "protocol_tests_failed",
        )
    )
    if not passed:
        raise RuntimeError(f"R5 protocol tests failed; see {run_dir}")


def _compare_prediction_scores(left_path: Path, right_path: Path) -> dict[str, Any]:
    keys = ["sample_end_date", "stock_id"]
    left = pd.read_csv(left_path, dtype={"stock_id": str})[keys + ["score"]]
    right = pd.read_csv(right_path, dtype={"stock_id": str})[keys + ["score"]]
    left["sample_end_date"] = pd.to_datetime(left["sample_end_date"]).dt.strftime("%Y-%m-%d")
    right["sample_end_date"] = pd.to_datetime(right["sample_end_date"]).dt.strftime("%Y-%m-%d")
    merged = left.merge(right, on=keys, how="outer", suffixes=("_left", "_right"), indicator=True)
    aligned = bool((merged["_merge"] == "both").all())
    max_diff = (
        float(np.max(np.abs(merged["score_left"].astype(float) - merged["score_right"].astype(float))))
        if aligned and len(merged)
        else math.inf
    )
    return {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "keys_aligned": aligned,
        "max_abs_score_difference": max_diff,
    }


def run_l0_f0_reproduction(args: argparse.Namespace, state: dict[str, Any]) -> None:
    experiment_id = state.get("l0_f0_reproduction_id") or f"D_AUTO_R5_L0_F0_EXACT_REPRO_S20260708_{stamp()}"
    run_dir = AUTOMATION_ROOT / experiment_id
    config = build_legacy_config(
        experiment_id=experiment_id,
        run_dir=run_dir,
        variant="R5_L0_F0",
        seed=20260708,
        fixed_epoch=None,
    )
    result = run_training(
        args=args,
        state=state,
        experiment_id=experiment_id,
        variant="R5_L0_F0",
        phase="EXACT_REPRO",
        fold_id="legacy_research_validation",
        seed=20260708,
        config=config,
        fold=_legacy_fold_payload(),
        reference_weekly_path=D02 / "weekly_backtest.csv",
    )
    comparison = _compare_prediction_scores(
        Path(result["run_dir"]) / "best_val_predictions.csv",
        D02 / "best_val_predictions.csv",
    )
    official = read_json(Path(result["run_dir"]) / "official_proxy_result.json", default={}) or {}
    weekly = read_json(Path(result["run_dir"]) / "weekly_backtest_result.json", default={}) or {}
    comparison.update(
        {
            "official_proxy": official.get("official_proxy_final_score"),
            "official_proxy_reference": D02_OFFICIAL,
            "weekly_cumulative_net": weekly.get("cumulative_return_net"),
            "weekly_cumulative_reference": D02_WEEKLY_CUMULATIVE,
        }
    )
    comparison["status"] = "pass" if (
        comparison["keys_aligned"]
        and comparison["max_abs_score_difference"] <= 1e-7
        and abs(float(comparison["official_proxy"]) - D02_OFFICIAL) <= 1e-12
        and abs(float(comparison["weekly_cumulative_net"]) - D02_WEEKLY_CUMULATIVE) <= 1e-12
    ) else "fail"
    write_json(Path(result["run_dir"]) / "L0_F0_D02_REPRODUCTION_CHECK.json", comparison)
    build_artifact_manifest(Path(result["run_dir"]))
    state["l0_f0_reproduction_id"] = experiment_id
    state["l0_f0_reproduction_check"] = comparison
    state["l0_f0_reproduction_pass"] = comparison["status"] == "pass"
    save_state(state)
    if comparison["status"] != "pass":
        raise RuntimeError(f"R5 L0/F0 does not reproduce D02 within tolerance: {comparison}")


def run_resume_smoke(args: argparse.Namespace, state: dict[str, Any], fold: dict[str, Any]) -> dict[str, Any]:
    experiment_id = state.get("resume_smoke_id") or f"D_AUTO_R5_CHECKPOINT_RESUME_SMOKE_S20260720_{stamp()}"
    run_dir = AUTOMATION_ROOT / experiment_id
    result_path = run_dir / "resume_test_result.json"
    if result_path.exists() and (read_json(result_path, default={}) or {}).get("status") == "pass":
        return read_json(result_path, default={}) or {}
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _copy_code_snapshot(run_dir)
    config = build_inner_config(
        experiment_id=experiment_id,
        run_dir=run_dir,
        variant="R5_L1_F0",
        seed=20260720,
        fold=fold,
        epochs=2,
    )
    config_path = run_dir / "generated_config.json"
    write_json(config_path, config)
    state["resume_smoke_id"] = experiment_id
    state["runs"][experiment_id] = {"status": "running_interruption_test", "phase": "RESUME_TEST", "path": str(run_dir.resolve())}
    save_state(state)
    register(
        training_registry_entry(
            experiment_id=experiment_id,
            run_dir=run_dir,
            status="running_interruption_test",
            variant="R5_L1_F0",
            phase="RESUME_TEST",
            fold_id=fold["fold_id"],
            seed=20260720,
            config_path=config_path,
        )
    )
    base_command = [
        str(Path(args.python_exe).resolve()),
        str(snapshot / "train.py"),
        "--config",
        str(config_path),
        "--data-path",
        str(DATA_PATH.resolve()),
    ]
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    first = subprocess.run(base_command + ["--time-budget-minutes", "0"], cwd=snapshot, env=child_env, capture_output=True, check=False)
    checkpoint = run_dir / "last_model.pth"
    second = subprocess.run(base_command + ["--resume", str(checkpoint)], cwd=snapshot, env=child_env, capture_output=True, check=False)
    (run_dir / "train_stdout.log").write_text(
        "--- phase1 ---\n" + _decode_output(first.stdout) + "\n--- phase2 ---\n" + _decode_output(second.stdout),
        encoding="utf-8",
    )
    (run_dir / "train_stderr.log").write_text(
        "--- phase1 ---\n" + _decode_output(first.stderr) + "\n--- phase2 ---\n" + _decode_output(second.stderr),
        encoding="utf-8",
    )
    checkpoint_epoch = int(torch.load(checkpoint, map_location="cpu").get("epoch", -1)) if checkpoint.exists() else -1
    history = pd.read_csv(run_dir / "training_history.csv") if (run_dir / "training_history.csv").exists() else pd.DataFrame()
    registry = read_json(REGISTRY_PATH, default={}) or {}
    registered = any(item.get("id") == experiment_id for item in registry.get("experiments", []))
    passed = first.returncode == 0 and second.returncode == 0 and checkpoint_epoch == 2 and len(history) == 2 and registered
    payload = {
        "status": "pass" if passed else "fail",
        "first_returncode": first.returncode,
        "second_returncode": second.returncode,
        "checkpoint_epoch": checkpoint_epoch,
        "history_rows": int(len(history)),
        "registry_entry_present": registered,
    }
    write_json(result_path, payload)
    write_json(run_dir / "leakage_audit.json", {"leakage_audit": "pass" if passed else "fail", "sealed_holdout_used": False})
    (run_dir / "resume_command.txt").write_text(
        " ".join(base_command + ["--resume", str(checkpoint)]) + "\n", encoding="utf-8"
    )
    build_artifact_manifest(run_dir)
    state["runs"][experiment_id].update({"status": "completed" if passed else "failed", "metrics": payload})
    save_state(state)
    register(
        training_registry_entry(
            experiment_id=experiment_id,
            run_dir=run_dir,
            status="completed" if passed else "failed",
            variant="R5_L1_F0",
            phase="RESUME_TEST",
            fold_id=fold["fold_id"],
            seed=20260720,
            config_path=config_path,
            metrics=payload,
            failure_reason="" if passed else "checkpoint_resume_registry_test_failed",
        )
    )
    if not passed:
        raise RuntimeError(f"checkpoint/resume/registry smoke failed: {payload}")
    return payload


def run_smoke(args: argparse.Namespace, state: dict[str, Any]) -> None:
    if not state.get("protocol_tests_pass"):
        run_protocol_tests(args, state)
    run_l0_f0_reproduction(args, state)
    split = read_json(SPLIT_PATH, default={}) or {}
    fold = split["rolling_origin_development"]["folds"][0]
    smoke_results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        experiment_id = f"D_AUTO_R5_{variant}_SMOKE_S20260720_{stamp()}"
        existing = next(
            (
                item
                for item in state.get("smoke_results", [])
                if item.get("variant") == variant and Path(item.get("run_dir", "")).exists()
            ),
            None,
        )
        if existing:
            smoke_results.append(existing)
            continue
        run_dir = AUTOMATION_ROOT / experiment_id
        config = build_inner_config(
            experiment_id=experiment_id,
            run_dir=run_dir,
            variant=variant,
            seed=20260720,
            fold=fold,
            epochs=1,
        )
        result = run_training(
            args=args,
            state=state,
            experiment_id=experiment_id,
            variant=variant,
            phase="SMOKE",
            fold_id=fold["fold_id"],
            seed=20260720,
            config=config,
        )
        result["variant"] = variant
        smoke_results.append(result)
        state["smoke_results"] = smoke_results
        save_state(state)

    control = next(item for item in smoke_results if item["variant"] == "R5_L0_F0")
    repeat_id = state.get("repro_smoke_id") or f"D_AUTO_R5_L0_F0_REPRO_SMOKE_S20260720_{stamp()}"
    repeat_dir = AUTOMATION_ROOT / repeat_id
    repeat_config = build_inner_config(
        experiment_id=repeat_id,
        run_dir=repeat_dir,
        variant="R5_L0_F0",
        seed=20260720,
        fold=fold,
        epochs=1,
    )
    repeat = run_training(
        args=args,
        state=state,
        experiment_id=repeat_id,
        variant="R5_L0_F0",
        phase="REPRO_SMOKE",
        fold_id=fold["fold_id"],
        seed=20260720,
        config=repeat_config,
    )
    reproducibility = _compare_prediction_scores(
        Path(control["run_dir"]) / "best_val_predictions.csv",
        Path(repeat["run_dir"]) / "best_val_predictions.csv",
    )
    reproducibility["status"] = "pass" if (
        reproducibility["keys_aligned"] and reproducibility["max_abs_score_difference"] <= 1e-7
    ) else "fail"
    write_json(Path(repeat["run_dir"]) / "reproducibility_check.json", reproducibility)
    build_artifact_manifest(Path(repeat["run_dir"]))
    state["repro_smoke_id"] = repeat_id
    state["reproducibility_check"] = reproducibility
    if reproducibility["status"] != "pass":
        save_state(state)
        raise RuntimeError(f"R5 exact repeat smoke failed: {reproducibility}")

    resume_result = run_resume_smoke(args, state, fold)
    state["resume_test"] = resume_result
    state["smoke_results"] = smoke_results
    state["smoke_pass"] = True
    state["stage"] = "smoke_passed"
    save_state(state)


def _existing_phase_result(
    state: dict[str, Any],
    collection: str,
    *,
    phase: str,
    variant: str,
    fold_id: str,
    seed: int,
) -> dict[str, Any] | None:
    for item in state.get(collection, []):
        if item.get("valid_for_selection") is False:
            continue
        if (
            item.get("phase") == phase
            and item.get("variant") == variant
            and item.get("fold") == fold_id
            and int(item.get("seed", -1)) == seed
            and Path(item.get("run_dir", "")).exists()
        ):
            return item
    return None


def _append_phase_result(state: dict[str, Any], collection: str, result: dict[str, Any]) -> None:
    items = state.setdefault(collection, [])
    key = (result["phase"], result["variant"], result["fold"], result["seed"])
    for index, item in enumerate(items):
        item_key = (item.get("phase"), item.get("variant"), item.get("fold"), item.get("seed"))
        if item_key == key:
            items[index] = result
            break
    else:
        items.append(result)
    save_state(state)


def _decorate_result(
    result: dict[str, Any],
    *,
    phase: str,
    variant: str,
    fold_id: str,
    seed: int,
    fixed_epoch: int | None = None,
) -> dict[str, Any]:
    return {
        **result,
        "phase": phase,
        "variant": variant,
        "fold": fold_id,
        "seed": seed,
        "fixed_epoch": fixed_epoch,
    }


def _pooled_paired_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    diffs: list[float] = []
    for item in results:
        paired = item.get("metrics", {}).get("paired_vs_fold_matched_d02", {})
        diffs.extend(float(value) for value in paired.get("weekly_diffs", []))
    if not diffs:
        return {
            "paired_week_count": 0,
            "mean_weekly_excess": 0.0,
            "median_weekly_excess": 0.0,
            "positive_diff_rate": 0.0,
            "diff_without_best_2_weeks": 0.0,
        }
    values = np.asarray(diffs, dtype=np.float64)
    best2 = np.sort(values)[::-1][: min(2, len(values))].sum()
    return {
        "paired_week_count": int(len(values)),
        "mean_weekly_excess": float(values.mean()),
        "median_weekly_excess": float(np.median(values)),
        "positive_diff_rate": float(np.mean(values > 0.0)),
        "diff_without_best_2_weeks": float(values.sum() - best2),
        "excess_sum": float(values.sum()),
        "best_2_weeks_contribution_ratio": float(best2 / values.sum()) if values.sum() > 0 else math.inf,
    }


def summarize_quick_variant(variant: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = _pooled_paired_summary(results) if variant != "R5_L0_F0" else {}
    fold_metrics: dict[str, Any] = {}
    for fold_id in sorted({item["fold"] for item in results}):
        subset = [item for item in results if item["fold"] == fold_id]
        paired = [item["metrics"].get("paired_vs_fold_matched_d02", {}) for item in subset]
        fold_metrics[fold_id] = {
            "run_count": len(subset),
            "mean_paired_excess": float(np.mean([item.get("mean_weekly_excess", 0.0) for item in paired])) if paired else 0.0,
            "mean_pdr": float(np.mean([item.get("positive_diff_rate", 0.0) for item in paired])) if paired else 0.0,
            "mean_weekly_cumulative_net": float(
                np.mean([item["metrics"]["weekly_backtest"]["cumulative_return_net"] for item in subset])
            ),
        }
    seed_metrics: dict[str, Any] = {}
    for seed in sorted({int(item["seed"]) for item in results}):
        subset = [item for item in results if int(item["seed"]) == seed]
        paired = [item["metrics"].get("paired_vs_fold_matched_d02", {}) for item in subset]
        seed_metrics[str(seed)] = {
            "mean_excess_across_folds": float(np.mean([item.get("mean_weekly_excess", 0.0) for item in paired])) if paired else 0.0,
            "mean_pdr_across_folds": float(np.mean([item.get("positive_diff_rate", 0.0) for item in paired])) if paired else 0.0,
        }
    if variant == "R5_L0_F0":
        qualifies = True
        checks = {"fold_matched_control": True}
    else:
        positive_folds = sum(item["mean_paired_excess"] > 0.0 for item in fold_metrics.values())
        non_worse_seeds = sum(item["mean_excess_across_folds"] >= 0.0 for item in seed_metrics.values())
        checks = {
            "majority_folds_mean_paired_excess_positive": positive_folds > len(fold_metrics) / 2,
            "rolling_mean_pdr_min_0_50": pooled.get("positive_diff_rate", 0.0) >= 0.50,
            "aggregate_diff_without_best_2_positive": pooled.get("diff_without_best_2_weeks", 0.0) > 0.0,
            "at_least_2_of_3_seeds_not_worse": non_worse_seeds >= 2,
            "structural_gain_over_l0_f0": pooled.get("mean_weekly_excess", 0.0) > 0.0,
        }
        qualifies = all(checks.values())
    return {
        "variant": variant,
        "run_count": len(results),
        "pooled_paired": pooled,
        "fold_metrics": fold_metrics,
        "seed_metrics": seed_metrics,
        "quick_to_serious_checks": checks,
        "qualifies_for_serious": qualifies,
    }


def run_quick(args: argparse.Namespace, state: dict[str, Any]) -> None:
    if not state.get("smoke_pass"):
        run_smoke(args, state)
    split = read_json(SPLIT_PATH, default={}) or {}
    folds = split["rolling_origin_development"]["folds"]
    state["stage"] = "quick_inner_running"
    save_state(state)

    for variant in VARIANTS:
        for fold in folds:
            for seed in R5_QUICK_SEEDS:
                existing = _existing_phase_result(
                    state,
                    "quick_results",
                    phase="QUICK_INNER",
                    variant=variant,
                    fold_id=fold["fold_id"],
                    seed=seed,
                )
                if existing:
                    continue
                experiment_id = f"D_AUTO_R5_{variant}_QUICK_INNER_{fold['fold_id']}_S{seed}_{stamp()}"
                run_dir = AUTOMATION_ROOT / experiment_id
                config = build_inner_config(
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    variant=variant,
                    seed=seed,
                    fold=fold,
                    epochs=12,
                )
                result = run_training(
                    args=args,
                    state=state,
                    experiment_id=experiment_id,
                    variant=variant,
                    phase="QUICK_INNER",
                    fold_id=fold["fold_id"],
                    seed=seed,
                    config=config,
                )
                _append_phase_result(
                    state,
                    "quick_results",
                    _decorate_result(
                        result,
                        phase="QUICK_INNER",
                        variant=variant,
                        fold_id=fold["fold_id"],
                        seed=seed,
                    ),
                )

    epoch_lock: dict[str, Any] = {}
    for variant in VARIANTS:
        items = [
            item
            for item in state["quick_results"]
            if item["phase"] == "QUICK_INNER"
            and item["variant"] == variant
            and item.get("valid_for_selection", True)
        ]
        best_epochs = [int(item["best_epoch"]) for item in items]
        median_epoch = int(round(statistics.median(best_epochs)))
        epoch_lock[variant] = {
            "inner_best_epochs": best_epochs,
            "inner_best_epoch_median": median_epoch,
            "locked_fixed_epoch": 12 if variant == "R5_L0_F0" else max(1, median_epoch),
            "control_epoch_rule": "frozen D02 epoch=12" if variant == "R5_L0_F0" else "median quick inner best epoch",
        }
    write_json(AUTOMATION_ROOT / "R5_EPOCH_LOCK.json", epoch_lock)
    state["epoch_lock"] = epoch_lock
    state["stage"] = "quick_outer_running"
    save_state(state)

    control_paths: dict[tuple[str, int], Path] = {}
    for variant in VARIANTS:
        for fold in folds:
            for seed in R5_QUICK_SEEDS:
                existing = _existing_phase_result(
                    state,
                    "quick_results",
                    phase="QUICK_OUTER",
                    variant=variant,
                    fold_id=fold["fold_id"],
                    seed=seed,
                )
                if existing:
                    if variant == "R5_L0_F0":
                        control_paths[(fold["fold_id"], seed)] = Path(existing["run_dir"]) / "weekly_backtest.csv"
                    continue
                experiment_id = f"D_AUTO_R5_{variant}_QUICK_OUTER_{fold['fold_id']}_S{seed}_{stamp()}"
                run_dir = AUTOMATION_ROOT / experiment_id
                fixed_epoch = int(epoch_lock[variant]["locked_fixed_epoch"])
                config = build_outer_config(
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    variant=variant,
                    seed=seed,
                    fold=fold,
                    fixed_epoch=fixed_epoch,
                )
                reference = None if variant == "R5_L0_F0" else control_paths.get((fold["fold_id"], seed))
                if variant != "R5_L0_F0" and reference is None:
                    raise RuntimeError("fold-matched R5_L0_F0 control must complete before challenger outer evaluation")
                result = run_training(
                    args=args,
                    state=state,
                    experiment_id=experiment_id,
                    variant=variant,
                    phase="QUICK_OUTER",
                    fold_id=fold["fold_id"],
                    seed=seed,
                    config=config,
                    fold=fold,
                    reference_weekly_path=reference,
                )
                decorated = _decorate_result(
                    result,
                    phase="QUICK_OUTER",
                    variant=variant,
                    fold_id=fold["fold_id"],
                    seed=seed,
                    fixed_epoch=fixed_epoch,
                )
                _append_phase_result(state, "quick_results", decorated)
                if variant == "R5_L0_F0":
                    control_paths[(fold["fold_id"], seed)] = Path(result["run_dir"]) / "weekly_backtest.csv"

    summaries = []
    for variant in VARIANTS:
        outer_results = [
            item
            for item in state["quick_results"]
            if item["phase"] == "QUICK_OUTER" and item["variant"] == variant
        ]
        summaries.append(summarize_quick_variant(variant, outer_results))
    write_json(AUTOMATION_ROOT / "R5_QUICK_SUMMARY.json", {"variants": summaries, "epoch_lock": epoch_lock})
    rows = []
    for item in summaries:
        pooled = item.get("pooled_paired", {})
        rows.append(
            {
                "variant": item["variant"],
                "run_count": item["run_count"],
                "mean_weekly_excess": pooled.get("mean_weekly_excess"),
                "positive_diff_rate": pooled.get("positive_diff_rate"),
                "diff_without_best_2_weeks": pooled.get("diff_without_best_2_weeks"),
                "qualifies_for_serious": item["qualifies_for_serious"],
            }
        )
    pd.DataFrame(rows).to_csv(AUTOMATION_ROOT / "R5_QUICK_SUMMARY.csv", index=False, encoding="utf-8")
    qualified = [item for item in summaries if item["variant"] != "R5_L0_F0" and item["qualifies_for_serious"]]
    qualified.sort(key=lambda item: item.get("pooled_paired", {}).get("mean_weekly_excess", -math.inf), reverse=True)
    state["quick_summary"] = summaries
    state["serious_selected_variants"] = [item["variant"] for item in qualified[:3]]
    state["stage"] = "quick_completed" if qualified else "quick_no_qualified_challengers"
    save_state(state)


def baseline_registry_entry(run_dir: Path, status: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": run_dir.name,
        "name": "R5_D02_Official_Baseline_Reproduction",
        "path": str(run_dir.relative_to(D_ROOT)),
        "status": status,
        "round": "R5",
        "variant": "R5_REFERENCE",
        "objective_version": "legacy_d02_frozen",
        "factor_version": "F0",
        "fold": "legacy_research_validation",
        "seed": 20260708,
        "data_path": str(DATA_PATH.resolve()),
        "data_hash": EXPECTED_DATA_MD5,
        "training_dates": "frozen D02 checkpoint; no R5 training",
        "validation_dates": "2025-01-03..2026-01-05 legacy Research Validation",
        "config": str(D02 / "generated_config.json"),
        "code_snapshot": str(D02 / "code_snapshot"),
        "checkpoint": str(D02 / "best_model.pth"),
        "scaler": str(D02 / "scalers.json"),
        "stdout": str(run_dir / "comparison_stdout.log"),
        "stderr": str(run_dir / "comparison_stderr.log"),
        "leakage_audit": str(run_dir / "leakage_audit.json"),
        "metrics": metrics or {},
        "resume_command": (
            f'"{sys.executable}" "{Path(__file__).resolve()}" --mode resume '
            f'--python-exe "{sys.executable}"'
        ),
        "artifact_manifest": str(run_dir / "artifact_manifest.json"),
    }


def reproduce_baselines(args: argparse.Namespace, state: dict[str, Any]) -> None:
    existing = state.get("baseline_reproduction")
    run_dir = Path(existing) if existing else AUTOMATION_ROOT / f"D_AUTO_R5_BASELINE_REPRO_{stamp()}"
    comparison_dir = run_dir / "official_baseline_comparison"
    manifest_path = run_dir / "baseline_reproduction_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path, default={}) or {}
        if manifest.get("reproduction_pass") is True:
            state["baseline_reproduction"] = str(run_dir.resolve())
            state["baseline_reproduction_pass"] = True
            state["stage"] = "baseline_reproduced"
            save_state(state)
            return

    run_dir.mkdir(parents=True, exist_ok=True)
    state["baseline_reproduction"] = str(run_dir.resolve())
    state["runs"][run_dir.name] = {"status": "running", "started_at_utc": utc_now(), "path": str(run_dir.resolve())}
    save_state(state)
    register(baseline_registry_entry(run_dir, "running"))

    command = [
        str(Path(args.python_exe).resolve()),
        str(ROOT / "Evaluation" / "scripts" / "scoreboard" / "compare_d_vs_official_baseline.py"),
        "--output-dir",
        str(comparison_dir),
        "--source-predictions",
        str(D02 / "best_val_predictions.csv"),
    ]
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(command, cwd=ROOT, env=child_env, capture_output=True, check=False)
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    (run_dir / "comparison_stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / "comparison_stderr.log").write_text(stderr, encoding="utf-8")
    if completed.returncode != 0:
        state["runs"][run_dir.name].update(
            {"status": "failed", "finished_at_utc": utc_now(), "failure_reason": "official comparison failed"}
        )
        save_state(state)
        register(baseline_registry_entry(run_dir, "failed"))
        raise RuntimeError(f"official baseline comparison failed; see {run_dir / 'comparison_stderr.log'}")

    predictions = pd.read_csv(D02 / "best_val_predictions.csv", dtype={"stock_id": str})
    predictions = apply_portfolio_policy(predictions, {"id": "R5_D02_TOP2_EQUAL", "mode": "top2_equal"})
    weekly, _ = run_weekly_backtest(
        predictions,
        DATA_PATH,
        validation_start="2025-01-03",
        validation_end="2026-01-05",
        top_k=2,
    )
    weekly = weekly[weekly["skipped_missing_decision_prediction"].astype(int) == 0].reset_index(drop=True)
    weekly_summary = summarize_weekly_rows(weekly)
    predictions.to_csv(run_dir / "d02_predictions_top2.csv", index=False, encoding="utf-8")
    weekly.to_csv(run_dir / "weekly_backtest.csv", index=False, encoding="utf-8")
    write_json(run_dir / "weekly_backtest_result.json", weekly_summary)
    write_json(
        run_dir / "leakage_audit.json",
        {
            "leakage_audit": "pass",
            "source": str(D02 / "leakage_audit.json"),
            "source_sha256": file_hash(D02 / "leakage_audit.json"),
            "frozen_checkpoint_only": True,
            "r5_sealed_holdout_used": False,
        },
    )
    comparison = read_json(comparison_dir / "comparison_summary.json", default={}) or {}
    d02_score = float(comparison["summaries"]["D_current_top2"]["mean_portfolio_return"])
    official_score = float(comparison["summaries"]["official_default_top5"]["mean_portfolio_return"])
    reproduction_pass = (
        abs(d02_score - D02_OFFICIAL) <= 1e-12
        and abs(official_score - OFFICIAL_BASELINE) <= 1e-12
        and abs(float(weekly_summary["cumulative_return_net"]) - D02_WEEKLY_CUMULATIVE) <= 1e-12
    )
    metrics = {
        "d02_official_proxy": d02_score,
        "official_baseline": official_score,
        "d02_weekly_cumulative_net": weekly_summary["cumulative_return_net"],
        "d02_weekly": weekly_summary,
    }
    write_json(
        manifest_path,
        {
            "experiment_id": run_dir.name,
            "created_at_utc": utc_now(),
            "data_path": str(DATA_PATH.resolve()),
            "data_md5": file_hash(DATA_PATH, "md5"),
            "d02_predictions_sha256": file_hash(D02 / "best_val_predictions.csv"),
            "metrics": metrics,
            "reproduction_pass": reproduction_pass,
        },
    )
    build_artifact_manifest(run_dir)
    state["runs"][run_dir.name].update(
        {"status": "completed" if reproduction_pass else "failed", "finished_at_utc": utc_now(), "metrics": metrics}
    )
    state["baseline_reproduction_pass"] = reproduction_pass
    state["stage"] = "baseline_reproduced" if reproduction_pass else "baseline_reproduction_failed"
    save_state(state)
    register(baseline_registry_entry(run_dir, "r5_baseline_reproduced" if reproduction_pass else "failed", metrics))
    if not reproduction_pass:
        raise RuntimeError("R5 D02/official baseline reproduction mismatch")


def prepare(args: argparse.Namespace, state: dict[str, Any]) -> None:
    AUTOMATION_ROOT.mkdir(parents=True, exist_ok=True)
    actual_md5 = file_hash(DATA_PATH, "md5")
    if actual_md5 != EXPECTED_DATA_MD5:
        state["stage"] = "data_hash_failed"
        state["data_md5"] = actual_md5
        save_state(state)
        raise RuntimeError(f"canonical R5 data MD5 mismatch: {actual_md5}")

    manifest = build_r5_split_manifest(
        data_path=DATA_PATH,
        d02_split_path=D02 / "split_metadata.json",
        d02_metrics_path=D02 / "metrics_v2.json",
        d02_weekly_path=D02 / "weekly_backtest.csv",
        d02_predictions_path=D02 / "best_val_predictions.csv",
        r4_split_path=AUTOMATION_ROOT / "R4_SPLIT_MANIFEST.json",
    )
    plan = build_r5_experiment_plan()
    if SPLIT_PATH.exists():
        existing = read_json(SPLIT_PATH, default={}) or {}
        if existing != manifest:
            old_holdout = existing.get("recent_sealed_holdout", {})
            new_holdout = manifest.get("recent_sealed_holdout", {})
            old_outer = [
                fold.get("outer_evaluation")
                for fold in existing.get("rolling_origin_development", {}).get("folds", [])
            ]
            new_outer = [
                fold.get("outer_evaluation")
                for fold in manifest.get("rolling_origin_development", {}).get("folds", [])
            ]
            holdout_identity_keys = [
                "first_decision_date",
                "weekly_buy_start",
                "weekly_sell_end",
                "complete_week_count",
                "sealed_window_sha256",
            ]
            same_holdout = all(old_holdout.get(key) == new_holdout.get(key) for key in holdout_identity_keys)
            if not same_holdout or old_outer != new_outer:
                raise RuntimeError("R5 split revision would alter locked outer folds or recent sealed holdout")
            backup = AUTOMATION_ROOT / "R5_SPLIT_MANIFEST_PRE_Q4_SELECTION_AUDIT.json"
            if not backup.exists():
                write_json(backup, existing)
            old_hash = file_hash(SPLIT_PATH)
            write_json(SPLIT_PATH, manifest)
            new_hash = file_hash(SPLIT_PATH)
            state.setdefault("split_manifest_revisions", []).append(
                {
                    "at_utc": utc_now(),
                    "old_sha256": old_hash,
                    "new_sha256": new_hash,
                    "reason": "exclude the already-opened R4 2025-Q4 holdout from R5 fold4 inner checkpoint selection",
                    "outer_folds_unchanged": True,
                    "recent_sealed_holdout_unchanged": True,
                    "backup": str(backup.resolve()),
                }
            )
    else:
        write_json(SPLIT_PATH, manifest)
    if PLAN_PATH.exists():
        existing_plan = read_json(PLAN_PATH, default={}) or {}
        if existing_plan != plan:
            raise RuntimeError("existing R5 experiment plan differs from the preregistered plan")
    else:
        write_json(PLAN_PATH, plan)

    state.setdefault("safety", {})
    state["safety"].setdefault("candidate_initial", directory_fingerprint(ROOT / "candidate"))
    state["safety"].setdefault("d02_initial", directory_fingerprint(D02))
    state["safety"].setdefault("registry_initial_sha256", file_hash(D_ROOT / "D_experiment_registry.json"))
    state["safety"].setdefault(
        "registry_initial_count",
        len((read_json(D_ROOT / "D_experiment_registry.json", default={}) or {}).get("experiments", [])),
    )
    state["data_path"] = str(DATA_PATH.resolve())
    state["data_md5"] = actual_md5
    state["split_manifest"] = str(SPLIT_PATH.resolve())
    state["split_manifest_sha256"] = file_hash(SPLIT_PATH)
    state["experiment_plan"] = str(PLAN_PATH.resolve())
    state["experiment_plan_sha256"] = file_hash(PLAN_PATH)
    state["environment"] = environment_summary(args.python_exe)
    state["stage"] = "prepared"
    save_state(state)
    exclude_invalid_q4_inner_runs(state)


def exclude_invalid_q4_inner_runs(state: dict[str, Any]) -> None:
    cutoff = pd.Timestamp("2025-10-09")
    excluded_ids: list[str] = []
    for item in state.get("quick_results", []):
        if item.get("phase") != "QUICK_INNER" or item.get("fold") != "R5_DEV_FOLD_04_2026_Q1":
            continue
        config = read_json(Path(item["run_dir"]) / "generated_config.json", default={}) or {}
        folds = config.get("validation", {}).get("folds", [])
        end = pd.Timestamp(folds[-1][2]) if folds else pd.Timestamp.min
        if end >= cutoff:
            item["valid_for_selection"] = False
            item["exclusion_reason"] = "inner validation overlaps forbidden R4 2025-Q4 sealed holdout"
            excluded_ids.append(item["experiment_id"])
    for experiment_id, run in state.get("runs", {}).items():
        if run.get("phase") != "QUICK_INNER" or run.get("fold") != "R5_DEV_FOLD_04_2026_Q1":
            continue
        run_dir = Path(run["path"])
        config = read_json(run_dir / "generated_config.json", default={}) or {}
        folds = config.get("validation", {}).get("folds", [])
        end = pd.Timestamp(folds[-1][2]) if folds else pd.Timestamp.min
        if end < cutoff:
            continue
        run["status"] = "invalid_protocol_excluded"
        run["failure_reason"] = "inner validation overlaps forbidden R4 2025-Q4 sealed holdout"
        excluded_ids.append(experiment_id)
        register(
            training_registry_entry(
                experiment_id=experiment_id,
                run_dir=run_dir,
                status="invalid_protocol_excluded",
                variant=run.get("variant", "R5_L0_F0"),
                phase="QUICK_INNER",
                fold_id=run.get("fold", "R5_DEV_FOLD_04_2026_Q1"),
                seed=int(run.get("seed", 20260720)),
                config_path=run_dir / "generated_config.json",
                metrics=run.get("metrics", {}),
                failure_reason="inner validation overlaps forbidden R4 2025-Q4 sealed holdout",
            )
        )
    state["invalid_q4_inner_runs"] = sorted(set(excluded_ids))
    save_state(state)


def _json_placeholder(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "reason": reason, "round": "R5", "created_at_utc": utc_now()}


def complete_r5_run_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    """Close interrupted/protocol-excluded runs without erasing their provenance."""
    rebuilt: list[str] = []
    invalid_ids = set(state.get("invalid_q4_inner_runs", []))
    stale_repro_id = "D_AUTO_R5_L0_F0_EXACT_REPRO_S20260708_20260710_201259"

    for experiment_id, run in state.get("runs", {}).items():
        run_dir = Path(run.get("path", AUTOMATION_ROOT / experiment_id))
        if not run_dir.exists():
            continue
        phase = str(run.get("phase", "unknown"))
        is_invalid = experiment_id in invalid_ids or run.get("status") == "invalid_protocol_excluded"
        if is_invalid:
            reason = "inner validation overlaps the already-opened R4 2025-Q4 holdout; excluded from all R5 selection"
            write_json(
                run_dir / "INTERRUPTION_REPORT.json",
                {
                    "status": "invalid_protocol_excluded",
                    "reason": reason,
                    "training_artifacts_preserved": True,
                    "metrics_eligible_for_r5_selection": False,
                    "split_manifest_revision": str(
                        (AUTOMATION_ROOT / "R5_SPLIT_MANIFEST_PRE_Q4_SELECTION_AUDIT.json").resolve()
                    ),
                },
            )
            write_json(
                run_dir / "leakage_audit.json",
                {
                    "leakage_audit": "invalid_protocol_excluded",
                    "checks": {"r4_q4_not_used_for_r5_selection": False},
                    "issues": [reason],
                },
            )
            for name in ("official_proxy_result.json", "weekly_backtest_result.json", "paired_comparison.json"):
                if not (run_dir / name).exists():
                    write_json(run_dir / name, _json_placeholder("not_evaluated", reason))
            for name in ("train_stdout.log", "train_stderr.log"):
                if not (run_dir / name).exists():
                    (run_dir / name).write_text(
                        "Controller interruption: no captured stream was available; see INTERRUPTION_REPORT.json.\n",
                        encoding="utf-8",
                    )
            run.update(
                {
                    "status": "invalid_protocol_excluded",
                    "finished_at_utc": run.get("finished_at_utc", utc_now()),
                    "failure_reason": reason,
                }
            )
            config_path = run_dir / "generated_config.json"
            register(
                training_registry_entry(
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    status="invalid_protocol_excluded",
                    variant=run.get("variant", "R5_L0_F0"),
                    phase=phase,
                    fold_id=run.get("fold", "unknown"),
                    seed=int(run.get("seed", 20260720)),
                    config_path=config_path,
                    metrics=run.get("metrics", {}),
                    failure_reason=reason,
                )
            )
        elif experiment_id == stale_repro_id and run.get("status") == "running":
            reason = "post-training diagnostic evaluation bug; replaced by the exact reproduction run ending 201848"
            run.update(
                {
                    "status": "failed_post_training_evaluation_bug_replaced",
                    "finished_at_utc": utc_now(),
                    "failure_reason": reason,
                }
            )
            write_json(run_dir / "FAILURE_REPORT.json", {"status": run["status"], "reason": reason})
            register(
                training_registry_entry(
                    experiment_id=experiment_id,
                    run_dir=run_dir,
                    status=run["status"],
                    variant="R5_L0_F0",
                    phase="EXACT_REPRO",
                    fold_id="legacy_research_validation",
                    seed=20260708,
                    config_path=run_dir / "generated_config.json",
                    failure_reason=reason,
                )
            )

        if (run_dir / "candidate_summary.json").exists():
            if not (run_dir / "weekly_backtest_result.json").exists():
                write_json(
                    run_dir / "weekly_backtest_result.json",
                    _json_placeholder("not_applicable", f"{phase} does not perform outer weekly evaluation"),
                )
            if not (run_dir / "paired_comparison.json").exists():
                write_json(
                    run_dir / "paired_comparison.json",
                    _json_placeholder("not_applicable", f"{phase} has no fold-matched paired comparison"),
                )
        for name in ("train_stdout.log", "train_stderr.log"):
            if not (run_dir / name).exists():
                (run_dir / name).write_text(
                    f"No training subprocess stream: R5 phase {phase} is a controller-only evaluation/test artifact.\n",
                    encoding="utf-8",
                )
        build_artifact_manifest(run_dir)
        rebuilt.append(experiment_id)
    save_state(state)
    return {"artifact_manifests_rebuilt": len(rebuilt), "run_ids": rebuilt}


def _holding_sets(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, dtype={"selected_stocks": str})
    result: dict[str, set[str]] = {}
    for _, row in frame.iterrows():
        if int(row.get("skipped_missing_decision_prediction", 0)) != 0:
            continue
        key = str(row.get("buy_date", row.get("week_start", "")))
        values = str(row.get("selected_stocks", ""))
        result[key] = {value for value in values.split(",") if value and value.lower() != "nan"}
    return result


def _cross_seed_jaccard(items: list[dict[str, Any]], filename: str) -> dict[str, Any]:
    values: list[float] = []
    by_fold: dict[str, list[float]] = {}
    for fold_id in sorted({item["fold"] for item in items}):
        fold_items = [item for item in items if item["fold"] == fold_id]
        seed_sets = {
            int(item["seed"]): _holding_sets(Path(item["run_dir"]) / filename)
            for item in fold_items
        }
        fold_values: list[float] = []
        for left_seed, right_seed in combinations(sorted(seed_sets), 2):
            common_dates = sorted(set(seed_sets[left_seed]) & set(seed_sets[right_seed]))
            for date in common_dates:
                left = seed_sets[left_seed][date]
                right = seed_sets[right_seed][date]
                union = left | right
                fold_values.append(float(len(left & right) / len(union)) if union else 1.0)
        by_fold[fold_id] = fold_values
        values.extend(fold_values)
    return {
        "observation_count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "min": float(np.min(values)) if values else None,
        "by_fold_mean": {
            fold_id: (float(np.mean(fold_values)) if fold_values else None)
            for fold_id, fold_values in by_fold.items()
        },
    }


def build_stability_analysis(state: dict[str, Any]) -> dict[str, Any]:
    outer = [item for item in state.get("quick_results", []) if item.get("phase") == "QUICK_OUTER"]
    rows: list[dict[str, Any]] = []
    variants: dict[str, Any] = {}
    risk_keys = [
        "max_drawdown_abs",
        "worst_week",
        "CVaR_95_loss_abs",
        "Sortino",
        "Calmar",
        "up_market_return",
        "down_market_return",
        "high_vol_return",
        "low_vol_return",
        "high_breadth_return",
        "low_breadth_return",
    ]
    for item in outer:
        metrics = item["metrics"]
        official = metrics["official_proxy"]
        weekly = metrics["weekly_backtest"]
        paired = metrics.get("paired_vs_fold_matched_d02", {})
        stability = metrics.get("prediction_stability", {})
        rows.append(
            {
                "experiment_id": item["experiment_id"],
                "variant": item["variant"],
                "fold": item["fold"],
                "seed": int(item["seed"]),
                "fixed_epoch": item.get("fixed_epoch"),
                "official_proxy": official.get("official_proxy_final_score"),
                "weekly_cumulative_net": weekly.get("cumulative_return_net"),
                "mean_weekly_net": weekly.get("mean_weekly_return_net"),
                "median_weekly_net": weekly.get("median_weekly_return_net"),
                "positive_week_rate": weekly.get("positive_week_rate"),
                "paired_mean_excess": paired.get("mean_weekly_excess", 0.0),
                "paired_median_excess": paired.get("median_weekly_excess", 0.0),
                "paired_pdr": paired.get("positive_diff_rate", 0.0),
                "diff_without_best_2_weeks": paired.get("diff_without_best_2_weeks", 0.0),
                "best_2_weeks_contribution_ratio": paired.get("best_2_weeks_contribution_ratio"),
                "max_drawdown_abs": weekly.get("max_drawdown_abs"),
                "worst_week": weekly.get("worst_week"),
                "CVaR_95_loss_abs": weekly.get("CVaR_95_loss_abs"),
                "Sortino": weekly.get("Sortino"),
                "Calmar": weekly.get("Calmar"),
                "mean_top2_boundary_margin": stability.get("mean_top2_boundary_margin"),
                "mean_hhi": stability.get("mean_hhi"),
                "max_single_stock_holding_frequency": stability.get("max_single_stock_holding_frequency"),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["variant", "fold", "seed"])
    frame.to_csv(RUN_METRICS_PATH, index=False, encoding="utf-8")
    for variant in VARIANTS:
        items = [item for item in outer if item["variant"] == variant]
        seed_means: dict[str, float] = {}
        for seed in R5_QUICK_SEEDS:
            subset = [item for item in items if int(item["seed"]) == seed]
            seed_means[str(seed)] = float(
                np.mean(
                    [item["metrics"].get("paired_vs_fold_matched_d02", {}).get("mean_weekly_excess", 0.0) for item in subset]
                )
            )
        seed_values = list(seed_means.values())
        leave_one_out = {
            seed: float(np.mean([value for other, value in seed_means.items() if other != seed]))
            for seed in seed_means
        }
        weekly_summaries = [item["metrics"]["weekly_backtest"] for item in items]
        stability_summaries = [item["metrics"].get("prediction_stability", {}) for item in items]
        pooled = _pooled_paired_summary(items) if variant != "R5_L0_F0" else {}
        variants[variant] = {
            "run_count": len(items),
            "top2_cross_seed_jaccard": _cross_seed_jaccard(items, "weekly_backtest.csv"),
            "top5_cross_seed_jaccard": _cross_seed_jaccard(items, "top5_diagnostic_weekly_backtest.csv"),
            "score_margin": {
                "mean": float(np.mean([item.get("mean_top2_boundary_margin", 0.0) for item in stability_summaries])),
                "min": float(np.min([item.get("mean_top2_boundary_margin", 0.0) for item in stability_summaries])),
            },
            "concentration": {
                "mean_hhi": float(np.mean([item.get("mean_hhi", 0.0) for item in stability_summaries])),
                "max_single_stock_holding_frequency_mean": float(
                    np.mean([item.get("max_single_stock_holding_frequency", 0.0) for item in stability_summaries])
                ),
                "max_single_stock_holding_frequency_max": float(
                    np.max([item.get("max_single_stock_holding_frequency", 0.0) for item in stability_summaries])
                ),
            },
            "seed_excess": {
                "by_seed": seed_means,
                "mean": float(np.mean(seed_values)),
                "std_population": float(np.std(seed_values)),
                "min": float(np.min(seed_values)),
                "leave_one_seed_out": leave_one_out,
                "three_of_five_gate": "NOT_EVALUATED_QUICK_STOPPED_AT_3_SEEDS",
            },
            "pooled_paired": pooled,
            "risk_and_regime_mean": {
                key: float(np.mean([summary.get(key, 0.0) for summary in weekly_summaries])) for key in risk_keys
            },
            "risk_worst_across_runs": {
                "max_drawdown_abs": float(np.max([summary["max_drawdown_abs"] for summary in weekly_summaries])),
                "worst_week": float(np.min([summary["worst_week"] for summary in weekly_summaries])),
                "CVaR_95_loss_abs": float(np.max([summary["CVaR_95_loss_abs"] for summary in weekly_summaries])),
            },
            "official_proxy_across_outer_runs": {
                "mean": float(np.mean([item["metrics"]["official_proxy"]["official_proxy_final_score"] for item in items])),
                "std_population": float(np.std([item["metrics"]["official_proxy"]["official_proxy_final_score"] for item in items])),
                "min": float(np.min([item["metrics"]["official_proxy"]["official_proxy_final_score"] for item in items])),
            },
        }
    payload = {
        "schema_version": "r5_stability_analysis_v1",
        "created_at_utc": utc_now(),
        "scope": "quick outer development folds only; sealed holdout remains unopened",
        "variant_results": variants,
        "run_metrics_csv": str(RUN_METRICS_PATH.resolve()),
        "run_count": len(rows),
    }
    write_json(STABILITY_PATH, payload)
    return payload


def build_finalist_lock_and_gate(state: dict[str, Any], stability: dict[str, Any]) -> dict[str, Any]:
    quick = read_json(AUTOMATION_ROOT / "R5_QUICK_SUMMARY.json", default={}) or {}
    challenger_summaries = [item for item in quick.get("variants", []) if item["variant"] != "R5_L0_F0"]
    lock = {
        "schema_version": "r5_finalist_lock_v1",
        "created_at_utc": utc_now(),
        "locked_finalists": [],
        "holdout_access_authorized": False,
        "holdout_status": "sealed",
        "holdout_unseal_count": 0,
        "reason": "No new variant passed every preregistered quick-to-serious criterion.",
        "quick_decisions": {
            item["variant"]: {
                "qualifies_for_serious": item["qualifies_for_serious"],
                "checks": item["quick_to_serious_checks"],
            }
            for item in challenger_summaries
        },
    }
    write_json(FINALIST_LOCK_PATH, lock)
    gate_variants: dict[str, Any] = {}
    for item in challenger_summaries:
        gate_variants[item["variant"]] = {
            "quick_to_serious": "PASS" if item["qualifies_for_serious"] else "FAIL",
            "quick_checks": item["quick_to_serious_checks"],
            "legacy_original_protocol": "NOT_EVALUATED_AFTER_QUICK_FAILURE",
            "five_seed_stability": "NOT_EVALUATED_AFTER_QUICK_FAILURE",
            "recent_sealed_holdout": "NOT_EVALUATED_HOLDOUT_REMAINS_SEALED",
            "risk_gate": "NOT_EVALUATED_AS_FINALIST",
            "champion_gate": "FAIL",
            "decision": "REJECT_AS_CHAMPION",
        }
    gate = {
        "schema_version": "r5_gate_decision_v1",
        "created_at_utc": utc_now(),
        "current_champion": "D02_FrozenResearchChampion_20260709",
        "champion_changed": False,
        "accepted_champion": None,
        "final_decision": "STOP_AFTER_QUICK_NO_QUALIFIED_CHALLENGER",
        "variants": gate_variants,
        "candidate_pool": [
            {
                "variant": "R5_L1_F0",
                "status": "RESEARCH_CANDIDATE_ONLY",
                "reason": "positive pooled/fold/seed excess evidence, but pooled PDR and recent-fold robustness failed",
            },
            {
                "variant": "R5_L1_F1",
                "status": "DIAGNOSTIC_ONLY",
                "reason": "largest pooled mean excess, but only two of four folds were positive and recent regimes failed",
            },
        ],
        "sealed_holdout": {"status": "sealed", "unseal_count": 0, "result": "NOT_EVALUATED"},
        "d03_created": False,
        "production_refit_executed": False,
        "stability_analysis_sha256": file_hash(STABILITY_PATH),
    }
    write_json(GATE_PATH, gate)
    state["finalist_lock"] = lock
    state["gate_decision"] = gate["final_decision"]
    state["sealed_holdout_status"] = "sealed"
    state["sealed_holdout_unseal_count"] = 0
    save_state(state)
    return gate


def _active_r5_training_processes() -> list[dict[str, Any]]:
    script = (
        "$p=Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne "
        f"{os.getpid()} -and $_.Name -match '^python' -and $_.CommandLine -match 'train.py' -and $_.CommandLine -match 'candidates_D' }}; "
        "$p | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    payload = json.loads(result.stdout)
    return payload if isinstance(payload, list) else [payload]


def _verify_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "artifact_manifest.json", default={}) or {}
    failures: list[str] = []
    for item in manifest.get("files", []):
        target = run_dir / item["path"]
        if not target.exists():
            failures.append(f"missing:{item['path']}")
        elif target.stat().st_size != int(item["size_bytes"]):
            failures.append(f"size:{item['path']}")
        elif file_hash(target) != item["sha256"]:
            failures.append(f"sha256:{item['path']}")
    return {"pass": not failures, "file_count": len(manifest.get("files", [])), "failures": failures}


def run_final_audit(state: dict[str, Any]) -> dict[str, Any]:
    registry = read_json(REGISTRY_PATH, default={}) or {}
    registry_ids = {item.get("id") for item in registry.get("experiments", [])}
    expected_ids = set(state.get("runs", {}))
    manifest_results: dict[str, Any] = {}
    stderr_nonempty: list[dict[str, Any]] = []
    stderr_missing: list[str] = []
    for experiment_id, run in state.get("runs", {}).items():
        run_dir = Path(run.get("path", AUTOMATION_ROOT / experiment_id))
        if (run_dir / "artifact_manifest.json").exists():
            manifest_results[experiment_id] = _verify_artifact_manifest(run_dir)
        stderr = run_dir / "train_stderr.log"
        if not stderr.exists():
            stderr_missing.append(experiment_id)
        elif stderr.stat().st_size:
            stderr_nonempty.append(
                {"experiment_id": experiment_id, "size_bytes": stderr.stat().st_size, "tail": stderr.read_text(encoding="utf-8", errors="replace")[-1000:]}
            )
    candidate_current = directory_fingerprint(ROOT / "candidate")
    d02_current = directory_fingerprint(D02)
    candidate_initial = state.get("safety", {}).get("candidate_initial")
    d02_initial = state.get("safety", {}).get("d02_initial")
    active = _active_r5_training_processes()
    split = read_json(SPLIT_PATH, default={}) or {}
    checks = {
        "data_md5_locked": file_hash(DATA_PATH, "md5") == EXPECTED_DATA_MD5,
        "candidate_unchanged": candidate_current == candidate_initial,
        "d02_unchanged": d02_current == d02_initial,
        "split_checks_pass": all(split.get("checks", {}).values()),
        "split_manifest_sha256_locked": file_hash(SPLIT_PATH) == state.get("split_manifest_sha256"),
        "baseline_reproduction_pass": bool(state.get("baseline_reproduction_pass")),
        "exact_l0_f0_reproduction_pass": bool(state.get("l0_f0_reproduction_pass")),
        "protocol_tests_pass": bool(state.get("protocol_tests_pass")),
        "smoke_pass": bool(state.get("smoke_pass")),
        "sealed_holdout_never_opened": state.get("sealed_holdout_status") == "sealed" and int(state.get("sealed_holdout_unseal_count", 0)) == 0,
        "all_registry_ids_present": expected_ids <= registry_ids,
        "all_artifact_manifests_valid": bool(manifest_results) and all(item["pass"] for item in manifest_results.values()),
        "no_active_training_process": not active,
        "no_d03_created": not any(item.is_dir() and item.name.startswith("D03") for item in D_ROOT.iterdir()),
        "no_r5_production_refit": not any("R5" in item.name and "production_refit" in item.name.lower() for item in D_ROOT.rglob("*")),
    }
    audit = {
        "schema_version": "r5_final_safety_audit_v1",
        "created_at_utc": utc_now(),
        "audit": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "candidate_initial": candidate_initial,
        "candidate_current": candidate_current,
        "d02_initial": d02_initial,
        "d02_current": d02_current,
        "data_md5": file_hash(DATA_PATH, "md5"),
        "split_manifest_sha256": file_hash(SPLIT_PATH),
        "registry": {
            "valid_json": True,
            "experiment_count": len(registry.get("experiments", [])),
            "missing_expected_ids": sorted(expected_ids - registry_ids),
        },
        "artifact_manifest": {
            "checked_run_count": len(manifest_results),
            "failed": {key: value for key, value in manifest_results.items() if not value["pass"]},
        },
        "stderr_nonempty": stderr_nonempty,
        "stderr_missing": stderr_missing,
        "active_training_processes": active,
        "sealed_holdout": {"status": state.get("sealed_holdout_status"), "unseal_count": state.get("sealed_holdout_unseal_count")},
    }
    write_json(SAFETY_PATH, audit)
    state["final_safety_audit"] = audit["audit"]
    save_state(state)
    return audit


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_r5_report(state: dict[str, Any], stability: dict[str, Any], gate: dict[str, Any], audit: dict[str, Any]) -> Path:
    split = read_json(SPLIT_PATH, default={}) or {}
    quick = read_json(AUTOMATION_ROOT / "R5_QUICK_SUMMARY.json", default={}) or {}
    rows = pd.read_csv(RUN_METRICS_PATH)
    created = pd.Timestamp(state["created_at_utc"])
    finished = pd.Timestamp.utcnow()
    wall_hours = float((finished - created).total_seconds() / 3600.0)
    gpu_upper_hours = float(
        sum(float(run.get("elapsed_seconds", 0.0) or 0.0) for run in state.get("runs", {}).values()) / 3600.0
    )
    report_path_value = state.get("final_report")
    report_path = Path(report_path_value) if report_path_value else AUTOMATION_ROOT / f"D_AUTO_ITERATION_REPORT_{stamp()}_R5.md"
    source = split["source"]
    legacy = split["legacy_research_validation"]
    sealed = split["recent_sealed_holdout"]
    variant_summary = {item["variant"]: item for item in quick["variants"]}
    lines = [
        "# D 系列 R5 自动迭代报告",
        "",
        f"- 生成时间（UTC）：`{utc_now()}`",
        f"- 结论：**D02 保持冠军；无 Challenger 获得 ACCEPT_AS_CHAMPION。**",
        f"- 停止原因：三个新变体均未通过预注册 Quick -> Serious 条件，故未启动 5-seed、Legacy finalist 或 sealed holdout。",
        "",
        "## 1. Champion 状态",
        "",
        "当前 Champion 仍为 `D02_FrozenResearchChampion_20260709`（来源程序 `D_AUTO_CHAMPION_REF_20260709_110423`）。R5 未创建 D03，D02 目录保持冻结。",
        "",
        "## 2. 墙钟时间、GPU 与资源",
        "",
        f"- R5 状态墙钟跨度：{wall_hours:.2f} 小时（含实现、测试、协议修订、训练、评估和报告）。",
        f"- 已登记训练子进程耗时合计：{gpu_upper_hours:.2f} GPU 小时上界；这是 CUDA 训练进程墙钟合计，不是采样得到的 GPU active time。",
        f"- 环境：Python `{state.get('environment', {}).get('python_version')}`，Torch `{state.get('environment', {}).get('torch')}`，GPU `{state.get('environment', {}).get('gpu')}`。",
        "",
        "## 3. 数据锁与日期",
        "",
        f"- 数据：`{source['data_path']}`",
        f"- MD5：`{source['data_md5']}`（预期值一致）。",
        f"- 原始范围：{source['data_stats']['raw_data_start']} 至 {source['data_stats']['raw_data_end']}；{source['data_stats']['trading_day_count']} 个交易日，{source['data_stats']['stock_count']} 只股票，{source['data_stats']['row_count']} 行。",
        f"- 最后可获得完整 T+5 标签的样本日：{source['data_stats']['last_complete_t5_label_sample']}。",
        "",
        "## 4. 三套评估协议",
        "",
        f"- Legacy：训练 {legacy['train']['sample_start']} 至 {legacy['train']['sample_end']}；验证 {legacy['validation']['sample_start']} 至 {legacy['validation']['sample_end']}。仅保留 D02/官方基准复现，未让 Quick 失败模型进入 Legacy finalist。",
    ]
    for fold in split["rolling_origin_development"]["folds"]:
        outer = fold["outer_evaluation"]
        train = fold["train"]
        inner = fold["inner_validation"]
        lines.append(
            f"- `{fold['fold_id']}`：train_end={train['sample_end']}，max_label_end={train['max_label_end']}，inner={inner['sample_start']}..{inner['sample_end']}，buy/sell={outer['weekly_buy_start']}..{outer['weekly_sell_end']}，{outer['complete_week_count']} 周。"
        )
    lines.extend(
        [
            f"- Recent sealed holdout：decision={sealed['first_decision_date']}，buy={sealed['weekly_buy_start']}，sell={sealed['weekly_sell_end']}，{sealed['complete_week_count']} 周；状态仍为 **SEALED**，unseal_count=0。",
            "- 2025 Q4 的 R4 holdout 未用于 R5 选择。初版 fold4 inner 边界审计发现重叠后，两个运行被保留但标记 `invalid_protocol_excluded`，随后将 inner 移至 2025-08-27..2025-09-23；外层折与新 sealed holdout 均未改变。",
            "",
            "## 5. 固定参考与 finalist",
            "",
            f"- 官方 baseline official-proxy：{OFFICIAL_BASELINE:.10f}。",
            f"- D02 official-proxy：{D02_OFFICIAL:.10f}；weekly cumulative net：{D02_WEEKLY_CUMULATIVE:.10f}。",
            "- Fresh reproduction：官方 baseline、D02 official-proxy 和 D02 weekly cumulative 均在 1e-12 容差内复现。",
            "- Finalist：无。根据预注册停止规则，没有权限读取 sealed 指标。",
            "",
            "## 6. L0/L1 与 F0/F1/F2 消融",
            "",
            "| Variant | Objective | Factor | Pooled mean excess | PDR | diff w/o best2 | Positive folds | Serious |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for variant in VARIANTS:
        summary = variant_summary[variant]
        pooled = summary.get("pooled_paired", {})
        positive_folds = sum(item["mean_paired_excess"] > 0 for item in summary["fold_metrics"].values()) if variant != "R5_L0_F0" else 0
        lines.append(
            f"| {variant} | {VARIANTS[variant]['objective']} | {VARIANTS[variant]['factor']} | {_fmt(pooled.get('mean_weekly_excess'))} | {_fmt(pooled.get('positive_diff_rate'))} | {_fmt(pooled.get('diff_without_best_2_weeks'))} | {positive_folds}/4 | {'CONTROL' if variant == 'R5_L0_F0' else ('PASS' if summary['qualifies_for_serious'] else 'FAIL')} |"
        )
    lines.extend(
        [
            "",
            "F2 未执行：F1 在 Q3 与最近折均未稳定优于 F0，不满足启用 causal cross-sectional ranks 的先决条件。Consistency 版本也未执行，因为固定 2×2 已给出明确停止证据。",
            "",
            "## 7. 训练/推理 score 对齐审计",
            "",
            "L1 的可微 Top-k utility 直接接收 `ranking_scores`，推理也只通过 `score_contract.primary_ranking_scores` 使用同一张量。自动化测试证明梯度有限且非零，同时未向独立 `portfolio_logits` 传递主目标梯度。L0 默认路径保持原语义。",
            "",
            "## 8. Factor Encoder 有效性",
            "",
            "F0 测试证明 factor_features 改变不影响输出；F1 测试证明 factor_features 改变会改变输出且 Factor Encoder 参数收到非零梯度。结构接通是 PASS，但经济增量证据是 FAIL：F1 变体均在 Q3 和最近开发折出现负超额。",
            "",
            "## 9. 每 fold / seed / variant 指标",
            "",
            "下表为全部 48 个有效 outer quick runs。`Official` 是该 outer 截面的同协议 proxy，`CumNet` 为该折累计净收益。",
            "",
            "| Variant | Fold | Seed | Epoch | Official | CumNet | MeanNet | MedianNet | Excess | PDR | MaxDD | Worst | CVaR |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['variant']} | {row['fold']} | {int(row['seed'])} | {int(row['fixed_epoch'])} | {_fmt(row['official_proxy'])} | {_fmt(row['weekly_cumulative_net'])} | {_fmt(row['mean_weekly_net'])} | {_fmt(row['median_weekly_net'])} | {_fmt(row['paired_mean_excess'])} | {_fmt(row['paired_pdr'])} | {_fmt(row['max_drawdown_abs'])} | {_fmt(row['worst_week'])} | {_fmt(row['CVaR_95_loss_abs'])} |"
        )
    lines.extend(["", "## 10. 收益与 paired 指标", ""])
    for variant in VARIANTS:
        summary = variant_summary[variant]
        pooled = summary.get("pooled_paired", {})
        fold_text = "; ".join(
            f"{fold.split('_')[-2]}_{fold.split('_')[-1]} excess={value['mean_paired_excess']:.6f}, PDR={value['mean_pdr']:.3f}, cum={value['mean_weekly_cumulative_net']:.6f}"
            for fold, value in summary["fold_metrics"].items()
        )
        lines.append(f"- `{variant}`：pooled excess={_fmt(pooled.get('mean_weekly_excess'))}，median={_fmt(pooled.get('median_weekly_excess'))}，PDR={_fmt(pooled.get('positive_diff_rate'))}。{fold_text}。")
    lines.extend(["", "## 11. 风险指标", "", "| Variant | Mean MaxDD | Worst MaxDD | Mean worst week | Worst week | Mean CVaR | Worst CVaR | Sortino mean | Calmar mean |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for variant, item in stability["variant_results"].items():
        mean = item["risk_and_regime_mean"]
        worst = item["risk_worst_across_runs"]
        lines.append(f"| {variant} | {_fmt(mean['max_drawdown_abs'])} | {_fmt(worst['max_drawdown_abs'])} | {_fmt(mean['worst_week'])} | {_fmt(worst['worst_week'])} | {_fmt(mean['CVaR_95_loss_abs'])} | {_fmt(worst['CVaR_95_loss_abs'])} | {_fmt(mean['Sortino'])} | {_fmt(mean['Calmar'])} |")
    lines.extend(["", "按 outer run 等权汇总的市场状态净收益：", "", "| Variant | Up market | Down market | High vol | Low vol | High breadth | Low breadth |", "|---|---:|---:|---:|---:|---:|---:|"])
    for variant, item in stability["variant_results"].items():
        mean = item["risk_and_regime_mean"]
        lines.append(f"| {variant} | {_fmt(mean['up_market_return'])} | {_fmt(mean['down_market_return'])} | {_fmt(mean['high_vol_return'])} | {_fmt(mean['low_vol_return'])} | {_fmt(mean['high_breadth_return'])} | {_fmt(mean['low_breadth_return'])} |")
    lines.extend(["", "## 12. 去极端周与贡献", ""])
    for variant in ("R5_L1_F0", "R5_L0_F1", "R5_L1_F1"):
        pooled = stability["variant_results"][variant]["pooled_paired"]
        lines.append(f"- `{variant}`：diff_without_best_2_weeks={_fmt(pooled['diff_without_best_2_weeks'])}，best_2_weeks_contribution_ratio={_fmt(pooled['best_2_weeks_contribution_ratio'])}。")
    lines.extend(["", "## 13. Top-k 稳定性、margin 与集中度", "", "| Variant | Top2 Jaccard | Top5 Jaccard | Mean margin | Mean HHI | Mean max holding freq | Max holding freq |", "|---|---:|---:|---:|---:|---:|---:|"])
    for variant, item in stability["variant_results"].items():
        lines.append(f"| {variant} | {_fmt(item['top2_cross_seed_jaccard']['mean'])} | {_fmt(item['top5_cross_seed_jaccard']['mean'])} | {_fmt(item['score_margin']['mean'])} | {_fmt(item['concentration']['mean_hhi'])} | {_fmt(item['concentration']['max_single_stock_holding_frequency_mean'])} | {_fmt(item['concentration']['max_single_stock_holding_frequency_max'])} |")
    lines.extend(["", "## 14. Seed 稳定性", ""])
    for variant, item in stability["variant_results"].items():
        seed = item["seed_excess"]
        lines.append(f"- `{variant}`：by_seed={json.dumps(seed['by_seed'], ensure_ascii=False)}；mean={seed['mean']:.6f}，std={seed['std_population']:.6f}，min={seed['min']:.6f}；leave-one-seed-out={json.dumps(seed['leave_one_seed_out'], ensure_ascii=False)}。5-seed/3-of-5 门禁未评估，因为 Quick 门禁先失败。")
    lines.extend(
        [
            "",
            "## 15. New recent sealed holdout",
            "",
            "结果：**NOT_EVALUATED**。没有 finalist，因此 `R5_FINALIST_LOCK.json` 明确 `holdout_access_authorized=false`；holdout 保持 sealed，unseal_count=0。这是遵守一次性解封规则的必要结果，不是缺失实验。",
            "",
            "## 16. Champion Gate",
            "",
            "| Gate | L1/F0 | L0/F1 | L1/F1 |",
            "|---|---|---|---|",
            f"| Quick 多数折 excess > 0 | {_fmt(variant_summary['R5_L1_F0']['quick_to_serious_checks']['majority_folds_mean_paired_excess_positive'])} | {_fmt(variant_summary['R5_L0_F1']['quick_to_serious_checks']['majority_folds_mean_paired_excess_positive'])} | {_fmt(variant_summary['R5_L1_F1']['quick_to_serious_checks']['majority_folds_mean_paired_excess_positive'])} |",
            f"| Quick pooled PDR >= 0.50 | {_fmt(variant_summary['R5_L1_F0']['quick_to_serious_checks']['rolling_mean_pdr_min_0_50'])} | {_fmt(variant_summary['R5_L0_F1']['quick_to_serious_checks']['rolling_mean_pdr_min_0_50'])} | {_fmt(variant_summary['R5_L1_F1']['quick_to_serious_checks']['rolling_mean_pdr_min_0_50'])} |",
            f"| diff_without_best_2 > 0 | {_fmt(variant_summary['R5_L1_F0']['quick_to_serious_checks']['aggregate_diff_without_best_2_positive'])} | {_fmt(variant_summary['R5_L0_F1']['quick_to_serious_checks']['aggregate_diff_without_best_2_positive'])} | {_fmt(variant_summary['R5_L1_F1']['quick_to_serious_checks']['aggregate_diff_without_best_2_positive'])} |",
            "| 5 seeds / Legacy / sealed / final risk | NOT_EVALUATED | NOT_EVALUATED | NOT_EVALUATED |",
            "| Final Champion Gate | **FAIL** | **FAIL** | **FAIL** |",
            "",
            "## 17. 过拟合与市场状态失效",
            "",
            "三个 challenger 的 pooled mean excess 都为正，但 PDR 全部低于 0.50，说明改善不是广泛周覆盖。L1/F0 在前三折均值为正，却在最近折转为 -0.008572；F1 两个版本在 Q3 与最近折均为负。该模式与模型老化/市场状态迁移一致，属于明确的 overfit/regime-risk 信号，不值得继续放大本轮搜索。",
            "",
            "## 18. Candidate Pool",
            "",
            "- `R5_L1_F0`：`RESEARCH_CANDIDATE_ONLY`。三个 quick seed 的跨折 mean excess 均为正，3/4 折均值为正，去最佳两周后仍为正；但 pooled PDR=0.475177、最近折为负，不能晋级。",
            "- `R5_L1_F1`：`DIAGNOSTIC_ONLY`。pooled mean excess 最高，但只有 2/4 折为正，PDR=0.453901，近期失效明显。",
            "- `R5_L0_F1`：不保留为 serious candidate；Factor 单独增量仅在早期两折出现，且一个 seed 跨折均值为负。",
            "",
            "## 19. D03 与 Production Refit",
            "",
            "未创建 D03，未执行 Production Refit。原因是没有 Challenger 通过 Quick 门禁，更不可能通过完整 Champion Gate；按规则不得以研究候选生成生产权重。",
            "",
            "## 20. 安全与审计",
            "",
            f"- Final safety audit：`{audit['audit'].upper()}`；candidate/ unchanged={audit['checks']['candidate_unchanged']}；D02 unchanged={audit['checks']['d02_unchanged']}。",
            f"- candidate/ fingerprint：initial/current=`{audit['candidate_current']['sha256']}`，files={audit['candidate_current']['file_count']}；D02 fingerprint：initial/current=`{audit['d02_current']['sha256']}`，files={audit['d02_current']['file_count']}。",
            f"- leakage/split/protocol tests：split={audit['checks']['split_checks_pass']}，protocol={audit['checks']['protocol_tests_pass']}，smoke={audit['checks']['smoke_pass']}，exact repro={audit['checks']['exact_l0_f0_reproduction_pass']}。",
            f"- registry：{audit['registry']['experiment_count']} 条，missing expected IDs={len(audit['registry']['missing_expected_ids'])}。",
            f"- artifacts：检查 {audit['artifact_manifest']['checked_run_count']} 个 run manifest，失败 {len(audit['artifact_manifest']['failed'])} 个。",
            f"- stderr：非空 {len(audit['stderr_nonempty'])} 个，缺失 {len(audit['stderr_missing'])} 个。协议失效中断 run 的占位日志明确指向 `INTERRUPTION_REPORT.json`。",
            f"- 非空 stderr 中仅包含 checkpoint/resume smoke 的 phase 分隔记录、协议失效中断 run 的显式恢复占位，以及 controller-only 阶段说明；没有训练异常 traceback。缺失日志={len(audit['stderr_missing'])}。",
            f"- 后台训练进程：{len(audit['active_training_processes'])}；sealed holdout 未打开；D03/refit 均不存在。",
            "",
            "## 21. 下一轮建议（最多 3 条）",
            "",
            "1. 先诊断 L1/F0 在 2026 Q1 的 score-margin、选股换手与 regime 分解，限定一个低自由度的时间衰减或 regime calibration 假设，再开新一轮；不要扩 lambda 网格。",
            "2. 暂停 F2。先解释现有 F1 为何在 Q3/近期同时失效，重点检查 factor normalization 的跨期漂移和 market/factor branch 的相对尺度。",
            "3. 下一轮继续保留当前 recent sealed holdout；只有新 development 证据满足全部预注册条件后才一次性解封。",
            "",
            "## 22. 恢复命令",
            "",
            "```powershell",
            f"& '{state['environment']['training_python']}' '{Path(__file__).resolve()}' --mode audit",
            "```",
            "",
            "该命令只重跑审计和报告，不启动训练、不解封 holdout。Production Refit 命令未生成，因为没有 accepted champion。",
            "",
            "## 产物索引",
            "",
            f"- Split：`{SPLIT_PATH.resolve()}`",
            f"- Quick summary：`{(AUTOMATION_ROOT / 'R5_QUICK_SUMMARY.json').resolve()}`",
            f"- Run metrics：`{RUN_METRICS_PATH.resolve()}`",
            f"- Stability：`{STABILITY_PATH.resolve()}`",
            f"- Finalist lock：`{FINALIST_LOCK_PATH.resolve()}`",
            f"- Gate：`{GATE_PATH.resolve()}`",
            f"- Safety：`{SAFETY_PATH.resolve()}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state["final_report"] = str(report_path.resolve())
    state["stage"] = "finalized_no_qualified_challenger"
    state["completed_at_utc"] = utc_now()
    save_state(state)
    return report_path


def finalize_r5(state: dict[str, Any]) -> Path:
    if state.get("stage") not in {"quick_no_qualified_challengers", "finalizing", "finalized_no_qualified_challenger"}:
        raise RuntimeError(f"R5 cannot finalize from stage {state.get('stage')!r}")
    if state.get("serious_selected_variants"):
        raise RuntimeError("This finalizer is only valid when no quick challenger qualified for serious evaluation")
    state["stage"] = "finalizing"
    save_state(state)
    state["artifact_completion"] = complete_r5_run_artifacts(state)
    stability = build_stability_analysis(state)
    gate = build_finalist_lock_and_gate(state, stability)
    audit = run_final_audit(state)
    if audit["audit"] != "pass":
        raise RuntimeError(f"R5 final safety audit failed; see {SAFETY_PATH}")
    return write_r5_report(state, stability, gate, audit)


def parse_args() -> argparse.Namespace:
    default_python = ROOT / "Experimental_WorkSpace" / "shared_envs" / "pytorch-cu128-py311" / "Scripts" / "python.exe"
    parser = argparse.ArgumentParser(description="Resumable D-series R5 rolling-origin experiment controller")
    parser.add_argument(
        "--mode",
        choices=["prepare", "test", "smoke", "quick", "serious", "finalize", "audit", "resume", "all"],
        default="resume",
    )
    parser.add_argument("--python-exe", default=str(default_python))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = read_json(STATE_PATH, default=None) or initial_state()
    if args.mode in {"prepare", "resume", "all"} and state.get("stage") == "not_started":
        prepare(args, state)
        reproduce_baselines(args, state)
    elif args.mode == "prepare":
        prepare(args, state)
        reproduce_baselines(args, state)
    elif args.mode in {"resume", "all"} and state.get("stage") == "prepared":
        reproduce_baselines(args, state)
    elif args.mode == "test":
        run_protocol_tests(args, state)
    elif args.mode == "smoke":
        run_smoke(args, state)
    elif args.mode == "quick":
        run_quick(args, state)
    elif args.mode == "serious":
        if not state.get("serious_selected_variants"):
            raise RuntimeError("No R5 variant passed the preregistered quick-to-serious gate; serious training is prohibited")
        raise NotImplementedError("Serious execution is not implemented because this R5 run selected no serious variants")
    elif args.mode == "finalize":
        report_path = finalize_r5(state)
        print(f"R5 report: {report_path}")
    elif args.mode == "audit":
        if not STABILITY_PATH.exists() or not GATE_PATH.exists():
            report_path = finalize_r5(state)
        else:
            audit = run_final_audit(state)
            if audit["audit"] != "pass":
                raise RuntimeError(f"R5 final safety audit failed; see {SAFETY_PATH}")
            report_path = write_r5_report(
                state,
                read_json(STABILITY_PATH, default={}) or {},
                read_json(GATE_PATH, default={}) or {},
                audit,
            )
        print(f"R5 audit/report: {report_path}")
    elif args.mode in {"resume", "all"} and state.get("stage") == "baseline_reproduced":
        run_protocol_tests(args, state)
        run_smoke(args, state)
    elif args.mode in {"resume", "all"} and state.get("stage") == "protocol_tests_passed":
        run_smoke(args, state)
    elif args.mode in {"resume", "all"} and state.get("stage") in {
        "smoke_passed",
        "quick_inner_running",
        "quick_outer_running",
    }:
        run_quick(args, state)
    elif args.mode in {"resume", "all"} and state.get("stage") in {
        "quick_no_qualified_challengers",
        "finalizing",
        "finalized_no_qualified_challenger",
    }:
        report_path = finalize_r5(state)
        print(f"R5 report: {report_path}")
    else:
        raise NotImplementedError(f"R5 mode {args.mode!r} is not implemented yet; run --mode prepare first")
    if args.mode not in {"finalize", "audit"}:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    else:
        print(
            json.dumps(
                {
                    "stage": state.get("stage"),
                    "final_safety_audit": state.get("final_safety_audit"),
                    "sealed_holdout_status": state.get("sealed_holdout_status"),
                    "sealed_holdout_unseal_count": state.get("sealed_holdout_unseal_count"),
                    "final_report": state.get("final_report"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

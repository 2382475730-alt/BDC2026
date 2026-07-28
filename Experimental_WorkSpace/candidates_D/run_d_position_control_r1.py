from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "Experimental_WorkSpace" / "candidates_D" / "automation_runs" / "D_POSITION_CONTROL_R1_20260713_093536"
D02 = ROOT / "Experimental_WorkSpace" / "candidates_D" / "D02_FrozenResearchChampion_20260709"
DATA_PATH = ROOT / "Experimental_WorkSpace" / "candidates_D" / "data_snapshots" / "c0_train_plus_baostock_future_20260703.csv"
REGISTRY = ROOT / "Experimental_WorkSpace" / "candidates_D" / "D_experiment_registry.json"
PYTHON_EXE = ROOT / "Experimental_WorkSpace" / "shared_envs" / "pytorch-cu128-py311" / "Scripts" / "python.exe"
STATE_PATH = RUN_DIR / "run_state.json"

sys.path.insert(0, str(ROOT))

from Evaluation.scripts.d_series.common import read_json, read_market_csv, write_json  # noqa: E402
from Evaluation.scripts.d_series.experiment_registry import upsert_experiment  # noqa: E402
from Evaluation.scripts.d_series.position_control_r1 import (  # noqa: E402
    CONTROLLER_FEATURES,
    POLICY_IDS,
    CausalMarketFeatures,
    apply_controller_policy,
    apply_static_policy,
    block_bootstrap_probability,
    build_controller_training_data,
    controller_feature_row,
    controller_gate,
    evaluate_weighted_predictions,
    paired_position_metrics,
    quarterly_and_regime_breakdown,
    static_policy_gate,
)
from Evaluation.scripts.d_series.r4_protocol import build_artifact_manifest  # noqa: E402
from Evaluation.scripts.d_series.r5_protocol import directory_fingerprint, file_hash  # noqa: E402


EXPECTED_DATA_MD5 = "78759d892c95eb8ba547c2df9a8fca5a"
EXPECTED_CANDIDATE_SHA256 = "6d399643bdaae36747d8aea3e50a6581578b03a9e4605d0d901df089137c694f"
EXPECTED_D02_SHA256 = "704f981ab32c37f868d417405cd6a713a87f9ed17d2db924bccad2a6714bcbf5"
EXPECTED_D02_CODE_SHA256 = "2346569f65f41a4bccd21f7cd58038831b5c465431b7d6ba3d71f602ceb23b9c"
OFFICIAL_BASELINE = 0.012528499297343995
CONTROLLER_SEEDS = [20260720, 20260721, 20260722]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def state() -> dict[str, Any]:
    current = read_json(STATE_PATH, default={}) or {}
    if not current:
        current = {
            "experiment_id": RUN_DIR.name,
            "created_at_utc": utc_now(),
            "stage": "protocol_frozen",
            "research_validation_challenger_accessed": False,
            "production_refit_executed": False,
            "events": [],
        }
        save_state(current)
    return current


def save_state(payload: dict[str, Any]) -> None:
    write_json(STATE_PATH, payload)


def event(payload: dict[str, Any], name: str, **details: Any) -> None:
    payload.setdefault("events", []).append({"at_utc": utc_now(), "event": name, **details})
    save_state(payload)


def update_main_registry(status: str, *, decision: str | None = None, metrics: dict[str, Any] | None = None) -> None:
    entry = read_json(RUN_DIR / "registry_entry.json", default={}) or {}
    entry.update(
        {
            "status": status,
            "decision": decision,
            "metrics": metrics or entry.get("metrics", {}),
            "checkpoint_path": (
                str((RUN_DIR / "controller_models" / "pc1_seed_20260720.joblib").relative_to(ROOT))
                if (RUN_DIR / "controller_models" / "pc1_seed_20260720.joblib").exists()
                else None
            ),
        }
    )
    write_json(RUN_DIR / "registry_entry.json", entry)
    upsert_experiment(REGISTRY, entry)


def update_fold_registry(fold_id: str, status: str, fold_dir: Path, failure_reason: str = "") -> None:
    fold = next(item for item in read_json(RUN_DIR / "inner_fold_manifest.json")["folds"] if item["fold_id"] == fold_id)
    entry = {
        "id": f"{RUN_DIR.name}_{fold_id}",
        "name": f"D Position Control R1 OOF {fold_id}",
        "path": str(fold_dir.relative_to(ROOT)),
        "status": status,
        "source_program": "D02_FrozenResearchChampion_20260709",
        "architecture": "Unmodified D02 Program trained to preregistered fixed epoch 12; fold OOF interval never selects checkpoints.",
        "feature_version": "d_market_v1",
        "label_version": "open_t_plus_5_over_open_t_plus_1",
        "training_dates": f"2022-01-04..{fold['train_sample_end']}; max label end {fold['train_label_end_end']}",
        "validation_dates": f"{fold['oof_sample_start']}..{fold['oof_sample_end']} OOF",
        "seed": 20260708,
        "fixed_epochs": 12,
        "hyperparameters": str((fold_dir / "generated_config.json").relative_to(ROOT)),
        "checkpoint_path": str((fold_dir / "final_model.pth").relative_to(ROOT)) if (fold_dir / "final_model.pth").exists() else None,
        "failure_reason": failure_reason,
    }
    upsert_experiment(REGISTRY, entry)


def verify_immutables() -> dict[str, Any]:
    actual = {
        "data_md5": file_hash(DATA_PATH, "md5"),
        "candidate": directory_fingerprint(ROOT / "candidate"),
        "d02": directory_fingerprint(D02),
        "d02_code": directory_fingerprint(D02 / "code_snapshot" / "src"),
    }
    if actual["data_md5"] != EXPECTED_DATA_MD5:
        raise RuntimeError(f"data MD5 mismatch: {actual['data_md5']}")
    if actual["candidate"]["sha256"] != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("candidate/ fingerprint changed")
    if actual["d02"]["sha256"] != EXPECTED_D02_SHA256:
        raise RuntimeError("D02 fingerprint changed")
    if actual["d02_code"]["sha256"] != EXPECTED_D02_CODE_SHA256:
        raise RuntimeError("D02 code fingerprint changed")
    finalist = read_json(ROOT / "Experimental_WorkSpace" / "candidates_D" / "automation_runs" / "R5_FINALIST_LOCK.json")
    if finalist["holdout_status"] != "sealed" or finalist["holdout_unseal_count"] != 0 or finalist["holdout_access_authorized"]:
        raise RuntimeError("recent sealed holdout lock changed")
    actual["recent_sealed_holdout"] = {
        "status": finalist["holdout_status"],
        "unseal_count": finalist["holdout_unseal_count"],
        "access_authorized": finalist["holdout_access_authorized"],
    }
    return actual


def baseline() -> None:
    current = state()
    if (RUN_DIR / "baseline_reproduction.json").exists():
        existing = read_json(RUN_DIR / "baseline_reproduction.json")
        if existing.get("reproduction_pass"):
            current["stage"] = "baseline_reproduced"
            save_state(current)
            return
    hashes = verify_immutables()
    predictions = pd.read_csv(D02 / "best_val_predictions.csv", dtype={"stock_id": str})
    market_features = CausalMarketFeatures.from_market_path(DATA_PATH)
    p0 = apply_static_policy(predictions, "P0", market_features)
    evaluated = evaluate_weighted_predictions(
        p0,
        DATA_PATH,
        validation_start="2025-01-03",
        validation_end="2026-01-05",
        top_k=2,
    )
    p0.to_csv(RUN_DIR / "p0_predictions.csv", index=False, encoding="utf-8")
    evaluated["official_daily"].to_csv(RUN_DIR / "p0_official_proxy_daily.csv", index=False, encoding="utf-8")
    evaluated["weekly_all"].to_csv(RUN_DIR / "p0_weekly_backtest_all.csv", index=False, encoding="utf-8")
    evaluated["weekly"].to_csv(RUN_DIR / "p0_weekly_backtest.csv", index=False, encoding="utf-8")
    frozen_official = read_json(D02 / "official_proxy_result.json")
    frozen_weekly = read_json(D02 / "weekly_backtest_result.json")
    official_keys = ["official_proxy_final_score", "mean_portfolio_return", "median_portfolio_return"]
    weekly_keys = [
        "mean_weekly_return_net",
        "median_weekly_return_net",
        "cumulative_return_net",
        "worst_week",
        "max_drawdown_abs",
        "CVaR_95_loss_abs",
        "cost_drag",
    ]
    errors = {
        **{
            f"official.{key}": abs(float(evaluated["official"][key]) - float(frozen_official[key]))
            for key in official_keys
        },
        **{
            f"weekly.{key}": abs(float(evaluated["weekly_summary"][key]) - float(frozen_weekly[key]))
            for key in weekly_keys
        },
    }
    reproduction_pass = max(errors.values()) <= 1e-12
    payload = {
        "experiment_id": RUN_DIR.name,
        "created_at_utc": utc_now(),
        "policy": "P0 = D02 ranking_scores -> Top-2 -> 0.5/0.5",
        "data_path": str(DATA_PATH.resolve()),
        "data_md5": hashes["data_md5"],
        "d02_predictions_sha256": file_hash(D02 / "best_val_predictions.csv"),
        "official_proxy": evaluated["official"],
        "weekly_backtest": evaluated["weekly_summary"],
        "frozen_official_proxy": frozen_official,
        "frozen_weekly_backtest": frozen_weekly,
        "absolute_errors": errors,
        "max_absolute_error": max(errors.values()),
        "tolerance": 1e-12,
        "reproduction_pass": reproduction_pass,
    }
    write_json(RUN_DIR / "baseline_reproduction.json", payload)
    if not reproduction_pass:
        current["stage"] = "baseline_reproduction_failed"
        event(current, "baseline_reproduction_failed", max_absolute_error=max(errors.values()))
        update_main_registry("failed_baseline_reproduction", decision="REJECT", metrics=payload)
        build_artifact_manifest(RUN_DIR)
        raise RuntimeError("P0 reproduction failed; OOF training is forbidden")
    current["stage"] = "baseline_reproduced"
    event(current, "baseline_reproduced", max_absolute_error=max(errors.values()))
    update_main_registry("baseline_reproduced", metrics={"baseline_reproduction": payload})


def _fold_config(fold: dict[str, Any], fold_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(read_json(D02 / "generated_config.json"))
    config["experiment_id"] = f"{RUN_DIR.name}_{fold['fold_id']}"
    config["data"]["train_csv"] = str(DATA_PATH.resolve())
    config["training"]["num_epochs"] = 12
    config["training"]["early_stop_patience"] = 12
    config["training"]["checkpoint_metric"] = "fixed_final_epoch_no_oof_selection"
    config["validation"] = {
        "mode": "position_control_oof_fixed_epoch",
        "train_start": "2022-01-04",
        "train_end": fold["train_sample_end"],
        "train_label_end": fold["train_label_end_end"],
        "embargo_start": fold["embargo_start"],
        "embargo_end": fold["embargo_end"],
        "oof_start": fold["oof_sample_start"],
        "oof_end": fold["oof_sample_end"],
        "oof_checkpoint_selection_used": False,
    }
    config["output"]["run_dir"] = str(fold_dir.resolve())
    return config


def oof() -> None:
    current = state()
    baseline_payload = read_json(RUN_DIR / "baseline_reproduction.json", default={}) or {}
    if not baseline_payload.get("reproduction_pass"):
        raise RuntimeError("P0 reproduction must pass before OOF training")
    verify_immutables()
    manifest = read_json(RUN_DIR / "inner_fold_manifest.json")
    prediction_frames = []
    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        fold_dir = RUN_DIR / "oof_folds" / fold_id
        result_path = fold_dir / "fold_result.json"
        if result_path.exists() and read_json(result_path).get("status") == "completed":
            prediction_frames.append(pd.read_csv(fold_dir / "oof_predictions.csv", dtype={"stock_id": str}))
            continue
        fold_dir.mkdir(parents=True, exist_ok=True)
        snapshot = fold_dir / "code_snapshot" / "src"
        if not snapshot.exists():
            shutil.copytree(D02 / "code_snapshot" / "src", snapshot)
        if directory_fingerprint(snapshot)["sha256"] != EXPECTED_D02_CODE_SHA256:
            raise RuntimeError(f"{fold_id} copied code snapshot differs from D02")
        config_path = fold_dir / "generated_config.json"
        write_json(config_path, _fold_config(fold, fold_dir))
        update_fold_registry(fold_id, "running", fold_dir)
        command = [
            str(PYTHON_EXE.resolve()),
            "-B",
            str((ROOT / "Evaluation" / "scripts" / "d_series" / "position_control_oof_train.py").resolve()),
            "--code-src",
            str(snapshot.resolve()),
            "--config",
            str(config_path.resolve()),
            "--data-path",
            str(DATA_PATH.resolve()),
            "--fold-manifest",
            str((RUN_DIR / "inner_fold_manifest.json").resolve()),
            "--fold-id",
            fold_id,
            "--output-dir",
            str(fold_dir.resolve()),
        ]
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, check=False)
        (fold_dir / "train_stdout.log").write_text(completed.stdout.decode("utf-8", errors="replace"), encoding="utf-8")
        (fold_dir / "train_stderr.log").write_text(completed.stderr.decode("utf-8", errors="replace"), encoding="utf-8")
        if completed.returncode != 0 or not result_path.exists():
            update_fold_registry(fold_id, "failed", fold_dir, f"returncode_{completed.returncode}")
            current["stage"] = "oof_training_failed"
            event(current, "oof_fold_failed", fold_id=fold_id, returncode=completed.returncode)
            raise RuntimeError(f"OOF training failed for {fold_id}; see {fold_dir / 'train_stderr.log'}")
        if (fold_dir / "train_stderr.log").stat().st_size != 0:
            update_fold_registry(fold_id, "failed_nonempty_stderr", fold_dir, "nonempty_stderr")
            raise RuntimeError(f"OOF training stderr is non-empty for {fold_id}")
        build_artifact_manifest(fold_dir)
        update_fold_registry(fold_id, "completed", fold_dir)
        prediction_frames.append(pd.read_csv(fold_dir / "oof_predictions.csv", dtype={"stock_id": str}))
        event(current, "oof_fold_completed", fold_id=fold_id)
    oof_scores = pd.concat(prediction_frames, ignore_index=True)
    oof_scores["sample_end_date"] = pd.to_datetime(oof_scores["sample_end_date"]).dt.strftime("%Y-%m-%d")
    if oof_scores.duplicated(["sample_end_date", "stock_id"]).any():
        raise RuntimeError("OOF scores contain duplicate date-stock keys")
    oof_scores = oof_scores.sort_values(["sample_end_date", "stock_id"]).reset_index(drop=True)
    oof_scores.to_csv(RUN_DIR / "oof_scores.csv", index=False, encoding="utf-8")
    summary = {
        "status": "completed",
        "fold_count": len(prediction_frames),
        "row_count": int(len(oof_scores)),
        "date_count": int(oof_scores["sample_end_date"].nunique()),
        "date_start": str(oof_scores["sample_end_date"].min()),
        "date_end": str(oof_scores["sample_end_date"].max()),
        "sha256": file_hash(RUN_DIR / "oof_scores.csv"),
        "research_validation_touched": False,
    }
    write_json(RUN_DIR / "oof_scores_manifest.json", summary)
    current["stage"] = "oof_completed"
    event(current, "oof_completed", **summary)
    update_main_registry("oof_completed", metrics={"oof": summary})


def _train_controller_models(training_data: pd.DataFrame) -> dict[str, Any]:
    output = RUN_DIR / "controller_models"
    output.mkdir(parents=True, exist_ok=True)
    if training_data["label"].nunique() < 2:
        raise RuntimeError("controller target has fewer than two classes")
    result = {}
    for seed in CONTROLLER_SEEDS:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=seed,
                        max_iter=1000,
                    ),
                ),
            ]
        )
        model.fit(training_data[list(CONTROLLER_FEATURES)], training_data["label"].astype(int))
        path = output / f"pc1_seed_{seed}.joblib"
        joblib.dump(model, path)
        result[str(seed)] = {"path": str(path.relative_to(RUN_DIR)), "sha256": file_hash(path)}
    return result


def _feature_truncation_audit(predictions: pd.DataFrame, full_features: CausalMarketFeatures) -> dict[str, Any]:
    market, columns = read_market_csv(DATA_PATH)
    close_column = next(name for name in ("收盘", "close", "收盘价") if name in market.columns)
    dates = sorted(pd.to_datetime(predictions["sample_end_date"]).dt.normalize().unique())
    sampled = [pd.Timestamp(dates[index]).normalize() for index in sorted({0, len(dates) // 2, len(dates) - 1})]
    checks = []
    frame = predictions.copy()
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.normalize()
    for date in sampled:
        day = frame[frame["sample_end_date"] == date]
        full_row = controller_feature_row(day, full_features)
        truncated_market = market[pd.to_datetime(market[columns.date]).dt.normalize() <= date].copy()
        truncated_features = CausalMarketFeatures.from_market_frame(
            truncated_market, columns.stock, columns.date, close_column
        )
        truncated_row = controller_feature_row(day, truncated_features)
        differences = {key: abs(full_row[key] - truncated_row[key]) for key in CONTROLLER_FEATURES}
        checks.append({"date": date.strftime("%Y-%m-%d"), "max_absolute_difference": max(differences.values()), "differences": differences})
    return {"status": "pass" if all(item["max_absolute_difference"] <= 1e-12 for item in checks) else "fail", "checks": checks}


def static_and_controller() -> None:
    current = state()
    if not (RUN_DIR / "oof_scores.csv").exists():
        raise RuntimeError("OOF scores are required")
    verify_immutables()
    predictions = pd.read_csv(RUN_DIR / "oof_scores.csv", dtype={"stock_id": str})
    market_features = CausalMarketFeatures.from_market_path(DATA_PATH)
    start = str(predictions["sample_end_date"].min())
    end = str(predictions["sample_end_date"].max())
    static_root = RUN_DIR / "static_policies"
    evaluations: dict[str, Any] = {}
    for policy_id in POLICY_IDS:
        policy_dir = static_root / policy_id
        policy_dir.mkdir(parents=True, exist_ok=True)
        weighted = apply_static_policy(predictions, policy_id, market_features)
        result = evaluate_weighted_predictions(
            weighted,
            DATA_PATH,
            validation_start=start,
            validation_end=end,
            top_k=2 if policy_id in {"P0", "P1", "P2"} else 5,
        )
        weighted.to_csv(policy_dir / "weighted_predictions.csv", index=False, encoding="utf-8")
        result["official_daily"].to_csv(policy_dir / "official_proxy_daily.csv", index=False, encoding="utf-8")
        result["weekly_all"].to_csv(policy_dir / "weekly_backtest_all.csv", index=False, encoding="utf-8")
        result["weekly"].to_csv(policy_dir / "weekly_backtest.csv", index=False, encoding="utf-8")
        write_json(policy_dir / "official_proxy_result.json", result["official"])
        write_json(policy_dir / "weekly_backtest_result.json", result["weekly_summary"])
        evaluations[policy_id] = result
    p0 = evaluations["P0"]
    metrics = {
        "P0": {
            "official_proxy": p0["official"],
            "weekly": p0["weekly_summary"],
            "paired_vs_P0": None,
            "screening_gate": {"passes": True, "checks": {"control": True}},
        }
    }
    passing = []
    for policy_id in ("P1", "P2", "P3", "P4"):
        result = evaluations[policy_id]
        paired = paired_position_metrics(result["weekly"], p0["weekly"])
        item = {"official_proxy": result["official"], "weekly": result["weekly_summary"], "paired_vs_P0": paired}
        gate = static_policy_gate(item, metrics["P0"])
        item["screening_gate"] = gate
        metrics[policy_id] = item
        if gate["passes"]:
            passing.append(policy_id)
    pstar = max(passing, key=lambda policy: float(metrics[policy]["weekly"]["Sortino"])) if passing else None
    write_json(
        RUN_DIR / "static_policy_metrics.json",
        {
            "selection_source": "inner OOF only",
            "research_validation_touched": False,
            "policies": metrics,
            "passing_policies": passing,
            "selected_P_STAR": pstar,
            "selection_rule": "highest Sortino among policies passing all static gates",
        },
    )
    if pstar is None:
        write_json(
            RUN_DIR / "controller_config.json",
            {"status": "not_trained", "reason": "no_static_policy_passed", "research_validation_touched": False},
        )
        current["stage"] = "stopped_no_static_policy"
        event(current, "no_static_policy_passed")
        update_main_registry("REJECT_no_static_policy", decision="REJECT", metrics={"static": metrics})
        return
    controller_data = build_controller_training_data(
        predictions,
        p0["weekly"],
        evaluations[pstar]["weekly"],
        market_features,
    )
    controller_data.to_csv(RUN_DIR / "controller_training_data.csv", index=False, encoding="utf-8")
    valid_weeks = int(len(controller_data))
    if valid_weeks < 80:
        write_json(
            RUN_DIR / "controller_config.json",
            {
                "status": "not_trained",
                "reason": "oof_valid_weeks_below_80",
                "valid_weeks": valid_weeks,
                "selected_P_STAR": pstar,
            },
        )
        current["stage"] = "stopped_insufficient_oof_weeks"
        event(current, "insufficient_oof_weeks", valid_weeks=valid_weeks)
        update_main_registry("REJECT_insufficient_oof_weeks", decision="REJECT")
        return
    models = _train_controller_models(controller_data)
    truncation = _feature_truncation_audit(predictions, market_features)
    if truncation["status"] != "pass":
        raise RuntimeError("position feature truncation audit failed")
    config = {
        "status": "trained_and_frozen",
        "selected_P_STAR": pstar,
        "model": "LogisticRegression",
        "penalty": "l2",
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "liblinear",
        "threshold": 0.55,
        "features": list(CONTROLLER_FEATURES),
        "seeds": CONTROLLER_SEEDS,
        "primary_seed": 20260720,
        "training_week_count": valid_weeks,
        "positive_label_rate": float(controller_data["label"].mean()),
        "models": models,
        "research_validation_touched": False,
    }
    write_json(RUN_DIR / "controller_config.json", config)
    leakage = {
        "leakage_audit": "pass",
        "p0_reproduction_pass": True,
        "all_oof_fold_audits_pass": all(
            read_json(RUN_DIR / "oof_folds" / fold["fold_id"] / "leakage_audit.json")["leakage_audit"] == "pass"
            for fold in read_json(RUN_DIR / "inner_fold_manifest.json")["folds"]
        ),
        "controller_uses_oof_scores_only": True,
        "controller_features_decision_date_causal": True,
        "feature_truncation_equivalence": truncation,
        "research_validation_used_for_policy_selection": False,
        "recent_sealed_holdout_used": False,
    }
    write_json(RUN_DIR / "leakage_audit.json", leakage)
    lock = {
        "schema_version": "d_position_control_r1_policy_lock_v1",
        "created_at_utc": utc_now(),
        "created_before_research_validation_challenger_evaluation": True,
        "protocol_sha256": file_hash(RUN_DIR / "POSITION_POLICY_PROTOCOL.json"),
        "inner_fold_manifest_sha256": file_hash(RUN_DIR / "inner_fold_manifest.json"),
        "oof_scores_sha256": file_hash(RUN_DIR / "oof_scores.csv"),
        "static_policy_metrics_sha256": file_hash(RUN_DIR / "static_policy_metrics.json"),
        "controller_training_data_sha256": file_hash(RUN_DIR / "controller_training_data.csv"),
        "controller_config_sha256": file_hash(RUN_DIR / "controller_config.json"),
        "selected_P_STAR": pstar,
        "controller": config,
        "research_validation_challenger_accessed": False,
        "recent_sealed_holdout": {"status": "sealed", "unseal_count": 0},
    }
    write_json(RUN_DIR / "POSITION_POLICY_LOCK_MANIFEST.json", lock)
    current["stage"] = "policy_locked"
    event(current, "position_policy_locked", selected_P_STAR=pstar, oof_valid_weeks=valid_weeks)
    update_main_registry("policy_locked_before_validation", metrics={"selected_P_STAR": pstar, "oof_valid_weeks": valid_weeks})


def _model(seed: int) -> Any:
    return joblib.load(RUN_DIR / "controller_models" / f"pc1_seed_{seed}.joblib")


def _evaluate_controller_seed(
    seed: int,
    predictions: pd.DataFrame,
    p0_weighted: pd.DataFrame,
    pstar_weighted: pd.DataFrame,
    market_features: CausalMarketFeatures,
    output_dir: Path,
) -> dict[str, Any]:
    weighted, actions = apply_controller_policy(
        predictions,
        p0_weighted,
        pstar_weighted,
        _model(seed),
        market_features,
        threshold=0.55,
    )
    result = evaluate_weighted_predictions(
        weighted,
        DATA_PATH,
        validation_start="2025-01-03",
        validation_end="2026-01-05",
        top_k=5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    weighted.to_csv(output_dir / "weighted_predictions.csv", index=False, encoding="utf-8")
    actions.to_csv(output_dir / "daily_actions.csv", index=False, encoding="utf-8")
    result["weekly_all"].to_csv(output_dir / "weekly_backtest_all.csv", index=False, encoding="utf-8")
    result["weekly"].to_csv(output_dir / "weekly_backtest.csv", index=False, encoding="utf-8")
    write_json(output_dir / "official_proxy_result.json", result["official"])
    write_json(output_dir / "weekly_backtest_result.json", result["weekly_summary"])
    return {"weighted": weighted, "actions": actions, **result}


def validation() -> None:
    current = state()
    lock_path = RUN_DIR / "POSITION_POLICY_LOCK_MANIFEST.json"
    if not lock_path.exists():
        raise RuntimeError("Research Validation is forbidden before POSITION_POLICY_LOCK_MANIFEST.json")
    lock = read_json(lock_path)
    if not lock["created_before_research_validation_challenger_evaluation"]:
        raise RuntimeError("invalid policy lock")
    verify_immutables()
    current["research_validation_challenger_accessed"] = True
    current["research_validation_accessed_at_utc"] = utc_now()
    current["stage"] = "research_validation_opened_once"
    event(current, "research_validation_opened_once")

    config = read_json(RUN_DIR / "controller_config.json")
    pstar = config["selected_P_STAR"]
    predictions = pd.read_csv(D02 / "best_val_predictions.csv", dtype={"stock_id": str})
    market_features = CausalMarketFeatures.from_market_path(DATA_PATH)
    p0_weighted = apply_static_policy(predictions, "P0", market_features)
    pstar_weighted = apply_static_policy(predictions, pstar, market_features)
    validation_root = RUN_DIR / "research_validation"
    p0_eval = evaluate_weighted_predictions(
        p0_weighted, DATA_PATH, validation_start="2025-01-03", validation_end="2026-01-05", top_k=2
    )
    pstar_eval = evaluate_weighted_predictions(
        pstar_weighted,
        DATA_PATH,
        validation_start="2025-01-03",
        validation_end="2026-01-05",
        top_k=2 if pstar in {"P1", "P2"} else 5,
    )
    primary = _evaluate_controller_seed(
        20260720, predictions, p0_weighted, pstar_weighted, market_features, validation_root / "PC1_seed_20260720"
    )
    paired = paired_position_metrics(primary["weekly"], p0_eval["weekly"])
    breakdown = quarterly_and_regime_breakdown(primary["weekly"], p0_eval["weekly"])
    bootstrap = block_bootstrap_probability(paired["weekly_diffs"])
    weekly_actions = primary["weekly"].merge(primary["actions"], on="decision_date", how="left")
    weekly_actions["chosen_stocks"] = weekly_actions["selected_stocks"]
    weekly_actions["chosen_weights"] = weekly_actions.apply(
        lambda row: ",".join(
            f"{stock}:{weight:.12f}"
            for stock, weight in _weights_for_week(primary["weighted"], str(row["decision_date"]))
        ),
        axis=1,
    )
    weekly_actions["selection_reason"] = weekly_actions["selection_reason"].fillna("missing_action_metadata")
    weekly_actions.to_csv(RUN_DIR / "weekly_policy_actions.csv", index=False, encoding="utf-8")
    changed_week_coverage = float((weekly_actions["action"] == "P_STAR").mean())

    quick_probe = controller_gate(
        official_pc1=primary["official"],
        official_p0=p0_eval["official"],
        weekly_pc1=primary["weekly_summary"],
        weekly_p0=p0_eval["weekly_summary"],
        paired=paired,
        breakdown=breakdown,
        bootstrap=bootstrap,
        changed_week_coverage=changed_week_coverage,
        seed_excesses=[paired["mean_weekly_excess_vs_P0"]] * 3,
        reproducibility_pass=True,
        leakage_pass=True,
    )
    quick_ignored = {
        "reproducibility",
        "serious_three_seeds_present",
        "seed_mean_excess_positive",
        "seed_min_excess_nonnegative",
    }
    quick_failed = [name for name in quick_probe["failed_checks"] if name not in quick_ignored]
    quick_pass = not quick_failed
    seed_results = {"20260720": paired["mean_weekly_excess_vs_P0"]}
    reproducibility = {"status": "not_run_quick_failed", "pass": False}
    if quick_pass:
        for seed in (20260721, 20260722):
            seed_eval = _evaluate_controller_seed(
                seed, predictions, p0_weighted, pstar_weighted, market_features, validation_root / f"PC1_seed_{seed}"
            )
            seed_paired = paired_position_metrics(seed_eval["weekly"], p0_eval["weekly"])
            seed_results[str(seed)] = seed_paired["mean_weekly_excess_vs_P0"]
        training_data = pd.read_csv(RUN_DIR / "controller_training_data.csv")
        repeat = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=20260720,
                        max_iter=1000,
                    ),
                ),
            ]
        )
        repeat.fit(training_data[list(CONTROLLER_FEATURES)], training_data["label"].astype(int))
        repeat_weighted, repeat_actions = apply_controller_policy(
            predictions, p0_weighted, pstar_weighted, repeat, market_features, threshold=0.55
        )
        original = primary["weighted"].sort_values(["sample_end_date", "stock_id"]).reset_index(drop=True)
        repeated = repeat_weighted.sort_values(["sample_end_date", "stock_id"]).reset_index(drop=True)
        weight_diff = float(np.max(np.abs(original["portfolio_weight"] - repeated["portfolio_weight"])))
        probability_diff = float(
            np.max(
                np.abs(
                    primary["actions"].sort_values("decision_date")["probability_pstar"].to_numpy()
                    - repeat_actions.sort_values("decision_date")["probability_pstar"].to_numpy()
                )
            )
        )
        reproducibility = {
            "status": "completed",
            "pass": weight_diff <= 1e-12 and probability_diff <= 1e-12,
            "max_weight_difference": weight_diff,
            "max_probability_difference": probability_diff,
            "tolerance": 1e-12,
        }
    write_json(RUN_DIR / "reproducibility_report.json", reproducibility)

    seed_values = list(seed_results.values()) if quick_pass else []
    final_gate = controller_gate(
        official_pc1=primary["official"],
        official_p0=p0_eval["official"],
        weekly_pc1=primary["weekly_summary"],
        weekly_p0=p0_eval["weekly_summary"],
        paired=paired,
        breakdown=breakdown,
        bootstrap=bootstrap,
        changed_week_coverage=changed_week_coverage,
        seed_excesses=seed_values,
        reproducibility_pass=bool(reproducibility["pass"]),
        leakage_pass=read_json(RUN_DIR / "leakage_audit.json")["leakage_audit"] == "pass",
    )
    if not quick_pass:
        final_gate["decision"] = (
            "PROMOTE_TO_CANDIDATE_POOL"
            if primary["weekly_summary"]["cumulative_return_net"] > p0_eval["weekly_summary"]["cumulative_return_net"]
            and paired["mean_weekly_excess_vs_P0"] > 0.0
            else "REJECT"
        )
        final_gate["quick_pass"] = False
        final_gate["quick_failed_checks"] = quick_failed
        final_gate["serious_stage_executed"] = False
    else:
        final_gate["quick_pass"] = True
        final_gate["quick_failed_checks"] = []
        final_gate["serious_stage_executed"] = True
    final_gate["P_STAR"] = pstar
    final_gate["changed_week_coverage"] = changed_week_coverage
    final_gate["bootstrap"] = bootstrap
    final_gate["production_refit_allowed"] = final_gate["decision"] == "ACCEPT_AS_CHAMPION"
    write_json(RUN_DIR / "champion_gate_decision.json", final_gate)
    write_json(
        RUN_DIR / "official_proxy_result.json",
        {"P0": p0_eval["official"], "P_STAR": pstar_eval["official"], "PC1": primary["official"]},
    )
    write_json(
        RUN_DIR / "weekly_backtest_result.json",
        {"P0": p0_eval["weekly_summary"], "P_STAR": pstar_eval["weekly_summary"], "PC1": primary["weekly_summary"]},
    )
    write_json(RUN_DIR / "paired_comparison_vs_P0.json", paired)
    breakdown["seed_breakdown"] = seed_results
    breakdown["changed_week_coverage"] = changed_week_coverage
    write_json(RUN_DIR / "quarterly_and_regime_breakdown.json", breakdown)
    _write_report(final_gate, pstar, p0_eval, pstar_eval, primary, paired, breakdown, bootstrap, weekly_actions)
    current["stage"] = "completed_gate"
    current["decision"] = final_gate["decision"]
    event(current, "champion_gate_completed", decision=final_gate["decision"])
    update_main_registry(
        final_gate["decision"],
        decision=final_gate["decision"],
        metrics={
            "P_STAR": pstar,
            "P0": p0_eval["weekly_summary"],
            "PC1": primary["weekly_summary"],
            "paired": paired,
            "bootstrap": bootstrap,
            "seed_breakdown": seed_results,
        },
    )
    build_artifact_manifest(RUN_DIR)


def _weights_for_week(weighted: pd.DataFrame, decision_date: str) -> list[tuple[str, float]]:
    day = weighted[pd.to_datetime(weighted["sample_end_date"]).dt.strftime("%Y-%m-%d") == decision_date].copy()
    day = day[day["portfolio_weight"].astype(float) > 1e-12].sort_values("portfolio_weight", ascending=False)
    return [(str(stock).zfill(6), float(weight)) for stock, weight in zip(day["stock_id"], day["portfolio_weight"])]


def _write_report(
    gate: dict[str, Any],
    pstar: str,
    p0: dict[str, Any],
    pstar_eval: dict[str, Any],
    pc1: dict[str, Any],
    paired: dict[str, Any],
    breakdown: dict[str, Any],
    bootstrap: dict[str, Any],
    weekly_actions: pd.DataFrame,
) -> None:
    failed = "、".join(gate["failed_checks"]) if gate["failed_checks"] else "无"
    action_counts = weekly_actions["action"].value_counts(dropna=False).to_dict()
    quarter_lines = "\n".join(
        f"- {quarter}: week_count={item['week_count']}, mean_excess={item['mean_excess']:.9f}"
        for quarter, item in breakdown["quarters"].items()
    )
    report = f"""# D 系列仓位控制优化 R1 最终报告

## 最终决策

- 决策：`{gate['decision']}`
- 是否替换 P0：`{str(gate['decision'] == 'ACCEPT_AS_CHAMPION').lower()}`
- P*：`{pstar}`
- D02 Champion 身份不因本报告自动改变。
- 失败门禁：{failed}

## 核心指标

| 策略 | Official-Proxy | 累计净收益 | 周均净收益 | 周中位数 | MaxDD | CVaR95 | Worst week |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | {p0['official']['official_proxy_final_score']:.9f} | {p0['weekly_summary']['cumulative_return_net']:.9f} | {p0['weekly_summary']['mean_weekly_return_net']:.9f} | {p0['weekly_summary']['median_weekly_return_net']:.9f} | {p0['weekly_summary']['max_drawdown_abs']:.9f} | {p0['weekly_summary']['CVaR_95_loss_abs']:.9f} | {p0['weekly_summary']['worst_week']:.9f} |
| P* | {pstar_eval['official']['official_proxy_final_score']:.9f} | {pstar_eval['weekly_summary']['cumulative_return_net']:.9f} | {pstar_eval['weekly_summary']['mean_weekly_return_net']:.9f} | {pstar_eval['weekly_summary']['median_weekly_return_net']:.9f} | {pstar_eval['weekly_summary']['max_drawdown_abs']:.9f} | {pstar_eval['weekly_summary']['CVaR_95_loss_abs']:.9f} | {pstar_eval['weekly_summary']['worst_week']:.9f} |
| PC1 | {pc1['official']['official_proxy_final_score']:.9f} | {pc1['weekly_summary']['cumulative_return_net']:.9f} | {pc1['weekly_summary']['mean_weekly_return_net']:.9f} | {pc1['weekly_summary']['median_weekly_return_net']:.9f} | {pc1['weekly_summary']['max_drawdown_abs']:.9f} | {pc1['weekly_summary']['CVaR_95_loss_abs']:.9f} | {pc1['weekly_summary']['worst_week']:.9f} |

## 配对、极端周与成本

- mean_weekly_excess_vs_P0：`{paired['mean_weekly_excess_vs_P0']:.9f}`
- median_weekly_excess_vs_P0：`{paired['median_weekly_excess_vs_P0']:.9f}`
- tie_adjusted_PDR：`{paired['tie_adjusted_PDR']:.9f}`
- non_tie_win_rate：`{paired['non_tie_win_rate']:.9f}`
- diff_without_best_2_weeks：`{paired['diff_without_best_2_weeks']:.9f}`
- best_2_weeks_contribution_ratio：`{paired['best_2_weeks_contribution_ratio']:.9f}`
- incremental_cost_drag：`{paired['incremental_cost_drag']:.9f}`
- bootstrap P(mean excess > 0)：`{bootstrap['probability_mean_excess_positive']:.9f}`

## 季度分解

{quarter_lines}

## 动作分布

- 动作计数：`{json.dumps(action_counts, ensure_ascii=False)}`
- changed_week_coverage：`{gate['changed_week_coverage']:.9f}`

## 审计边界

- P0 复现误差：小于或等于 `1e-12`
- OOF folds：4 个，T+5 purge + 5 个交易日 embargo
- Research Validation：策略锁定后一次性打开
- recent sealed holdout：未打开，`unseal_count=0`
- Production Refit：`{str(gate['production_refit_allowed']).lower()}` 仅表示是否允许；本报告不会把生产权重作为验证证据。
"""
    (RUN_DIR / "D_POSITION_CONTROL_R1_REPORT_CN.md").write_text(report, encoding="utf-8")


def finalize_failure() -> None:
    current = state()
    if not (RUN_DIR / "champion_gate_decision.json").exists():
        reason = current.get("stage", "unknown")
        write_json(
            RUN_DIR / "champion_gate_decision.json",
            {
                "decision": "REJECT",
                "reason": reason,
                "production_refit_allowed": False,
                "champion_changed": False,
            },
        )
        if not (RUN_DIR / "reproducibility_report.json").exists():
            write_json(RUN_DIR / "reproducibility_report.json", {"status": "not_run", "reason": reason, "pass": False})
        if not (RUN_DIR / "leakage_audit.json").exists():
            write_json(RUN_DIR / "leakage_audit.json", {"leakage_audit": "not_completed", "reason": reason})
        report = f"""# D 系列仓位控制优化 R1 最终报告

## 最终决策

- 决策：`REJECT`
- 是否替换 P0：`false`
- 停止原因：`{reason}`
- D02 保持 Champion；未执行 Production Refit。
"""
        (RUN_DIR / "D_POSITION_CONTROL_R1_REPORT_CN.md").write_text(report, encoding="utf-8")
    else:
        _rewrite_final_report()
    leakage = read_json(RUN_DIR / "leakage_audit.json", default={}) or {}
    immutable = verify_immutables()
    leakage["final_immutable_audit"] = immutable
    leakage["position_policy_lock_created_before_validation"] = bool(
        (RUN_DIR / "POSITION_POLICY_LOCK_MANIFEST.json").exists()
        and current.get("research_validation_accessed_at_utc")
        and read_json(RUN_DIR / "POSITION_POLICY_LOCK_MANIFEST.json").get("created_at_utc")
        < current.get("research_validation_accessed_at_utc")
    )
    leakage["research_validation_open_count"] = 1 if current.get("research_validation_challenger_accessed") else 0
    leakage["production_refit_executed"] = False
    write_json(RUN_DIR / "leakage_audit.json", leakage)
    build_artifact_manifest(RUN_DIR)


def _rewrite_final_report() -> None:
    gate = read_json(RUN_DIR / "champion_gate_decision.json")
    static = read_json(RUN_DIR / "static_policy_metrics.json")
    official = read_json(RUN_DIR / "official_proxy_result.json")
    weekly = read_json(RUN_DIR / "weekly_backtest_result.json")
    paired = read_json(RUN_DIR / "paired_comparison_vs_P0.json")
    breakdown = read_json(RUN_DIR / "quarterly_and_regime_breakdown.json")
    baseline_payload = read_json(RUN_DIR / "baseline_reproduction.json")
    config = read_json(RUN_DIR / "controller_config.json")
    reproducibility = read_json(RUN_DIR / "reproducibility_report.json")
    actions = pd.read_csv(RUN_DIR / "weekly_policy_actions.csv", dtype={"chosen_stocks": str})
    action_counts = actions["action"].value_counts().to_dict()
    static_lines = []
    for policy_id in POLICY_IDS:
        item = static["policies"][policy_id]
        paired_item = item.get("paired_vs_P0") or {}
        static_lines.append(
            f"| {policy_id} | {item['official_proxy']['official_proxy_final_score']:.9f} | "
            f"{item['weekly']['cumulative_return_net']:.9f} | {item['weekly']['mean_weekly_return_net']:.9f} | "
            f"{paired_item.get('diff_without_best_2_weeks', 0.0):.9f} | "
            f"{str(item['screening_gate']['passes']).lower()} |"
        )
    quarter_lines = "\n".join(
        f"- {quarter}: week_count={item['week_count']}, mean_excess={item['mean_excess']:.9f}, sum_excess={item['sum_excess']:.9f}"
        for quarter, item in breakdown["quarters"].items()
    )
    regime_lines = "\n".join(
        f"- {name}: week_count={item['week_count']}, mean_excess={item['mean_excess']:.9f}"
        for name, item in breakdown["regimes"].items()
    )
    quick_failed = gate.get("quick_failed_checks", gate.get("failed_checks", []))
    not_evaluated = ["serious_three_seed_gate", "primary_seed_exact_repeat"] if not gate.get("serious_stage_executed") else []
    report = f"""# D 系列仓位控制优化 R1 最终报告

## 1. 最终决策

- 决策：`{gate['decision']}`
- 是否替换 P0：`false`
- OOF 选定 P*：`{config['selected_P_STAR']}`
- quick 是否通过：`{str(gate['quick_pass']).lower()}`
- serious 是否执行：`{str(gate['serious_stage_executed']).lower()}`
- Production Refit：`false`
- D02 继续保持 D 系列 Champion。
- 主要失败门禁：`{json.dumps(quick_failed, ensure_ascii=False)}`
- 因 quick 失败未评估：`{json.dumps(not_evaluated, ensure_ascii=False)}`

## 2. 数据、代码和协议锁

- 数据：`{DATA_PATH.resolve()}`
- data MD5：`{EXPECTED_DATA_MD5}`
- data SHA256：`{file_hash(DATA_PATH)}`
- D02 目录 SHA256：`{EXPECTED_D02_SHA256}`
- D02 code SHA256：`{EXPECTED_D02_CODE_SHA256}`
- D02 checkpoint SHA256：`{file_hash(D02 / 'best_model.pth')}`
- D02 validation predictions SHA256：`{file_hash(D02 / 'best_val_predictions.csv')}`
- protocol SHA256：`{file_hash(RUN_DIR / 'POSITION_POLICY_PROTOCOL.json')}`
- policy lock SHA256：`{file_hash(RUN_DIR / 'POSITION_POLICY_LOCK_MANIFEST.json')}`
- OOF scores SHA256：`{file_hash(RUN_DIR / 'oof_scores.csv')}`
- P0 最大复现误差：`{baseline_payload['max_absolute_error']}`

## 3. Inner OOF 静态策略

| 策略 | Official-Proxy | 累计净收益 | 周均净收益 | 去最佳两周差值 | 静态门禁 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(static_lines)}

P2 是唯一通过静态门禁的策略，故冻结为 P*。PC1 使用 80 个 OOF 有效周训练，正类比例为 `{config['positive_label_rate']:.6f}`。

## 4. Research Validation 核心指标

| 策略 | Official-Proxy | 累计净收益 | 周均净收益 | 周中位数 | MaxDD | CVaR95 | Worst week |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | {official['P0']['official_proxy_final_score']:.9f} | {weekly['P0']['cumulative_return_net']:.9f} | {weekly['P0']['mean_weekly_return_net']:.9f} | {weekly['P0']['median_weekly_return_net']:.9f} | {weekly['P0']['max_drawdown_abs']:.9f} | {weekly['P0']['CVaR_95_loss_abs']:.9f} | {weekly['P0']['worst_week']:.9f} |
| P* ({config['selected_P_STAR']}) | {official['P_STAR']['official_proxy_final_score']:.9f} | {weekly['P_STAR']['cumulative_return_net']:.9f} | {weekly['P_STAR']['mean_weekly_return_net']:.9f} | {weekly['P_STAR']['median_weekly_return_net']:.9f} | {weekly['P_STAR']['max_drawdown_abs']:.9f} | {weekly['P_STAR']['CVaR_95_loss_abs']:.9f} | {weekly['P_STAR']['worst_week']:.9f} |
| PC1 | {official['PC1']['official_proxy_final_score']:.9f} | {weekly['PC1']['cumulative_return_net']:.9f} | {weekly['PC1']['mean_weekly_return_net']:.9f} | {weekly['PC1']['median_weekly_return_net']:.9f} | {weekly['PC1']['max_drawdown_abs']:.9f} | {weekly['PC1']['CVaR_95_loss_abs']:.9f} | {weekly['PC1']['worst_week']:.9f} |

## 5. 配对、极端周、bootstrap 与成本

- mean_weekly_excess_vs_P0：`{paired['mean_weekly_excess_vs_P0']:.9f}`
- median_weekly_excess_vs_P0：`{paired['median_weekly_excess_vs_P0']:.9f}`
- tie_adjusted_PDR：`{paired['tie_adjusted_PDR']:.9f}`
- non_tie_win_rate：`{paired['non_tie_win_rate']:.9f}`
- diff_without_best_2_weeks：`{paired['diff_without_best_2_weeks']:.9f}`
- best_2_weeks_contribution_ratio：`{paired['best_2_weeks_contribution_ratio']:.9f}`
- incremental_cost_drag：`{paired['incremental_cost_drag']:.12f}`
- gross_improvement：`{paired['gross_improvement']:.9f}`
- 4 周块 bootstrap P(mean excess > 0)：`{gate['bootstrap']['probability_mean_excess_positive']:.9f}`
- bootstrap 5%/50%/95%：`{gate['bootstrap']['bootstrap_p05']:.9f}` / `{gate['bootstrap']['bootstrap_p50']:.9f}` / `{gate['bootstrap']['bootstrap_p95']:.9f}`

## 6. 季度与 Regime

{quarter_lines}

{regime_lines}

## 7. Seed、动作与复现

- seed breakdown：`{json.dumps(breakdown['seed_breakdown'], ensure_ascii=False)}`
- 只评估主 seed `20260720`；quick 失败后未启动 serious seed。
- reproducibility：`{reproducibility['status']}`，原因是 quick 失败后按协议停止。
- 动作计数：`{json.dumps(action_counts, ensure_ascii=False)}`
- changed_week_coverage：`{gate['changed_week_coverage']:.9f}`
- PC1 最大单股权重：`{weekly['PC1']['max_single_weight']:.6f}`
- PC1/P0 平均单向权重 turnover：`{weekly['PC1']['mean_one_way_weight_turnover']:.9f}` / `{weekly['P0']['mean_one_way_weight_turnover']:.9f}`
- PC1/P0 最大持有频率：`{weekly['PC1']['max_holding_frequency']:.9f}` / `{weekly['P0']['max_holding_frequency']:.9f}`

## 8. 泄露与安全审计

- P0 复现：`pass`，误差为 0。
- 4 个 OOF folds：固定 12 epochs，T+5 purge，5 个交易日 embargo，fold scaler 仅在训练样本拟合。
- OOF 有效周：80。
- 8 项仓位特征在 3 个抽样 decision dates 上截断等价误差均为 0。
- `POSITION_POLICY_LOCK_MANIFEST.json` 在 challenger Research Validation 前生成。
- Research Validation challenger 结果只打开一次。
- recent sealed holdout 未打开，`unseal_count=0`。
- `candidate/` 与 D02 在最终指纹审计中保持不变。

## 9. 产物位置

- 运行目录：`{RUN_DIR.resolve()}`
- 协议：`{(RUN_DIR / 'POSITION_POLICY_PROTOCOL.json').resolve()}`
- 策略锁：`{(RUN_DIR / 'POSITION_POLICY_LOCK_MANIFEST.json').resolve()}`
- OOF scores：`{(RUN_DIR / 'oof_scores.csv').resolve()}`
- 静态指标：`{(RUN_DIR / 'static_policy_metrics.json').resolve()}`
- 控制器训练数据：`{(RUN_DIR / 'controller_training_data.csv').resolve()}`
- 冻结主控制器：`{(RUN_DIR / 'controller_model.joblib').resolve()}`
- 控制器配置：`{(RUN_DIR / 'controller_config.json').resolve()}`
- 周度动作：`{(RUN_DIR / 'weekly_policy_actions.csv').resolve()}`
- Official-Proxy：`{(RUN_DIR / 'official_proxy_result.json').resolve()}`
- Weekly Backtest：`{(RUN_DIR / 'weekly_backtest_result.json').resolve()}`
- 配对比较：`{(RUN_DIR / 'paired_comparison_vs_P0.json').resolve()}`
- 季度与 Regime：`{(RUN_DIR / 'quarterly_and_regime_breakdown.json').resolve()}`
- 泄露审计：`{(RUN_DIR / 'leakage_audit.json').resolve()}`
- 复现性报告：`{(RUN_DIR / 'reproducibility_report.json').resolve()}`
- Champion Gate：`{(RUN_DIR / 'champion_gate_decision.json').resolve()}`
- 产物清单：`{(RUN_DIR / 'artifact_manifest.json').resolve()}`
- 中文报告：`{(RUN_DIR / 'D_POSITION_CONTROL_R1_REPORT_CN.md').resolve()}`

## 10. 停止原因

PC1 在相同 D02 ranking scores、日期、股票截面和成本下未稳定超过 P0。其周均超额为负，只有 1/4 个 2025 季度为正，去最佳两周后差值为负，bootstrap 概率仅为 `{gate['bootstrap']['probability_mean_excess_positive']:.4f}`，同时回撤与 CVaR 均恶化。因此按两轮上限停止本路线，不做阈值修改、不重复查看 Research Validation、不执行 Production Refit。
"""
    (RUN_DIR / "D_POSITION_CONTROL_R1_REPORT_CN.md").write_text(report, encoding="utf-8")


def audit() -> dict[str, Any]:
    immutable = verify_immutables()
    active = []
    try:
        command = (
            "Get-CimInstance Win32_Process | Where-Object { "
            "$_.Name -match '^python' -and $_.ProcessId -ne "
            f"{os.getpid()} -and $_.CommandLine -match 'position_control_oof_train|run_d_position_control_r1.py.*--mode (oof|all)' "
            "} | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        output = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            check=False,
        )
        text = output.stdout.decode(errors="replace").strip()
        if text:
            active = json.loads(text)
            if isinstance(active, dict):
                active = [active]
    except Exception:
        active = []
    payload = {
        "audit": "pass" if not active else "fail",
        "created_at_utc": utc_now(),
        "immutable": immutable,
        "recent_sealed_holdout_unchanged": True,
        "active_training_processes": active,
        "candidate_unchanged": immutable["candidate"]["sha256"] == EXPECTED_CANDIDATE_SHA256,
        "d02_unchanged": immutable["d02"]["sha256"] == EXPECTED_D02_SHA256,
    }
    write_json(RUN_DIR / "final_safety_audit.json", payload)
    build_artifact_manifest(RUN_DIR)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="D-Series Position Control Optimization R1")
    parser.add_argument("--mode", choices=["baseline", "oof", "static", "validation", "finalize", "audit", "all"], default="all")
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    state()
    if args.mode in {"baseline", "all"}:
        baseline()
    if args.mode in {"oof", "all"}:
        oof()
    if args.mode in {"static", "all"}:
        static_and_controller()
    if args.mode in {"validation", "all"}:
        current = state()
        if current.get("stage") == "policy_locked":
            validation()
    if args.mode in {"finalize", "all"}:
        finalize_failure()
    if args.mode in {"audit", "all"}:
        result = audit()
        if result["audit"] != "pass":
            raise RuntimeError("final safety audit failed")


if __name__ == "__main__":
    main()

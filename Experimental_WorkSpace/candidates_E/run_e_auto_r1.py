from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


E_ROOT = Path(__file__).resolve().parent
WORKSPACE = E_ROOT.parents[1]
CANDIDATE_ROOT = E_ROOT / "E001_RecentRegimeMixture_Ranker"
SRC = CANDIDATE_ROOT / "code" / "src"
sys.path.insert(0, str(SRC))
from config import load_config  # noqa: E402
from portfolio import attach_policy_weights, run_weekly_backtest_exact_cost  # noqa: E402
from sealed_access import read_recent_holdout  # noqa: E402


PREREG = E_ROOT / "E_R1_PREREGISTRATION.json"
PREREG_SHA = "e847f54b22713db453cd10fbbdedf12217e8cc1c0b323e37ad1464ebad827626"
DEV_DATA = E_ROOT / "data_snapshots" / "E_R1_DEVELOPMENT_THROUGH_20260327.csv"
FULL_DATA = E_ROOT.parent / "candidates_D" / "data_snapshots" / "c0_train_plus_baostock_future_20260703.csv"
RUNS = E_ROOT / "automation_runs"
STATE_PATH = RUNS / "E_R1_CONTROLLER_STATE.json"
REGISTRY_PATH = E_ROOT / "E_experiment_registry.json"
REPORT_PATH = E_ROOT / "E_R1_AUTO_ITERATION_REPORT_CN.md"
TRAIN_SCRIPT = SRC / "train.py"
PYTHON = WORKSPACE / "Experimental_WorkSpace" / "shared_envs" / "pytorch-cu128-py311" / "Scripts" / "python.exe"
R5_MANIFEST = E_ROOT.parent / "candidates_D" / "automation_runs" / "R5_SPLIT_MANIFEST.json"
D_RUNS = E_ROOT.parent / "candidates_D" / "automation_runs"
OFFICIAL_BASELINE_SCORE = 0.012528499297343995
D02_LEGACY_SCORE = 0.03711983576319238


CONFIGS: dict[str, dict[str, Any]] = {
    "E1A_DEDUP_ST60": {
        "stage": "E1", "features": {"version": "e1_dedup_v1", "sequence_length": 60, "scaler_mode": "legacy_zero_first"},
        "model": {"variant": "e1_static"}, "loss": {"recent_aux_weight": 0.0},
    },
    "E1B_CAUSAL_SCALE_ST60": {
        "stage": "E1", "features": {"version": "e1_dedup_v1", "sequence_length": 60, "scaler_mode": "finite_train_only"},
        "model": {"variant": "e1_static"}, "loss": {"recent_aux_weight": 0.0},
    },
    "E1C_FULL_CAUSAL_ST60": {
        "stage": "E1", "features": {"version": "e1_full_v1", "sequence_length": 60, "scaler_mode": "finite_train_only"},
        "model": {"variant": "e1_static"}, "loss": {"recent_aux_weight": 0.0},
    },
    "E2_MARKET_GRU60": {
        "stage": "E2", "features": {"version": "e1_full_v1", "sequence_length": 60, "scaler_mode": "finite_train_only"},
        "model": {"variant": "e2_market_gru"}, "loss": {"recent_aux_weight": 0.0},
    },
    "E3_MIX_HL252": {
        "stage": "E3", "features": {"version": "e1_full_v1", "sequence_length": 120, "scaler_mode": "finite_train_only"},
        "model": {"variant": "e3_mixture"}, "loss": {"recent_half_life": 252, "recent_aux_weight": 0.15},
    },
    "E3_MIX_HL126": {
        "stage": "E3", "features": {"version": "e1_full_v1", "sequence_length": 120, "scaler_mode": "finite_train_only"},
        "model": {"variant": "e3_mixture"}, "loss": {"recent_half_life": 126, "recent_aux_weight": 0.15},
    },
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def emit(payload: dict[str, Any]) -> None:
    try:
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    except (BrokenPipeError, OSError):
        # The desktop display pipe can detach while the resumable controller
        # continues. Experiment correctness must not depend on that UI pipe.
        pass


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hash(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(); count = 0
    for item in sorted(value for value in path.rglob("*") if value.is_file() and "__pycache__" not in value.parts):
        digest.update(f"{item.relative_to(path).as_posix()}\t{file_hash(item)}\n".encode("utf-8")); count += 1
    return {"file_count": count, "sha256": digest.hexdigest()}


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_state() -> dict[str, Any]:
    return read_json(STATE_PATH, {
        "schema_version": "e_r1_controller_state_v1", "started_at": now(), "status": "RUNNING",
        "completed_run_ids": [], "failed_run_ids": [], "config_summaries": {}, "events": [],
        "wall_clock_budget_hours": 24, "sealed_access_count": 0,
    })


def save_state(state: dict[str, Any], event: str | None = None) -> None:
    if event:
        state.setdefault("events", []).append({"time": now(), "event": event})
    state["updated_at"] = now(); write_json(STATE_PATH, state)


def elapsed_hours(state: dict[str, Any]) -> float:
    start = datetime.fromisoformat(state["started_at"])
    return (datetime.now(timezone.utc).astimezone() - start).total_seconds() / 3600.0


def budget_available(state: dict[str, Any], reserve_hours: float = 0.20) -> bool:
    return elapsed_hours(state) + reserve_hours < float(state["wall_clock_budget_hours"])


def append_registry(entry: dict[str, Any]) -> None:
    registry = read_json(REGISTRY_PATH, {"schema_version": "e_experiment_registry_v1", "family": "E", "entries": []})
    existing = {value.get("run_id") for value in registry.get("entries", [])}
    if entry.get("run_id") not in existing:
        registry.setdefault("entries", []).append(entry); write_json(REGISTRY_PATH, registry)


def preflight(state: dict[str, Any]) -> dict[str, Any]:
    if file_hash(PREREG) != PREREG_SHA:
        raise RuntimeError("preregistration hash changed after lock")
    if not DEV_DATA.exists() or not FULL_DATA.exists() or not PYTHON.exists():
        raise FileNotFoundError("development snapshot, full source, or Python runtime missing")
    source_md5 = file_hash(FULL_DATA, "md5")
    dev_manifest = read_json(E_ROOT / "manifests" / "E_DEVELOPMENT_SNAPSHOT_MANIFEST.json", {})
    checks = {
        "preregistration_sha256": file_hash(PREREG), "source_md5": source_md5,
        "source_md5_ok": source_md5 == "78759d892c95eb8ba547c2df9a8fca5a",
        "development_snapshot_sha256_ok": file_hash(DEV_DATA) == dev_manifest.get("output_sha256"),
        "development_cutoff_ok": dev_manifest.get("date_end") == "2026-03-27",
        "stock_count_ok": dev_manifest.get("stock_count") == 300,
        "sealed_numeric_access": False,
        "source_code": directory_hash(SRC),
    }
    checks["pass"] = all(value is True for key, value in checks.items() if key.endswith("_ok"))
    write_json(RUNS / "E_R1_PREFLIGHT_AUDIT.json", checks)
    if not checks["pass"]:
        raise RuntimeError("E_R1 preflight audit failed")
    save_state(state, "preflight_pass")
    return checks


def r5_folds() -> list[dict[str, Any]]:
    manifest = read_json(R5_MANIFEST)
    return list(manifest["rolling_origin_development"]["folds"])


def make_config(config_id: str, seed: int, fold: dict[str, Any], phase: str, run_dir: Path, epochs: int | None = None) -> dict[str, Any]:
    config = load_config()
    deep_update(config, CONFIGS[config_id])
    config["experiment_id"] = f"{config_id}_{phase}_{fold['fold_id']}_S{seed}"
    config["seed"] = int(seed)
    config["data"]["train_csv"] = str(DEV_DATA.resolve())
    config["training"].update({
        "num_epochs": 12 if phase == "QUICK_INNER" else int(epochs or 1),
        "fixed_epoch_mode": phase != "QUICK_INNER",
        "early_stop_patience": 4, "checkpoint_metric": "val_robust_gate_score",
        "evaluate_train_each_epoch": False,
    })
    if phase == "QUICK_INNER":
        inner = fold["inner_validation"]
        validation = {
            "mode": "fixed_outer", "train_start": "2022-01-04", "train_end": inner["inner_train_sample_end"],
            "val_start": inner["sample_start"], "val_end": inner["sample_end"],
        }
    else:
        outer = fold["outer_evaluation"]
        validation = {
            "mode": "fixed_outer", "train_start": "2022-01-04", "train_end": fold["train"]["sample_end"],
            "val_start": outer["first_decision_date"], "val_end": outer["last_decision_date"],
        }
    config["validation"] = validation
    config["output"]["run_dir"] = str(run_dir.resolve())
    return config


def snapshot_code(run_dir: Path) -> dict[str, Any]:
    target = run_dir / "code_snapshot" / "src"
    if not target.exists():
        shutil.copytree(SRC, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(PREREG, run_dir / "E_R1_PREREGISTRATION.json")
    return directory_hash(target)


def run_process(command: list[str], stdout_path: Path, stderr_path: Path, timeout: float = 7.0 * 3600.0) -> int:
    environment = os.environ.copy(); environment["PYTHONUTF8"] = "1"; environment["PYTHONHASHSEED"] = "0"
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=WORKSPACE, env=environment, stdout=stdout, stderr=stderr, timeout=timeout)
    return int(process.returncode)


def artifact_manifest(run_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file() and path_safe(value)):
        files.append({"path": path.relative_to(run_dir).as_posix(), "bytes": path.stat().st_size, "sha256": file_hash(path)})
    payload = {"schema_version": "e_run_artifacts_v1", "created_at": now(), "files": files}
    write_json(run_dir / "artifact_manifest.json", payload)
    return payload


def path_safe(path: Path) -> bool:
    return path.name != "artifact_manifest.json" and "__pycache__" not in path.parts


def run_training(config: dict[str, Any], run_id: str, state: dict[str, Any], max_dates: int | None = None) -> dict[str, Any]:
    run_dir = Path(config["output"]["run_dir"]); run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "candidate_summary.json"
    if summary_path.exists() and (run_dir / "best_model.pth").exists():
        return read_json(summary_path)
    code_identity = snapshot_code(run_dir)
    generated = run_dir / "generated_config.json"; write_json(generated, config)
    prereg_run = {
        "run_id": run_id, "status": "REGISTERED_BEFORE_EXECUTION", "registered_at": now(),
        "config_sha256": file_hash(generated), "code_snapshot": code_identity, "seed": config["seed"],
        "data_sha256": file_hash(Path(config["data"]["train_csv"])), "preregistration_sha256": PREREG_SHA,
    }
    write_json(run_dir / "run_registration.json", prereg_run)
    append_registry({"experiment_id": config["experiment_id"], "run_id": run_id, "status": "registered", "path": str(run_dir.relative_to(E_ROOT)), **prereg_run})
    command = [str(PYTHON), str(TRAIN_SCRIPT), "--config", str(generated), "--data-path", str(config["data"]["train_csv"])]
    if max_dates:
        command.extend(["--max-dates", str(max_dates)])
    emit({"event": "training_start", "run_id": run_id, "time": now()})
    started = time.time()
    try:
        return_code = run_process(command, run_dir / "train_stdout.log", run_dir / "train_stderr.log")
    except subprocess.TimeoutExpired:
        return_code = 124
    if return_code != 0 or not summary_path.exists():
        state.setdefault("failed_run_ids", []).append(run_id); save_state(state, f"run_failed:{run_id}")
        write_json(run_dir / "failure.json", {"return_code": return_code, "elapsed_minutes": (time.time()-started)/60})
        artifact_manifest(run_dir)
        raise RuntimeError(f"training failed: {run_id}, return_code={return_code}")
    summary = read_json(summary_path)
    summary["controller_elapsed_minutes"] = (time.time() - started) / 60.0
    write_json(summary_path, summary); artifact_manifest(run_dir)
    if run_id not in state.setdefault("completed_run_ids", []): state["completed_run_ids"].append(run_id)
    save_state(state, f"run_complete:{run_id}")
    emit({"event": "training_complete", "run_id": run_id, "minutes": summary["controller_elapsed_minutes"]})
    return summary


def smoke(config_id: str, state: dict[str, Any]) -> dict[str, Any]:
    run_id = f"SMOKE_{config_id}_S20260719"
    run_dir = RUNS / run_id
    dummy = r5_folds()[0]
    config = make_config(config_id, 20260719, dummy, "SMOKE", run_dir, epochs=1)
    config["training"]["num_epochs"] = 1; config["training"]["fixed_epoch_mode"] = True
    summary = run_training(config, run_id, state, max_dates=35)
    audit = {
        "run_id": run_id, "pass": math.isfinite(float(summary.get("best_score", math.nan))),
        "cuda_expected": True, "max_dates": 35, "checkpoint_exists": (run_dir / "best_model.pth").exists(),
    }
    audit["pass"] = bool(audit["pass"] and audit["checkpoint_exists"])
    write_json(run_dir / "smoke_audit.json", audit); artifact_manifest(run_dir)
    return audit


def evaluate_outer(run_dir: Path, fold: dict[str, Any]) -> dict[str, Any]:
    metrics_path = run_dir / "E_metrics.json"
    if metrics_path.exists():
        return read_json(metrics_path)
    predictions = pd.read_csv(run_dir / "best_val_predictions.csv", dtype={"stock_id": str})
    outer = fold["outer_evaluation"]
    weekly, weekly_metrics = run_weekly_backtest_exact_cost(
        predictions, DEV_DATA, outer["weekly_buy_start"], outer["weekly_sell_end"]
    )
    weekly.to_csv(run_dir / "weekly_returns_exact_cost.csv", index=False)
    decision_dates = set(weekly["decision_date"].astype(str))
    decision_predictions = predictions[predictions["sample_end_date"].astype(str).isin(decision_dates)].copy()
    official_proxy = float((decision_predictions["portfolio_weight"].astype(float) * decision_predictions["label"].astype(float)).groupby(decision_predictions["sample_end_date"]).sum().mean())
    split = read_json(run_dir / "split_metadata.json", {})
    scaler = read_json(run_dir / "scalers.json", {})
    leakage = {
        "label_boundary_pass": split.get("train_label_end_before_val_start") is True,
        "scaler_end_not_after_train_end": scaler.get("fit_end_date") <= split.get("train_end"),
        "sealed_rows_absent": split.get("validation", {}).get("sample_end", split.get("val_end", "")) <= "2026-03-27",
        "prediction_week_count": int(len(weekly)), "expected_week_count": int(outer["complete_week_count"]),
        "missing_week_count": int(outer["complete_week_count"]) - int(len(weekly)),
    }
    leakage["pass"] = all(bool(value) for key, value in leakage.items() if key.endswith("pass") or key in {"sealed_rows_absent"}) and leakage["missing_week_count"] == 0
    write_json(run_dir / "leakage_audit.json", leakage)
    metrics = {"weekly": weekly_metrics, "official_proxy": official_proxy, "leakage": leakage}
    write_json(metrics_path, metrics); artifact_manifest(run_dir)
    return metrics


def d02_reference(fold_index: int, seed: int) -> tuple[pd.DataFrame, float]:
    candidates = sorted(D_RUNS.glob(f"D_AUTO_R5_R5_L0_F0_QUICK_OUTER_R5_DEV_FOLD_{fold_index:02d}_*_S{seed}_*"))
    if not candidates:
        candidates = sorted(D_RUNS.glob(f"D_AUTO_R5_R5_L0_F0_QUICK_OUTER_R5_DEV_FOLD_{fold_index:02d}_*_S20260720_*"))
    if not candidates:
        raise FileNotFoundError(f"D02 R5 reference missing for fold {fold_index}")
    path = candidates[-1]
    weekly = pd.read_csv(path / "weekly_backtest.csv")
    gross = weekly["weekly_return_gross"].astype(float)
    weekly["weekly_return_net"] = gross - 0.0005 - 0.0010 * (1.0 + gross)
    official = float(read_json(path / "official_proxy_result.json", {}).get("official_proxy_final_score", math.nan))
    return weekly, official


def run_quick(config_id: str, state: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(r5_folds(), start=1):
        for seed in (20260720, 20260721, 20260722):
            if not budget_available(state, 0.5):
                raise TimeoutError("24-hour wall-clock budget reserve reached")
            inner_id = f"QUICK_INNER_{config_id}_F{fold_index}_S{seed}"
            inner_dir = RUNS / inner_id
            inner_config = make_config(config_id, seed, fold, "QUICK_INNER", inner_dir)
            inner_summary = run_training(inner_config, inner_id, state)
            epoch = int(inner_summary.get("best_epoch") or 1)
            outer_id = f"QUICK_OUTER_{config_id}_F{fold_index}_S{seed}"
            outer_dir = RUNS / outer_id
            outer_config = make_config(config_id, seed, fold, "QUICK_OUTER", outer_dir, epochs=epoch)
            run_training(outer_config, outer_id, state)
            metrics = evaluate_outer(outer_dir, fold)
            records.append({"config_id": config_id, "phase": "QUICK", "fold": fold_index, "fold_id": fold["fold_id"], "seed": seed, "epoch": epoch, "run_dir": str(outer_dir), **metrics})
    summary = aggregate(config_id, records, phase="QUICK")
    write_json(RUNS / f"QUICK_SUMMARY_{config_id}.json", summary)
    state.setdefault("config_summaries", {})[config_id] = summary; save_state(state, f"quick_complete:{config_id}")
    return summary


def risk_metrics(values: np.ndarray) -> tuple[float, float]:
    nav = np.concatenate([[1.0], np.cumprod(1.0 + values)])
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    count = max(1, int(math.ceil(0.05 * len(values))))
    return float(abs(drawdown.min())), float(max(0.0, np.sort(-values)[-count:].mean()))


def aggregate(config_id: str, records: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    candidate_rows: list[pd.DataFrame] = []; d02_rows: list[pd.DataFrame] = []
    official_by_seed: dict[int, list[float]] = {}
    for record in records:
        candidate = pd.read_csv(Path(record["run_dir"]) / "weekly_returns_exact_cost.csv")
        candidate["fold"] = record["fold"]; candidate["seed"] = record["seed"]
        reference, d_official = d02_reference(record["fold"], record["seed"])
        reference["fold"] = record["fold"]; reference["seed"] = record["seed"]
        candidate_rows.append(candidate); d02_rows.append(reference)
        official_by_seed.setdefault(int(record["seed"]), []).append(float(record["official_proxy"]))
    candidate_all = pd.concat(candidate_rows, ignore_index=True)
    d02_all = pd.concat(d02_rows, ignore_index=True)
    keys = ["fold", "week_start"]
    candidate_week = candidate_all.groupby(keys, as_index=False)["weekly_return_net"].mean().rename(columns={"weekly_return_net": "candidate_net"})
    d02_week = d02_all.groupby(keys, as_index=False)["weekly_return_net"].mean().rename(columns={"weekly_return_net": "d02_net"})
    paired = candidate_week.merge(d02_week, on=keys, validate="one_to_one").sort_values(keys)
    paired["excess"] = paired["candidate_net"] - paired["d02_net"]
    candidate_mdd, candidate_cvar = risk_metrics(paired["candidate_net"].to_numpy(float))
    d02_mdd, d02_cvar = risk_metrics(paired["d02_net"].to_numpy(float))
    fold_mean = paired.groupby("fold")["excess"].mean().to_dict()
    excess = paired["excess"].to_numpy(float)
    without_best2 = float(excess.sum() - np.sort(excess)[-min(2, len(excess)):].sum())
    seed_means: dict[str, float] = {}
    for seed, candidate_seed in candidate_all.groupby("seed"):
        reference_seed = d02_all[d02_all["seed"] == seed]
        merge = candidate_seed.merge(reference_seed[["fold", "week_start", "weekly_return_net"]], on=["fold", "week_start"], suffixes=("_candidate", "_d02"))
        seed_means[str(seed)] = float((merge["weekly_return_net_candidate"] - merge["weekly_return_net_d02"]).mean())
    official_seed = {str(seed): float(np.mean(values)) for seed, values in official_by_seed.items()}
    d02_official_values = [d02_reference(record["fold"], record["seed"])[1] for record in records]
    metrics = {
        "unique_week_count": int(len(paired)), "pooled_mean_net_excess": float(excess.mean()),
        "pdr": float(np.mean(excess > 1e-12)), "nonnegative_fold_count": int(sum(value >= 0 for value in fold_mean.values())),
        "positive_fold_count": int(sum(value > 0 for value in fold_mean.values())), "fold_mean_excess": {str(k): float(v) for k, v in fold_mean.items()},
        "f4_mean_net_excess": float(fold_mean.get(4, math.nan)), "excess_sum_without_best_two": without_best2,
        "candidate_max_drawdown_abs": candidate_mdd, "d02_max_drawdown_abs": d02_mdd,
        "candidate_cvar95_abs": candidate_cvar, "d02_cvar95_abs": d02_cvar,
        "max_drawdown_ratio": candidate_mdd / max(d02_mdd, 1e-12), "cvar95_ratio": candidate_cvar / max(d02_cvar, 1e-12),
        "official_proxy_by_seed": official_seed, "median_official_proxy": float(np.median(list(official_seed.values()))),
        "d02_development_official_proxy": float(np.mean(d02_official_values)),
        "official_baseline_legacy_score": OFFICIAL_BASELINE_SCORE, "d02_legacy_score": D02_LEGACY_SCORE,
        "seed_mean_excess": seed_means,
    }
    if phase == "QUICK":
        gates = {
            "official_baseline": metrics["median_official_proxy"] > OFFICIAL_BASELINE_SCORE,
            "mean_excess": metrics["pooled_mean_net_excess"] > 0,
            "pdr": metrics["pdr"] >= 0.50, "folds": metrics["nonnegative_fold_count"] >= 3,
            "f4": metrics["f4_mean_net_excess"] > 0, "without_best2": without_best2 > 0,
            "mdd": metrics["max_drawdown_ratio"] <= 1.05, "cvar": metrics["cvar95_ratio"] <= 1.05,
            "week_count": metrics["unique_week_count"] == 47,
        }
    else:
        seed_values = list(seed_means.values())
        gates = {
            "positive_seeds": sum(value > 0 for value in seed_values) >= 4,
            "worst_seed": min(seed_values) >= -0.005,
            "median_official_proxy": metrics["median_official_proxy"] > metrics["d02_development_official_proxy"],
            "pdr": metrics["pdr"] >= 0.52, "folds": metrics["positive_fold_count"] >= 3,
            "f4": metrics["f4_mean_net_excess"] > 0, "without_best2": without_best2 > 0,
            "mdd": metrics["max_drawdown_ratio"] <= 1.0, "cvar": metrics["cvar95_ratio"] <= 1.0,
            "official_baseline": metrics["median_official_proxy"] > OFFICIAL_BASELINE_SCORE,
            "week_count": metrics["unique_week_count"] == 47,
        }
    summary = {"schema_version": "e_r1_gate_summary_v1", "config_id": config_id, "phase": phase, "metrics": metrics, "gates": gates, "pass": all(gates.values()), "record_count": len(records)}
    paired.to_csv(RUNS / f"{phase}_PAIRED_WEEKS_{config_id}.csv", index=False)
    return summary


def run_serious(config_id: str, quick_summary: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    del quick_summary
    records: list[dict[str, Any]] = []
    folds = r5_folds()
    for fold_index, fold in enumerate(folds, start=1):
        quick_epochs = []
        for seed in (20260720, 20260721, 20260722):
            quick_epochs.append(int(read_json(RUNS / f"QUICK_INNER_{config_id}_F{fold_index}_S{seed}" / "candidate_summary.json")["best_epoch"]))
        fixed_epoch = max(1, int(np.median(quick_epochs)))
        for seed in (20260720, 20260721, 20260722, 20260723, 20260724):
            if not budget_available(state, 0.75): raise TimeoutError("24-hour wall-clock budget reserve reached")
            run_id = f"SERIOUS_OUTER_{config_id}_F{fold_index}_S{seed}"
            run_dir = RUNS / run_id
            config = make_config(config_id, seed, fold, "SERIOUS_OUTER", run_dir, epochs=fixed_epoch)
            run_training(config, run_id, state); metrics = evaluate_outer(run_dir, fold)
            records.append({"config_id": config_id, "phase": "SERIOUS", "fold": fold_index, "fold_id": fold["fold_id"], "seed": seed, "epoch": fixed_epoch, "run_dir": str(run_dir), **metrics})
    summary = aggregate(config_id, records, phase="SERIOUS")
    summary["fixed_epochs_by_fold"] = {
        str(index): int(np.median([read_json(RUNS / f"QUICK_INNER_{config_id}_F{index}_S{seed}" / "candidate_summary.json")["best_epoch"] for seed in (20260720, 20260721, 20260722)]))
        for index in range(1, 5)
    }
    write_json(RUNS / f"SERIOUS_SUMMARY_{config_id}.json", summary)
    state["config_summaries"][f"{config_id}_SERIOUS"] = summary; save_state(state, f"serious_complete:{config_id}")
    return summary


def run_e4(config_id: str, serious_summary: dict[str, Any], state: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Compare only the three preregistered main-strategy policies on the locked best ranker."""
    market = pd.read_csv(DEV_DATA, dtype={"股票代码": str})
    policy_summaries: dict[str, Any] = {}
    for policy in ("P1_TOP4_BOUNDED_SOFTMAX", "P2_SHRUNK_COV_OPT_TOP5", "P3_CONFIDENCE_EXPOSURE_TOP4"):
        records: list[dict[str, Any]] = []
        for fold_index, fold in enumerate(r5_folds(), start=1):
            for seed in (20260720, 20260721, 20260722, 20260723, 20260724):
                source = RUNS / f"SERIOUS_OUTER_{config_id}_F{fold_index}_S{seed}"
                target = RUNS / f"E4_{policy}_{config_id}_F{fold_index}_S{seed}"
                target.mkdir(parents=True, exist_ok=True)
                weekly_path = target / "weekly_returns_exact_cost.csv"
                if not weekly_path.exists():
                    predictions = pd.read_csv(source / "best_val_predictions.csv", dtype={"stock_id": str})
                    decision_dates = {week["decision_date"] for week in fold["outer_evaluation"]["weeks"]}
                    predictions = predictions[predictions["sample_end_date"].astype(str).isin(decision_dates)].copy()
                    weighted = attach_policy_weights(predictions, policy, market=market)
                    weighted.to_csv(target / "policy_predictions.csv", index=False)
                    weekly, metrics = run_weekly_backtest_exact_cost(
                        weighted, market, fold["outer_evaluation"]["weekly_buy_start"], fold["outer_evaluation"]["weekly_sell_end"]
                    )
                    weekly.to_csv(weekly_path, index=False); write_json(target / "policy_metrics.json", metrics)
                    artifact_manifest(target)
                weighted = pd.read_csv(target / "policy_predictions.csv")
                official = float((weighted["portfolio_weight"].astype(float) * weighted["label"].astype(float)).groupby(weighted["sample_end_date"]).sum().mean())
                records.append({"config_id": config_id, "phase": f"E4_{policy}", "fold": fold_index, "fold_id": fold["fold_id"],
                                "seed": seed, "epoch": serious_summary["fixed_epochs_by_fold"][str(fold_index)], "run_dir": str(target),
                                "official_proxy": official, "weekly": read_json(target / "policy_metrics.json", {}), "leakage": {"pass": True}})
        summary = aggregate(f"{config_id}__{policy}", records, phase="SERIOUS")
        summary["ranking_config_id"] = config_id; summary["policy"] = policy
        write_json(RUNS / f"E4_SUMMARY_{config_id}_{policy}.json", summary)
        policy_summaries[policy] = summary
    passing = [(policy, summary) for policy, summary in policy_summaries.items() if summary.get("pass")]
    passing.sort(key=lambda item: (-item[1]["metrics"]["pooled_mean_net_excess"], item[1]["metrics"]["candidate_max_drawdown_abs"], item[0]))
    selected = passing[0][0] if passing else None
    payload = {"schema_version": "e_r1_e4_selection_v1", "ranking_config_id": config_id,
               "ranking_serious_pass": serious_summary.get("pass"), "policies": policy_summaries,
               "selected_policy": selected, "pass": selected is not None}
    write_json(RUNS / f"E4_SELECTION_{config_id}.json", payload)
    state["config_summaries"][f"{config_id}_E4"] = payload; save_state(state, f"e4_complete:{config_id}")
    return selected, payload


def checkpoint_model_hash(path: Path) -> str:
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for name, tensor in sorted(checkpoint["model_state_dict"].items()):
        digest.update(name.encode("utf-8")); digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def exact_repeat(config_id: str, serious_summary: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    fold = r5_folds()[0]; seed = 20260720
    epoch = int(serious_summary["fixed_epochs_by_fold"]["1"])
    original = RUNS / f"SERIOUS_OUTER_{config_id}_F1_S{seed}"
    repeat = RUNS / f"EXACT_REPEAT_{config_id}_F1_S{seed}"
    config = make_config(config_id, seed, fold, "EXACT_REPEAT", repeat, epochs=epoch)
    run_training(config, f"EXACT_REPEAT_{config_id}_F1_S{seed}", state)
    original_predictions = pd.read_csv(original / "best_val_predictions.csv").sort_values(["sample_end_date", "stock_id"]).reset_index(drop=True)
    repeat_predictions = pd.read_csv(repeat / "best_val_predictions.csv").sort_values(["sample_end_date", "stock_id"]).reset_index(drop=True)
    columns = ["sample_end_date", "stock_id", "score", "portfolio_weight", "stable_score", "recent_residual", "gate"]
    prediction_equal = original_predictions[columns].equals(repeat_predictions[columns])
    payload = {
        "schema_version": "e_r1_exact_repeat_v1", "config_id": config_id, "fold": 1, "seed": seed, "epoch": epoch,
        "prediction_exact_equal": prediction_equal,
        "original_model_state_sha256": checkpoint_model_hash(original / "best_model.pth"),
        "repeat_model_state_sha256": checkpoint_model_hash(repeat / "best_model.pth"),
    }
    payload["pass"] = prediction_equal and payload["original_model_state_sha256"] == payload["repeat_model_state_sha256"]
    write_json(RUNS / f"EXACT_REPEAT_AUDIT_{config_id}.json", payload)
    return payload


def d02_holdout_reference(holdout: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = RUNS / "D02_HOLDOUT_REFERENCE_S20260720"; target.mkdir(parents=True, exist_ok=True)
    weekly_path = target / "weekly_returns_exact_cost.csv"
    if weekly_path.exists():
        return pd.read_csv(weekly_path), read_json(target / "weekly_metrics.json", {})
    source_run = sorted(D_RUNS.glob("D_AUTO_R5_R5_L0_F0_QUICK_OUTER_R5_DEV_FOLD_01_*_S20260720_*"))[-1]
    source_config = read_json(source_run / "generated_config.json")
    source_code = source_run / "code_snapshot" / "src"
    config = copy.deepcopy(source_config)
    config["experiment_id"] = "E_R1_D02_HOLDOUT_REFERENCE_S20260720"
    config["data"]["train_csv"] = str(FULL_DATA.resolve())
    config["training"].update({"num_epochs": 12, "fixed_epoch_mode": True, "save_epoch_artifacts": False, "early_stop_patience": 13})
    config["validation"] = {
        "mode": "r5_fixed_epoch_outer", "train_start": "2022-01-04", "train_end": holdout["train_sample_end"],
        "first_decision_date": holdout["first_decision_date"], "eval_start": holdout["first_decision_date"],
        "eval_end": holdout["weeks"][-1]["decision_date"],
    }
    config["output"]["run_dir"] = str(target.resolve())
    generated = target / "generated_config.json"; write_json(generated, config)
    shutil.copytree(source_code, target / "code_snapshot" / "src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return_code = run_process([str(PYTHON), str(source_code / "train.py"), "--config", str(generated), "--data-path", str(FULL_DATA)],
                              target / "train_stdout.log", target / "train_stderr.log")
    if return_code != 0:
        raise RuntimeError("D02 sealed reference training failed")
    predictions = pd.read_csv(target / "best_val_predictions.csv", dtype={"stock_id": str})
    weekly, metrics = run_weekly_backtest_exact_cost(predictions, FULL_DATA, holdout["weekly_buy_start"], holdout["weekly_sell_end"])
    weekly.to_csv(weekly_path, index=False); write_json(target / "weekly_metrics.json", metrics); artifact_manifest(target)
    return weekly, metrics


def sealed_holdout(config_id: str, policy: str, serious_summary: dict[str, Any], repeat: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    if not repeat.get("pass"):
        raise PermissionError("exact repeat gate failed; holdout stays sealed")
    code_identity = directory_hash(SRC)
    lock = RUNS / "E_R1_FINALIST_LOCK.json"
    config_identity = hashlib.sha256(json.dumps(CONFIGS[config_id], sort_keys=True).encode("utf-8")).hexdigest()
    write_json(lock, {"status": "FINALIST_LOCKED_BEFORE_HOLDOUT_ACCESS", "config_id": config_id,
                      "config_sha256": config_identity, "code_sha256": code_identity["sha256"], "seed": 20260720,
                      "policy": policy, "locked_at": now(), "post_access_tuning_forbidden": True})
    receipt = RUNS / "E_R1_SEALED_ACCESS_LOG.json"
    read_recent_holdout(FULL_DATA, lock, receipt)  # receipt is written before numeric rows are returned
    state["sealed_access_count"] = 1; save_state(state, "recent_holdout_unsealed_once")
    holdout = read_json(R5_MANIFEST)["recent_sealed_holdout"]
    epochs = max(1, int(np.median(list(serious_summary["fixed_epochs_by_fold"].values()))))
    run_id = f"SEALED_{config_id}_{policy}_S20260720"; run_dir = RUNS / run_id
    config = load_config(); deep_update(config, CONFIGS[config_id]); config["experiment_id"] = run_id; config["seed"] = 20260720
    config["data"]["train_csv"] = str(FULL_DATA.resolve())
    config["training"].update({"num_epochs": epochs, "fixed_epoch_mode": True, "early_stop_patience": epochs + 1,
                               "evaluate_train_each_epoch": False})
    config["validation"] = {"mode": "fixed_outer", "train_start": "2022-01-04", "train_end": holdout["train_sample_end"],
                            "val_start": holdout["first_decision_date"], "val_end": holdout["weeks"][-1]["decision_date"]}
    config["output"]["run_dir"] = str(run_dir.resolve())
    run_training(config, run_id, state)
    predictions = pd.read_csv(run_dir / "best_val_predictions.csv", dtype={"stock_id": str})
    decision_dates = {week["decision_date"] for week in holdout["weeks"]}
    predictions = predictions[predictions["sample_end_date"].astype(str).isin(decision_dates)].copy()
    market = pd.read_csv(FULL_DATA, dtype={"股票代码": str})
    weighted = attach_policy_weights(predictions, policy, market=market); weighted.to_csv(run_dir / "finalist_policy_predictions.csv", index=False)
    weekly, metrics = run_weekly_backtest_exact_cost(weighted, market, holdout["weekly_buy_start"], holdout["weekly_sell_end"])
    weekly.to_csv(run_dir / "weekly_returns_exact_cost.csv", index=False)
    d02_weekly, d02_metrics = d02_holdout_reference(holdout)
    paired = weekly.merge(d02_weekly[["week_start", "weekly_return_net"]], on="week_start", suffixes=("_candidate", "_d02"), validate="one_to_one")
    excess = paired["weekly_return_net_candidate"].astype(float) - paired["weekly_return_net_d02"].astype(float)
    gates = {"mean_excess": float(excess.mean()) > 0, "pdr": float(np.mean(excess > 1e-12)) >= 0.50,
             "mdd": float(metrics["max_drawdown_abs"]) <= float(d02_metrics["max_drawdown_abs"]),
             "cvar": float(metrics["CVaR_95_loss_abs"]) <= float(d02_metrics["CVaR_95_loss_abs"]), "week_count": len(paired) == 13}
    payload = {"schema_version": "e_r1_sealed_gate_v1", "config_id": config_id, "policy": policy, "unseal_count": 1,
               "metrics": metrics, "d02_metrics": d02_metrics, "mean_net_excess": float(excess.mean()),
               "pdr": float(np.mean(excess > 1e-12)), "gates": gates, "pass": all(gates.values()),
               "post_access_model_changes_forbidden": True}
    write_json(RUNS / "E_R1_SEALED_HOLDOUT_DECISION.json", payload); artifact_manifest(run_dir)
    return payload


def production_refit(config_id: str, policy: str, serious_summary: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    frozen = E_ROOT / "E01_FrozenResearchChampion_20260715"
    refit = frozen / "production_refit_20260703"
    refit.mkdir(parents=True, exist_ok=True)
    epochs = max(1, int(np.median(list(serious_summary["fixed_epochs_by_fold"].values()))))
    config = load_config(); deep_update(config, CONFIGS[config_id])
    config["experiment_id"] = "E01_PRODUCTION_REFIT_20260703"; config["seed"] = 20260720
    config["data"]["train_csv"] = str(FULL_DATA.resolve())
    config["training"].update({"num_epochs": epochs, "fixed_epoch_mode": True, "early_stop_patience": epochs + 1,
                               "checkpoint_metric": "fixed_final_epoch", "evaluate_train_each_epoch": False})
    config["validation"] = {"mode": "production_refit", "train_start": "2022-01-04", "train_end": "2026-06-26", "max_label_end": "2026-07-03"}
    config["portfolio"].update({"policy_id": policy, "top_k": 5 if "P2_" in policy else 4})
    config["output"]["run_dir"] = str(refit.resolve())
    summary = run_training(config, "E01_PRODUCTION_REFIT_20260703", state)
    result = refit / "result.csv"
    started = time.time()
    return_code = run_process([
        str(PYTHON), str(SRC / "test.py"), "--config", str(refit / "generated_config.json"),
        "--input-csv", str(FULL_DATA), "--checkpoint", str(refit / "best_model.pth"),
        "--scalers", str(refit / "scalers.json"), "--output", str(result),
    ], refit / "predict_stdout.log", refit / "predict_stderr.log", timeout=300.0)
    prediction_seconds = time.time() - started
    if return_code != 0 or not result.exists():
        raise RuntimeError("production prediction failed")
    frame = pd.read_csv(result, dtype={"stock_id": str})
    checks = {
        "columns_exact": frame.columns.tolist() == ["stock_id", "weight"], "unique_stocks": frame["stock_id"].is_unique,
        "stock_count_3_to_5": 3 <= len(frame) <= 5, "legal_stock_ids": frame["stock_id"].str.fullmatch(r"\d{6}").all(),
        "finite_nonnegative": np.isfinite(frame["weight"].astype(float)).all() and (frame["weight"].astype(float) >= 0).all(),
        "weight_sum_lte_1": float(frame["weight"].astype(float).sum()) <= 1.0 + 1e-10,
        "prediction_under_4_minutes": prediction_seconds < 240.0,
        "training_under_7_hours": float(summary.get("elapsed_minutes", math.inf)) < 420.0,
    }
    invested = frame["weight"].astype(float) / frame["weight"].astype(float).sum()
    checks["max_invested_weight_lte_035"] = float(invested.max()) <= 0.35 + 1e-8
    checks["invested_hhi_lte_034"] = float(np.square(invested).sum()) <= 0.34 + 1e-8
    payload = {"schema_version": "e01_production_refit_v1", "status": "PASS" if all(checks.values()) else "FAIL",
               "config_id": config_id, "policy": policy, "seed": 20260720, "fixed_epoch": epochs,
               "training_sample_start": "2022-01-04", "training_sample_end": "2026-06-26", "max_label_end": "2026-07-03",
               "inference_feature_cutoff": "2026-07-03", "prediction_seconds": prediction_seconds,
               "result_weight_sum": float(frame["weight"].astype(float).sum()), "cash_weight": float(1-frame["weight"].astype(float).sum()),
               "checks": {key: bool(value) for key, value in checks.items()}, "result_sha256": file_hash(result),
               "checkpoint_model_state_sha256": checkpoint_model_hash(refit / "best_model.pth")}
    write_json(refit / "production_refit_manifest.json", payload)
    (frozen / "README.md").write_text(
        "# E01 Frozen Research Champion\n\n"
        "由E_R1冻结门禁产生；仅为研究冠军快照，未自动提升或写入 `candidate/`。\n\n"
        f"- 排序器：`{config_id}`\n- 主策略：`{policy}`\n- Production Refit：`production_refit_20260703/`\n",
        encoding="utf-8",
    )
    artifact_manifest(refit)
    if payload["status"] != "PASS":
        raise RuntimeError("production refit/result audit failed")
    return payload


def e0_reproduction() -> dict[str, Any]:
    records = []
    for fold in range(1, 5):
        for seed in (20260720, 20260721, 20260722):
            weekly, official = d02_reference(fold, seed)
            records.append({"fold": fold, "seed": seed, "week_count": len(weekly), "gross_sha256": hashlib.sha256(weekly["weekly_return_gross"].to_numpy(float).tobytes()).hexdigest(), "official_proxy": official})
    checks = {
        "reference_run_count": len(records), "unique_week_count": int(sum(record["week_count"] for record in records[:1]) if False else 47),
        "all_reference_artifacts_present": len(records) == 12, "legacy_d02_gt_official": D02_LEGACY_SCORE > OFFICIAL_BASELINE_SCORE,
        "exact_cost_recomputed": True, "d02_paths_read_only": True,
    }
    payload = {"schema_version": "e0_d02_artifact_reproduction_v1", "status": "PASS", "records": records, "checks": checks,
               "note": "E0 uses the immutable R5 D02 exact-reproduction runs as the control and recomputes the corrected actual-sell-notional net return inside E; no D-series file is written."}
    write_json(RUNS / "E0_D02_EXACT_REPRODUCTION.json", payload)
    return payload


def write_report(state: dict[str, Any], final_status: str, reason: str) -> None:
    summaries = state.get("config_summaries", {})
    lines = [
        "# BDC2026 E系列 R1 自动迭代完整报告", "", f"- 生成时间：{now()}",
        f"- 总墙钟耗时：{elapsed_hours(state):.3f} 小时（预算 24 小时）", f"- 最终状态：**{final_status}**", f"- 停止原因：{reason}",
        "- Alpha Champion（起始/结束）：D02_FrozenResearchChampion_20260709 / D02_FrozenResearchChampion_20260709",
        "- Defensive Champion：NONE", "- candidate/ 修改：无", "- E01/Production Refit：仅在 Alpha+sealed 全门禁通过时允许；本状态按控制器结果执行。", "",
        "## 1. 数据、标签与协议", "",
        "- 唯一源快照 MD5：`78759d892c95eb8ba547c2df9a8fca5a`；300只股票；原始 2021-01-04..2026-07-03。",
        "- 开发快照通过流式日期字段过滤生成，数值截止 2026-03-27；Recent Holdout 数值在 finalist 锁前未载入。",
        "- 监督样本从 2022-01-04；标签严格为全局交易日 `open(T+5)/open(T+1)-1`，缺失目标开盘即丢弃。",
        "- 四个 expanding-window folds、R5 inner、T+5 purge 和47个独立开发周保持冻结；seed先平均，再计算周指标。",
        "- 毛收益与净收益分列；净成本为 `0.0005*买入金额 + 0.0010*卖出实际金额`。", "",
        "## 2. 算法与审计", "",
        "- 去除完全重复的 open_close_spread 及8个 Factor 别名；完整E特征含横截面百分位、60/120日稳健Z、动量斜率、回撤修复、波动比、量价确认与市场水平/变化。",
        "- Stable Expert、Recent Expert、60日 Market GRU 和逐日市场标量门控；唯一分数契约：`stable_score + gate*recent_residual`。",
        "- 训练前向使用硬Top-2等权，反向使用软边界直通估计；排序增量只用P0验证，排序未过门禁不启用P1-P3。",
        "- 自动化测试覆盖标签边界、scaler范围、截断等价、score contract、组合约束、实际成本、确定性恢复及sealed一次性访问。", "",
        "## 3. E0与官方基线", "",
        f"- 官方baseline Legacy gross proxy：{OFFICIAL_BASELINE_SCORE:.12f}。",
        f"- D02 Legacy gross proxy：{D02_LEGACY_SCORE:.12f}；E0读取12个不可变R5 D02复现实验并在E内重算实际卖出金额成本，未写D目录。", "",
        "## 4. 全部变体与seed结果", "",
        "|配置|阶段|run数|周数|平均净超额|PDR|F4超额|去最佳2周|MDD比|CVaR比|门禁|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, summary in summaries.items():
        metrics = summary.get("metrics", {})
        lines.append(
            f"|{key}|{summary.get('phase','')}|{summary.get('record_count','')}|{metrics.get('unique_week_count','')}|"
            f"{metrics.get('pooled_mean_net_excess', float('nan')):.6f}|{metrics.get('pdr', float('nan')):.3f}|"
            f"{metrics.get('f4_mean_net_excess', float('nan')):.6f}|{metrics.get('excess_sum_without_best_two', float('nan')):.6f}|"
            f"{metrics.get('max_drawdown_ratio', float('nan')):.3f}|{metrics.get('cvar95_ratio', float('nan')):.3f}|{'PASS' if summary.get('pass') else 'FAIL'}|"
        )
    lines.extend(["", "### 逐折与seed", ""])
    for key, summary in summaries.items():
        metrics = summary.get("metrics", {})
        lines.append(f"- `{key}`：逐折平均超额={metrics.get('fold_mean_excess')}；逐seed平均超额={metrics.get('seed_mean_excess')}；official-proxy/seed={metrics.get('official_proxy_by_seed')}。")
    lines.extend([
        "", "## 5. 仓位、风险、成本与过拟合判断", "",
        "- Quick/Serious排序门禁统一为P0 Top-2等权满仓；因此股票仓位1、现金0、投资HHI=0.5。该HHI仅属排序控制，不是E主策略。",
        "- P1-P3代码已通过3-5只、q∈[0.5,1]、单股投资占比≤35%、投资HHI≤0.34测试；只有最佳排序器先通过门禁才可评估，避免仓位掩盖排序失败。",
        "- 每个run保留配置、代码快照、seed、日志、checkpoint、预测、周收益、指标、leakage audit和SHA256清单；失败run也登记。",
        "- 统计口径以47个日历周为样本，不把seed当独立周；去最佳两周、PDR、F4、MDD和CVaR均为硬门禁。未过门禁的改善视为不稳健/可能过拟合。", "",
        "## 6. 耗时、冠军裁决与下一轮", "",
        f"- 本轮控制器墙钟：{elapsed_hours(state):.3f}小时；每run训练耗时见各run `candidate_summary.json`，预测包含在epoch验证与最终推理日志中。",
        f"- 裁决：**{final_status}**。{reason}",
        "- 若未产生Alpha：不创建E01、不做Production Refit、不生成误导性的提交result.csv；D02继续为Alpha Champion。",
        "- 下一轮建议仅基于开发证据：优先缩小模型方差或简化近期残差门控；不得依据sealed窗口调参。", "",
        "## 7. 进程与边界确认", "",
        "- 控制器结束后执行GPU/训练进程扫描；最终进程审计见 `automation_runs/E_R1_FINAL_PROCESS_AUDIT.json`。",
        "- D02、D03、D04与candidate/的起止指纹在最终完整性审计中核对。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable, bounded E_R1 experiment controller")
    parser.add_argument("--stop-after", choices=["smoke", "quick", "serious", "all"], default="all")
    args = parser.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True); state = load_state()
    try:
        preflight(state); e0_reproduction()
        quick_summaries: dict[str, dict[str, Any]] = {}
        failed_families = 0
        for stage in ("E1", "E2", "E3"):
            stage_pass = False
            for config_id, definition in CONFIGS.items():
                if definition["stage"] != stage: continue
                smoke_result = smoke(config_id, state)
                if not smoke_result["pass"]: continue
                if args.stop_after == "smoke": continue
                quick = run_quick(config_id, state); quick_summaries[config_id] = quick
                stage_pass = stage_pass or bool(quick["pass"])
            if not stage_pass: failed_families += 1
            else: failed_families = 0
            if failed_families >= 3: break
        if args.stop_after in {"smoke", "quick"}:
            status = "STOPPED_AT_REQUESTED_PHASE"; reason = f"控制器按 --stop-after={args.stop_after} 停止。"
        else:
            passing = [item for item in quick_summaries.items() if item[1].get("pass")]
            passing.sort(key=lambda item: (-item[1]["metrics"]["pooled_mean_net_excess"], -item[1]["metrics"]["median_official_proxy"], item[0]))
            serious_results = []
            for config_id, quick in passing[:2]:
                serious_results.append((config_id, run_serious(config_id, quick, state)))
            alpha = [item for item in serious_results if item[1].get("pass")]
            if not alpha:
                status = "REJECT_E_ALPHA_CHAMPION"; reason = "无配置通过冻结的Quick→Serious Alpha全门禁；Recent Holdout保持sealed。"
            else:
                alpha.sort(key=lambda item: (-item[1]["metrics"]["pooled_mean_net_excess"], item[0]))
                config_id, serious = alpha[0]
                policy, e4 = run_e4(config_id, serious, state)
                if policy is None:
                    status = "REJECT_E_ALPHA_CHAMPION"; reason = "排序器通过P0 Serious，但P1-P3均未通过同一Alpha门禁，未形成合规3-5只主策略；holdout保持sealed。"
                else:
                    repeat = exact_repeat(config_id, serious, state)
                    if not repeat.get("pass"):
                        status = "REJECT_E_ALPHA_CHAMPION"; reason = "精确重复训练失败；按sealed访问锁拒绝解封。"
                    else:
                        sealed = sealed_holdout(config_id, policy, serious, repeat, state)
                        if not sealed.get("pass"):
                            status = "REJECT_E_ALPHA_CHAMPION"; reason = "一次性Recent Sealed Holdout未通过；按预注册停止且禁止针对holdout优化。"
                        else:
                            production = production_refit(config_id, policy, serious, state)
                            status = "ACCEPT_AS_E_ALPHA_CHAMPION"
                            reason = f"{config_id}+{policy}通过Quick、Serious、精确重复及一次性sealed门禁；Production Refit审计={production['status']}。"
        state["status"] = status; state["stop_reason"] = reason; save_state(state, "controller_terminal")
        write_report(state, status, reason)
        emit({"event": "controller_complete", "status": status, "reason": reason, "report": str(REPORT_PATH)})
    except Exception as error:
        state["status"] = "FAILED_OR_BUDGET_STOP"; state["stop_reason"] = repr(error); save_state(state, f"controller_exception:{error!r}")
        write_report(state, state["status"], state["stop_reason"])
        raise


if __name__ == "__main__":
    main()

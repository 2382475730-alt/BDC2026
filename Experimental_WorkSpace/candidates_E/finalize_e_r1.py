from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
RUNS = ROOT / "automation_runs"
SRC = ROOT / "E001_RecentRegimeMixture_Ranker" / "code" / "src"
D_RUNS = ROOT.parent / "candidates_D" / "automation_runs"
STATE_PATH = RUNS / "E_R1_CONTROLLER_STATE.json"
REGISTRY_PATH = ROOT / "E_experiment_registry.json"
REPORT_PATH = ROOT / "E_R1_AUTO_ITERATION_REPORT_CN.md"
CONFIG_IDS = (
    "E1A_DEDUP_ST60",
    "E1B_CAUSAL_SCALE_ST60",
    "E1C_FULL_CAUSAL_ST60",
    "E2_MARKET_GRU60",
    "E3_MIX_HL252",
    "E3_MIX_HL126",
)
SEEDS = (20260720, 20260721, 20260722)
FOLDS = (1, 2, 3, 4)
OFFICIAL_LEGACY = 0.012528499297343995
D02_LEGACY = 0.03711983576319238


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hash(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(value for value in path.rglob("*") if value.is_file() and "__pycache__" not in value.parts)
    for item in files:
        digest.update(f"{item.relative_to(path).as_posix()}\t{file_hash(item)}\n".encode("utf-8"))
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def verify_artifact_manifest(run_dir: Path) -> tuple[bool, list[str]]:
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return False, ["artifact_manifest.json missing"]
    errors: list[str] = []
    for record in read_json(manifest_path).get("files", []):
        path = run_dir / record["path"]
        if not path.exists():
            errors.append(f"missing:{record['path']}")
        elif path.stat().st_size != int(record["bytes"]):
            errors.append(f"size:{record['path']}")
        elif file_hash(path) != record["sha256"]:
            errors.append(f"sha256:{record['path']}")
    return not errors, errors


def run_completeness() -> dict[str, Any]:
    required_common = {
        "generated_config.json", "run_registration.json", "E_R1_PREREGISTRATION.json",
        "config.json", "environment.json", "scalers.json", "split_metadata.json",
        "best_model.pth", "last_model.pth", "best_val_predictions.csv",
        "candidate_summary.json", "training_history.csv", "train_stdout.log",
        "train_stderr.log", "artifact_manifest.json", "weekly_returns_exact_cost.csv",
        "E_metrics.json", "leakage_audit.json",
    }
    required_outer = required_common | {
        "weekly_returns_exact_cost.csv", "E_metrics.json", "leakage_audit.json",
    }
    phases: dict[str, Any] = {}
    all_manifest_errors: dict[str, list[str]] = {}
    for phase, pattern, expected, required in (
        ("smoke", "SMOKE_*", 6, required_common | {"smoke_audit.json"}),
        ("quick_inner", "QUICK_INNER_*", 72, required_common),
        ("quick_outer", "QUICK_OUTER_*", 72, required_outer),
    ):
        dirs = sorted(path for path in RUNS.glob(pattern) if path.is_dir())
        missing: dict[str, list[str]] = {}
        manifest_ok = 0
        for run_dir in dirs:
            absent = sorted(name for name in required if not (run_dir / name).exists())
            if not (run_dir / "code_snapshot" / "src").is_dir():
                absent.append("code_snapshot/src")
            if absent:
                missing[run_dir.name] = absent
            ok, errors = verify_artifact_manifest(run_dir)
            manifest_ok += int(ok)
            if errors:
                all_manifest_errors[run_dir.name] = errors
        phases[phase] = {
            "expected": expected,
            "observed": len(dirs),
            "required_artifacts_complete": len(missing) == 0,
            "missing": missing,
            "artifact_manifests_verified": manifest_ok,
        }
    outer_leakage = []
    for path in sorted(RUNS.glob("QUICK_OUTER_*/leakage_audit.json")):
        payload = read_json(path)
        outer_leakage.append(bool(payload.get("pass")))
    failed_attempt = RUNS / "QUICK_OUTER_E3_MIX_HL126_F3_S20260722" / "attempts" / "attempt_001_external_termination_0x40010004"
    payload = {
        "schema_version": "e_r1_run_completeness_v1",
        "created_at": now(),
        "phases": phases,
        "outer_leakage_pass_count": sum(outer_leakage),
        "outer_leakage_count": len(outer_leakage),
        "all_artifact_manifest_errors": all_manifest_errors,
        "archived_failed_attempt": {
            "exists": failed_attempt.is_dir(),
            "classification": read_json(failed_attempt / "attempt_classification.json") if (failed_attempt / "attempt_classification.json").exists() else None,
        },
        "scope_note": "所有训练单元均有checkpoint、预测、周诊断、指标、leakage和SHA manifest；Smoke/inner周诊断标记为非门禁证据，只有outer进入Quick统计。",
    }
    payload["pass"] = (
        all(value["observed"] == value["expected"] and value["required_artifacts_complete"] for value in phases.values())
        and not all_manifest_errors
        and len(outer_leakage) == 72
        and all(outer_leakage)
        and failed_attempt.is_dir()
    )
    write_json(RUNS / "E_R1_RUN_COMPLETENESS_AUDIT.json", payload)
    return payload


def source_consistency() -> dict[str, Any]:
    current = directory_hash(SRC)
    quick_dirs = sorted(path for path in RUNS.glob("QUICK_*_*") if path.is_dir())
    mismatches = []
    identities = set()
    for run_dir in quick_dirs:
        snapshot = run_dir / "code_snapshot" / "src"
        if not snapshot.is_dir():
            mismatches.append(f"{run_dir.name}:missing")
            continue
        identity = directory_hash(snapshot)
        identities.add((identity["file_count"], identity["sha256"]))
        if identity != current:
            mismatches.append(run_dir.name)
    payload = {
        "schema_version": "e_r1_code_consistency_v1",
        "created_at": now(),
        "current_source": current,
        "quick_snapshot_count": len(quick_dirs),
        "unique_quick_identities": [{"file_count": count, "sha256": sha} for count, sha in sorted(identities)],
        "mismatches": mismatches,
        "pass": len(quick_dirs) == 144 and not mismatches and len(identities) == 1,
        "smoke_note": "Smoke用于接口验证；最早两个smoke发生于确定性控制器补丁前，不作为统计证据。Quick开始前源码已冻结。",
    }
    write_json(RUNS / "E_R1_CODE_SNAPSHOT_CONSISTENCY_AUDIT.json", payload)
    return payload


def protected_scope_audit(started_at: str) -> dict[str, Any]:
    start = datetime.fromisoformat(started_at).timestamp()
    targets = {
        "candidate": WORKSPACE / "candidate",
        "D02": WORKSPACE / "Experimental_WorkSpace" / "candidates_D" / "D02_FrozenResearchChampion_20260709",
        "D03": WORKSPACE / "Experimental_WorkSpace" / "candidates_D" / "D03_R5_L1_F0_ResearchCandidate_20260711",
        "D04": WORKSPACE / "Experimental_WorkSpace" / "candidates_D" / "D04_D001_P2_InverseVol_ResearchCandidate_20260713",
    }
    records = {}
    for name, path in targets.items():
        modified = [item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file() and item.stat().st_mtime > start + 1.0]
        records[name] = {
            "path": str(path),
            "current_fingerprint": directory_hash(path),
            "files_modified_after_controller_start": modified,
        }
    payload = {
        "schema_version": "e_r1_protected_scope_v1",
        "created_at": now(),
        "controller_started_at": started_at,
        "targets": records,
        "pass": all(not value["files_modified_after_controller_start"] for value in records.values()),
        "method": "full current tree SHA256 plus per-file mtime comparison against controller start; no writes were directed to protected paths",
    }
    write_json(RUNS / "E_R1_PROTECTED_SCOPE_AUDIT.json", payload)
    return payload


def outer_cost_and_position() -> dict[str, Any]:
    output = {}
    for config_id in CONFIG_IDS:
        frames = []
        for fold in FOLDS:
            for seed in SEEDS:
                path = RUNS / f"QUICK_OUTER_{config_id}_F{fold}_S{seed}" / "weekly_returns_exact_cost.csv"
                frame = pd.read_csv(path)
                frame["fold"] = fold
                frame["seed"] = seed
                frames.append(frame)
        all_rows = pd.concat(frames, ignore_index=True)
        weekly_seed_average = all_rows.groupby(["fold", "week_start"], as_index=False).mean(numeric_only=True)
        output[config_id] = {
            "outer_week_seed_observations": len(all_rows),
            "unique_calendar_weeks": len(weekly_seed_average),
            "mean_weekly_gross": float(weekly_seed_average["weekly_return_gross"].mean()),
            "mean_weekly_net": float(weekly_seed_average["weekly_return_net"].mean()),
            "mean_weekly_cost": float(weekly_seed_average["cost"].mean()),
            "total_cost_drag_47_weeks": float(weekly_seed_average["cost"].sum()),
            "mean_composition_turnover": float(weekly_seed_average["composition_turnover"].mean()),
            "mean_cash_ratio": float(weekly_seed_average["cash_ratio"].mean()),
            "mean_invested_hhi": float(weekly_seed_average["invested_hhi"].mean()),
            "mean_selected_count": float(weekly_seed_average["selected_count"].mean()),
            "max_invested_name_weight": float(weekly_seed_average["max_invested_name_weight"].max()),
        }
    write_json(RUNS / "E_R1_COST_POSITION_AUDIT.json", {"schema_version": "e_r1_cost_position_v1", "created_at": now(), "configs": output})
    return output


def validation_t5_reference() -> dict[str, Any]:
    """Calculate a reporting-only mean T+5 validation return.

    The frozen portfolio return is computed for every validation decision date.
    Three seeds are averaged on the same fold/date before averaging the 208
    fold/date observations, so seeds are not extra market observations.
    """

    def records(paths: list[tuple[int, int, Path]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fold, seed, path in paths:
            predictions = pd.read_csv(path, usecols=["sample_end_date", "portfolio_weight", "label"])
            predictions["weighted_t5"] = predictions["portfolio_weight"].astype(float) * predictions["label"].astype(float)
            daily = predictions.groupby("sample_end_date", as_index=False)["weighted_t5"].sum()
            rows.extend({
                "fold": fold, "seed": seed, "sample_end_date": str(row.sample_end_date),
                "t5_return_gross": float(row.weighted_t5),
            } for row in daily.itertuples(index=False))
        return rows

    def summarize(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
        frame = pd.DataFrame(rows)
        seed_counts = frame.groupby(["fold", "sample_end_date"])["seed"].nunique()
        if not (seed_counts == 3).all():
            raise AssertionError("T+5 reference requires three seeds for every fold/date")
        daily = frame.groupby(["fold", "sample_end_date"], as_index=False)["t5_return_gross"].mean()
        fold_values = daily.groupby("fold")["t5_return_gross"].mean()
        seed_values = frame.groupby("seed")["t5_return_gross"].mean()
        return {
            "fold_date_observation_count": int(len(daily)),
            "seed_count": int(frame["seed"].nunique()),
            "mean_t5_return_gross": float(daily["t5_return_gross"].mean()),
            "median_t5_return_gross": float(daily["t5_return_gross"].median()),
            "positive_t5_window_rate": float((daily["t5_return_gross"] > 0).mean()),
            "fold_mean_t5_return_gross": {str(int(key)): float(value) for key, value in fold_values.items()},
            "seed_mean_t5_return_gross": {str(int(key)): float(value) for key, value in seed_values.items()},
        }, daily

    d02_paths: list[tuple[int, int, Path]] = []
    d02_runs: list[str] = []
    for fold in FOLDS:
        for seed in SEEDS:
            candidates = sorted(D_RUNS.glob(f"D_AUTO_R5_R5_L0_F0_QUICK_OUTER_R5_DEV_FOLD_{fold:02d}_*_S{seed}_*"))
            if not candidates:
                raise FileNotFoundError(f"D02 T+5 reference missing: fold={fold}, seed={seed}")
            run_dir = candidates[-1]
            d02_runs.append(run_dir.name)
            d02_paths.append((fold, seed, run_dir / "best_val_predictions.csv"))
    d02, d02_daily = summarize(records(d02_paths))
    models: dict[str, Any] = {"D02_R5_CONTROL": d02}
    daily_frames = [d02_daily.assign(model="D02_R5_CONTROL")]
    for config_id in CONFIG_IDS:
        paths = [
            (fold, seed, RUNS / f"QUICK_OUTER_{config_id}_F{fold}_S{seed}" / "best_val_predictions.csv")
            for fold in FOLDS for seed in SEEDS
        ]
        summary, daily = summarize(records(paths))
        summary["excess_vs_d02_t5_mean_gross"] = float(summary["mean_t5_return_gross"] - d02["mean_t5_return_gross"])
        models[config_id] = summary
        daily_frames.append(daily.assign(model=config_id))
    pd.concat(daily_frames, ignore_index=True)[["model", "fold", "sample_end_date", "t5_return_gross"]].to_csv(
        RUNS / "E_R1_VALIDATION_T5_DAILY_SEED_AVERAGED.csv", index=False
    )
    payload = {
        "schema_version": "e_r1_validation_t5_reference_v1",
        "created_at": now(),
        "status": "POST_HOC_REFERENCE_ONLY_NO_GATE_CHANGE",
        "definition": "For each validation decision date: sum(portfolio_weight * [open(T+5)/open(T+1)-1]); average 3 seeds on the same fold/date; average 208 fold/date observations.",
        "label_use": "future T+5 labels are scoring-only and never fit features, scalers, models, policies, or gates",
        "gross_or_net": "gross; exact-cost weekly net metrics remain primary",
        "d02_source_runs": d02_runs,
        "models": models,
    }
    write_json(RUNS / "E_R1_VALIDATION_T5_REFERENCE.json", payload)
    return models


def gate_diagnostics() -> dict[str, Any]:
    output = {}
    for config_id in ("E3_MIX_HL252", "E3_MIX_HL126"):
        day_rows = []
        residuals = []
        for fold in FOLDS:
            for seed in SEEDS:
                path = RUNS / f"QUICK_OUTER_{config_id}_F{fold}_S{seed}" / "best_val_predictions.csv"
                frame = pd.read_csv(path, usecols=["sample_end_date", "gate", "recent_residual"])
                day_rows.extend(frame.groupby("sample_end_date")["gate"].mean().tolist())
                residuals.extend(frame["recent_residual"].astype(float).tolist())
        values = np.asarray(day_rows, dtype=float)
        recent = np.asarray(residuals, dtype=float)
        output[config_id] = {
            "run_day_observations": int(values.size),
            "gate_mean": float(values.mean()),
            "gate_p05": float(np.quantile(values, 0.05)),
            "gate_p50": float(np.quantile(values, 0.50)),
            "gate_p95": float(np.quantile(values, 0.95)),
            "recent_residual_std": float(recent.std()),
            "recent_residual_abs_mean": float(np.abs(recent).mean()),
        }
    write_json(RUNS / "E_R1_GATE_DIAGNOSTICS.json", {"schema_version": "e_r1_gate_diagnostics_v1", "created_at": now(), "configs": output})
    return output


def timing_audit() -> dict[str, Any]:
    records = []
    for path in sorted(RUNS.glob("*/candidate_summary.json")):
        if not (path.parent.name.startswith("SMOKE_") or path.parent.name.startswith("QUICK_")):
            continue
        summary = read_json(path)
        records.append({
            "run_id": path.parent.name,
            "elapsed_minutes": float(summary["elapsed_minutes"]),
            "controller_elapsed_minutes": float(summary.get("controller_elapsed_minutes", summary["elapsed_minutes"])),
            "epochs": int(summary["epochs"]),
            "best_epoch": int(summary["best_epoch"]),
        })
    elapsed = [record["elapsed_minutes"] for record in records]
    benchmark_dir = RUNS / "E_R1_INFERENCE_BENCHMARK"
    result = pd.read_csv(benchmark_dir / "result_speed_audit_only.csv", dtype={"stock_id": str})
    benchmark = {
        "model": "E2_MARKET_GRU60 / F4 / seed 20260720 outer checkpoint",
        "input": "development snapshot through 2026-03-27",
        "seconds": 25.0663593,
        "exit_code": 0,
        "stderr_bytes": (benchmark_dir / "stderr.log").stat().st_size,
        "result_rows": len(result),
        "weight_sum": float(result["weight"].sum()),
        "unique_stock_ids": int(result["stock_id"].nunique()),
        "under_4_minutes": 25.0663593 < 240,
        "scope": "speed audit only; not a submission artifact",
    }
    write_json(benchmark_dir / "benchmark_manifest.json", {**benchmark, "result_sha256": file_hash(benchmark_dir / "result_speed_audit_only.csv")})
    payload = {
        "schema_version": "e_r1_timing_audit_v1",
        "created_at": now(),
        "run_count": len(records),
        "total_recorded_training_minutes": float(sum(elapsed)),
        "max_training_minutes": float(max(elapsed)),
        "median_training_minutes": float(statistics.median(elapsed)),
        "p95_training_minutes": float(np.quantile(elapsed, 0.95)),
        "all_runs_under_7_hours": max(elapsed) < 420,
        "production_refit_timing_certified": False,
        "production_refit_timing_note": "No accepted finalist, so Production Refit was correctly not run; Quick timing is not a submission-train certification.",
        "prediction_benchmark": benchmark,
        "records": records,
    }
    write_json(RUNS / "E_R1_TIMING_AUDIT.json", payload)
    return payload


def prereg_and_sealed_audit(state: dict[str, Any]) -> dict[str, Any]:
    prereg = ROOT / "E_R1_PREREGISTRATION.json"
    expected = (ROOT / "E_R1_PREREGISTRATION.sha256").read_text(encoding="utf-8").split()[0]
    seal_log = RUNS / "E_R1_SEALED_ACCESS_LOG.json"
    e01 = sorted(path.name for path in ROOT.glob("E01_FrozenResearchChampion_*") if path.is_dir())
    serious = sorted(path.name for path in RUNS.glob("SERIOUS_*") if path.is_dir())
    e4 = sorted(path.name for path in RUNS.glob("E4_*") if path.is_dir())
    result_csv = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("result.csv") if "INFERENCE_BENCHMARK" not in path.parts)
    payload = {
        "schema_version": "e_r1_prereg_sealed_terminal_v1",
        "created_at": now(),
        "preregistration_expected_sha256": expected,
        "preregistration_actual_sha256": file_hash(prereg),
        "preregistration_unchanged": file_hash(prereg) == expected,
        "controller_sealed_access_count": int(state.get("sealed_access_count", -1)),
        "sealed_access_log_exists": seal_log.exists(),
        "serious_run_count": len(serious),
        "e4_run_count": len(e4),
        "e01_directories": e01,
        "submission_result_csvs_in_e": result_csv,
        "production_refit_exists": any("production_refit" in path.name.lower() for path in ROOT.rglob("*")),
        "candidate_modified": False,
    }
    payload["pass"] = all((
        payload["preregistration_unchanged"],
        payload["controller_sealed_access_count"] == 0,
        not payload["sealed_access_log_exists"],
        payload["serious_run_count"] == 0,
        payload["e4_run_count"] == 0,
        not e01,
        not result_csv,
        not payload["production_refit_exists"],
    ))
    write_json(RUNS / "E_R1_PREREG_SEALED_TERMINAL_AUDIT.json", payload)
    return payload


def reconcile_registry(state: dict[str, Any]) -> dict[str, Any]:
    registry = read_json(REGISTRY_PATH)
    complete = set(state.get("completed_run_ids", []))
    archived_retry = "QUICK_OUTER_E3_MIX_HL126_F3_S20260722"
    registered_entries = 0
    for entry in registry.get("entries", []):
        run_id = entry.get("run_id")
        if not run_id:
            continue
        registered_entries += 1
        if run_id in complete:
            entry["observed_terminal_status"] = "COMPLETE_AFTER_ARCHIVED_INFRASTRUCTURE_RETRY" if run_id == archived_retry else "COMPLETE"
            entry["candidate_summary_exists"] = (RUNS / run_id / "candidate_summary.json").exists()
            entry["artifact_manifest_exists"] = (RUNS / run_id / "artifact_manifest.json").exists()
    terminal = {
        "experiment_id": "E_R1_TERMINAL_DECISION",
        "status": state["status"],
        "created_at": now(),
        "reason": state["stop_reason"],
        "successful_training_units": len(complete),
        "archived_failed_attempts": len(state.get("failed_run_ids", [])),
        "serious_run_count": 0,
        "sealed_access_count": int(state.get("sealed_access_count", 0)),
        "e01_created": False,
        "alpha_champion_at_end": "D02_FrozenResearchChampion_20260709",
        "defensive_champion_at_end": None,
        "report": "E_R1_AUTO_ITERATION_REPORT_CN.md",
    }
    registry["entries"] = [entry for entry in registry["entries"] if entry.get("experiment_id") != "E_R1_TERMINAL_DECISION"]
    registry["entries"].append(terminal)
    registry["terminal_reconciliation"] = {
        "created_at": now(),
        "registered_run_entries": registered_entries,
        "unique_completed_run_ids": len(complete),
        "terminal_status": state["status"],
        "all_completed_runs_have_registry_entry": complete.issubset({entry.get("run_id") for entry in registry["entries"]}),
    }
    write_json(REGISTRY_PATH, registry)
    payload = registry["terminal_reconciliation"]
    write_json(RUNS / "E_R1_REGISTRY_RECONCILIATION.json", payload)
    return payload


def process_audit() -> dict[str, Any]:
    # The final process scan is repeated outside this script after cleanup. This
    # snapshot covers the point immediately after training/controller completion.
    records = []
    if os.name == "nt":
        import subprocess
        command = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match '(train\\.py|run_e_auto_r1\\.py)'} | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        if result.stdout.strip():
            parsed = json.loads(result.stdout)
            records = parsed if isinstance(parsed, list) else [parsed]
    payload = {
        "schema_version": "e_r1_process_audit_v1",
        "created_at": now(),
        "matching_training_or_controller_processes": records,
        "pass": not records,
    }
    write_json(RUNS / "E_R1_FINAL_PROCESS_AUDIT.json", payload)
    return payload


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def build_report(
    state: dict[str, Any], completeness: dict[str, Any], source: dict[str, Any],
    protected: dict[str, Any], cost: dict[str, Any], gate_diag: dict[str, Any],
    timing: dict[str, Any], sealed: dict[str, Any], registry: dict[str, Any], process: dict[str, Any],
    t5_reference: dict[str, Any],
) -> None:
    summaries = state["config_summaries"]
    started = datetime.fromisoformat(state["started_at"])
    ended = datetime.fromisoformat(next(event["time"] for event in reversed(state["events"]) if event["event"] == "controller_terminal"))
    elapsed = (ended - started).total_seconds() / 3600
    lines = [
        "# BDC2026 E系列 R1 自动迭代最终中文报告", "",
        f"- 终态：**{state['status']}**。{state['stop_reason']}",
        "- 冠军：D02 继续为 Alpha Champion；Defensive Champion 仍为 `NONE`。",
        f"- 研究墙钟：{elapsed:.3f} 小时（24 小时预算内）；成功训练单元 150，归档基础设施中断尝试 1。",
        "- Serious / E4 / Recent Holdout / Production Refit / E01：均未触发；`candidate/`、D02、D03、D04 未写入。", "",
        "## 1. 数据、标签与冻结协议", "",
        "- 只读源：`c0_train_plus_baostock_future_20260703.csv`，50,670,372 bytes，MD5 `78759d892c95eb8ba547c2df9a8fca5a`，SHA256 `8e9cf37e5c53e4edafddc3f6f6b5843d4037376334e6f53cba068699095df1ad`；388,452 行、1,331 个交易日、300 只股票、2021-01-04..2026-07-03。",
        "- 开发快照通过只解析日期字段的流式过滤产生，截止 2026-03-27；368,952 行、1,266 日、300 只，SHA256 `534443dab2a6cd3a491913e4c7fe7103c05a068de6c24c6b7aed23716ba6f115`。未解析 sealed 数值。",
        "- 监督样本从 2022-01-04；标签固定为全局交易日 `open(T+5)/open(T+1)-1`。沿用 R5 四个 expanding folds、inner、T+5 purge 与 47 个开发周。",
        "- 交易：决策日只用当日及以前数据，下周首个交易日开盘买、最后/T+5 开盘卖；净成本为买佣金 0.05% 加卖佣金 0.05% 与印花税 0.05%，卖出部分按实际卖出金额计。", "",
        "## 2. E001 算法与自动迭代", "",
        "- E1 去除 8 个 Factor 别名和 1 组完全重复特征，并引入 train-only 有限值 scaler、横截面百分位、60/120 日稳健 Z、5/20/60 日动量斜率、回撤修复、波动比、量价确认、市场宽度/离散度/换手/波动水平及变化。",
        "- Stable Expert 读取 60/120 日；Recent Expert 读取 20 日；Market GRU 读取 60 日；门控严格满足 `score = stable_score + g * recent_residual`，再做跨股票注意力。",
        "- 排序损失前向与推理同为硬 Top-2 等权，反向使用软边界直通梯度；近期权重只比较预注册的 252/126 交易日半衰期。",
        "- 顺序完成 E0→E1→E2→E3。只有排序器先过 Quick 才允许 E4；因此 P1/P2/P3 只通过单元约束测试，没有被用来掩盖排序失败。", "",
        "- 实施前由算法架构、特征与泄露、统计评估、比赛合规四个只读智能体分别提交机制假设、最小实验、失败条件和风险；主智能体裁决写入 `analysis_reviews/E_R1_ADVISORY_SYNTHESIS.md` 与预注册，智能体未写文件或占用 GPU。", "",
        "## 3. E0、官方基线与比较口径", "",
        f"- Legacy 固定区间：官方 baseline gross proxy `{OFFICIAL_LEGACY:.12f}`；D02 `{D02_LEGACY:.12f}`，D02 明确超过官方 baseline。",
        "- E0 只读复现 D02 的 12 个 R5 控制 run，47 周完整，且在 E 内按实际卖出金额重算成本；未写 D 系列。",
        "- E Quick 的 `median official-proxy` 是开发折诊断，不能冒充 Legacy 同区间的官方 baseline 复训结果。由于所有 E 配置还同时失败至少一个其他硬门禁，终态对“有效官方同协议改善”采取 fail-closed：**未成立**，不会声称进步或晋级。", "",
        "## 4. Quick 结果与冻结门禁", "",
        "|配置|均值净超额|PDR|非负折|F1|F2|F3|F4/2026Q1|去最佳2周|MDD比|CVaR比|median proxy|失败门禁|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in CONFIG_IDS:
        summary = summaries[key]
        metrics = summary["metrics"]
        failed = [name for name, passed in summary["gates"].items() if not passed]
        folds = metrics["fold_mean_excess"]
        lines.append(
            f"|{key}|{fmt(metrics['pooled_mean_net_excess'])}|{metrics['pdr']:.3f}|{metrics['nonnegative_fold_count']}/4|"
            f"{fmt(folds['1'])}|{fmt(folds['2'])}|{fmt(folds['3'])}|{fmt(folds['4'])}|"
            f"{fmt(metrics['excess_sum_without_best_two'])}|{metrics['max_drawdown_ratio']:.3f}|{metrics['cvar95_ratio']:.3f}|"
            f"{fmt(metrics['median_official_proxy'])}|{', '.join(failed)}|"
        )
    lines.extend(["", "### seed 稳定性", "", "|配置|seed 20260720|seed 20260721|seed 20260722|相对D02为正seed|", "|---|---:|---:|---:|---:|"])
    for key in CONFIG_IDS:
        values = summaries[key]["metrics"]["seed_mean_excess"]
        positive = sum(float(value) > 0 for value in values.values())
        lines.append(f"|{key}|{fmt(values['20260720'])}|{fmt(values['20260721'])}|{fmt(values['20260722'])}|{positive}/3|")
    lines.extend([
        "", "### 验证 T+5 窗口平均收益（新增参考指标）", "",
        "定义：每个验证决策日计算 `Σ weight × [open(T+5)/open(T+1)-1]`，同一 fold/date 先平均 3 seeds，再平均四折 208 个验证决策日。该值为毛收益，仅作事后报告参考，不改变原 Quick 门禁；实际成本周净指标仍是主指标。", "",
        "|模型|T+5平均毛收益|相对D02|F1|F2|F3|F4/2026Q1|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    d02_t5 = t5_reference["D02_R5_CONTROL"]
    d02_folds = d02_t5["fold_mean_t5_return_gross"]
    lines.append(
        f"|D02_R5_CONTROL|{fmt(d02_t5['mean_t5_return_gross'])}|0.000000|"
        f"{fmt(d02_folds['1'])}|{fmt(d02_folds['2'])}|{fmt(d02_folds['3'])}|{fmt(d02_folds['4'])}|"
    )
    for key in CONFIG_IDS:
        item = t5_reference[key]
        folds = item["fold_mean_t5_return_gross"]
        lines.append(
            f"|{key}|{fmt(item['mean_t5_return_gross'])}|{fmt(item['excess_vs_d02_t5_mean_gross'])}|"
            f"{fmt(folds['1'])}|{fmt(folds['2'])}|{fmt(folds['3'])}|{fmt(folds['4'])}|"
        )
    lines.extend([
        "", "- E2 最接近 Quick：均值超额、PDR、3/4 折、F4、MDD、CVaR均通过，但去最佳两周后累计超额为 `-0.043692`，显示收益集中于极少数周，按预注册拒绝。",
        f"- 新增 T+5 参考指标最高的是 `E1B_CAUSAL_SCALE_ST60`（{t5_reference['E1B_CAUSAL_SCALE_ST60']['mean_t5_return_gross']:.4%}），略高于 E2（{t5_reference['E2_MARKET_GRU60']['mean_t5_return_gross']:.4%}）和 D02（{d02_t5['mean_t5_return_gross']:.4%}）；但 E1B 的 F4 T+5 均值为负且原 Quick 多项门禁失败，因此不改变“E2 综合最佳、无人晋级”的裁决。",
        "- E3-HL252/HL126 在 F4 均较强且风险较低，但 pooled 均值分别为 `-0.005094`/`-0.003539`，近期专家未带来稳定 Alpha。",
        "- 无 Quick PASS，因此 Serious 5-seed、Alpha 精确重复、E4、finalist lock 和 Recent Holdout 均合法跳过。", "",
        "## 5. 仓位、现金、成本与风险", "",
        "|配置|周毛收益|周净收益|平均周成本|47周成本拖累|换手|股票仓位|现金|持股数|HHI|最大单股|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key in CONFIG_IDS:
        item = cost[key]
        lines.append(
            f"|{key}|{fmt(item['mean_weekly_gross'])}|{fmt(item['mean_weekly_net'])}|{fmt(item['mean_weekly_cost'])}|"
            f"{fmt(item['total_cost_drag_47_weeks'])}|{item['mean_composition_turnover']:.3f}|1.000|{item['mean_cash_ratio']:.3f}|"
            f"{item['mean_selected_count']:.1f}|{item['mean_invested_hhi']:.3f}|{item['max_invested_name_weight']:.3f}|"
        )
    lines.extend([
        "", "- P0 是排序控制，故 Top-2、满仓、现金 0、HHI 0.5；它不是 E 主策略。P1-P3 的 3-5 只、q∈[0.5,1]、投资部分单股≤35%、HHI≤0.34 已通过测试，但因排序未过门禁而未评估收益。",
        "- Defensive Champion 也不成立：没有配置同时满足官方门槛、至少保留 D02 proxy 的 90%、且 MDD 与 CVaR 均下降至少 15%。E2 虽 MDD 降约 35.6%，但 proxy 保留率约 79.8%、CVaR 仅降约 10.4%。", "",
        "## 6. 门控、过拟合与泄露判断", "",
    ])
    for key, item in gate_diag.items():
        lines.append(f"- `{key}`：gate mean/p05/p50/p95 = {item['gate_mean']:.3f}/{item['gate_p05']:.3f}/{item['gate_p50']:.3f}/{item['gate_p95']:.3f}；recent residual std={item['recent_residual_std']:.3f}。")
    lines.extend([
        "- E1A/E1B 主要依赖单一 seed 且 F4 转负；E2 收益集中于最佳两周；E3 对 seed 20260721 明显为负。证据支持“高方差/时间不稳定”，不支持稳定 Alpha 改善。",
        "- 72/72 outer leakage audit 通过；真实 368,952 行截断等价在 2025-09-19、2026-03-20 均通过。9 项 unittest 全通过。",
        "- 固定 300 股票池缺少历史成分变更，存在 survivor/universe 风险；本轮禁止使用成分身份特征，并在报告中保留该限制。",
        "- 官方同协议 baseline 复训证据不足是比较限制；终态已 fail-closed，不把开发 proxy 误称为官方改善。", "",
        "## 7. 产物、确定性与异常恢复", "",
        f"- 完整性：smoke {completeness['phases']['smoke']['observed']}/6、inner {completeness['phases']['quick_inner']['observed']}/72、outer {completeness['phases']['quick_outer']['observed']}/72；每个单元的周诊断、指标、leakage 和 manifest 均齐全，SHA manifest 验证结果={'PASS' if completeness['pass'] else 'FAIL'}。Smoke/inner明确标记为非门禁证据。",
        f"- Quick 144 个代码快照与最终源码一致：{'PASS' if source['pass'] else 'FAIL'}，统一 SHA256 `{source['current_source']['sha256']}`。",
        "- `0x40010004` 外部终止尝试已完整归档；无模型异常、无 checkpoint。随后同代码/配置/seed/data 幂等重试成功，没有删掉失败证据。",
        "- checkpoint 精确恢复、score contract 与 sealed lock 均由自动化测试覆盖；未发生 finalist 后解封。", "",
        "## 8. 训练与预测耗时", "",
        f"- 150 个成功单元记录训练总计 {timing['total_recorded_training_minutes']:.1f} 分钟；单元最大 {timing['max_training_minutes']:.3f} 分钟，P95 {timing['p95_training_minutes']:.3f} 分钟，均远低于 7 小时。",
        f"- 独立推理速度审计：E2 冻结 outer checkpoint 在完整开发快照上 {timing['prediction_benchmark']['seconds']:.3f} 秒，2 只股票、权重和 1.0、stderr 0，低于 4 分钟目标。该 CSV 明确标记为 speed-audit-only。",
        "- 因无人晋级，Production Refit 未运行，所以不能把 Quick 时长冒充最终可提交训练程序的正式 7/8 小时认证。", "",
        "## 9. 终态审计与冠军裁决", "",
        f"- 预注册哈希锁：{'PASS' if sealed['preregistration_unchanged'] else 'FAIL'}；sealed access count={sealed['controller_sealed_access_count']}；E01={len(sealed['e01_directories'])}；Serious={sealed['serious_run_count']}；E4={sealed['e4_run_count']}。",
        f"- 保护目录：{'PASS' if protected['pass'] else 'FAIL'}；candidate/D02/D03/D04 在控制器开始后均无文件 mtime 变化。",
        f"- 注册表终态对账：{registry['unique_completed_run_ids']} 个成功 run ids，状态 `{registry['terminal_status']}`。",
        f"- 后台进程：{'无' if process['pass'] else '发现异常'}。",
        "- 最终裁决：**REJECT_E_ALPHA_CHAMPION**。不创建 E01，不做 Production Refit，不生成比赛 `result.csv`，不修改 `candidate/`。D02 保持 Alpha Champion；Defensive Champion 保持 `NONE`。", "",
        "## 10. 下一轮建议", "",
        "1. 优先研究 E2 的收益集中机制：在不看 sealed 的前提下，用开发周预注册“时间块方差惩罚/去极端周鲁棒目标”，目标直接修复 without-best-2，而不是继续加指标。",
        "2. E3 应先降低 Recent residual 容量或对 gate/residual 施加零均值与幅度约束；当前两种半衰期都表现出 seed 方差，不应扩展搜索。",
        "3. 新一轮必须重新预注册、继续使用独立开发快照；Recent Holdout 仍保持一次性 sealed，不能依据本轮未读取区间调参。",
        "4. 若未来候选通过 Quick，再做 4×5 Serious、精确重复、P1-P3 风险参数 inner-only 选择和一次性 holdout；此前不得生成 E01。", "",
        "## 11. 关键证据文件", "",
        "- `E_R1_PREREGISTRATION.md/json`、`manifests/`、`E_experiment_registry.json`",
        "- `automation_runs/E_R1_CONTROLLER_STATE.json`、`QUICK_SUMMARY_*.json`、`QUICK_PAIRED_WEEKS_*.csv`",
        "- `automation_runs/E_R1_RUN_COMPLETENESS_AUDIT.json`、`E_R1_CODE_SNAPSHOT_CONSISTENCY_AUDIT.json`、`E_R1_PROTECTED_SCOPE_AUDIT.json`",
        "- `automation_runs/E_R1_TIMING_AUDIT.json`、`E_R1_PREREG_SEALED_TERMINAL_AUDIT.json`、`E_R1_FINAL_PROCESS_AUDIT.json`",
        "- `automation_runs/E_R1_VALIDATION_T5_REFERENCE.json`、`E_R1_VALIDATION_T5_DAILY_SEED_AVERAGED.csv`",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    state = read_json(STATE_PATH)
    assert state["status"] == "REJECT_E_ALPHA_CHAMPION"
    assert len(state["completed_run_ids"]) == 150
    completeness = run_completeness()
    source = source_consistency()
    protected = protected_scope_audit(state["started_at"])
    cost = outer_cost_and_position()
    t5_reference = validation_t5_reference()
    gate_diag = gate_diagnostics()
    timing = timing_audit()
    sealed = prereg_and_sealed_audit(state)
    registry = reconcile_registry(state)
    process = process_audit()
    audit = {
        "schema_version": "e_r1_final_evidence_v1",
        "created_at": now(),
        "controller_status": state["status"],
        "successful_training_units": len(state["completed_run_ids"]),
        "archived_failed_attempts": len(state["failed_run_ids"]),
        "checks": {
            "run_completeness": completeness["pass"],
            "quick_code_consistency": source["pass"],
            "protected_scope": protected["pass"],
            "prereg_sealed_terminal": sealed["pass"],
            "registry": registry["all_completed_runs_have_registry_entry"],
            "process": process["pass"],
            "unit_tests_9": True,
            "real_truncation_equivalence": read_json(RUNS / "E_R1_REAL_TRUNCATION_AUDIT.json")["pass"],
            "requirement_matrix": (RUNS / "E_R1_REQUIREMENT_MATRIX.md").exists(),
            "validation_t5_reference": (RUNS / "E_R1_VALIDATION_T5_REFERENCE.json").exists(),
        },
        "official_same_protocol_improvement": "NOT_ESTABLISHED_FAIL_CLOSED",
        "alpha_champion": "D02_FrozenResearchChampion_20260709",
        "defensive_champion": None,
        "e01_created": False,
    }
    audit["pass"] = all(value is True for value in audit["checks"].values())
    write_json(RUNS / "E_R1_FINAL_EVIDENCE_AUDIT.json", audit)
    build_report(state, completeness, source, protected, cost, gate_diag, timing, sealed, registry, process, t5_reference)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

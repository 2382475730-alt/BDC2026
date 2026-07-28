from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
D_ROOT = ROOT / "Experimental_WorkSpace" / "candidates_D"
AUTOMATION = D_ROOT / "automation_runs"
RUN_ID = "D_DEFENSIVE_POSITION_R3D_R4_ENSEMBLE_20260716"
RUN_DIR = AUTOMATION / RUN_ID
PROTOCOL = RUN_DIR / "POSITION_R3D_PROTOCOL.json"
STATE = RUN_DIR / "RUN_STATE.json"
REGISTRY = D_ROOT / "D_experiment_registry.json"
DATA_PATH = D_ROOT / "data_snapshots" / "c0_train_plus_baostock_future_20260703.csv"
R4_SPLIT = AUTOMATION / "R4_SPLIT_MANIFEST.json"
R5_SPLIT = AUTOMATION / "R5_SPLIT_MANIFEST.json"
R3B_RUN = AUTOMATION / "D_DEFENSIVE_POSITION_R3B_20260716"
GLOBAL_SEALED_LEDGER = AUTOMATION / "D_RECENT_HOLDOUT_GLOBAL_ACCESS.json"
D02 = D_ROOT / "D02_FrozenResearchChampion_20260709"
D03 = D_ROOT / "D03_R5_L1_F0_ResearchCandidate_20260711"
D04 = D_ROOT / "D04_D001_P2_InverseVol_ResearchCandidate_20260713"
CANDIDATE = ROOT / "candidate"
OFFICIAL_BASELINE = 0.012528499297343995
SOURCE_RUN_IDS = (
    "D_AUTO_R4_A_PDR_D02_LOSS_SERIOUS_S20260710_20260710_133055",
    "D_AUTO_R4_A_PDR_D02_LOSS_SERIOUS_S20260711_20260710_133540",
    "D_AUTO_R4_A_PDR_D02_LOSS_SERIOUS_S20260712_20260710_134017",
    "D_AUTO_R4_A_PDR_D02_LOSS_SERIOUS_S20260713_20260710_134512",
    "D_AUTO_R4_A_PDR_D02_LOSS_SERIOUS_S20260714_20260710_134950",
)
SOURCE_SEEDS = (20260710, 20260711, 20260712, 20260713, 20260714)

sys.path.insert(0, str(ROOT))
from Evaluation.scripts.d_series.common import read_json, write_json  # noqa: E402
from Evaluation.scripts.d_series.defensive_position_r3 import (  # noqa: E402
    attach_top2_control,
    canonical_hash,
    directory_fingerprint,
    evaluate_explicit_weeks,
    file_hash,
    moving_block_probability,
    official_proxy_summary,
    paired_weekly,
    project_invested_weights,
    weekly_metrics,
    weighted_turnover,
)
from Evaluation.scripts.d_series.r4_protocol import aggregate_predictions  # noqa: E402


def utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {"id": "R3D_TOP5_EQUAL", "weight_mode": "equal"},
        {"id": "R3D_RANK_TILT_30", "weight_mode": "rank", "weights": [0.30, 0.25, 0.20, 0.15, 0.10]},
        {"id": "R3D_RANK_TILT_28", "weight_mode": "rank", "weights": [0.28, 0.24, 0.20, 0.16, 0.12]},
        {"id": "R3D_SOFTMAX_T050", "weight_mode": "softmax", "temperature": 0.50},
        {"id": "R3D_SOFTMAX_T100", "weight_mode": "softmax", "temperature": 1.00},
        {"id": "R3D_SOFTMAX_T150", "weight_mode": "softmax", "temperature": 1.50},
    ]


def config_hash() -> str:
    return canonical_hash(candidate_grid())


def protected_fingerprints() -> dict[str, Any]:
    return {
        "candidate": directory_fingerprint(CANDIDATE),
        "D02": directory_fingerprint(D02),
        "D03": directory_fingerprint(D03),
        "D04": directory_fingerprint(D04),
    }


def source_manifest() -> dict[str, Any]:
    rows = []
    for expected_seed, run_id in zip(SOURCE_SEEDS, SOURCE_RUN_IDS):
        path = AUTOMATION / run_id
        config = read_json(path / "generated_config.json")
        split = read_json(path / "split_metadata.json")
        leakage = read_json(path / "leakage_audit.json")
        if int(config["seed"]) != expected_seed:
            raise RuntimeError(f"seed mismatch: {run_id}")
        if split["train"]["sample_end"] != "2024-12-25":
            raise RuntimeError(f"unexpected training cutoff: {run_id}")
        if leakage["leakage_audit"] != "pass":
            raise RuntimeError(f"source leakage audit failed: {run_id}")
        files = {
            "checkpoint": path / "best_model.pth",
            "scalers": path / "scalers.json",
            "config": path / "generated_config.json",
            "code_src": path / "code_snapshot" / "src",
        }
        rows.append(
            {
                "run_id": run_id,
                "seed": expected_seed,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "train_sample_end": split["train"]["sample_end"],
                "train_label_end": split["train"]["label_end_end"],
                "checkpoint_sha256": file_hash(files["checkpoint"]),
                "scalers_sha256": file_hash(files["scalers"]),
                "config_sha256": file_hash(files["config"]),
                "code_fingerprint": directory_fingerprint(files["code_src"]),
            }
        )
    return {"source_count": len(rows), "sources": rows}


def initial_state() -> dict[str, Any]:
    return {
        "experiment_id": RUN_ID,
        "created_at_utc": utc_now(),
        "stage": "new",
        "completed_stages": [],
        "oot_finalist": None,
        "sealed_holdout_accessed": False,
        "d05_created": False,
        "events": [],
    }


def load_state() -> dict[str, Any]:
    return read_json(STATE, default=None) or initial_state()


def save_state(state: Mapping[str, Any]) -> None:
    write_json(STATE, dict(state))


def event(state: dict[str, Any], name: str, **details: Any) -> None:
    state["events"].append({"at_utc": utc_now(), "event": name, **details})
    save_state(state)


def update_registry(status: str, decision: str | None, metrics: Mapping[str, Any] | None = None) -> None:
    registry = read_json(REGISTRY, default={}) or {}
    entries = registry.setdefault("experiments", [])
    entry = {
        "id": RUN_ID,
        "name": "D Defensive Position R3D R4 five-seed ensemble study",
        "path": str(RUN_DIR.relative_to(ROOT)).replace("\\", "/"),
        "status": status,
        "decision": decision,
        "source_program": "R4 A_PDR_D02_LOSS, five frozen seeds",
        "architecture": "unchanged R4 rankers; trimmed standardized score consensus; exact-five bounded policy",
        "training_dates": "Frozen checkpoints trained through 2024-12-25",
        "validation_dates": "R4 development diagnostic; primary OOT 2025Q4 plus 2026Q1",
        "seed": list(SOURCE_SEEDS),
        "hyperparameters": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/") if PROTOCOL.exists() else None,
        "metrics": dict(metrics or {}),
        "updated_at_utc": utc_now(),
    }
    for index, existing in enumerate(entries):
        if existing.get("id") == RUN_ID:
            entries[index] = entry
            break
    else:
        entries.append(entry)
    write_json(REGISTRY, registry)


def protocol_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "d_defensive_position_r3d_r4_ensemble_v1",
        "experiment_id": RUN_ID,
        "created_at_utc": utc_now(),
        "locked_before_full_inference_or_oot_return_computation": True,
        "objective": "Retest the strongest unpromoted R4 Top-5 lineage under strict diversification and later out-of-time windows.",
        "source": {
            "variant": "A_PDR_D02_LOSS",
            "seed_count": 5,
            "train_sample_end": "2024-12-25",
            "train_max_label_end": "2025-01-02",
            "checkpoint_selected_on": "R4 development 2025-01-03..2025-09-23",
            "consensus": "trimmed_mean_standardized",
            "source_manifest_sha256": canonical_hash(manifest),
        },
        "evaluation": {
            "diagnostic_only": "R4 development 38 weeks, 2025-01-06..2025-09-26",
            "primary_oot_q4": "13 weeks, 2025-10-09..2025-12-31",
            "primary_oot_q1": "11 weeks, 2026-01-05..2026-03-27",
            "primary_total_weeks": 24,
            "current_recent_holdout": "2026-04-07..2026-07-03 remains sealed and forbidden",
            "inference_sample_range": "2025-01-03..2026-03-20",
        },
        "configuration_count": len(candidate_grid()),
        "configuration_hash": config_hash(),
        "configurations": candidate_grid(),
        "position_contract": {
            "exactly_five": True,
            "full_investment": True,
            "long_only": True,
            "single_stock_bounds": [0.10, 0.30],
            "hhi_max": 0.24,
            "same_costs_as_D02": True,
        },
        "primary_oot_gate": {
            "official_proxy_above_official_baseline": True,
            "official_retention_vs_same_consensus_top2": ">=0.85",
            "mean_return_retention_vs_alpha_reference": ">=0.80",
            "cumulative_return_retention_vs_alpha_reference": ">=0.70",
            "maxdd_ratio_vs_alpha_reference": "<=0.75",
            "cvar95_ratio_vs_alpha_reference": "<=0.85",
            "worst_week_abs_ratio_vs_alpha_reference": "<=0.80",
            "positive_week_rate": ">=alpha reference - 1/24",
            "candidate_without_best2": ">0",
            "paired_without_best2_vs_alpha_reference": ">=0",
            "q4_and_q1_mean_noninferiority": "each >= alpha reference segment mean - 0.005",
            "bootstrap_probability_of_80pct_alpha_mean": ">=0.60",
            "exact_position_contract": True,
            "loso": "top two diagnostic; a passing finalist additionally requires >=4/5 LOSO runs with >=75% same-consensus mean retention and no worse CVaR",
        },
        "promotion": {
            "oot_pass_does_not_create_D05": True,
            "new fixed-epoch refit through 2026-03-26 and one-time recent holdout required": True,
            "D02_remains_Alpha_Champion": True,
            "D05_if_all_gates_pass": "Defensive Champion only",
        },
        "data_sha256": file_hash(DATA_PATH),
        "production_refit_forbidden_at_R3D": True,
    }


def preflight() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if "preflight" in state["completed_stages"]:
        return read_json(RUN_DIR / "PREFLIGHT_AUDIT.json")
    ledger = read_json(GLOBAL_SEALED_LEDGER)
    if ledger["status"] != "sealed" or int(ledger["unseal_count"]) != 0:
        raise RuntimeError("current recent holdout is not sealed")
    manifest = source_manifest()
    manifest["protected_before"] = protected_fingerprints()
    manifest["data_sha256"] = file_hash(DATA_PATH)
    write_json(RUN_DIR / "INPUT_MANIFEST.json", manifest)
    write_json(PROTOCOL, protocol_payload(manifest))
    (RUN_DIR / "POSITION_R3D_PROTOCOL.sha256").write_text(file_hash(PROTOCOL) + "\n", encoding="ascii")
    command = [sys.executable, "-m", "unittest", "Evaluation.scripts.d_series.test_defensive_position_r3d", "-v"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (RUN_DIR / "unit_test_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (RUN_DIR / "unit_test_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError("R3D unit tests failed")
    audit = {
        "status": "pass",
        "protocol_sha256": file_hash(PROTOCOL),
        "configuration_hash": config_hash(),
        "source_count": manifest["source_count"],
        "source_train_end": "2024-12-25",
        "unit_tests_pass": True,
        "current_recent_holdout_unseal_count": 0,
    }
    write_json(RUN_DIR / "PREFLIGHT_AUDIT.json", audit)
    state["completed_stages"].append("preflight")
    state["stage"] = "preflighted"
    event(state, "preflight_completed", protocol_sha256=audit["protocol_sha256"])
    update_registry("preflighted", None)
    return audit


def run_inference() -> dict[str, Any]:
    state = load_state()
    if "inference" in state["completed_stages"]:
        return read_json(RUN_DIR / "inference" / "INFERENCE_MANIFEST.json")
    preflight()
    output = RUN_DIR / "inference"
    output.mkdir(parents=True, exist_ok=True)
    source = read_json(RUN_DIR / "INPUT_MANIFEST.json")
    rows = []
    for item in source["sources"]:
        run = ROOT / item["path"]
        seed_output = output / f"seed_{item['seed']}"
        seed_output.mkdir(parents=True, exist_ok=True)
        prediction_path = seed_output / "predictions_20250103_20260320.csv"
        timing_path = seed_output / "timing.json"
        command = [
            sys.executable,
            "-B",
            str(ROOT / "Evaluation" / "scripts" / "d_series" / "r4_checkpoint_inference.py"),
            "--code-src", str(run / "code_snapshot" / "src"),
            "--config", str(run / "generated_config.json"),
            "--checkpoint", str(run / "best_model.pth"),
            "--scalers", str(run / "scalers.json"),
            "--data", str(DATA_PATH),
            "--sample-start", "2025-01-03",
            "--sample-end", "2026-03-20",
            "--output", str(prediction_path),
            "--timing-output", str(timing_path),
        ]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPYCACHEPREFIX"] = str(seed_output / "pycache")
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        (seed_output / "stdout.log").write_text(completed.stdout, encoding="utf-8")
        (seed_output / "stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"R3D inference failed for seed {item['seed']}")
        frame = pd.read_csv(prediction_path, dtype={"stock_id": str})
        if frame.duplicated(["sample_end_date", "stock_id"]).any():
            raise RuntimeError("duplicate inference prediction keys")
        rows.append(
            {
                "seed": item["seed"],
                "source_run": item["run_id"],
                "prediction_path": str(prediction_path.relative_to(ROOT)).replace("\\", "/"),
                "prediction_sha256": file_hash(prediction_path),
                "row_count": int(len(frame)),
                "date_count": int(frame["sample_end_date"].nunique()),
                "date_start": str(frame["sample_end_date"].min()),
                "date_end": str(frame["sample_end_date"].max()),
                "elapsed_seconds": elapsed,
                "timing": read_json(timing_path),
            }
        )
    manifest = {
        "status": "pass",
        "source_count": len(rows),
        "sample_start": "2025-01-03",
        "sample_end": "2026-03-20",
        "predictions": rows,
        "total_elapsed_seconds": sum(item["elapsed_seconds"] for item in rows),
    }
    write_json(output / "INFERENCE_MANIFEST.json", manifest)
    state["completed_stages"].append("inference")
    state["stage"] = "inference_completed"
    event(state, "frozen_checkpoint_inference_completed", seed_count=len(rows))
    update_registry("inference_completed", None)
    return manifest


def attach_weights(predictions: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.copy().reset_index(drop=True)
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
    frame["portfolio_weight"] = 0.0
    frame["portfolio_selected"] = False
    decisions = []
    for date, day in frame.groupby("sample_end_date", sort=True):
        ranked = day.dropna(subset=["score"]).sort_values(
            ["score", "stock_id"], ascending=[False, True], kind="mergesort"
        )
        if len(ranked) < 5:
            raise RuntimeError(f"fewer than five candidates at {date}")
        selected = ranked.head(5)
        mode = config["weight_mode"]
        if mode == "equal":
            weights = np.full(5, 0.20, dtype=float)
        elif mode == "rank":
            weights = project_invested_weights(config["weights"])
        elif mode == "softmax":
            all_scores = ranked["score"].to_numpy(dtype=float)
            normalized = (all_scores - float(all_scores.mean())) / (float(all_scores.std()) + 1e-12)
            raw = np.exp((normalized[:5] - float(normalized[:5].max())) / float(config["temperature"]))
            weights = project_invested_weights(raw)
        else:
            raise ValueError(f"unsupported weight mode: {mode}")
        frame.loc[selected.index, "portfolio_weight"] = weights
        frame.loc[selected.index, "portfolio_selected"] = True
        decisions.append(
            {
                "decision_date": date,
                "strategy": config["id"],
                "selected_stocks": ",".join(selected["stock_id"].astype(str)),
                "selected_count": 5,
                "weight_sum": 1.0,
                "cash_ratio": 0.0,
                "invested_hhi": float(np.square(weights).sum()),
                "max_invested_weight": float(weights.max()),
                "max_actual_weight": float(weights.max()),
                "weights": ",".join(f"{value:.17g}" for value in weights),
            }
        )
    return frame, pd.DataFrame(decisions)


def _weeks_from_r4(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    market_dates = sorted(pd.to_datetime(pd.read_csv(DATA_PATH, usecols=["日期"])["日期"]).dt.normalize().unique())
    market_dates = [pd.Timestamp(value).normalize() for value in market_dates]
    index = {value: position for position, value in enumerate(market_dates)}
    weeks = []
    for item in items:
        buy = pd.Timestamp(item["start"]).normalize()
        sell = pd.Timestamp(item["end"]).normalize()
        decision = market_dates[index[buy] - 1]
        weeks.append(
            {
                "decision_date": decision.strftime("%Y-%m-%d"),
                "buy_date": buy.strftime("%Y-%m-%d"),
                "sell_date": sell.strftime("%Y-%m-%d"),
                "trading_day_count": int(item["trading_day_count"]),
            }
        )
    return weeks


def evaluation_weeks() -> dict[str, list[dict[str, Any]]]:
    r4 = read_json(R4_SPLIT)
    r5 = read_json(R5_SPLIT)
    q1 = next(
        item for item in r5["rolling_origin_development"]["folds"]
        if item["fold_id"] == "R5_DEV_FOLD_04_2026_Q1"
    )["outer_evaluation"]["weeks"]
    return {
        "R4_DEVELOPMENT": _weeks_from_r4(r4["development"]["weeks"]),
        "OOT_2025_Q4": _weeks_from_r4(r4["sealed_holdout"]["weeks"]),
        "OOT_2026_Q1": q1,
    }


def _evaluate_segments(
    weighted: pd.DataFrame,
    decisions: pd.DataFrame,
    weeks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    weekly_parts = []
    segments = {}
    for segment, segment_weeks in weeks.items():
        weekly = evaluate_explicit_weeks(weighted, segment_weeks, DATA_PATH, fold_id=segment)
        segment_decisions = decisions[decisions["decision_date"].isin([item["decision_date"] for item in segment_weeks])]
        segments[segment] = weekly_metrics(weekly, segment_decisions)
        weekly_parts.append(weekly)
    oot = pd.concat([weekly_parts[1], weekly_parts[2]], ignore_index=True)
    oot_decision_dates = set(oot["decision_date"].astype(str))
    oot_decisions = decisions[decisions["decision_date"].astype(str).isin(oot_decision_dates)]
    return {
        "weekly_all": pd.concat(weekly_parts, ignore_index=True),
        "weekly_oot": oot,
        "metrics_oot": weekly_metrics(oot, oot_decisions),
        "segments": segments,
    }


def _alpha_reference(weeks: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    predictions = pd.read_csv(D02 / "best_val_predictions.csv", dtype={"stock_id": str})
    weighted, decisions = attach_top2_control(predictions)
    q4 = evaluate_explicit_weeks(weighted, weeks["OOT_2025_Q4"], DATA_PATH, fold_id="OOT_2025_Q4")
    q1_all = pd.read_csv(R3B_RUN / "development" / "R3_DP0_TOP2_EQUAL_ensemble_weekly.csv")
    q1 = q1_all[q1_all["fold_id"] == "R5_DEV_FOLD_04_2026_Q1"].copy()
    q1["fold_id"] = "OOT_2026_Q1"
    combined = pd.concat([q4, q1], ignore_index=True)
    return {
        "weekly": combined,
        "metrics": weekly_metrics(combined),
        "segments": {
            "OOT_2025_Q4": weekly_metrics(q4),
            "OOT_2026_Q1": weekly_metrics(q1),
        },
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > 1e-12 else 0.0


def _gate(
    candidate: Mapping[str, Any],
    weighted: pd.DataFrame,
    decisions: pd.DataFrame,
    consensus_control: Mapping[str, Any],
    alpha: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = candidate["metrics_oot"]
    alpha_metrics = alpha["metrics"]
    _, paired_alpha = paired_weekly(candidate["weekly_oot"], alpha["weekly"])
    _, paired_control = paired_weekly(candidate["weekly_oot"], consensus_control["weekly_oot"])
    decision_dates = set(candidate["weekly_oot"]["decision_date"].astype(str))
    oot_decisions = decisions[decisions["decision_date"].astype(str).isin(decision_dates)]
    oot_predictions = weighted[weighted["sample_end_date"].astype(str).between("2025-09-30", "2026-03-20")]
    official = official_proxy_summary(oot_predictions)
    control_official = consensus_control["official_oot"]
    bootstrap_values = (
        candidate["weekly_oot"]["weekly_return_net"].to_numpy(dtype=float)
        - 0.80 * alpha["weekly"]["weekly_return_net"].to_numpy(dtype=float)
    )
    checks = {
        "official_above_baseline": official["official_proxy_final_score"] > OFFICIAL_BASELINE,
        "official_retention_same_consensus_85pct": (
            official["official_proxy_final_score"] >= 0.85 * control_official["official_proxy_final_score"]
        ),
        "mean_retention_alpha_80pct": metrics["mean_weekly_return_net"] >= 0.80 * alpha_metrics["mean_weekly_return_net"],
        "cumulative_retention_alpha_70pct": metrics["cumulative_return_net"] >= 0.70 * alpha_metrics["cumulative_return_net"],
        "maxdd_alpha_75pct": metrics["max_drawdown_abs"] <= 0.75 * alpha_metrics["max_drawdown_abs"],
        "cvar_alpha_85pct": metrics["CVaR_95_loss_abs"] <= 0.85 * alpha_metrics["CVaR_95_loss_abs"],
        "worst_week_alpha_80pct": abs(metrics["worst_week"]) <= 0.80 * abs(alpha_metrics["worst_week"]),
        "positive_week_noninferiority": metrics["positive_week_rate"] >= alpha_metrics["positive_week_rate"] - 1.0 / 24.0,
        "candidate_without_best2_positive": metrics["cumulative_without_best_2_weeks"] > 0.0,
        "paired_alpha_without_best2_nonnegative": paired_alpha["diff_without_best_2_weeks"] >= 0.0,
        "q4_noninferiority_minus_0_005": (
            candidate["segments"]["OOT_2025_Q4"]["mean_weekly_return_net"]
            >= alpha["segments"]["OOT_2025_Q4"]["mean_weekly_return_net"] - 0.005
        ),
        "q1_noninferiority_minus_0_005": (
            candidate["segments"]["OOT_2026_Q1"]["mean_weekly_return_net"]
            >= alpha["segments"]["OOT_2026_Q1"]["mean_weekly_return_net"] - 0.005
        ),
        "bootstrap_probability_60pct": moving_block_probability(bootstrap_values) >= 0.60,
        "exact_position_contract": bool(
            (oot_decisions["selected_count"] == 5).all()
            and np.allclose(oot_decisions["weight_sum"], 1.0, atol=1e-12)
            and (oot_decisions["max_invested_weight"] <= 0.30 + 1e-12).all()
            and (oot_decisions["invested_hhi"] <= 0.24 + 1e-12).all()
        ),
    }
    official_retention = _safe_ratio(
        official["official_proxy_final_score"], control_official["official_proxy_final_score"]
    )
    mean_retention = _safe_ratio(metrics["mean_weekly_return_net"], alpha_metrics["mean_weekly_return_net"])
    maxdd_gain = 1.0 - _safe_ratio(metrics["max_drawdown_abs"], alpha_metrics["max_drawdown_abs"])
    cvar_gain = 1.0 - _safe_ratio(metrics["CVaR_95_loss_abs"], alpha_metrics["CVaR_95_loss_abs"])
    utility = 0.35 * official_retention + 0.35 * mean_retention + 0.15 * maxdd_gain + 0.15 * cvar_gain
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "official": official,
        "control_official": control_official,
        "official_retention": official_retention,
        "mean_retention_alpha": mean_retention,
        "paired_alpha": paired_alpha,
        "paired_same_consensus": paired_control,
        "bootstrap_probability": moving_block_probability(bootstrap_values),
        "utility": utility,
    }


def _loso(
    configs: Sequence[Mapping[str, Any]],
    frames: Sequence[pd.DataFrame],
    weeks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    output = {}
    for config in configs:
        runs = []
        for index, excluded_seed in enumerate(SOURCE_SEEDS):
            consensus = aggregate_predictions(
                [frame for frame_index, frame in enumerate(frames) if frame_index != index],
                method="trimmed_mean_standardized",
            )
            candidate_weighted, candidate_decisions = attach_weights(consensus, config)
            candidate = _evaluate_segments(candidate_weighted, candidate_decisions, weeks)
            control_weighted, control_decisions = attach_top2_control(consensus)
            control = _evaluate_segments(control_weighted, control_decisions, weeks)
            _, paired = paired_weekly(candidate["weekly_oot"], control["weekly_oot"])
            c = candidate["metrics_oot"]
            m = control["metrics_oot"]
            pass_run = (
                c["mean_weekly_return_net"] >= 0.75 * m["mean_weekly_return_net"]
                and c["CVaR_95_loss_abs"] <= m["CVaR_95_loss_abs"]
            )
            runs.append(
                {
                    "excluded_seed": excluded_seed,
                    "passes": pass_run,
                    "candidate": c,
                    "control": m,
                    "paired": paired,
                }
            )
        output[str(config["id"])] = {
            "pass_count": sum(item["passes"] for item in runs),
            "required": 4,
            "runs": runs,
        }
    return output


def evaluate() -> dict[str, Any]:
    state = load_state()
    if "evaluation" in state["completed_stages"]:
        return read_json(RUN_DIR / "evaluation" / "OOT_GATE_DECISION.json")
    inference = run_inference()
    output = RUN_DIR / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for item in inference["predictions"]:
        path = ROOT / item["prediction_path"]
        if file_hash(path) != item["prediction_sha256"]:
            raise RuntimeError("inference prediction changed")
        frames.append(pd.read_csv(path, dtype={"stock_id": str}))
    consensus = aggregate_predictions(frames, method="trimmed_mean_standardized")
    weeks = evaluation_weeks()
    control_weighted, control_decisions = attach_top2_control(consensus)
    control = _evaluate_segments(control_weighted, control_decisions, weeks)
    oot_mask = control_weighted["sample_end_date"].astype(str).between("2025-09-30", "2026-03-20")
    control["official_oot"] = official_proxy_summary(control_weighted[oot_mask])
    alpha = _alpha_reference(weeks)
    alpha["weekly"].to_csv(output / "alpha_reference_oot_weekly.csv", index=False, encoding="utf-8")
    write_json(output / "alpha_reference_metrics.json", {"metrics": alpha["metrics"], "segments": alpha["segments"]})
    control["weekly_oot"].to_csv(output / "same_consensus_top2_oot_weekly.csv", index=False, encoding="utf-8")
    write_json(
        output / "same_consensus_top2_metrics.json",
        {"metrics_oot": control["metrics_oot"], "segments": control["segments"], "official_oot": control["official_oot"]},
    )

    rows = []
    candidates = {}
    gates = {}
    configs = candidate_grid()
    for config in configs:
        weighted, decisions = attach_weights(consensus, config)
        result = _evaluate_segments(weighted, decisions, weeks)
        gate = _gate(result, weighted, decisions, control, alpha)
        identifier = str(config["id"])
        candidates[identifier] = result
        gates[identifier] = gate
        result["weekly_all"].to_csv(output / f"{identifier}_weekly_all.csv", index=False, encoding="utf-8")
        decisions.to_csv(output / f"{identifier}_decisions.csv", index=False, encoding="utf-8")
        metrics = result["metrics_oot"]
        rows.append(
            {
                "id": identifier,
                "weight_mode": config["weight_mode"],
                "official_proxy_oot": gate["official"]["official_proxy_final_score"],
                "official_retention": gate["official_retention"],
                "mean_weekly_net_oot": metrics["mean_weekly_return_net"],
                "cumulative_net_oot": metrics["cumulative_return_net"],
                "positive_week_rate_oot": metrics["positive_week_rate"],
                "max_drawdown_abs_oot": metrics["max_drawdown_abs"],
                "cvar95_oot": metrics["CVaR_95_loss_abs"],
                "worst_week_oot": metrics["worst_week"],
                "without_best2_oot": metrics["cumulative_without_best_2_weeks"],
                "q4_mean": result["segments"]["OOT_2025_Q4"]["mean_weekly_return_net"],
                "q1_mean": result["segments"]["OOT_2026_Q1"]["mean_weekly_return_net"],
                "paired_alpha_mean_excess": gate["paired_alpha"]["mean_weekly_excess"],
                "paired_alpha_pdr": gate["paired_alpha"]["positive_diff_rate"],
                "utility": gate["utility"],
                "passes": gate["passes"],
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["passes", "utility", "official_proxy_oot"], ascending=[False, False, False]
    )
    table.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8")
    write_json(
        output / "candidate_metrics.json",
        {
            identifier: {"metrics_oot": result["metrics_oot"], "segments": result["segments"]}
            for identifier, result in candidates.items()
        },
    )
    write_json(output / "oot_gates.json", gates)
    top_ids = table.head(2)["id"].astype(str).tolist()
    by_id = {str(config["id"]): config for config in configs}
    loso = _loso([by_id[identifier] for identifier in top_ids], frames, weeks)
    write_json(output / "LOSO_TOP2.json", loso)

    passing_ids = table[table["passes"]]["id"].astype(str).tolist()
    finalist = None
    for identifier in passing_ids:
        if loso.get(identifier, {}).get("pass_count", 0) >= 4:
            finalist = identifier
            break
        gates[identifier]["passes"] = False
        gates[identifier]["failed_checks"].append("loso_4_of_5")
    write_json(output / "oot_gates.json", gates)
    decision = {
        "status": "OOT_FINALIST_LOCKED" if finalist else "NO_OOT_FINALIST",
        "finalist": finalist,
        "evaluated_count": len(configs),
        "diagnostic_loso_ids": top_ids,
        "recent_refit_authorized": finalist is not None,
        "recent_holdout_authorized": False,
        "d05_created": False,
    }
    if finalist:
        write_json(
            RUN_DIR / "D05_OOT_FINALIST_LOCK.json",
            {
                "finalist": finalist,
                "config": by_id[finalist],
                "source_program": "A_PDR_D02_LOSS five-seed ensemble",
                "oot_gate_sha256": file_hash(output / "oot_gates.json"),
                "loso_sha256": file_hash(output / "LOSO_TOP2.json"),
                "recent_refit_authorized": True,
                "sealed_access_authorized": False,
            },
        )
    write_json(output / "OOT_GATE_DECISION.json", decision)
    state["completed_stages"].append("evaluation")
    state["stage"] = "oot_finalist_locked" if finalist else "rejected_oot"
    state["oot_finalist"] = finalist
    event(state, "oot_evaluation_completed", finalist=finalist, loso_ids=top_ids)
    update_registry(
        state["stage"],
        None if finalist else "REJECT",
        {"decision": decision, "top2": table.head(2).to_dict(orient="records")},
    )
    return decision


def build_report() -> Path:
    state = load_state()
    decision = read_json(RUN_DIR / "evaluation" / "OOT_GATE_DECISION.json", default={}) or {}
    table_path = RUN_DIR / "evaluation" / "candidate_summary.csv"
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    gates = read_json(RUN_DIR / "evaluation" / "oot_gates.json", default={}) or {}
    loso = read_json(RUN_DIR / "evaluation" / "LOSO_TOP2.json", default={}) or {}
    lines = [
        "# D 系列 R4 五 seed 五股防御仓位 Position R3D 报告",
        "",
        f"- Experiment ID: `{RUN_ID}`",
        f"- 状态：`{state.get('stage')}`",
        "- 排名源：R4 `A_PDR_D02_LOSS`，5 个冻结 seed，训练样本截止 `2024-12-25`",
        "- 主评估：2025Q4 13 周 + 2026Q1 11 周，共 24 周样本外",
        "- Current Recent Holdout：未访问",
        "- Alpha Champion：D02；Defensive Champion：NONE",
        "",
        "## 样本外结果",
        "",
        "| 方案 | Official | 周均净收益 | 累计净收益 | 正收益周 | MaxDD | CVaR95 | Worst | Q4均值 | Q1均值 | Alpha周均超额 | PASS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"|{row['id']}|{row['official_proxy_oot']:.6f}|{row['mean_weekly_net_oot']:.4%}|{row['cumulative_net_oot']:.4f}|{row['positive_week_rate_oot']:.2%}|{row['max_drawdown_abs_oot']:.2%}|{row['cvar95_oot']:.2%}|{row['worst_week_oot']:.2%}|{row['q4_mean']:.4%}|{row['q1_mean']:.4%}|{row['paired_alpha_mean_excess']:.4%}|{'PASS' if str(row['passes']).lower() == 'true' else 'FAIL'}|"
        )
    lines.extend(["", "## Top-2 门禁与 LOSO", ""])
    for identifier in decision.get("diagnostic_loso_ids", []):
        lines.append(
            f"- `{identifier}`：失败项 `{gates.get(identifier, {}).get('failed_checks', [])}`；LOSO `{loso.get(identifier, {}).get('pass_count', 0)}/5`。"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                f"样本外 finalist `{decision.get('finalist')}` 已锁定；下一阶段仅允许做预注册近期 refit，当前仍不创建 D05。"
                if decision.get("finalist")
                else "没有方案通过 24 周样本外与 LOSO 门禁，因此不执行近期 refit、不访问当前 Recent Holdout、不创建 D05。"
            ),
        ]
    )
    path = RUN_DIR / "D_DEFENSIVE_POSITION_R3D_REPORT_CN.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def artifact_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(RUN_DIR.rglob("*")):
        if (
            not path.is_file()
            or path.name == "ARTIFACT_MANIFEST.json"
            or "pycache" in {part.lower() for part in path.parts}
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        files.append(
            {
                "path": str(path.relative_to(RUN_DIR)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": file_hash(path),
            }
        )
    payload = {"file_count": len(files), "files": files}
    write_json(RUN_DIR / "ARTIFACT_MANIFEST.json", payload)
    return payload


def final_audit() -> dict[str, Any]:
    manifest = read_json(RUN_DIR / "INPUT_MANIFEST.json")
    ledger = read_json(GLOBAL_SEALED_LEDGER)
    state = load_state()
    current = protected_fingerprints()
    sources = source_manifest()
    checks = {
        "protocol_locked": file_hash(PROTOCOL)
        == (RUN_DIR / "POSITION_R3D_PROTOCOL.sha256").read_text(encoding="ascii").strip(),
        "source_manifest_unchanged": canonical_hash(sources["sources"]) == canonical_hash(manifest["sources"]),
        "candidate_unchanged": current["candidate"] == manifest["protected_before"]["candidate"],
        "D02_unchanged": current["D02"] == manifest["protected_before"]["D02"],
        "D03_unchanged": current["D03"] == manifest["protected_before"]["D03"],
        "D04_unchanged": current["D04"] == manifest["protected_before"]["D04"],
        "current_recent_holdout_still_sealed": ledger["status"] == "sealed" and int(ledger["unseal_count"]) == 0,
        "no_d05_created": not state["d05_created"],
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "protected_initial": manifest["protected_before"],
        "protected_final": current,
        "global_sealed_ledger": ledger,
        "state": state,
    }
    write_json(RUN_DIR / "FINAL_SAFETY_AUDIT.json", payload)
    artifact_manifest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R3D R4 five-seed exact-five OOT study.")
    parser.add_argument("--mode", choices=("preflight", "inference", "evaluation", "resume", "audit"), default="resume")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.mode == "preflight":
        result: Any = preflight()
    elif args.mode == "inference":
        result = run_inference()
    elif args.mode == "evaluation":
        result = evaluate()
    elif args.mode == "audit":
        build_report()
        result = final_audit()
    else:
        preflight()
        inference = run_inference()
        decision = evaluate()
        report = build_report()
        audit = final_audit()
        result = {"inference": inference, "decision": decision, "report": str(report), "audit": audit}
    if isinstance(result, dict):
        result["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

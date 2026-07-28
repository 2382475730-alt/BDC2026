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
RUN_ID = "D_DEFENSIVE_POSITION_R3C_CONSENSUS_20260716"
RUN_DIR = AUTOMATION / RUN_ID
PROTOCOL = RUN_DIR / "POSITION_R3C_PROTOCOL.json"
STATE = RUN_DIR / "RUN_STATE.json"
REGISTRY = D_ROOT / "D_experiment_registry.json"
DATA_PATH = D_ROOT / "data_snapshots" / "c0_train_plus_baostock_future_20260703.csv"
R2_RUN = AUTOMATION / "D_DEFENSIVE_POSITION_R2_20260714_092957"
R3B_RUN = AUTOMATION / "D_DEFENSIVE_POSITION_R3B_20260716"
GLOBAL_SEALED_LEDGER = AUTOMATION / "D_RECENT_HOLDOUT_GLOBAL_ACCESS.json"
D02 = D_ROOT / "D02_FrozenResearchChampion_20260709"
D03 = D_ROOT / "D03_R5_L1_F0_ResearchCandidate_20260711"
D04 = D_ROOT / "D04_D001_P2_InverseVol_ResearchCandidate_20260713"
CANDIDATE = ROOT / "candidate"
OFFICIAL_BASELINE = 0.012528499297343995
SEEDS = (20260720, 20260721, 20260722)
FOLDS = (
    "R5_DEV_FOLD_01_2025_Q1",
    "R5_DEV_FOLD_02_2025_Q2",
    "R5_DEV_FOLD_03_2025_Q3",
    "R5_DEV_FOLD_04_2026_Q1",
)
METHODS = (
    "mean_percentile_rank",
    "trimmed_mean_standardized",
    "borda_rank_average",
    "disagreement_penalized_consensus",
)
WEIGHT_MODES = ("equal", "projected_softmax", "empirical_cvar")

sys.path.insert(0, str(ROOT))
from Evaluation.scripts.d_series.common import read_json, write_json  # noqa: E402
from Evaluation.scripts.d_series.defensive_position_r3 import (  # noqa: E402
    CausalRiskData,
    attach_candidate_weights,
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
    configs: list[dict[str, Any]] = []
    for method in METHODS:
        method_id = method.upper()
        for mode in WEIGHT_MODES:
            config: dict[str, Any] = {
                "id": f"R3C_{method_id}_{mode.upper()}",
                "consensus_method": method,
                "weight_mode": mode,
                "selection_mode": "top5",
                "exposure_mode": "full",
            }
            if mode == "projected_softmax":
                config["temperature"] = 0.35
            if mode == "empirical_cvar":
                config["tail_fraction"] = 0.05
            configs.append(config)
    return configs


def config_hash() -> str:
    return canonical_hash(candidate_grid())


def source_manifest() -> dict[str, Any]:
    source = read_json(R2_RUN / "ranking_source_manifest.json")
    if not source.get("reuse_approved"):
        raise RuntimeError("R5 outer prediction source is not approved")
    rows = []
    for item in source["sources"]:
        path = Path(item["path"]) / "best_val_predictions.csv"
        expected = item["hashes"]["best_val_predictions_sha256"]
        if file_hash(path) != expected:
            raise RuntimeError(f"source prediction changed: {path}")
        rows.append(
            {
                "fold_id": item["fold_id"],
                "seed": int(item["seed"]),
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": expected,
            }
        )
    rows.sort(key=lambda item: (item["fold_id"], item["seed"]))
    expected_matrix = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if {(item["fold_id"], item["seed"]) for item in rows} != expected_matrix:
        raise RuntimeError("incomplete fold/seed source matrix")
    return {"source_count": len(rows), "sources": rows}


def initial_state() -> dict[str, Any]:
    return {
        "experiment_id": RUN_ID,
        "created_at_utc": utc_now(),
        "stage": "new",
        "completed_stages": [],
        "development_finalist": None,
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


def _fold_weeks() -> dict[str, list[dict[str, Any]]]:
    split = read_json(AUTOMATION / "R5_SPLIT_MANIFEST.json")
    return {
        item["fold_id"]: item["outer_evaluation"]["weeks"]
        for item in split["rolling_origin_development"]["folds"]
    }


def _protected_fingerprints() -> dict[str, Any]:
    return {
        "candidate": directory_fingerprint(CANDIDATE),
        "D02": directory_fingerprint(D02),
        "D03": directory_fingerprint(D03),
        "D04": directory_fingerprint(D04),
    }


def protocol_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "d_defensive_position_r3c_consensus_v1",
        "experiment_id": RUN_ID,
        "created_at_utc": utc_now(),
        "locked_before_r3c_return_computation": True,
        "objective": (
            "Build one actually tradable exact-five portfolio after cross-seed score consensus, with lower "
            "drawdown and tail risk and near-D02 return."
        ),
        "prior_evidence_only": {
            "R4": "trimmed standardized consensus plus Top-5 softmax was promising on R4 development but was not an R4 finalist",
            "R3B": "single-seed empirical CVaR improved risk but failed late development folds",
            "new_hypothesis": "consensus before one-time selection can improve ranks 3-5 and avoid seed-average pseudo-diversification",
        },
        "immutable": {
            "ranking_variant": "R5_L0_F0 fixed epoch 12, D02 program lineage",
            "ranking_predictions_unchanged": True,
            "data_path": str(DATA_PATH.relative_to(ROOT)).replace("\\", "/"),
            "data_sha256": file_hash(DATA_PATH),
            "label": "open(T+5)/open(T+1)-1",
            "recent_holdout_access_forbidden": True,
            "production_refit_forbidden": True,
        },
        "search": {
            "configuration_count": len(candidate_grid()),
            "configuration_hash": config_hash(),
            "configurations": candidate_grid(),
            "selection": "all configurations evaluated once on 47 development weeks; one passing finalist maximum",
            "diagnostic_loso": "top three by utility receive three leave-one-seed-out stress tests; LOSO cannot rescue a failed main gate",
        },
        "position_contract": {
            "exactly_five_stocks": True,
            "long_only": True,
            "full_investment": True,
            "single_stock_bounds": [0.10, 0.30],
            "hhi_max": 0.24,
            "costs": {"buy_commission": 0.0005, "sell_commission": 0.0005, "sell_stamp_tax": 0.0005},
        },
        "development_gate": {
            "official_above_official_baseline": True,
            "official_retention_vs_same_consensus_top2": ">=0.85",
            "official_retention_vs_R5_DP0": ">=0.85",
            "mean_return_retention_vs_same_consensus_top2": ">=0.80",
            "mean_return_retention_vs_R5_DP0": ">=0.85",
            "cumulative_return_retention_vs_R5_DP0": ">=0.70",
            "maxdd_vs_R5_DP0": "<=0.75",
            "cvar95_vs_R5_DP0": "<=0.85",
            "worst_week_abs_vs_R5_DP0": "<=0.80",
            "positive_week_rate": ">=R5_DP0 - 1/47",
            "without_best2": "candidate > 0, best-two contribution <=0.50, and paired vs same-consensus control >= 0",
            "fold_stability": "at least 3/4 within -0.005 of R5_DP0; no fold mean below -0.010; 2026Q1 within -0.005",
            "bootstrap_probability_of_80pct_R5_DP0_mean": ">=0.60",
            "exact_position_contract": True,
        },
        "promotion": {
            "development_pass_does_not_create_D05": True,
            "Legacy_and_one_time_recent_holdout_required": True,
            "D02_remains_Alpha_Champion": True,
            "D05_identity_if_all_gates_pass": "Defensive Champion only",
        },
        "source_manifest_sha256": canonical_hash(manifest),
    }


def update_registry(status: str, decision: str | None, metrics: Mapping[str, Any] | None = None) -> None:
    registry = read_json(REGISTRY, default={}) or {}
    entries = registry.setdefault("experiments", [])
    entry = {
        "id": RUN_ID,
        "name": "D Defensive Position R3C consensus-first exact-five study",
        "path": str(RUN_DIR.relative_to(ROOT)).replace("\\", "/"),
        "status": status,
        "decision": decision,
        "source_program": "R5_L0_F0 / D02 program lineage",
        "architecture": "unchanged ranker; score-only cross-seed consensus and exact-five position policy",
        "training_dates": "No retraining",
        "validation_dates": "R5 rolling development folds 2025Q1..2026Q1",
        "seed": list(SEEDS),
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


def preflight() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if "preflight" in state["completed_stages"]:
        return read_json(RUN_DIR / "PREFLIGHT_AUDIT.json")
    ledger = read_json(GLOBAL_SEALED_LEDGER)
    if ledger["status"] != "sealed" or int(ledger["unseal_count"]) != 0:
        raise RuntimeError("global recent holdout is not sealed")
    manifest = source_manifest()
    manifest["protected_before"] = _protected_fingerprints()
    manifest["data_sha256"] = file_hash(DATA_PATH)
    manifest["r3b_development_metrics_sha256"] = file_hash(R3B_RUN / "development" / "development_metrics.json")
    write_json(RUN_DIR / "INPUT_MANIFEST.json", manifest)
    write_json(PROTOCOL, protocol_payload(manifest))
    (RUN_DIR / "POSITION_R3C_PROTOCOL.sha256").write_text(file_hash(PROTOCOL) + "\n", encoding="ascii")
    command = [sys.executable, "-m", "unittest", "Evaluation.scripts.d_series.test_defensive_position_r3c", "-v"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (RUN_DIR / "unit_test_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (RUN_DIR / "unit_test_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError("R3C unit tests failed")
    audit = {
        "status": "pass",
        "protocol_sha256": file_hash(PROTOCOL),
        "configuration_hash": config_hash(),
        "source_count": manifest["source_count"],
        "unit_tests_pass": True,
        "global_sealed_unseal_count": int(ledger["unseal_count"]),
        "recent_return_access": 0,
    }
    write_json(RUN_DIR / "PREFLIGHT_AUDIT.json", audit)
    state["completed_stages"].append("preflight")
    state["stage"] = "preflighted"
    event(state, "preflight_completed", protocol_sha256=audit["protocol_sha256"])
    update_registry("preflighted", None)
    return audit


def attach_projected_softmax(
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
    risk: CausalRiskData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.copy().reset_index(drop=True)
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
    frame["portfolio_weight"] = 0.0
    frame["portfolio_selected"] = False
    decisions: list[dict[str, Any]] = []
    temperature = float(config["temperature"])
    for date, day in frame.groupby("sample_end_date", sort=True):
        ranked = day.dropna(subset=["score"]).sort_values(
            ["score", "stock_id"], ascending=[False, True], kind="mergesort"
        )
        selected = ranked.head(5)
        values = selected["score"].to_numpy(dtype=float)
        raw = np.exp((values - float(values.max())) / temperature)
        weights = project_invested_weights(raw)
        frame.loc[selected.index, "portfolio_weight"] = weights
        frame.loc[selected.index, "portfolio_selected"] = True
        state = risk.state_row(date)
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
                **state,
            }
        )
    return frame, pd.DataFrame(decisions)


def attach_weights(
    consensus: pd.DataFrame,
    config: Mapping[str, Any],
    risk: CausalRiskData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mode = config["weight_mode"]
    if mode == "projected_softmax":
        return attach_projected_softmax(consensus, config, risk)
    helper_config = {
        "id": config["id"],
        "selection_mode": "top5",
        "weight_mode": "equal" if mode == "equal" else "empirical_cvar",
        "tail_fraction": float(config.get("tail_fraction", 0.05)),
        "exposure_mode": "full",
    }
    return attach_candidate_weights(consensus, helper_config, risk)


def _read_source(item: Mapping[str, Any]) -> pd.DataFrame:
    path = ROOT / item["path"]
    if file_hash(path) != item["sha256"]:
        raise RuntimeError(f"source changed: {path}")
    return pd.read_csv(path, dtype={"stock_id": str})


def _combined_official(parts: Sequence[Mapping[str, Any]]) -> float:
    counts = np.asarray([item["sample_count"] for item in parts], dtype=float)
    scores = np.asarray([item["official_proxy_final_score"] for item in parts], dtype=float)
    return float(np.average(scores, weights=counts))


def _evaluate_consensus_config(
    source_by_fold: Mapping[str, Sequence[pd.DataFrame]],
    config: Mapping[str, Any],
    risk: CausalRiskData,
    weeks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    weekly_parts = []
    decision_parts = []
    official_parts = []
    for fold in FOLDS:
        consensus = aggregate_predictions(
            list(source_by_fold[fold]), method=str(config["consensus_method"])
        )
        weighted, decisions = attach_weights(consensus, config, risk)
        weekly = evaluate_explicit_weeks(weighted, weeks[fold], DATA_PATH, fold_id=fold)
        decisions["fold_id"] = fold
        weekly_parts.append(weekly)
        decision_parts.append(decisions)
        official_parts.append(official_proxy_summary(weighted))
    weekly_all = pd.concat(weekly_parts, ignore_index=True)
    decisions_all = pd.concat(decision_parts, ignore_index=True)
    metrics = weekly_metrics(weekly_all, decisions_all)
    metrics["fold_metrics"] = {
        fold: weekly_metrics(group)
        for fold, group in weekly_all.groupby("fold_id", sort=True)
    }
    metrics["official_proxy"] = _combined_official(official_parts)
    metrics["turnover"] = weighted_turnover(decisions_all)
    return {"weekly": weekly_all, "decisions": decisions_all, "metrics": metrics}


def _evaluate_consensus_control(
    source_by_fold: Mapping[str, Sequence[pd.DataFrame]],
    method: str,
    weeks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    weekly_parts = []
    decision_parts = []
    official_parts = []
    for fold in FOLDS:
        consensus = aggregate_predictions(list(source_by_fold[fold]), method=method)
        weighted, decisions = attach_top2_control(consensus)
        weekly = evaluate_explicit_weeks(weighted, weeks[fold], DATA_PATH, fold_id=fold)
        decisions["fold_id"] = fold
        weekly_parts.append(weekly)
        decision_parts.append(decisions)
        official_parts.append(official_proxy_summary(weighted))
    weekly_all = pd.concat(weekly_parts, ignore_index=True)
    decisions_all = pd.concat(decision_parts, ignore_index=True)
    metrics = weekly_metrics(weekly_all, decisions_all)
    metrics["fold_metrics"] = {
        fold: weekly_metrics(group)
        for fold, group in weekly_all.groupby("fold_id", sort=True)
    }
    metrics["official_proxy"] = _combined_official(official_parts)
    return {"weekly": weekly_all, "decisions": decisions_all, "metrics": metrics}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > 1e-12 else 0.0


def _gate(
    candidate: Mapping[str, Any],
    method_control: Mapping[str, Any],
    dp0: Mapping[str, Any],
) -> dict[str, Any]:
    c = candidate["metrics"]
    m = method_control["metrics"]
    paired_frame, paired = paired_weekly(candidate["weekly"], method_control["weekly"])
    fold_passes = {
        fold: c["fold_metrics"][fold]["mean_weekly_return_net"]
        >= dp0["fold_metrics"][fold]["mean_weekly_return_net"] - 0.005
        for fold in FOLDS
    }
    bootstrap_values = (
        paired_frame["weekly_return_net"].to_numpy(dtype=float)
        - 0.80 * paired_frame["control_weekly_return_net"].to_numpy(dtype=float)
    )
    checks = {
        "official_above_baseline": c["official_proxy"] > OFFICIAL_BASELINE,
        "official_retention_same_consensus_85pct": c["official_proxy"] >= 0.85 * m["official_proxy"],
        "official_retention_dp0_85pct": c["official_proxy"] >= 0.85 * dp0["official_proxy_median_across_seeds"],
        "mean_retention_same_consensus_80pct": c["mean_weekly_return_net"] >= 0.80 * m["mean_weekly_return_net"],
        "mean_retention_dp0_85pct": c["mean_weekly_return_net"] >= 0.85 * dp0["mean_weekly_return_net"],
        "cumulative_retention_dp0_70pct": c["cumulative_return_net"] >= 0.70 * dp0["cumulative_return_net"],
        "maxdd_dp0_75pct": c["max_drawdown_abs"] <= 0.75 * dp0["max_drawdown_abs"],
        "cvar_dp0_85pct": c["CVaR_95_loss_abs"] <= 0.85 * dp0["CVaR_95_loss_abs"],
        "worst_week_dp0_80pct": abs(c["worst_week"]) <= 0.80 * abs(dp0["worst_week"]),
        "positive_week_noninferiority": c["positive_week_rate"] >= dp0["positive_week_rate"] - 1.0 / 47.0,
        "candidate_without_best2_positive": c["cumulative_without_best_2_weeks"] > 0.0,
        "best2_contribution_50pct": c["best_2_weeks_contribution_ratio"] <= 0.50,
        "paired_without_best2_nonnegative": paired["diff_without_best_2_weeks"] >= 0.0,
        "fold_noninferiority_3_of_4": sum(fold_passes.values()) >= 3,
        "no_fold_below_minus_0_010": min(
            c["fold_metrics"][fold]["mean_weekly_return_net"] for fold in FOLDS
        ) >= -0.010,
        "f4_noninferiority_minus_0_005": (
            c["fold_metrics"][FOLDS[-1]]["mean_weekly_return_net"]
            >= dp0["fold_metrics"][FOLDS[-1]]["mean_weekly_return_net"] - 0.005
        ),
        "bootstrap_probability_60pct": moving_block_probability(bootstrap_values) >= 0.60,
        "exact_position_contract": bool(
            (candidate["decisions"]["selected_count"] == 5).all()
            and np.allclose(candidate["decisions"]["weight_sum"], 1.0, atol=1e-12)
            and (candidate["decisions"]["max_invested_weight"] <= 0.30 + 1e-12).all()
            and (candidate["decisions"]["invested_hhi"] <= 0.24 + 1e-12).all()
        ),
    }
    official_retention = _safe_ratio(c["official_proxy"], dp0["official_proxy_median_across_seeds"])
    mean_retention = _safe_ratio(c["mean_weekly_return_net"], dp0["mean_weekly_return_net"])
    maxdd_gain = 1.0 - _safe_ratio(c["max_drawdown_abs"], dp0["max_drawdown_abs"])
    cvar_gain = 1.0 - _safe_ratio(c["CVaR_95_loss_abs"], dp0["CVaR_95_loss_abs"])
    utility = 0.40 * official_retention + 0.30 * mean_retention + 0.15 * maxdd_gain + 0.15 * cvar_gain
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "paired": paired,
        "fold_passes": fold_passes,
        "bootstrap_probability": moving_block_probability(bootstrap_values),
        "official_retention_vs_dp0": official_retention,
        "mean_retention_vs_dp0": mean_retention,
        "utility": utility,
    }


def _loso_diagnostics(
    finalists: Sequence[Mapping[str, Any]],
    source_items: Sequence[Mapping[str, Any]],
    risk: CausalRiskData,
    weeks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for config in finalists:
        rows = []
        for excluded in SEEDS:
            source_by_fold = {
                fold: [
                    _read_source(item)
                    for item in source_items
                    if item["fold_id"] == fold and int(item["seed"]) != excluded
                ]
                for fold in FOLDS
            }
            candidate = _evaluate_consensus_config(source_by_fold, config, risk, weeks)
            control = _evaluate_consensus_control(
                source_by_fold, str(config["consensus_method"]), weeks
            )
            _, paired = paired_weekly(candidate["weekly"], control["weekly"])
            rows.append(
                {
                    "excluded_seed": excluded,
                    "candidate": candidate["metrics"],
                    "control": control["metrics"],
                    "paired": paired,
                    "exact_position_contract": bool(
                        (candidate["decisions"]["selected_count"] == 5).all()
                        and np.allclose(candidate["decisions"]["weight_sum"], 1.0, atol=1e-12)
                        and (candidate["decisions"]["max_invested_weight"] <= 0.30 + 1e-12).all()
                        and (candidate["decisions"]["invested_hhi"] <= 0.24 + 1e-12).all()
                    ),
                }
            )
        output[str(config["id"])] = {
            "runs": rows,
            "positive_mean_excess_count": sum(
                item["paired"]["mean_weekly_excess"] >= 0.0 for item in rows
            ),
            "worst_mean_excess": min(item["paired"]["mean_weekly_excess"] for item in rows),
            "all_exact_position_contract": all(item["exact_position_contract"] for item in rows),
        }
    return output


def development_search() -> dict[str, Any]:
    state = load_state()
    if "development_search" in state["completed_stages"]:
        return read_json(RUN_DIR / "development" / "DEVELOPMENT_DECISION.json")
    preflight()
    output = RUN_DIR / "development"
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(RUN_DIR / "INPUT_MANIFEST.json")
    source_items = manifest["sources"]
    source_by_fold = {
        fold: [_read_source(item) for item in source_items if item["fold_id"] == fold]
        for fold in FOLDS
    }
    risk = CausalRiskData.from_market_path(DATA_PATH, cutoff="2026-03-27")
    weeks = _fold_weeks()
    dp0 = read_json(R3B_RUN / "development" / "development_metrics.json")["R3_DP0_TOP2_EQUAL"]

    controls = {
        method: _evaluate_consensus_control(source_by_fold, method, weeks)
        for method in METHODS
    }
    candidates: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    rows = []
    configs = candidate_grid()
    for config in configs:
        result = _evaluate_consensus_config(source_by_fold, config, risk, weeks)
        gate = _gate(result, controls[str(config["consensus_method"])], dp0)
        identifier = str(config["id"])
        candidates[identifier] = result
        gates[identifier] = gate
        metrics = result["metrics"]
        rows.append(
            {
                "id": identifier,
                "consensus_method": config["consensus_method"],
                "weight_mode": config["weight_mode"],
                "official_proxy": metrics["official_proxy"],
                "mean_weekly_net": metrics["mean_weekly_return_net"],
                "cumulative_net": metrics["cumulative_return_net"],
                "positive_week_rate": metrics["positive_week_rate"],
                "max_drawdown_abs": metrics["max_drawdown_abs"],
                "cvar95": metrics["CVaR_95_loss_abs"],
                "worst_week": metrics["worst_week"],
                "without_best2": metrics["cumulative_without_best_2_weeks"],
                "paired_mean_excess": gate["paired"]["mean_weekly_excess"],
                "paired_pdr": gate["paired"]["positive_diff_rate"],
                "utility": gate["utility"],
                "passes": gate["passes"],
            }
        )
        result["weekly"].to_csv(output / f"{identifier}_weekly.csv", index=False, encoding="utf-8")
        result["decisions"].to_csv(output / f"{identifier}_decisions.csv", index=False, encoding="utf-8")

    table = pd.DataFrame(rows).sort_values(
        ["passes", "utility", "official_proxy"], ascending=[False, False, False]
    )
    table.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8")
    serializable_metrics = {
        identifier: result["metrics"] for identifier, result in candidates.items()
    }
    control_metrics = {method: value["metrics"] for method, value in controls.items()}
    write_json(output / "candidate_metrics.json", serializable_metrics)
    write_json(output / "consensus_top2_controls.json", control_metrics)
    write_json(output / "development_gates.json", gates)

    diagnostic_ids = table.head(3)["id"].astype(str).tolist()
    config_by_id = {str(config["id"]): config for config in configs}
    loso = _loso_diagnostics(
        [config_by_id[identifier] for identifier in diagnostic_ids], source_items, risk, weeks
    )
    write_json(output / "LOSO_DIAGNOSTICS_TOP3.json", loso)

    passing = table[table["passes"]]
    finalist = str(passing.iloc[0]["id"]) if not passing.empty else None
    if finalist is not None:
        loso_gate = loso.get(finalist)
        if loso_gate is None or loso_gate["positive_mean_excess_count"] < 2:
            gates[finalist]["passes"] = False
            gates[finalist]["failed_checks"].append("loso_positive_mean_excess_2_of_3")
            finalist = None
            write_json(output / "development_gates.json", gates)
    decision = {
        "status": "DEVELOPMENT_FINALIST_LOCKED" if finalist else "NO_DEVELOPMENT_FINALIST",
        "finalist": finalist,
        "evaluated_count": len(configs),
        "diagnostic_loso_ids": diagnostic_ids,
        "legacy_authorized": finalist is not None,
        "recent_holdout_authorized": False,
        "d05_created": False,
    }
    if finalist:
        write_json(
            RUN_DIR / "D05_FINALIST_LOCK.json",
            {
                "finalist": finalist,
                "config": config_by_id[finalist],
                "development_gate_sha256": file_hash(output / "development_gates.json"),
                "loso_sha256": file_hash(output / "LOSO_DIAGNOSTICS_TOP3.json"),
                "sealed_access_authorized": False,
            },
        )
    write_json(output / "DEVELOPMENT_DECISION.json", decision)
    state["completed_stages"].append("development_search")
    state["stage"] = "development_finalist_locked" if finalist else "rejected_development"
    state["development_finalist"] = finalist
    event(state, "development_search_completed", finalist=finalist, diagnostic_loso=diagnostic_ids)
    update_registry(
        state["stage"],
        None if finalist else "REJECT",
        {"decision": decision, "top3": table.head(3).to_dict(orient="records")},
    )
    return decision


def build_report() -> Path:
    state = load_state()
    decision = read_json(RUN_DIR / "development" / "DEVELOPMENT_DECISION.json", default={}) or {}
    table_path = RUN_DIR / "development" / "candidate_summary.csv"
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame()
    gates = read_json(RUN_DIR / "development" / "development_gates.json", default={}) or {}
    loso = read_json(RUN_DIR / "development" / "LOSO_DIAGNOSTICS_TOP3.json", default={}) or {}
    lines = [
        "# D 系列五股共识仓位 Position R3C 报告",
        "",
        f"- Experiment ID: `{RUN_ID}`",
        f"- 状态：`{state.get('stage')}`",
        "- Alpha Champion：`D02_FrozenResearchChampion_20260709`",
        "- Defensive Champion：`NONE`",
        "- Recent Holdout：未访问",
        "",
        "## 方法",
        "",
        "对同一折的三个 R5_L0_F0 seed 先做分数共识，再只生成一次严格五股、满仓、单股 10%-30%、HHI<=0.24 的组合。比较 4 种共识与 3 种配权，共 12 个预注册组合。",
        "",
        "## 开发结果",
        "",
        "| 排名 | 方案 | Official | 周均净收益 | 累计净收益 | 正收益周 | MaxDD | CVaR95 | Worst | 同共识 Top2 周均超额 | PASS |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for index, row in table.head(12).reset_index(drop=True).iterrows():
        lines.append(
            f"|{index + 1}|{row['id']}|{row['official_proxy']:.6f}|{row['mean_weekly_net']:.4%}|{row['cumulative_net']:.4f}|{row['positive_week_rate']:.2%}|{row['max_drawdown_abs']:.2%}|{row['cvar95']:.2%}|{row['worst_week']:.2%}|{row['paired_mean_excess']:.4%}|{'PASS' if str(row['passes']).lower() == 'true' else 'FAIL'}|"
        )
    lines.extend(["", "## 门禁与 LOSO", ""])
    for identifier in decision.get("diagnostic_loso_ids", []):
        lines.append(
            f"- `{identifier}`：主门禁失败项 `{gates.get(identifier, {}).get('failed_checks', [])}`；LOSO 正超额 `{loso.get(identifier, {}).get('positive_mean_excess_count', 0)}/3`，最差周均超额 `{loso.get(identifier, {}).get('worst_mean_excess', 0):.4%}`。"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                f"开发 finalist `{decision.get('finalist')}` 已锁定，但仍需 Legacy 与一次性近期封存门禁；当前不创建 D05。"
                if decision.get("finalist")
                else "没有方案通过全部开发门禁，因此不进入 Legacy、不访问近期封存集、不创建 D05。"
            ),
        ]
    )
    path = RUN_DIR / "D_DEFENSIVE_POSITION_R3C_REPORT_CN.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def artifact_manifest() -> dict[str, Any]:
    files = []
    for path in sorted(RUN_DIR.rglob("*")):
        if (
            not path.is_file()
            or path.name == "ARTIFACT_MANIFEST.json"
            or "__pycache__" in path.parts
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
    current = _protected_fingerprints()
    source_files_unchanged = all(
        file_hash(ROOT / item["path"]) == item["sha256"] for item in manifest["sources"]
    )
    checks = {
        "protocol_locked": file_hash(PROTOCOL)
        == (RUN_DIR / "POSITION_R3C_PROTOCOL.sha256").read_text(encoding="ascii").strip(),
        "candidate_unchanged": current["candidate"] == manifest["protected_before"]["candidate"],
        "D02_unchanged": current["D02"] == manifest["protected_before"]["D02"],
        "D03_unchanged": current["D03"] == manifest["protected_before"]["D03"],
        "D04_unchanged": current["D04"] == manifest["protected_before"]["D04"],
        "all_source_predictions_unchanged": source_files_unchanged,
        "recent_holdout_still_sealed": ledger["status"] == "sealed" and int(ledger["unseal_count"]) == 0,
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
    parser = argparse.ArgumentParser(description="Run R3C consensus-first exact-five defensive study.")
    parser.add_argument("--mode", choices=("preflight", "development", "resume", "audit"), default="resume")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.mode == "preflight":
        result: Any = preflight()
    elif args.mode == "development":
        result = development_search()
    elif args.mode == "audit":
        build_report()
        result = final_audit()
    else:
        preflight()
        decision = development_search()
        report = build_report()
        audit = final_audit()
        result = {"decision": decision, "report": str(report), "audit": audit}
    if isinstance(result, dict):
        result["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

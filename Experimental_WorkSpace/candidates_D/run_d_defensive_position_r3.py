from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
D_ROOT = ROOT / "Experimental_WorkSpace" / "candidates_D"
AUTOMATION = D_ROOT / "automation_runs"
RUN_ID = "D_DEFENSIVE_POSITION_R3_20260716"
RUN_DIR = AUTOMATION / RUN_ID
D02 = D_ROOT / "D02_FrozenResearchChampion_20260709"
D03 = D_ROOT / "D03_R5_L1_F0_ResearchCandidate_20260711"
D04 = D_ROOT / "D04_D001_P2_InverseVol_ResearchCandidate_20260713"
CANDIDATE = ROOT / "candidate"
DATA_PATH = D_ROOT / "data_snapshots" / "c0_train_plus_baostock_future_20260703.csv"
R5_SPLIT = AUTOMATION / "R5_SPLIT_MANIFEST.json"
R5_FINALIST = AUTOMATION / "R5_FINALIST_LOCK.json"
R2_RUN = AUTOMATION / "D_DEFENSIVE_POSITION_R2_20260714_092957"
R1_RUN = AUTOMATION / "D_POSITION_CONTROL_R1_20260713_093536"
REGISTRY = D_ROOT / "D_experiment_registry.json"
STATE = RUN_DIR / "RUN_STATE.json"
PROTOCOL = RUN_DIR / "POSITION_R3_PROTOCOL.json"
INPUT_MANIFEST = RUN_DIR / "INPUT_MANIFEST.json"
SEALED_LOG = RUN_DIR / "SEALED_ACCESS_LOG.json"
GLOBAL_SEALED_LEDGER = AUTOMATION / "D_RECENT_HOLDOUT_GLOBAL_ACCESS.json"
OFFICIAL_BASELINE = 0.012528499297343995
SEEDS = (20260720, 20260721, 20260722)
FOLDS = (
    "R5_DEV_FOLD_01_2025_Q1",
    "R5_DEV_FOLD_02_2025_Q2",
    "R5_DEV_FOLD_03_2025_Q3",
    "R5_DEV_FOLD_04_2026_Q1",
)

sys.path.insert(0, str(ROOT))
from Evaluation.scripts.d_series.common import read_json, write_json  # noqa: E402
from Evaluation.scripts.d_series.defensive_position_r3 import (  # noqa: E402
    CausalRiskData,
    assign_fold_ids,
    attach_candidate_weights,
    attach_top2_control,
    candidate_grid,
    canonical_hash,
    decision_config_hash,
    directory_fingerprint,
    evaluate_explicit_weeks,
    file_hash,
    moving_block_probability,
    official_proxy_summary,
    paired_weekly,
    weekly_metrics,
    weighted_turnover,
)
from Evaluation.scripts.d_series.r5_protocol import assert_sealed_holdout_access  # noqa: E402


def utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def initial_state() -> dict[str, Any]:
    return {
        "experiment_id": RUN_ID,
        "created_at_utc": utc_now(),
        "stage": "new",
        "completed_stages": [],
        "sealed_holdout_status": "sealed",
        "sealed_holdout_unseal_count": 0,
        "sealed_return_data_access_count": 0,
        "finalist": None,
        "legacy_gate_passed": False,
        "d05_created": False,
        "events": [],
    }


def load_state() -> dict[str, Any]:
    return read_json(STATE, default=None) or initial_state()


def save_state(state: Mapping[str, Any]) -> None:
    write_json(STATE, dict(state))


def event(state: dict[str, Any], name: str, **details: Any) -> None:
    state.setdefault("events", []).append({"at_utc": utc_now(), "event": name, **details})
    save_state(state)


def global_sealed_ledger() -> dict[str, Any]:
    existing = read_json(GLOBAL_SEALED_LEDGER, default=None)
    if existing is not None:
        if existing.get("sealed_window_sha256") != _split_payload()["recent_sealed_holdout"]["sealed_window_sha256"]:
            raise RuntimeError("global sealed ledger window identity mismatch")
        return existing
    r5_lock = read_json(R5_FINALIST)
    r2_log = read_json(R2_RUN / "sealed_access_log.json")
    historical_counts = {
        "R5_FINALIST_LOCK": int(r5_lock["holdout_unseal_count"]),
        "D_DEFENSIVE_POSITION_R2": int(r2_log["unseal_count"]),
    }
    if any(historical_counts.values()):
        raise RuntimeError("a historical run already unsealed the recent holdout")
    payload = {
        "schema_version": "d_recent_holdout_global_access_v1",
        "sealed_window_sha256": _split_payload()["recent_sealed_holdout"]["sealed_window_sha256"],
        "status": "sealed",
        "unseal_count": 0,
        "return_data_access_count": 0,
        "historical_counts_at_creation": historical_counts,
        "events": [{"at_utc": utc_now(), "event": "global_ledger_initialized_without_return_access"}],
    }
    write_json(GLOBAL_SEALED_LEDGER, payload)
    return payload


def source_manifest() -> dict[str, Any]:
    source = read_json(R2_RUN / "ranking_source_manifest.json")
    if not source or not source.get("reuse_approved"):
        raise RuntimeError("R2 ranking source manifest is not reusable")
    sources = []
    for item in source["sources"]:
        path = Path(item["path"]) / "best_val_predictions.csv"
        sources.append(
            {
                "fold_id": item["fold_id"],
                "seed": int(item["seed"]),
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": item["hashes"]["best_val_predictions_sha256"],
            }
        )
    sources = sorted(sources, key=lambda item: (item["fold_id"], item["seed"]))
    expected = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    actual = {(item["fold_id"], item["seed"]) for item in sources}
    if actual != expected:
        raise RuntimeError("development source matrix is incomplete")
    for item in sources:
        path = ROOT / item["path"]
        if file_hash(path) != item["sha256"]:
            raise RuntimeError(f"prediction hash mismatch: {path}")
    return {"source_count": len(sources), "sources": sources}


def protocol_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    configs = candidate_grid()
    return {
        "schema_version": "d_defensive_position_r3_protocol_v1",
        "experiment_id": RUN_ID,
        "created_at_utc": utc_now(),
        "locked_before_any_r3_return_computation": True,
        "objective": "Find an exact-five-stock D02 defensive portfolio with materially lower tail risk and near-D02 return.",
        "time_boundary": {
            "execution_date": "2026-07-16 Asia/Shanghai",
            "wall_clock_budget_hours": 24,
            "user_controls_pause_or_stop": True,
        },
        "immutable": {
            "ranking_program": "D02_FrozenResearchChampion_20260709",
            "ranking_training_and_scores_unchanged": True,
            "data_path": str(DATA_PATH.relative_to(ROOT)).replace("\\", "/"),
            "data_md5": file_hash(DATA_PATH, "md5"),
            "data_sha256": file_hash(DATA_PATH),
            "label": "open(T+5)/open(T+1)-1",
            "candidate_modified": False,
            "production_refit_forbidden": True,
        },
        "research_design": {
            "stage_1_inner_oof": {
                "path": str((R1_RUN / "oof_scores.csv").relative_to(ROOT)).replace("\\", "/"),
                "date_range": "2023-05-30..2024-12-25",
                "week_count": 80,
                "use": f"search {len(configs)} preregistered low-capacity configurations; retain one representative per new mechanism family",
            },
            "stage_2_development": {
                "folds": list(FOLDS),
                "seeds": list(SEEDS),
                "week_count": 47,
                "use": "one-time evaluation of OOF-locked finalists",
            },
            "legacy": "2025-01-03..2026-01-05, only after development finalist lock",
            "recent_sealed": "2026-04-07..2026-07-03, one-time only after Legacy gate",
        },
        "position_contract": {
            "exactly_five_stocks": True,
            "long_only": True,
            "no_leverage": True,
            "invested_fraction_bounds": [0.10, 0.30],
            "invested_hhi_max": 0.24,
            "total_stock_exposure": 1.00,
            "cash_allowed": False,
        },
        "costs": {
            "buy_commission": 0.0005,
            "sell_commission": 0.0005,
            "sell_stamp_tax": 0.0005,
            "sell_notional_includes_portfolio_return": True,
        },
        "configuration_count": len(configs),
        "configuration_hash": decision_config_hash(configs),
        "configurations": configs,
        "oof_family_selection": {
            "maximum_finalists_per_family": 1,
            "eligibility": [
                "official_proxy_retention >= 0.75",
                "candidate cumulative_without_best_2_weeks > -0.05",
                "paired diff_without_best_2_weeks >= -0.10",
                "max_drawdown <= OOF control",
                "CVaR95 <= OOF control",
                "worst_week >= OOF control",
            ],
            "family_winner": "highest defensive utility; ties by higher official proxy then lower CVaR",
        },
        "development_gate": {
            "official_proxy_retention_vs_dp0": ">=0.85",
            "official_proxy_above_official_baseline": True,
            "mean_weekly_return_retention_vs_dp0": ">=0.85 when DP0 mean is positive",
            "cumulative_return_retention_vs_dp0": ">=0.70 when DP0 cumulative is positive",
            "max_drawdown_ratio": "<=0.75",
            "cvar95_ratio": "<=0.85",
            "worst_week_abs_loss_ratio": "<=0.80",
            "positive_week_rate": ">= DP0 positive-week-rate - 1/47",
            "candidate_cumulative_without_best_2_weeks": ">0",
            "paired_diff_without_best_2_weeks": ">=0",
            "best_2_weeks_contribution_ratio": "<=0.50",
            "fold_return_noninferiority": "candidate mean >= DP0 mean - 0.005 in at least 3/4 folds; no fold mean below -0.010",
            "f4_mean_noninferiority_delta": ">=-0.005",
            "seed_rule": "at least 2/3 seeds pass risk gates and mean >= DP0 mean - 0.005; no seed mean below -0.010",
            "bootstrap_probability_mean_return_retention_80pct": ">=0.60",
        },
        "promotion": {
            "identity": "Defensive Champion only; D02 remains Alpha Champion",
            "create_d05_only_after_development_legacy_and_sealed_all_pass": True,
            "target": "Experimental_WorkSpace/candidates_D/D05_DefensiveTop5_ResearchCandidate_20260716",
            "candidate_submission_package": False,
        },
        "input_manifest_sha256": canonical_hash(manifest),
    }


def preflight() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    if "preflight" in state.get("completed_stages", []):
        return read_json(RUN_DIR / "preflight_audit.json")
    manifest = source_manifest()
    manifest.update(
        {
            "data_md5": file_hash(DATA_PATH, "md5"),
            "data_sha256": file_hash(DATA_PATH),
            "r5_split_sha256": file_hash(R5_SPLIT),
            "inner_oof_sha256": file_hash(R1_RUN / "oof_scores.csv"),
            "d02_predictions_sha256": file_hash(D02 / "best_val_predictions.csv"),
            "protected_before": {
                "candidate": directory_fingerprint(CANDIDATE),
                "D02": directory_fingerprint(D02),
                "D03": directory_fingerprint(D03),
                "D04": directory_fingerprint(D04),
            },
        }
    )
    write_json(INPUT_MANIFEST, manifest)
    protocol = protocol_payload(manifest)
    write_json(PROTOCOL, protocol)
    (RUN_DIR / "POSITION_R3_PROTOCOL.sha256").write_text(file_hash(PROTOCOL) + "\n", encoding="ascii")
    sealed = read_json(R5_FINALIST)
    global_ledger = global_sealed_ledger()
    sealed_audit = {
        "status": sealed["holdout_status"],
        "unseal_count": int(sealed["holdout_unseal_count"]),
        "return_data_access_count": 0,
        "global_ledger_path": str(GLOBAL_SEALED_LEDGER.relative_to(ROOT)).replace("\\", "/"),
        "global_ledger_sha256": file_hash(GLOBAL_SEALED_LEDGER),
        "events": [{"at_utc": utc_now(), "event": "protocol_locked", "return_data_accessed": False}],
    }
    if (
        sealed_audit["status"] != "sealed"
        or sealed_audit["unseal_count"] != 0
        or global_ledger["status"] != "sealed"
        or int(global_ledger["unseal_count"]) != 0
    ):
        raise RuntimeError("R5 recent holdout is not sealed")
    write_json(SEALED_LOG, sealed_audit)
    command = [sys.executable, "-m", "unittest", "Evaluation.scripts.d_series.test_defensive_position_r3", "-v"]
    test = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (RUN_DIR / "unit_test_stdout.log").write_text(test.stdout, encoding="utf-8")
    (RUN_DIR / "unit_test_stderr.log").write_text(test.stderr, encoding="utf-8")
    if test.returncode != 0:
        raise RuntimeError("R3 unit tests failed")
    audit = {
        "status": "pass",
        "protocol_sha256": file_hash(PROTOCOL),
        "input_manifest_sha256": file_hash(INPUT_MANIFEST),
        "source_count": manifest["source_count"],
        "unit_tests_pass": True,
        "sealed_holdout": sealed_audit,
    }
    write_json(RUN_DIR / "preflight_audit.json", audit)
    state["completed_stages"].append("preflight")
    state["stage"] = "preflighted"
    event(state, "preflight_completed", protocol_sha256=audit["protocol_sha256"])
    update_registry("preflighted", decision=None)
    return audit


def _split_payload() -> dict[str, Any]:
    return read_json(R5_SPLIT)


def _fold_weeks() -> dict[str, list[dict[str, Any]]]:
    return {
        item["fold_id"]: item["outer_evaluation"]["weeks"]
        for item in _split_payload()["rolling_origin_development"]["folds"]
    }


def _legacy_weeks() -> list[dict[str, Any]]:
    market = pd.read_csv(DATA_PATH, usecols=["日期"])
    dates = sorted(pd.to_datetime(market["日期"]).dt.normalize().unique())
    start, end = pd.Timestamp("2025-01-03"), pd.Timestamp("2026-01-05")
    dates = [pd.Timestamp(value).normalize() for value in dates if start <= value <= end]
    groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for date in dates:
        iso = date.isocalendar()
        groups.setdefault((int(iso.year), int(iso.week)), []).append(date)
    all_dates = sorted(pd.to_datetime(pd.read_csv(DATA_PATH, usecols=["日期"])["日期"]).dt.normalize().unique())
    index = {pd.Timestamp(value).normalize(): i for i, value in enumerate(all_dates)}
    weeks = []
    for values in (groups[key] for key in sorted(groups)):
        buy, sell = values[0], values[-1]
        decision = pd.Timestamp(all_dates[index[buy] - 1]).normalize()
        weeks.append(
            {
                "decision_date": decision.strftime("%Y-%m-%d"),
                "buy_date": buy.strftime("%Y-%m-%d"),
                "sell_date": sell.strftime("%Y-%m-%d"),
                "trading_day_count": len(values),
            }
        )
    return weeks


def _read_source(item: Mapping[str, Any]) -> pd.DataFrame:
    path = ROOT / str(item["path"])
    if file_hash(path) != item["sha256"]:
        raise RuntimeError(f"source changed: {path}")
    frame = pd.read_csv(path, dtype={"stock_id": str})
    frame["fold_id"] = item["fold_id"]
    frame["seed"] = int(item["seed"])
    return frame


def _evaluate_one(
    predictions: pd.DataFrame,
    config: Mapping[str, Any] | None,
    risk: CausalRiskData,
    weeks: Sequence[Mapping[str, Any]],
    *,
    fold_id: str,
    control: bool = False,
) -> dict[str, Any]:
    weighted, decisions = attach_top2_control(predictions) if control else attach_candidate_weights(predictions, config or {}, risk)
    weekly = evaluate_explicit_weeks(weighted, weeks, DATA_PATH, fold_id=fold_id)
    return {
        "weighted": weighted,
        "decisions": decisions,
        "weekly": weekly,
        "official": official_proxy_summary(weighted),
        "metrics": weekly_metrics(weekly, decisions),
    }


def _family_utility(metrics: Mapping[str, Any], control: Mapping[str, Any], official_retention: float) -> float:
    maxdd_gain = 1.0 - safe_ratio(metrics["max_drawdown_abs"], control["max_drawdown_abs"])
    cvar_gain = 1.0 - safe_ratio(metrics["CVaR_95_loss_abs"], control["CVaR_95_loss_abs"])
    worst_gain = 1.0 - safe_ratio(abs(metrics["worst_week"]), abs(control["worst_week"]))
    return 0.35 * official_retention + 0.25 * maxdd_gain + 0.25 * cvar_gain + 0.15 * worst_gain


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > 1e-12 else 0.0


def oof_search() -> dict[str, Any]:
    state = load_state()
    if "oof_search" in state.get("completed_stages", []):
        return read_json(RUN_DIR / "oof" / "OOF_FAMILY_LOCK.json")
    preflight()
    output = RUN_DIR / "oof"
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(R1_RUN / "oof_scores.csv", dtype={"stock_id": str})
    risk = CausalRiskData.from_market_path(DATA_PATH, cutoff="2024-12-25")
    manifest = read_json(R1_RUN / "inner_fold_manifest.json")
    all_weeks = []
    for fold in manifest["folds"]:
        frame = pd.read_csv(R1_RUN / "oof_folds" / fold["fold_id"] / "oof_predictions.csv", dtype={"stock_id": str})
        start, end = fold["oof_sample_start"], fold["oof_sample_end"]
        market = pd.read_csv(DATA_PATH, usecols=["日期"])
        dates = [pd.Timestamp(value).normalize() for value in sorted(pd.to_datetime(market["日期"]).unique())]
        validation = [value for value in dates if pd.Timestamp(start) <= value <= pd.Timestamp(end)]
        groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
        for date in validation:
            iso = date.isocalendar()
            groups.setdefault((int(iso.year), int(iso.week)), []).append(date)
        index = {date: i for i, date in enumerate(dates)}
        weeks = []
        for values in (groups[key] for key in sorted(groups)):
            buy, sell = values[0], values[-1]
            decision = dates[index[buy] - 1]
            if decision.strftime("%Y-%m-%d") not in set(pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")):
                continue
            weeks.append(
                {
                    "decision_date": decision.strftime("%Y-%m-%d"),
                    "buy_date": buy.strftime("%Y-%m-%d"),
                    "sell_date": sell.strftime("%Y-%m-%d"),
                    "trading_day_count": len(values),
                }
            )
        all_weeks.extend({**week, "fold_id": fold["fold_id"]} for week in weeks)
    control_weighted, control_decisions = attach_top2_control(predictions)
    control_weekly_frames = []
    for fold in manifest["folds"]:
        weeks = [item for item in all_weeks if item["fold_id"] == fold["fold_id"]]
        control_weekly_frames.append(evaluate_explicit_weeks(control_weighted, weeks, DATA_PATH, fold_id=fold["fold_id"]))
    control_weekly = pd.concat(control_weekly_frames, ignore_index=True)
    control_metrics = weekly_metrics(control_weekly, control_decisions)
    control_official = official_proxy_summary(control_weighted)
    rows = []
    full_results: dict[str, Any] = {}
    for config in candidate_grid():
        weighted, decisions = attach_candidate_weights(predictions, config, risk)
        weekly_frames = []
        for fold in manifest["folds"]:
            weeks = [item for item in all_weeks if item["fold_id"] == fold["fold_id"]]
            weekly_frames.append(evaluate_explicit_weeks(weighted, weeks, DATA_PATH, fold_id=fold["fold_id"]))
        weekly = pd.concat(weekly_frames, ignore_index=True)
        metrics = weekly_metrics(weekly, decisions)
        official = official_proxy_summary(weighted)
        paired_frame, paired = paired_weekly(weekly, control_weekly)
        official_retention = safe_ratio(official["official_proxy_final_score"], control_official["official_proxy_final_score"])
        checks = {
            "official_retention_75pct": official_retention >= 0.75,
            "candidate_without_best2_above_minus_0_05": metrics["cumulative_without_best_2_weeks"] > -0.05,
            "paired_without_best2_above_minus_0_10": paired["diff_without_best_2_weeks"] >= -0.10,
            "maxdd_not_worse": metrics["max_drawdown_abs"] <= control_metrics["max_drawdown_abs"],
            "cvar_not_worse": metrics["CVaR_95_loss_abs"] <= control_metrics["CVaR_95_loss_abs"],
            "worst_week_not_worse": metrics["worst_week"] >= control_metrics["worst_week"],
        }
        utility = _family_utility(metrics, control_metrics, official_retention)
        row = {
            "id": config["id"],
            "family": config["family"],
            "official_proxy": official["official_proxy_final_score"],
            "official_retention": official_retention,
            "mean_weekly_net": metrics["mean_weekly_return_net"],
            "cumulative_net": metrics["cumulative_return_net"],
            "max_drawdown_abs": metrics["max_drawdown_abs"],
            "cvar95": metrics["CVaR_95_loss_abs"],
            "worst_week": metrics["worst_week"],
            "positive_week_rate": metrics["positive_week_rate"],
            "candidate_without_best2": metrics["cumulative_without_best_2_weeks"],
            "paired_mean_excess": paired["mean_weekly_excess"],
            "paired_pdr": paired["positive_diff_rate"],
            "average_cash": metrics.get("average_cash_ratio", 0.0),
            "turnover": weighted_turnover(decisions),
            "utility": utility,
            "passes": all(checks.values()),
            "checks": checks,
        }
        rows.append(row)
        full_results[config["id"]] = {"config": config, "summary": row, "metrics": metrics, "official": official, "paired": paired}
    table = pd.DataFrame(rows).sort_values(["passes", "utility", "official_proxy"], ascending=[False, False, False])
    table.to_csv(output / "oof_candidate_summary.csv", index=False, encoding="utf-8")
    write_json(output / "oof_candidate_metrics.json", {"control": {"metrics": control_metrics, "official": control_official}, "candidates": full_results})
    locked = []
    eligible_table = table[(table["passes"]) & (table["family"] != "top5_control")]
    for family, group in eligible_table.groupby("family", sort=True):
        winner = group.sort_values(["utility", "official_proxy", "cvar95"], ascending=[False, False, True]).iloc[0]
        locked.append(str(winner["id"]))
    if len(locked) > 6:
        locked = (
            table[table["id"].isin(locked)]
            .sort_values(["utility", "official_proxy"], ascending=[False, False])
            .head(6)["id"]
            .astype(str)
            .tolist()
        )
    lock = {
        "status": "OOF_FAMILY_FINALISTS_LOCKED" if locked else "NO_OOF_FINALIST",
        "locked_finalists": locked,
        "candidate_count": len(rows),
        "family_count": int(table["family"].nunique()),
        "configuration_hash": decision_config_hash(candidate_grid()),
        "summary_sha256": file_hash(output / "oof_candidate_summary.csv"),
    }
    write_json(output / "OOF_FAMILY_LOCK.json", lock)
    state["completed_stages"].append("oof_search")
    state["stage"] = "oof_finalists_locked" if locked else "rejected_oof"
    event(state, "oof_search_completed", locked_finalists=locked)
    update_registry(state["stage"], decision="REJECT" if not locked else None)
    return lock


def _seed_average_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold_id", "week_start", "buy_date", "sell_date", "decision_date"]
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [column for column in numeric if column not in keys]
    return frame.groupby(keys, as_index=False)[numeric].mean()


def development() -> dict[str, Any]:
    state = load_state()
    if "development" in state.get("completed_stages", []):
        return read_json(RUN_DIR / "development" / "DEVELOPMENT_GATE_DECISION.json")
    lock = oof_search()
    if not lock["locked_finalists"]:
        return {"status": "NO_DEVELOPMENT_RUN", "finalist": None}
    output = RUN_DIR / "development"
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(INPUT_MANIFEST)
    configs = {config["id"]: config for config in candidate_grid()}
    weeks = _fold_weeks()
    risk = CausalRiskData.from_market_path(DATA_PATH, cutoff="2026-03-27")
    weekly_all = []
    official_all = []
    decision_all = []
    for source in manifest["sources"]:
        predictions = _read_source(source)
        control = _evaluate_one(predictions, None, risk, weeks[source["fold_id"]], fold_id=source["fold_id"], control=True)
        for kind, identifier, result in [("control", "R3_DP0_TOP2_EQUAL", control)]:
            frame = result["weekly"].copy()
            frame["seed"] = source["seed"]
            frame["strategy"] = identifier
            weekly_all.append(frame)
            decision = result["decisions"].copy()
            decision["fold_id"] = source["fold_id"]
            decision["seed"] = source["seed"]
            decision_all.append(decision)
            official_all.append({"strategy": identifier, "fold_id": source["fold_id"], "seed": source["seed"], **result["official"]})
        for identifier in lock["locked_finalists"]:
            result = _evaluate_one(predictions, configs[identifier], risk, weeks[source["fold_id"]], fold_id=source["fold_id"])
            frame = result["weekly"].copy()
            frame["seed"] = source["seed"]
            frame["strategy"] = identifier
            weekly_all.append(frame)
            decision = result["decisions"].copy()
            decision["fold_id"] = source["fold_id"]
            decision["seed"] = source["seed"]
            decision_all.append(decision)
            official_all.append({"strategy": identifier, "fold_id": source["fold_id"], "seed": source["seed"], **result["official"]})
    weekly_raw = pd.concat(weekly_all, ignore_index=True)
    decisions_raw = pd.concat(decision_all, ignore_index=True)
    official_raw = pd.DataFrame(official_all)
    weekly_raw.to_csv(output / "weekly_returns_all_strategies.csv", index=False, encoding="utf-8")
    decisions_raw.to_csv(output / "portfolio_decisions_all_strategies.csv", index=False, encoding="utf-8")
    official_raw.to_csv(output / "official_proxy_all_strategies.csv", index=False, encoding="utf-8")
    strategies = ["R3_DP0_TOP2_EQUAL", *lock["locked_finalists"]]
    metrics: dict[str, Any] = {}
    ensemble: dict[str, pd.DataFrame] = {}
    for strategy in strategies:
        raw = weekly_raw[weekly_raw["strategy"] == strategy].copy()
        mean_weekly = _seed_average_weekly(raw)
        ensemble[strategy] = mean_weekly
        decision = decisions_raw[decisions_raw["strategy"] == strategy]
        summary = weekly_metrics(mean_weekly, decision)
        official_strategy = official_raw[official_raw["strategy"] == strategy]
        official_by_seed = pd.Series(
            {
                int(seed): float(np.average(group["official_proxy_final_score"], weights=group["sample_count"]))
                for seed, group in official_strategy.groupby("seed", sort=True)
            },
            dtype=float,
        )
        summary["official_proxy_median_across_seeds"] = float(official_by_seed.median())
        summary["official_proxy_by_seed"] = {str(int(index)): float(value) for index, value in official_by_seed.items()}
        summary["fold_metrics"] = {}
        for fold, group in mean_weekly.groupby("fold_id", sort=True):
            summary["fold_metrics"][str(fold)] = weekly_metrics(group)
        summary["seed_metrics"] = {}
        for seed, group in raw.groupby("seed", sort=True):
            summary["seed_metrics"][str(int(seed))] = weekly_metrics(group)
        metrics[strategy] = summary
        mean_weekly.to_csv(output / f"{strategy}_ensemble_weekly.csv", index=False, encoding="utf-8")
    control = metrics["R3_DP0_TOP2_EQUAL"]
    gates: dict[str, Any] = {}
    for strategy in lock["locked_finalists"]:
        summary = metrics[strategy]
        paired_frame, paired = paired_weekly(ensemble[strategy], ensemble["R3_DP0_TOP2_EQUAL"])
        fold_deltas = {}
        fold_passes = {}
        for fold in FOLDS:
            control_fold = control["fold_metrics"][fold]
            candidate_fold = summary["fold_metrics"][fold]
            candidate_fold_mean = candidate_fold["mean_weekly_return_net"]
            control_fold_mean = control_fold["mean_weekly_return_net"]
            fold_deltas[fold] = candidate_fold_mean - control_fold_mean
            fold_passes[fold] = candidate_fold_mean >= control_fold_mean - 0.005
        f4_candidate = summary["fold_metrics"][FOLDS[-1]]["mean_weekly_return_net"]
        f4_control = control["fold_metrics"][FOLDS[-1]]["mean_weekly_return_net"]
        seed_passes = []
        for seed in SEEDS:
            candidate_seed = summary["seed_metrics"][str(seed)]
            control_seed = control["seed_metrics"][str(seed)]
            seed_passes.append(
                candidate_seed["max_drawdown_abs"] <= 0.85 * control_seed["max_drawdown_abs"]
                and candidate_seed["CVaR_95_loss_abs"] <= 0.90 * control_seed["CVaR_95_loss_abs"]
                and candidate_seed["mean_weekly_return_net"] >= control_seed["mean_weekly_return_net"] - 0.005
                and candidate_seed["mean_weekly_return_net"] >= -0.010
            )
        bootstrap_values = paired_frame["weekly_return_net"].to_numpy(float) - 0.80 * paired_frame["control_weekly_return_net"].to_numpy(float)
        official_retention = safe_ratio(summary["official_proxy_median_across_seeds"], control["official_proxy_median_across_seeds"])
        mean_retention = safe_ratio(summary["mean_weekly_return_net"], control["mean_weekly_return_net"])
        cumulative_retention = safe_ratio(summary["cumulative_return_net"], control["cumulative_return_net"])
        checks = {
            "official_proxy_retention_85pct": official_retention >= 0.85,
            "official_baseline_exceeded": summary["official_proxy_median_across_seeds"] > OFFICIAL_BASELINE,
            "mean_weekly_return_retention_85pct": mean_retention >= 0.85,
            "cumulative_return_retention_70pct": cumulative_retention >= 0.70,
            "maxdd_75pct": summary["max_drawdown_abs"] <= 0.75 * control["max_drawdown_abs"],
            "cvar_85pct": summary["CVaR_95_loss_abs"] <= 0.85 * control["CVaR_95_loss_abs"],
            "worst_week_80pct": abs(summary["worst_week"]) <= 0.80 * abs(control["worst_week"]),
            "positive_week_rate_noninferiority_one_week": summary["positive_week_rate"] >= control["positive_week_rate"] - 1.0 / 47.0,
            "candidate_without_best2_positive": summary["cumulative_without_best_2_weeks"] > 0.0,
            "paired_diff_without_best2_nonnegative": paired["diff_without_best_2_weeks"] >= 0.0,
            "best2_contribution_50pct": summary["best_2_weeks_contribution_ratio"] <= 0.50,
            "fold_noninferiority_3_of_4": sum(fold_passes.values()) >= 3,
            "no_fold_mean_below_minus_0_010": min(
                summary["fold_metrics"][fold]["mean_weekly_return_net"] for fold in FOLDS
            ) >= -0.010,
            "f4_noninferiority_minus_0_005": f4_candidate >= f4_control - 0.005,
            "seed_rule_2_of_3": sum(seed_passes) >= 2,
            "bootstrap_probability_60pct": moving_block_probability(bootstrap_values) >= 0.60,
            "exactly_five_and_constraints": bool(
                (decisions_raw[decisions_raw["strategy"] == strategy]["selected_count"] == 5).all()
                and np.isclose(
                    decisions_raw[decisions_raw["strategy"] == strategy]["weight_sum"].to_numpy(dtype=float),
                    1.0,
                    atol=1e-12,
                ).all()
                and (decisions_raw[decisions_raw["strategy"] == strategy]["max_invested_weight"] <= 0.30 + 1e-12).all()
                and (decisions_raw[decisions_raw["strategy"] == strategy]["invested_hhi"] <= 0.24 + 1e-12).all()
            ),
        }
        gates[strategy] = {
            "passes": all(checks.values()),
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "official_retention": official_retention,
            "mean_return_retention": mean_retention,
            "cumulative_return_retention": cumulative_retention,
            "fold_mean_deltas": fold_deltas,
            "fold_passes": fold_passes,
            "f4_mean_delta": f4_candidate - f4_control,
            "seed_passes": {str(seed): passed for seed, passed in zip(SEEDS, seed_passes)},
            "paired": paired,
            "bootstrap_probability": moving_block_probability(bootstrap_values),
        }
    write_json(output / "development_metrics.json", metrics)
    write_json(output / "development_gate_details.json", gates)
    eligible = [strategy for strategy in lock["locked_finalists"] if gates[strategy]["passes"]]
    finalist = None
    if eligible:
        finalist = sorted(
            eligible,
            key=lambda strategy: (
                -metrics[strategy]["official_proxy_median_across_seeds"],
                metrics[strategy]["CVaR_95_loss_abs"],
                metrics[strategy]["max_drawdown_abs"],
            ),
        )[0]
        write_json(
            RUN_DIR / "D05_FINALIST_LOCK.json",
            {
                "locked_finalists": [finalist],
                "locked_config": configs[finalist],
                "development_gate_passed": True,
                "development_metrics_sha256": file_hash(output / "development_metrics.json"),
                "development_gate_sha256": file_hash(output / "development_gate_details.json"),
                "sealed_access_authorized": False,
            },
        )
    decision = {
        "status": "DEVELOPMENT_FINALIST_LOCKED" if finalist else "NO_DEVELOPMENT_FINALIST",
        "finalist": finalist,
        "passing_strategies": eligible,
        "evaluated_count": len(lock["locked_finalists"]),
        "legacy_authorized": finalist is not None,
        "sealed_authorized": False,
    }
    write_json(output / "DEVELOPMENT_GATE_DECISION.json", decision)
    state["completed_stages"].append("development")
    state["stage"] = "development_finalist_locked" if finalist else "rejected_development"
    state["finalist"] = finalist
    event(state, "development_completed", finalist=finalist)
    update_registry(state["stage"], decision="REJECT" if not finalist else None, metrics={"development": metrics, "gates": gates})
    return decision


def legacy() -> dict[str, Any]:
    state = load_state()
    if "legacy" in state.get("completed_stages", []):
        return read_json(RUN_DIR / "legacy" / "LEGACY_GATE_DECISION.json")
    decision = development()
    if not decision.get("finalist"):
        return {"status": "NOT_RUN", "reason": "no development finalist"}
    output = RUN_DIR / "legacy"
    output.mkdir(parents=True, exist_ok=True)
    config = next(item for item in candidate_grid() if item["id"] == decision["finalist"])
    predictions = pd.read_csv(D02 / "best_val_predictions.csv", dtype={"stock_id": str})
    risk = CausalRiskData.from_market_path(DATA_PATH, cutoff="2026-01-05")
    weeks = _legacy_weeks()
    control = _evaluate_one(predictions, None, risk, weeks, fold_id="LEGACY", control=True)
    candidate = _evaluate_one(predictions, config, risk, weeks, fold_id="LEGACY")
    control["weekly"].to_csv(output / "control_weekly.csv", index=False, encoding="utf-8")
    candidate["weekly"].to_csv(output / "candidate_weekly.csv", index=False, encoding="utf-8")
    candidate["decisions"].to_csv(output / "candidate_decisions.csv", index=False, encoding="utf-8")
    control_metrics = weekly_metrics(control["weekly"], control["decisions"])
    candidate_metrics = weekly_metrics(candidate["weekly"], candidate["decisions"])
    _, paired = paired_weekly(candidate["weekly"], control["weekly"])
    expected = read_json(D02 / "official_proxy_result.json")["official_proxy_final_score"]
    if abs(control["official"]["official_proxy_final_score"] - expected) > 1e-12:
        raise RuntimeError("Legacy D02 control did not exactly reproduce")
    official_retention = safe_ratio(candidate["official"]["official_proxy_final_score"], control["official"]["official_proxy_final_score"])
    checks = {
        "d02_exact_reproduction": True,
        "candidate_exceeds_official_baseline": candidate["official"]["official_proxy_final_score"] > OFFICIAL_BASELINE,
        "official_retention_85pct": official_retention >= 0.85,
        "mean_return_retention_75pct": candidate_metrics["mean_weekly_return_net"] >= 0.75 * control_metrics["mean_weekly_return_net"],
        "maxdd_80pct": candidate_metrics["max_drawdown_abs"] <= 0.80 * control_metrics["max_drawdown_abs"],
        "cvar_85pct": candidate_metrics["CVaR_95_loss_abs"] <= 0.85 * control_metrics["CVaR_95_loss_abs"],
        "worst_week_85pct": abs(candidate_metrics["worst_week"]) <= 0.85 * abs(control_metrics["worst_week"]),
        "positive_week_rate_50pct": candidate_metrics["positive_week_rate"] >= 0.50,
        "candidate_without_best2_positive": candidate_metrics["cumulative_without_best_2_weeks"] > 0.0,
        "best2_contribution_50pct": candidate_metrics["best_2_weeks_contribution_ratio"] <= 0.50,
        "paired_diff_without_best2_nonnegative": paired["diff_without_best_2_weeks"] >= 0.0,
    }
    gate = {
        "status": "LEGACY_PASS" if all(checks.values()) else "LEGACY_FAIL",
        "passes": all(checks.values()),
        "finalist": decision["finalist"],
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "official_retention": official_retention,
        "control": {"official": control["official"], "metrics": control_metrics},
        "candidate": {"official": candidate["official"], "metrics": candidate_metrics},
        "paired": paired,
        "sealed_holdout_authorized": all(checks.values()),
    }
    write_json(output / "LEGACY_GATE_DECISION.json", gate)
    state["completed_stages"].append("legacy")
    state["legacy_gate_passed"] = gate["passes"]
    state["stage"] = "waiting_for_explicit_holdout_unseal" if gate["passes"] else "rejected_legacy"
    event(state, "legacy_completed", passes=gate["passes"])
    update_registry(state["stage"], decision=None if gate["passes"] else "REJECT", metrics={"legacy": gate})
    return gate


def holdout() -> dict[str, Any]:
    state = load_state()
    if "holdout" in state.get("completed_stages", []):
        return read_json(RUN_DIR / "holdout" / "HOLDOUT_GATE_DECISION.json")
    gate = legacy()
    if not gate.get("passes"):
        return {"status": "NOT_RUN", "reason": "Legacy gate did not pass"}
    ledger = global_sealed_ledger()
    if (
        state["sealed_holdout_status"] != "sealed"
        or int(state["sealed_holdout_unseal_count"]) != 0
        or ledger["status"] != "sealed"
        or int(ledger["unseal_count"]) != 0
    ):
        raise RuntimeError("sealed holdout state is not pristine")
    state["sealed_holdout_status"] = "unsealing"
    state["sealed_holdout_unseal_count"] = 1
    event(state, "sealed_holdout_unsealing_authorized", finalist=state["finalist"])
    ledger["status"] = "unsealing"
    ledger["unseal_count"] = 1
    ledger["events"].append({"at_utc": utc_now(), "event": "R3_finalist_unsealing_authorized", "finalist": state["finalist"]})
    write_json(GLOBAL_SEALED_LEDGER, ledger)
    assert_sealed_holdout_access(split_manifest=_split_payload(), finalist_lock_path=RUN_DIR / "D05_FINALIST_LOCK.json", run_state=state)
    output = RUN_DIR / "holdout"
    output.mkdir(parents=True, exist_ok=True)
    holdout_spec = _split_payload()["recent_sealed_holdout"]
    last_decision_date = holdout_spec["weeks"][-1]["decision_date"]
    prediction_path = output / "d02_frozen_checkpoint_predictions.csv"
    timing_path = output / "inference_timing.json"
    command = [
        sys.executable,
        "-B",
        str(ROOT / "Evaluation" / "scripts" / "d_series" / "r4_checkpoint_inference.py"),
        "--code-src", str(D02 / "code_snapshot" / "src"),
        "--config", str(D02 / "generated_config.json"),
        "--checkpoint", str(D02 / "best_model.pth"),
        "--scalers", str(D02 / "scalers.json"),
        "--data", str(DATA_PATH),
        "--sample-start", holdout_spec["first_decision_date"],
        "--sample-end", last_decision_date,
        "--output", str(prediction_path),
        "--timing-output", str(timing_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPYCACHEPREFIX"] = str(output / "pycache")
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
    (output / "inference_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "inference_stderr.log").write_text(completed.stderr, encoding="utf-8")
    state["sealed_return_data_access_count"] = 1
    save_state(state)
    ledger["status"] = "unsealed"
    ledger["return_data_access_count"] = 1
    ledger["events"].append({"at_utc": utc_now(), "event": "R3_return_data_accessed_once", "return_data_accessed": True})
    write_json(GLOBAL_SEALED_LEDGER, ledger)
    if completed.returncode != 0:
        raise RuntimeError("sealed holdout D02 checkpoint inference failed")
    sealed_log = read_json(SEALED_LOG)
    sealed_log["status"] = "unsealed"
    sealed_log["unseal_count"] = 1
    sealed_log["return_data_access_count"] = 1
    sealed_log["events"].append({"at_utc": utc_now(), "event": "return_data_accessed_once", "return_data_accessed": True})
    write_json(SEALED_LOG, sealed_log)
    predictions = pd.read_csv(prediction_path, dtype={"stock_id": str})
    config = next(item for item in candidate_grid() if item["id"] == state["finalist"])
    risk = CausalRiskData.from_market_path(DATA_PATH, cutoff=last_decision_date)
    weeks = holdout_spec["weeks"]
    control = _evaluate_one(predictions, None, risk, weeks, fold_id="RECENT_SEALED", control=True)
    candidate = _evaluate_one(predictions, config, risk, weeks, fold_id="RECENT_SEALED")
    control["weekly"].to_csv(output / "control_weekly.csv", index=False, encoding="utf-8")
    candidate["weekly"].to_csv(output / "candidate_weekly.csv", index=False, encoding="utf-8")
    candidate["decisions"].to_csv(output / "candidate_decisions.csv", index=False, encoding="utf-8")
    control_metrics = weekly_metrics(control["weekly"], control["decisions"])
    candidate_metrics = weekly_metrics(candidate["weekly"], candidate["decisions"])
    _, paired = paired_weekly(candidate["weekly"], control["weekly"])
    checks = {
        "mean_weekly_noninferiority_minus_0_005": candidate_metrics["mean_weekly_return_net"] >= control_metrics["mean_weekly_return_net"] - 0.005,
        "positive_control_mean_retention_70pct": control_metrics["mean_weekly_return_net"] <= 0 or candidate_metrics["mean_weekly_return_net"] >= 0.70 * control_metrics["mean_weekly_return_net"],
        "maxdd_not_worse": candidate_metrics["max_drawdown_abs"] <= control_metrics["max_drawdown_abs"],
        "cvar_not_worse": candidate_metrics["CVaR_95_loss_abs"] <= control_metrics["CVaR_95_loss_abs"],
        "worst_week_delta_minus_0_005": candidate_metrics["worst_week"] >= control_metrics["worst_week"] - 0.005,
        "positive_week_rate_50pct": candidate_metrics["positive_week_rate"] >= 0.50,
        "candidate_without_best2_positive": candidate_metrics["cumulative_without_best_2_weeks"] > 0.0,
        "exactly_five": bool((candidate["decisions"]["selected_count"] == 5).all()),
        "single_unseal": int(sealed_log["unseal_count"]) == 1 and int(sealed_log["return_data_access_count"]) == 1,
    }
    result = {
        "status": "ACCEPT_AS_DEFENSIVE_CHAMPION" if all(checks.values()) else "HOLDOUT_FAIL",
        "passes": all(checks.values()),
        "finalist": state["finalist"],
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "control": control_metrics,
        "candidate": candidate_metrics,
        "paired": paired,
        "inference_timing": read_json(timing_path),
    }
    write_json(output / "HOLDOUT_GATE_DECISION.json", result)
    state["completed_stages"].append("holdout")
    state["sealed_holdout_status"] = "unsealed"
    state["stage"] = "accepted_not_promoted" if result["passes"] else "rejected_holdout"
    event(state, "holdout_completed", passes=result["passes"])
    update_registry(state["stage"], decision="ACCEPT_AS_DEFENSIVE_CHAMPION" if result["passes"] else "REJECT", metrics={"holdout": result})
    return result


def promote() -> dict[str, Any]:
    result = holdout()
    if not result.get("passes"):
        return {"status": "NOT_PROMOTED", "reason": "holdout gate failed"}
    state = load_state()
    target = D_ROOT / "D05_DefensiveTop5_ResearchCandidate_20260716"
    if target.exists():
        return {"status": "ALREADY_PROMOTED", "path": str(target)}
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(D02 / "code_snapshot", staging / "code_snapshot")
    for name in ("best_model.pth", "scalers.json", "generated_config.json", "config.json"):
        shutil.copy2(D02 / name, staging / name)
    shutil.copy2(PROTOCOL, staging / "POSITION_R3_PROTOCOL.json")
    shutil.copy2(RUN_DIR / "D05_FINALIST_LOCK.json", staging / "D05_FINALIST_LOCK.json")
    shutil.copy2(RUN_DIR / "development" / "DEVELOPMENT_GATE_DECISION.json", staging / "DEVELOPMENT_GATE_DECISION.json")
    shutil.copy2(RUN_DIR / "legacy" / "LEGACY_GATE_DECISION.json", staging / "LEGACY_GATE_DECISION.json")
    shutil.copy2(RUN_DIR / "holdout" / "HOLDOUT_GATE_DECISION.json", staging / "HOLDOUT_GATE_DECISION.json")
    card = (
        "# D05 Defensive Top-5 Research Candidate\n\n"
        "D05 freezes the unchanged D02 ranking program plus the accepted Position R3 exact-five-stock defensive policy. "
        "It is the Defensive Champion only; D02 remains Alpha Champion. This directory is not a submission package and contains no Production Refit.\n"
    )
    (staging / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    manifest = {
        "identity": "D05_DefensiveTop5_ResearchCandidate_20260716",
        "status": "ACCEPT_AS_DEFENSIVE_CHAMPION",
        "alpha_champion": "D02_FrozenResearchChampion_20260709",
        "source_d02_sha256": directory_fingerprint(D02),
        "accepted_policy": state["finalist"],
        "run_path": str(RUN_DIR.relative_to(ROOT)).replace("\\", "/"),
        "production_refit": False,
        "candidate_submission_package": False,
    }
    write_json(staging / "D05_SNAPSHOT_MANIFEST.json", manifest)
    staging.rename(target)
    state["d05_created"] = True
    state["stage"] = "promoted_d05_defensive_champion"
    event(state, "d05_promoted", path=str(target.relative_to(ROOT)))
    update_registry("frozen_defensive_champion_snapshot", decision="ACCEPT_AS_DEFENSIVE_CHAMPION", checkpoint=str(target / "best_model.pth"))
    return {"status": "PROMOTED_D05", "path": str(target)}


def update_registry(
    status: str,
    *,
    decision: str | None,
    metrics: Mapping[str, Any] | None = None,
    checkpoint: str | None = None,
) -> None:
    registry = read_json(REGISTRY, default={}) or {}
    entries = registry.setdefault("experiments", [])
    entry = {
        "id": RUN_ID,
        "name": "D Defensive Position R3 exact-five-stock study",
        "path": str(RUN_DIR.relative_to(ROOT)).replace("\\", "/"),
        "status": status,
        "decision": decision,
        "source_program": "D02_FrozenResearchChampion_20260709",
        "architecture": "Unchanged D02 ranker; inference-only exact-five portfolio policy",
        "feature_version": "D02 scores plus causal close-return risk state",
        "label_version": "open_t_plus_5_over_open_t_plus_1",
        "training_dates": "No ranker retraining; Position R3 uses 2023-05-30..2024-12-25 OOF search",
        "validation_dates": "R5 four development folds; conditional Legacy and one-time Recent sealed holdout",
        "seed": list(SEEDS),
        "hyperparameters": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/") if PROTOCOL.exists() else None,
        "checkpoint_path": checkpoint,
        "metrics": dict(metrics or {}),
        "updated_at_utc": utc_now(),
    }
    replaced = False
    for index, existing in enumerate(entries):
        if existing.get("id") == RUN_ID:
            entries[index] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)
    write_json(REGISTRY, registry)


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
        files.append({"path": str(path.relative_to(RUN_DIR)).replace("\\", "/"), "size": path.stat().st_size, "sha256": file_hash(path)})
    payload = {"file_count": len(files), "files": files}
    write_json(RUN_DIR / "ARTIFACT_MANIFEST.json", payload)
    return payload


def final_audit() -> dict[str, Any]:
    state = load_state()
    initial = read_json(INPUT_MANIFEST)["protected_before"]
    final = {
        "candidate": directory_fingerprint(CANDIDATE),
        "D02": directory_fingerprint(D02),
        "D03": directory_fingerprint(D03),
        "D04": directory_fingerprint(D04),
    }
    sealed = read_json(SEALED_LOG)
    global_ledger = global_sealed_ledger()
    checks = {
        "protocol_locked": PROTOCOL.exists() and file_hash(PROTOCOL) == (RUN_DIR / "POSITION_R3_PROTOCOL.sha256").read_text(encoding="ascii").strip(),
        "candidate_unchanged": final["candidate"] == initial["candidate"],
        "D02_unchanged": final["D02"] == initial["D02"],
        "D03_unchanged": final["D03"] == initial["D03"],
        "D04_unchanged": final["D04"] == initial["D04"],
        "sealed_access_valid": int(sealed["unseal_count"]) in {0, 1} and int(sealed["return_data_access_count"]) == int(sealed["unseal_count"]),
        "global_sealed_access_matches_local": (
            int(global_ledger["unseal_count"]) == int(sealed["unseal_count"])
            and int(global_ledger["return_data_access_count"]) == int(sealed["return_data_access_count"])
        ),
        "d05_only_if_accepted": not state.get("d05_created") or state.get("stage") == "promoted_d05_defensive_champion",
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "protected_initial": initial,
        "protected_final": final,
        "sealed": sealed,
        "global_sealed_ledger": global_ledger,
        "state": state,
    }
    write_json(RUN_DIR / "FINAL_SAFETY_AUDIT.json", payload)
    artifact_manifest()
    return payload


def build_report() -> Path:
    state = load_state()
    oof = read_json(RUN_DIR / "oof" / "OOF_FAMILY_LOCK.json", default={}) or {}
    dev = read_json(RUN_DIR / "development" / "DEVELOPMENT_GATE_DECISION.json", default={}) or {}
    dev_metrics = read_json(RUN_DIR / "development" / "development_metrics.json", default={}) or {}
    dev_gates = read_json(RUN_DIR / "development" / "development_gate_details.json", default={}) or {}
    legacy_gate = read_json(RUN_DIR / "legacy" / "LEGACY_GATE_DECISION.json", default={}) or {}
    holdout_gate = read_json(RUN_DIR / "holdout" / "HOLDOUT_GATE_DECISION.json", default={}) or {}
    sealed = read_json(SEALED_LOG, default={}) or {}
    lines = [
        "# D 系列五股防御仓位 Position R3 最终报告",
        "",
        f"- Experiment ID: `{RUN_ID}`",
        f"- 状态：`{state.get('stage')}`",
        "- Alpha Champion：`D02_FrozenResearchChampion_20260709`",
        f"- Defensive Champion：`{'D05' if state.get('d05_created') else 'NONE'}`",
        f"- Recent Holdout unseal_count：`{sealed.get('unseal_count', 0)}`",
        "",
        "## 1. 方法",
        "",
        f"冻结 D02 排名程序，先在 80 周 purged OOF 上搜索 {len(candidate_grid())} 个预注册低自由度五股仓位配置，每个新机制族最多锁定一个；再对锁定方案执行四折三 seed 的 47 周滚动开发评估。",
        "策略族包括下行捕获卫星股、方向可靠性筛选、最差区段残差筛选，以及经验 CVaR 配权。每周严格满仓持有五只股票，单股 10%-30%、HHI<=0.24，与满仓 Top-2 等权作同敞口比较。",
        "",
        "## 2. 阶段结果",
        "",
        f"- OOF：`{oof.get('status', 'NOT_RUN')}`；锁定 `{len(oof.get('locked_finalists', []))}` 个方案：`{oof.get('locked_finalists', [])}`。",
        f"- Development：`{dev.get('status', 'NOT_RUN')}`；finalist=`{dev.get('finalist')}`。",
        f"- Legacy：`{legacy_gate.get('status', 'NOT_RUN')}`。",
        f"- Recent sealed holdout：`{holdout_gate.get('status', 'NOT_RUN')}`。",
        f"- D05：`{'已创建' if state.get('d05_created') else '未创建'}`。",
        "",
        "## 3. Development 指标",
        "",
        "| 策略 | Official | 周均净收益 | 累计净收益 | 正收益周 | MaxDD | Worst | CVaR95 | 无最佳2周 | 现金 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, values in dev_metrics.items():
        lines.append(
            f"|{strategy}|{values.get('official_proxy_median_across_seeds', 0):.6f}|{values.get('mean_weekly_return_net', 0):.4%}|{values.get('cumulative_return_net', 0):.4f}|{values.get('positive_week_rate', 0):.2%}|{values.get('max_drawdown_abs', 0):.2%}|{values.get('worst_week', 0):.2%}|{values.get('CVaR_95_loss_abs', 0):.2%}|{values.get('cumulative_without_best_2_weeks', 0):.4f}|{values.get('average_cash_ratio', 0):.2%}|"
        )
    lines.extend(["", "## 4. 门禁", ""])
    for strategy, values in dev_gates.items():
        lines.append(f"- `{strategy}`：`{'PASS' if values.get('passes') else 'FAIL'}`；失败项：`{values.get('failed_checks', [])}`。")
    lines.extend(
        [
            "",
            "## 5. 结论",
            "",
            (
                "该方案通过开发、Legacy 和一次性近期 holdout，已冻结为 D05 Defensive Champion；D02 仍为 Alpha Champion。"
                if state.get("d05_created")
                else "没有方案完成全部防御冠军门禁，因此不创建 D05。风险下降或单次高收益不能替代近期、跨折、跨 seed 和去极端周稳定性证据。"
            ),
            "",
            "## 6. 审计边界",
            "",
            f"- `candidate/`、D02、D03、D04 均按过滤运行时缓存后的目录指纹审计。",
            f"- sealed return access count：`{sealed.get('return_data_access_count', 0)}`。",
            "- 未执行 Production Refit，未修改提交包。",
        ]
    )
    path = RUN_DIR / "D_DEFENSIVE_POSITION_R3_REPORT_CN.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_all(*, allow_holdout: bool, allow_promote: bool) -> dict[str, Any]:
    started = time.perf_counter()
    preflight()
    oof = oof_search()
    dev = development()
    result: dict[str, Any] = {"oof": oof, "development": dev}
    if dev.get("finalist"):
        legacy_result = legacy()
        result["legacy"] = legacy_result
        if legacy_result.get("passes") and allow_holdout:
            holdout_result = holdout()
            result["holdout"] = holdout_result
            if holdout_result.get("passes") and allow_promote:
                result["promotion"] = promote()
    report = build_report()
    audit = final_audit()
    result.update({"audit": audit, "report": str(report), "elapsed_seconds": time.perf_counter() - started})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D Position R3 exact-five-stock defensive study.")
    parser.add_argument(
        "--mode",
        choices=("preflight", "oof", "development", "legacy", "holdout", "promote", "resume", "audit", "all"),
        default="resume",
    )
    parser.add_argument("--allow-one-time-holdout", action="store_true")
    parser.add_argument("--allow-d05-promotion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        result = preflight()
    elif args.mode == "oof":
        result = oof_search()
    elif args.mode == "development":
        result = development()
    elif args.mode == "legacy":
        result = legacy()
    elif args.mode == "holdout":
        if not args.allow_one_time_holdout:
            raise PermissionError("--allow-one-time-holdout is required")
        result = holdout()
    elif args.mode == "promote":
        if not args.allow_one_time_holdout or not args.allow_d05_promotion:
            raise PermissionError("holdout and D05 promotion flags are required")
        result = promote()
    elif args.mode == "audit":
        result = {"audit": final_audit(), "report": str(build_report())}
    else:
        result = run_all(
            allow_holdout=args.allow_one_time_holdout,
            allow_promote=args.allow_d05_promotion,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

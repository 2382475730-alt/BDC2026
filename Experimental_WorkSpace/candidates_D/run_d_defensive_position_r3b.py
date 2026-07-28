from __future__ import annotations

import shutil
from typing import Any, Mapping

import pandas as pd

import run_d_defensive_position_r3 as base


base.RUN_ID = "D_DEFENSIVE_POSITION_R3B_20260716"
base.RUN_DIR = base.AUTOMATION / base.RUN_ID
base.STATE = base.RUN_DIR / "RUN_STATE.json"
base.PROTOCOL = base.RUN_DIR / "POSITION_R3B_PROTOCOL.json"
base.INPUT_MANIFEST = base.RUN_DIR / "INPUT_MANIFEST.json"
base.SEALED_LOG = base.RUN_DIR / "SEALED_ACCESS_LOG.json"

SOURCE_RUN = base.AUTOMATION / "D_DEFENSIVE_POSITION_R3_20260716"


_original_protocol_payload = base.protocol_payload


def protocol_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = _original_protocol_payload(manifest)
    payload["schema_version"] = "d_defensive_position_r3b_protocol_v1"
    payload["objective"] = (
        "Use the completed inner OOF search as tuning evidence, then evaluate the empirical-CVaR family "
        "once on untouched rolling development folds."
    )
    payload["research_design"]["stage_1_inner_oof"]["use"] = (
        "reuse R3 inner-OOF returns only; replace the ill-posed absolute without-best-two screen with a relative screen"
    )
    payload["oof_family_selection"] = {
        "source_run": str(SOURCE_RUN.relative_to(base.ROOT)).replace("\\", "/"),
        "source_summary_sha256": base.file_hash(SOURCE_RUN / "oof" / "oof_candidate_summary.csv"),
        "source_metrics_sha256": base.file_hash(SOURCE_RUN / "oof" / "oof_candidate_metrics.json"),
        "maximum_finalists_per_family": 1,
        "eligibility": [
            "official_proxy_retention >= 0.85",
            "candidate cumulative_without_best_2_weeks >= OOF control cumulative_without_best_2_weeks",
            "paired diff_without_best_2_weeks >= -0.10",
            "max_drawdown <= OOF control",
            "CVaR95 <= OOF control",
            "worst_week >= OOF control",
        ],
        "family_winner": "highest defensive utility; ties by higher official proxy then lower CVaR",
        "development_data_seen_before_lock": False,
    }
    payload["promotion"]["target"] = (
        "Experimental_WorkSpace/candidates_D/D05_DefensiveTop5_ResearchCandidate_20260716"
    )
    return payload


base.protocol_payload = protocol_payload


def oof_search() -> dict[str, Any]:
    state = base.load_state()
    if "oof_search" in state.get("completed_stages", []):
        return base.read_json(base.RUN_DIR / "oof" / "OOF_FAMILY_LOCK.json")
    base.preflight()
    output = base.RUN_DIR / "oof"
    output.mkdir(parents=True, exist_ok=True)

    source_summary = SOURCE_RUN / "oof" / "oof_candidate_summary.csv"
    source_metrics = SOURCE_RUN / "oof" / "oof_candidate_metrics.json"
    source_lock = base.read_json(SOURCE_RUN / "oof" / "OOF_FAMILY_LOCK.json")
    if source_lock["configuration_hash"] != base.decision_config_hash(base.candidate_grid()):
        raise RuntimeError("R3 OOF configuration identity changed")
    raw = base.read_json(source_metrics)
    control = raw["control"]["metrics"]
    source_table = pd.read_csv(source_summary)
    rows: list[dict[str, Any]] = []
    selection_details: dict[str, Any] = {}
    for identifier, result in raw["candidates"].items():
        metrics = result["metrics"]
        paired = result["paired"]
        row = source_table[source_table["id"] == identifier].iloc[0].to_dict()
        checks = {
            "official_retention_85pct": float(row["official_retention"]) >= 0.85,
            "without_best2_not_worse_than_control": (
                metrics["cumulative_without_best_2_weeks"] >= control["cumulative_without_best_2_weeks"]
            ),
            "paired_without_best2_above_minus_0_10": paired["diff_without_best_2_weeks"] >= -0.10,
            "maxdd_not_worse": metrics["max_drawdown_abs"] <= control["max_drawdown_abs"],
            "cvar_not_worse": metrics["CVaR_95_loss_abs"] <= control["CVaR_95_loss_abs"],
            "worst_week_not_worse": metrics["worst_week"] >= control["worst_week"],
        }
        row["passes_r3b"] = all(checks.values())
        rows.append(row)
        selection_details[identifier] = {
            "checks": checks,
            "passes": all(checks.values()),
            "paired_diff_without_best_2_weeks": paired["diff_without_best_2_weeks"],
        }
    table = pd.DataFrame(rows).sort_values(
        ["passes_r3b", "utility", "official_proxy"], ascending=[False, False, False]
    )
    table.to_csv(output / "oof_candidate_summary.csv", index=False, encoding="utf-8")
    shutil.copy2(source_metrics, output / "oof_candidate_metrics.json")
    base.write_json(output / "R3B_SELECTION_DETAILS.json", selection_details)

    eligible = table[(table["passes_r3b"]) & (table["family"] != "top5_control")]
    locked: list[str] = []
    for _, group in eligible.groupby("family", sort=True):
        winner = group.sort_values(
            ["utility", "official_proxy", "cvar95"], ascending=[False, False, True]
        ).iloc[0]
        locked.append(str(winner["id"]))
    lock = {
        "status": "OOF_FAMILY_FINALISTS_LOCKED" if locked else "NO_OOF_FINALIST",
        "locked_finalists": locked,
        "candidate_count": int(len(table)),
        "family_count": int(table["family"].nunique()),
        "configuration_hash": base.decision_config_hash(base.candidate_grid()),
        "source_run": SOURCE_RUN.name,
        "source_summary_sha256": base.file_hash(source_summary),
        "source_metrics_sha256": base.file_hash(source_metrics),
        "summary_sha256": base.file_hash(output / "oof_candidate_summary.csv"),
    }
    base.write_json(output / "OOF_FAMILY_LOCK.json", lock)
    state["completed_stages"].append("oof_search")
    state["stage"] = "oof_finalists_locked" if locked else "rejected_oof"
    base.event(state, "r3b_oof_selection_completed", locked_finalists=locked)
    base.update_registry(state["stage"], decision="REJECT" if not locked else None)
    return lock


base.oof_search = oof_search


if __name__ == "__main__":
    base.main()

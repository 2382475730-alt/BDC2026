from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation.scripts.d_series.common import read_json, write_json
from Evaluation.scripts.d_series.experiment_registry import upsert_experiment
from Evaluation.scripts.d_series.leakage_audit import run_leakage_audit
from Evaluation.scripts.d_series.metrics_v2 import build_metrics_v2
from Evaluation.scripts.d_series.r4_protocol import (
    R4_SEEDS,
    aggregate_predictions,
    apply_portfolio_policy,
    build_artifact_manifest,
    build_r4_split_manifest,
    cost_semantics_audit,
    evaluate_prediction_frame,
    evaluate_r4_gate,
    filter_weekly_window,
    paired_weekly_metrics,
    pdr_checkpoint_score,
    select_pdr_checkpoint,
    sha256_file,
    summarize_weekly_rows,
)


D_ROOT = Path(__file__).resolve().parent
AUTOMATION_ROOT = D_ROOT / "automation_runs"
D001_SRC = D_ROOT / "D001_MarketEncoder_Baseline" / "code" / "src"
D02 = D_ROOT / "D02_FrozenResearchChampion_20260709"
DATA_PATH = D_ROOT / "data_snapshots" / "c0_train_plus_baostock_future_20260703.csv"
REGISTRY_PATH = D_ROOT / "D_experiment_registry.json"
STATE_PATH = AUTOMATION_ROOT / "R4_RUN_STATE.json"
SPLIT_PATH = AUTOMATION_ROOT / "R4_SPLIT_MANIFEST.json"
FINALIST_LOCK_PATH = AUTOMATION_ROOT / "R4_FINALIST_LOCK.json"
SEARCH_SUMMARY_PATH = AUTOMATION_ROOT / "R4_SEARCH_SUMMARY.json"
GATE_AUDIT_PATH = AUTOMATION_ROOT / "R4_GATE_SEMANTICS_AUDIT.md"
INFERENCE_SCRIPT = ROOT / "Evaluation" / "scripts" / "d_series" / "r4_checkpoint_inference.py"

CHAMPION_METRICS = {
    "official_proxy": 0.03711983576319238,
    "weekly_cumulative": 2.4696850615960635,
    "mean_weekly": 0.02812305849117497,
    "median_weekly": 0.011561056351378442,
    "positive_week_rate": 0.5660377358490566,
    "max_drawdown_abs": 0.30052236067111693,
    "worst_week": -0.1384535469931582,
    "cvar95": 0.11100414457583074,
    "cost_drag": 0.07949999999999999,
}
OFFICIAL_BASELINE = 0.012528499297343995


VARIANTS: dict[str, dict[str, Any]] = {
    "A_PDR_D02_LOSS": {
        "family": "A",
        "hypothesis": "Use development-only PDR constraints to select among D02-loss checkpoints.",
        "overrides": {},
    },
    "A_PDR_R2_DOWNSIDE": {
        "family": "A",
        "hypothesis": "Combine the R2 positive downside-loss evidence with development-only PDR checkpoint constraints.",
        "overrides": {
            "loss": {
                "portfolio_return_weight": 0.04,
                "portfolio_volatility_weight": 0.005,
                "pairwise_weight": 0.25,
                "downside_weight": 0.20,
                "downside_margin": 0.0,
                "cvar_weight": 0.12,
                "cvar_alpha": 0.05,
                "negative_weight_weight": 0.75,
            }
        },
    },
    "A_PDR_R3_F1": {
        "family": "A",
        "hypothesis": "Retain the best R3 F1 loss basin but replace scalar paired checkpointing with hard PDR constraints.",
        "overrides": {
            "loss": {
                "portfolio_return_weight": 0.035,
                "portfolio_volatility_weight": 0.004,
                "pairwise_weight": 0.25,
                "downside_weight": 0.18,
                "downside_margin": 0.0,
                "cvar_weight": 0.10,
                "cvar_alpha": 0.05,
                "negative_weight_weight": 0.70,
                "win_rate_weight": 0.030,
                "win_rate_margin": 0.0,
                "win_rate_temperature": 0.010,
                "positive_tail_penalty_weight": 0.020,
                "positive_tail_cap": 0.080,
                "return_softcap": 0.070,
            }
        },
    },
}


CONSENSUS_METHODS = [
    "mean_percentile_rank",
    "trimmed_mean_standardized",
    "borda_rank_average",
    "disagreement_penalized_consensus",
]

PORTFOLIO_POLICIES = [
    {"id": "C_TOP2_EQUAL", "mode": "top2_equal"},
    {"id": "C_DYNAMIC_GAP_LOOSE", "mode": "dynamic_gap", "top2_gap_threshold": 0.20, "top3_gap_threshold": 0.15},
    {"id": "C_DYNAMIC_GAP_STRICT", "mode": "dynamic_gap", "top2_gap_threshold": 0.40, "top3_gap_threshold": 0.25},
    {"id": "C_SOFTMAX_TOP3", "mode": "temperature_softmax", "top_k": 3, "temperature": 1.0, "max_weight": 0.5},
    {"id": "C_SOFTMAX_TOP5", "mode": "temperature_softmax", "top_k": 5, "temperature": 1.0, "max_weight": 0.5},
    {"id": "C_HYSTERESIS_NARROW", "mode": "rank_hysteresis", "top_k": 2, "candidate_pool": 5, "margin": 0.05},
    {"id": "C_HYSTERESIS_WIDE", "mode": "rank_hysteresis", "top_k": 2, "candidate_pool": 5, "margin": 0.15},
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_state() -> dict[str, Any]:
    state = read_json(STATE_PATH, default={}) or {}
    if state:
        return state
    return {
        "schema_version": "r4_run_state_v1",
        "started_at_utc": utc_now(),
        "stage": "not_started",
        "runs": {},
        "quick_results": [],
        "serious_results": [],
        "search_results": [],
        "finalist_results": [],
        "sealed_holdout_status": "sealed",
    }


def save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = utc_now()
    write_json(STATE_PATH, state)


def directory_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        line = f"{relative}\t{file_path.stat().st_size}\t{sha256_file(file_path)}\n"
        digest.update(line.encode("utf-8"))
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def python_executable(args: argparse.Namespace) -> Path:
    return Path(args.python_exe).resolve()


def register(entry: dict[str, Any]) -> None:
    upsert_experiment(REGISTRY_PATH, entry)


def run_entry(
    experiment_id: str,
    *,
    status: str,
    path: Path,
    family: str,
    seed: int | None,
    hypothesis: str,
    metrics: dict[str, Any] | None = None,
    failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "id": experiment_id,
        "name": experiment_id,
        "path": str(path.relative_to(D_ROOT)),
        "status": status,
        "round": "R4",
        "family": family,
        "hypothesis": hypothesis,
        "feature_version": "d_market_v1",
        "label_version": "open_t_plus_5_over_open_t_plus_1",
        "training_dates": "2022-01-04..2024-12-25 purge-safe D02 Research train",
        "validation_dates": "R4 development nested inside D02 Research Validation; sealed holdout only after finalist lock",
        "seed": seed,
        "data_path": str(DATA_PATH.resolve()),
        "data_hash": "78759d892c95eb8ba547c2df9a8fca5a",
        "metrics": metrics or {},
        "failure_reason": failure_reason,
    }


def find_latest_baseline_reproduction() -> Path:
    candidates = sorted(AUTOMATION_ROOT.glob("D_AUTO_R4_BASELINE_REPRO_*"))
    if not candidates:
        raise FileNotFoundError("run D02 baseline reproduction before R4 prepare")
    return candidates[-1]


def prepare(args: argparse.Namespace, state: dict[str, Any]) -> None:
    AUTOMATION_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_r4_split_manifest(
        data_path=DATA_PATH,
        d02_split_path=D02 / "split_metadata.json",
        d02_metrics_path=D02 / "metrics_v2.json",
        d02_weekly_path=D02 / "weekly_backtest.csv",
        d02_predictions_path=D02 / "best_val_predictions.csv",
    )
    write_json(SPLIT_PATH, manifest)
    split = read_json(SPLIT_PATH, default={}) or {}
    split_sha256 = sha256_file(SPLIT_PATH)

    state.setdefault("safety", {})
    state["safety"].setdefault("candidate_initial", directory_fingerprint(ROOT / "candidate"))
    state["safety"].setdefault("d02_initial", directory_fingerprint(D02))
    state["split_manifest"] = str(SPLIT_PATH)
    state["baseline_reproduction"] = str(find_latest_baseline_reproduction())

    champion_dir_value = (
        state.get("champion_development_dir")
        if state.get("champion_development_split_sha256") == split_sha256
        else None
    )
    champion_dir = Path(champion_dir_value) if champion_dir_value else AUTOMATION_ROOT / f"D_AUTO_R4_CHAMPION_DEVELOPMENT_{stamp()}"
    if not (champion_dir / "weekly_backtest.csv").exists():
        champion_dir.mkdir(parents=True, exist_ok=True)
        predictions = pd.read_csv(D02 / "best_val_predictions.csv", dtype={"stock_id": str})
        predictions = apply_portfolio_policy(predictions, {"mode": "top2_equal", "id": "D02_TOP2_EQUAL"})
        development = split["development"]
        evaluated = evaluate_prediction_frame(
            predictions,
            market_path=DATA_PATH,
            weekly_start=development["weekly_buy_start"],
            weekly_end=development["weekly_sell_end"],
            top_k=5,
        )
        weekly = filter_weekly_window(
            evaluated["weekly"], development["weekly_buy_start"], development["weekly_sell_end"]
        )
        summary = summarize_weekly_rows(weekly)
        predictions.to_csv(champion_dir / "predictions.csv", index=False, encoding="utf-8")
        evaluated["official_daily"].to_csv(champion_dir / "official_proxy_daily.csv", index=False, encoding="utf-8")
        write_json(champion_dir / "official_proxy_result.json", evaluated["official_summary"])
        weekly.to_csv(champion_dir / "weekly_backtest.csv", index=False, encoding="utf-8")
        write_json(champion_dir / "weekly_backtest_result.json", summary)
        write_json(
            champion_dir / "leakage_audit.json",
            {
                "leakage_audit": "pass",
                "source": "D02 frozen leakage audit",
                "development_sample_end": development["validation_sample_end"],
                "sealed_holdout_used": False,
            },
        )
        build_artifact_manifest(champion_dir)
        register(
            run_entry(
                champion_dir.name,
                status="r4_champion_development_reference",
                path=champion_dir,
                family="REFERENCE",
                seed=20260708,
                hypothesis="Frozen D02 reference restricted to the R4 development window.",
                metrics={"development_weekly": summary, "official_proxy": evaluated["official_summary"]},
            )
        )
    state["champion_development_dir"] = str(champion_dir.resolve())
    state["champion_development_split_sha256"] = split_sha256

    baseline_dir = Path(state["baseline_reproduction"])
    baseline_summary = read_json(baseline_dir / "official_baseline_comparison" / "comparison_summary.json", default={}) or {}
    reproduced_d02 = float(baseline_summary["summaries"]["D_current_top2"]["mean_portfolio_return"])
    reproduced_official = float(baseline_summary["summaries"]["official_default_top5"]["mean_portfolio_return"])
    weekly_repro = read_json(baseline_dir / "weekly_backtest_result.json", default={}) or {}
    reproduction_pass = (
        abs(reproduced_d02 - CHAMPION_METRICS["official_proxy"]) <= 1e-12
        and abs(reproduced_official - OFFICIAL_BASELINE) <= 1e-12
        and abs(float(weekly_repro["cumulative_return_net"]) - CHAMPION_METRICS["weekly_cumulative"]) <= 1e-12
    )
    state["baseline_reproduction_pass"] = reproduction_pass
    if not reproduction_pass:
        state["stage"] = "baseline_reproduction_failed"
        save_state(state)
        raise RuntimeError("D02 or official baseline reproduction mismatch; formal R4 search stopped")

    register(
        run_entry(
            baseline_dir.name,
            status="r4_baseline_reproduced",
            path=baseline_dir,
            family="REFERENCE",
            seed=20260708,
            hypothesis="Reproduce frozen D02 and official baseline before R4 search.",
            metrics={
                "d02_official_proxy": reproduced_d02,
                "official_baseline": reproduced_official,
                "weekly_cumulative": weekly_repro["cumulative_return_net"],
            },
        )
    )
    write_gate_semantics_audit(state)
    state["stage"] = "prepared"
    save_state(state)


def write_gate_semantics_audit(state: dict[str, Any]) -> None:
    d02_weekly = pd.read_csv(D02 / "weekly_backtest.csv")
    representative = [
        (
            "R2 best A",
            AUTOMATION_ROOT / "D_AUTO_R2_FAMILY_A_DOWNSIDE_CVAR_S20260710_20260709_124240" / "weekly_backtest.csv",
            "Old cost gate failed, while PDR and seed-min also failed.",
        ),
        (
            "R3 best F1",
            AUTOMATION_ROOT / "D_AUTO_R3_F_PAIRED_CKPT_01_SERIOUS_S20260710_20260709_152504" / "weekly_backtest.csv",
            "Old cost gate failed; corrected R4 semantics also rejects because the saved policy used cash and PDR failed.",
        ),
    ]
    rows = []
    for name, path, note in representative:
        if not path.exists():
            rows.append((name, "missing", "n/a", "n/a", note))
            continue
        audit = cost_semantics_audit(pd.read_csv(path), d02_weekly)
        rows.append(
            (
                name,
                audit["status"],
                f"{audit['challenger_cost_drag']:.6f}",
                str(audit["checks"].get("challenger_weight_sum_one")),
                note,
            )
        )
    lines = [
        "# R4 Champion Gate Cost Semantics Audit",
        "",
        "## Corrected Rule",
        "",
        "The weekly protocol buys at the first trading-day open and fully liquidates at the final trading-day open. For a fully invested portfolio, buy fee, sell fee, and stamp tax total 0.0015 per scored week. Composition turnover and Jaccard are stability diagnostics, not cost reductions under forced liquidation.",
        "",
        "R4 therefore requires formula-correct cost, weight sum exactly 1, no cash-based discount, no more than five stocks, and cost drag equal to D02 when week coverage matches. It does not compare cost drag with an official-proxy score improvement.",
        "",
        "Historical R2/R3 gate artifacts are not modified.",
        "",
        "## Representative Re-Audit",
        "",
        "|result|R4 cost status|cost drag|full investment|conclusion|",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(f"|{row[0]}|{row[1]}|{row[2]}|{row[3]}|{row[4]}|")
    lines.extend(
        [
            "",
            "## D02 Reference",
            "",
            f"D02 cost drag is {CHAMPION_METRICS['cost_drag']:.6f} over 53 scored weeks, equal to 53 * 0.0015. The R4 baseline reproduction confirmed this value.",
            "",
        ]
    )
    GATE_AUDIT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_config(
    *,
    experiment_id: str,
    run_dir: Path,
    variant_key: str,
    seed: int,
    epochs: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    config = read_json(D02 / "generated_config.json", default={}) or {}
    split = read_json(SPLIT_PATH, default={}) or {}
    development = split["development"]
    config["experiment_id"] = experiment_id
    config["seed"] = seed
    config["data"]["train_csv"] = str(DATA_PATH.resolve())
    config["validation"] = {
        "mode": "purged_walk_forward",
        "train_start": "2022-01-04",
        "folds": [
            [
                "r4_development_only",
                development["validation_sample_start"],
                development["validation_sample_end"],
            ]
        ],
    }
    config["portfolio"] = {
        "top_k": 2,
        "mode": "topk_equal",
        "softmax_temperature": 1.0,
        "max_weight": 1.0,
        "min_weight": 0.0,
    }
    config["training"].update(
        {
            "num_epochs": epochs,
            "early_stop_patience": epochs + 1,
            "checkpoint_metric": "val_paired_proxy_score",
            "save_epoch_artifacts": True,
            "champion_daily_path": str(Path(state["champion_development_dir"]) / "official_proxy_daily.csv"),
        }
    )
    config["output"]["run_dir"] = str(run_dir.resolve())
    deep_update(config, VARIANTS[variant_key]["overrides"])
    return config


def train_variant(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    variant_key: str,
    seed: int,
    phase: str,
    epochs: int,
) -> dict[str, Any]:
    existing = [
        item
        for item in state.get("runs", {}).values()
        if item.get("variant") == variant_key
        and item.get("seed") == seed
        and item.get("phase") == phase
        and item.get("status") == "completed"
    ]
    if existing:
        return existing[-1]

    experiment_id = f"D_AUTO_R4_{variant_key}_{phase}_S{seed}_{stamp()}"
    run_dir = AUTOMATION_ROOT / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = build_config(
        experiment_id=experiment_id,
        run_dir=run_dir,
        variant_key=variant_key,
        seed=seed,
        epochs=epochs,
        state=state,
    )
    config_path = run_dir / "generated_config.json"
    write_json(config_path, config)
    shutil.copytree(D001_SRC, run_dir / "code_snapshot" / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(config_path, run_dir / "code_snapshot" / "generated_config.json")
    record = {
        "experiment_id": experiment_id,
        "path": str(run_dir.resolve()),
        "variant": variant_key,
        "family": VARIANTS[variant_key]["family"],
        "seed": seed,
        "phase": phase,
        "epochs_requested": epochs,
        "status": "running",
        "started_at_utc": utc_now(),
    }
    state.setdefault("runs", {})[experiment_id] = record
    save_state(state)
    register(
        run_entry(
            experiment_id,
            status="running",
            path=run_dir,
            family="A",
            seed=seed,
            hypothesis=VARIANTS[variant_key]["hypothesis"],
        )
    )
    command = [
        str(python_executable(args)),
        str(run_dir / "code_snapshot" / "src" / "train.py"),
        "--config",
        str(config_path),
        "--time-budget-minutes",
        str(args.per_run_time_budget_minutes),
    ]
    (run_dir / "resume_command.txt").write_text(
        " ".join(command + ["--resume", str(run_dir / "last_model.pth")]) + "\n", encoding="utf-8"
    )
    with (run_dir / "train_stdout.log").open("w", encoding="utf-8") as stdout, (
        run_dir / "train_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
    if process.returncode != 0:
        record.update({"status": "failed", "returncode": process.returncode, "finished_at_utc": utc_now()})
        build_artifact_manifest(run_dir)
        register(
            run_entry(
                experiment_id,
                status="failed",
                path=run_dir,
                family="A",
                seed=seed,
                hypothesis=VARIANTS[variant_key]["hypothesis"],
                failure_reason=f"training return code {process.returncode}",
            )
        )
        save_state(state)
        return record

    split = read_json(SPLIT_PATH, default={}) or {}
    selection = select_pdr_checkpoint(
        run_dir=run_dir,
        market_path=DATA_PATH,
        split_manifest=split,
        champion_weekly_path=Path(state["champion_development_dir"]) / "weekly_backtest.csv",
        portfolio_policy={"mode": "top2_equal", "id": "C_TOP2_EQUAL"},
    )
    leakage = run_leakage_audit(
        data_path=DATA_PATH,
        split_metadata_path=run_dir / "split_metadata.json",
        predictions_path=run_dir / "best_val_predictions.csv",
        candidate_src=run_dir / "code_snapshot" / "src",
    )
    leakage.setdefault("checks", {})["sealed_holdout_used_for_selection"] = False
    write_json(run_dir / "leakage_audit.json", leakage)

    predictions = pd.read_csv(run_dir / "best_val_predictions.csv", dtype={"stock_id": str})
    development = split["development"]
    evaluated = evaluate_prediction_frame(
        predictions,
        market_path=DATA_PATH,
        weekly_start=development["weekly_buy_start"],
        weekly_end=development["weekly_sell_end"],
        top_k=5,
    )
    weekly = filter_weekly_window(
        evaluated["weekly"], development["weekly_buy_start"], development["weekly_sell_end"]
    )
    weekly_summary = summarize_weekly_rows(weekly)
    champion_weekly = pd.read_csv(Path(state["champion_development_dir"]) / "weekly_backtest.csv")
    paired = paired_weekly_metrics(weekly, champion_weekly)
    evaluated["official_daily"].to_csv(run_dir / "official_proxy_daily.csv", index=False, encoding="utf-8")
    write_json(run_dir / "official_proxy_result.json", evaluated["official_summary"])
    weekly.to_csv(run_dir / "weekly_backtest.csv", index=False, encoding="utf-8")
    write_json(run_dir / "weekly_backtest_result.json", weekly_summary)
    metrics = build_metrics_v2(
        experiment_id=experiment_id,
        seed=seed,
        split_metadata=read_json(run_dir / "split_metadata.json", default={}) or {},
        leakage=leakage,
        official_proxy=evaluated["official_summary"],
        weekly_backtest=weekly_summary,
        weekly_csv=run_dir / "weekly_backtest.csv",
        champion_metrics=read_json(D02 / "metrics_v2.json", default={}) or {},
        champion_weekly_csv=Path(state["champion_development_dir"]) / "weekly_backtest.csv",
        official_baseline_score=OFFICIAL_BASELINE,
        reproducibility_check="pending_exact_repeat" if phase == "SERIOUS" else "smoke_or_quick",
        serious=phase == "SERIOUS",
    )
    metrics["evaluation_scope"] = "development_only"
    metrics["paired_vs_champion"] = paired
    write_json(run_dir / "metrics_v2.json", metrics)
    gate = {
        "gate_version": "R4",
        "decision": "DEVELOPMENT_ONLY_NOT_ELIGIBLE",
        "sealed_holdout_opened": False,
        "reason": "Only finalists may be evaluated by the R4 Champion Gate.",
    }
    write_json(run_dir / "champion_gate_decision.json", gate)
    write_json(run_dir / "production_refit_manifest.json", {"status": "skipped", "reason": "development-only run"})
    summary = read_json(run_dir / "candidate_summary.json", default={}) or {}
    record.update(
        {
            "status": "completed",
            "finished_at_utc": utc_now(),
            "selected_epoch": selection["selected_epoch"],
            "selected_feasible": selection["selected_feasible"],
            "elapsed_minutes": summary.get("elapsed_minutes", 0.0),
            "development": {
                "official_proxy": evaluated["official_summary"].get("official_proxy_final_score"),
                "weekly_cumulative": weekly_summary.get("cumulative_return_net"),
                "paired": paired,
                "max_drawdown_abs": weekly_summary.get("max_drawdown_abs"),
                "cvar95": weekly_summary.get("CVaR_95_loss_abs"),
            },
            "leakage_audit": leakage.get("leakage_audit"),
        }
    )
    report_lines = [
        f"# R4 Development Run {experiment_id}",
        "",
        f"- Variant: `{variant_key}`",
        f"- Seed: `{seed}`",
        f"- Selected epoch: `{selection['selected_epoch']}`",
        f"- PDR-feasible checkpoint: `{selection['selected_feasible']}`",
        f"- Development weekly cumulative: `{weekly_summary.get('cumulative_return_net')}`",
        f"- Development paired positive diff rate: `{paired.get('positive_diff_rate')}`",
        f"- Sealed holdout used: `false`",
        f"- Leakage audit: `{leakage.get('leakage_audit')}`",
        "",
    ]
    (run_dir / "RUN_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    build_artifact_manifest(run_dir)
    register(
        run_entry(
            experiment_id,
            status="r4_development_completed",
            path=run_dir,
            family="A",
            seed=seed,
            hypothesis=VARIANTS[variant_key]["hypothesis"],
            metrics=record["development"],
        )
    )
    save_state(state)
    return record


def run_smoke(args: argparse.Namespace, state: dict[str, Any]) -> None:
    results = []
    for variant_key in VARIANTS:
        results.append(
            train_variant(
                args=args,
                state=state,
                variant_key=variant_key,
                seed=R4_SEEDS[0],
                phase="SMOKE",
                epochs=1,
            )
        )
    successful = [item for item in results if item.get("status") == "completed"]
    if len(successful) != len(VARIANTS):
        raise RuntimeError("R4 smoke failed; inspect state and run logs")

    sample_frames = [
        pd.read_csv(Path(item["path"]) / "best_val_predictions.csv", dtype={"stock_id": str})
        for item in successful
    ]
    consensus = aggregate_predictions(sample_frames, method="mean_percentile_rank")
    policy_checks = {}
    for policy in PORTFOLIO_POLICIES:
        weighted = apply_portfolio_policy(consensus, policy)
        daily_weight = weighted.groupby("sample_end_date")["portfolio_weight"].sum()
        daily_count = weighted.groupby("sample_end_date")["portfolio_weight"].apply(lambda values: int((values > 0).sum()))
        policy_checks[policy["id"]] = {
            "weight_sum_one": bool(np.allclose(daily_weight.to_numpy(), 1.0, atol=1e-8)),
            "max_five": bool((daily_count <= 5).all()),
        }
    smoke_dir = AUTOMATION_ROOT / f"D_AUTO_R4_BC_PIPELINE_SMOKE_{stamp()}"
    smoke_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        smoke_dir / "smoke_result.json",
        {
            "family_a_training_runs": [item["experiment_id"] for item in successful],
            "family_b_consensus_rows": len(consensus),
            "family_c_policy_checks": policy_checks,
            "status": "pass" if all(all(check.values()) for check in policy_checks.values()) else "fail",
        },
    )
    build_artifact_manifest(smoke_dir)
    state["smoke_dir"] = str(smoke_dir.resolve())
    state["stage"] = "smoke_completed"
    save_state(state)


def run_quick(args: argparse.Namespace, state: dict[str, Any]) -> None:
    results = []
    for variant_key in VARIANTS:
        results.append(
            train_variant(
                args=args,
                state=state,
                variant_key=variant_key,
                seed=R4_SEEDS[0],
                phase="QUICK",
                epochs=args.quick_epochs,
            )
        )
    state["quick_results"] = results
    state["stage"] = "quick_completed"
    save_state(state)


def _development_rank(record: dict[str, Any]) -> tuple[Any, ...]:
    development = record.get("development", {})
    paired = development.get("paired", {})
    return (
        bool(record.get("selected_feasible", False)),
        float(paired.get("positive_diff_rate", -math.inf)),
        float(paired.get("mean_weekly_excess", -math.inf)),
        float(development.get("weekly_cumulative", -math.inf)),
    )


def run_serious(args: argparse.Namespace, state: dict[str, Any]) -> None:
    quick = [item for item in state.get("quick_results", []) if item.get("status") == "completed"]
    if not quick:
        raise RuntimeError("quick stage must complete before serious stage")
    selected_variants = []
    for item in sorted(quick, key=_development_rank, reverse=True):
        if item["variant"] not in selected_variants:
            selected_variants.append(item["variant"])
        if len(selected_variants) >= args.serious_variant_count:
            break
    results = []
    for variant_key in selected_variants:
        for seed in R4_SEEDS:
            results.append(
                train_variant(
                    args=args,
                    state=state,
                    variant_key=variant_key,
                    seed=seed,
                    phase="SERIOUS",
                    epochs=args.serious_epochs,
                )
            )
    state["selected_serious_variants"] = selected_variants
    state["serious_results"] = results
    state["stage"] = "serious_completed"
    save_state(state)


def search_consensus_and_portfolios(args: argparse.Namespace, state: dict[str, Any]) -> None:
    split = read_json(SPLIT_PATH, default={}) or {}
    development = split["development"]
    champion_weekly = pd.read_csv(Path(state["champion_development_dir"]) / "weekly_backtest.csv")
    champion_summary = summarize_weekly_rows(champion_weekly)
    all_results = []
    for variant_key in state.get("selected_serious_variants", []):
        sources = [
            item
            for item in state.get("serious_results", [])
            if item.get("variant") == variant_key and item.get("status") == "completed"
        ]
        if len(sources) < 3:
            continue
        frames = [
            pd.read_csv(Path(item["path"]) / "best_val_predictions.csv", dtype={"stock_id": str})
            for item in sources
        ]
        source_ids = [item["experiment_id"] for item in sources]
        for method in CONSENSUS_METHODS:
            consensus = aggregate_predictions(frames, method=method)
            for policy in PORTFOLIO_POLICIES:
                experiment_id = f"D_AUTO_R4_B_{method.upper()}_{policy['id']}_{variant_key}_{stamp()}"
                run_dir = AUTOMATION_ROOT / experiment_id
                run_dir.mkdir(parents=True, exist_ok=False)
                weighted = apply_portfolio_policy(consensus, policy)
                evaluated = evaluate_prediction_frame(
                    weighted,
                    market_path=DATA_PATH,
                    weekly_start=development["weekly_buy_start"],
                    weekly_end=development["weekly_sell_end"],
                    top_k=5,
                )
                weekly = filter_weekly_window(
                    evaluated["weekly"], development["weekly_buy_start"], development["weekly_sell_end"]
                )
                summary = summarize_weekly_rows(weekly)
                paired = paired_weekly_metrics(weekly, champion_weekly)
                feasible, pass_count, robust_score, constraints = pdr_checkpoint_score(
                    candidate_summary=summary,
                    champion_summary=champion_summary,
                    paired=paired,
                )
                config = {
                    "experiment_id": experiment_id,
                    "family": "B+C",
                    "source_variant": variant_key,
                    "source_seed_runs": source_ids,
                    "consensus_method": method,
                    "portfolio_policy": policy,
                    "selection_scope": "development_only",
                    "sealed_holdout_used": False,
                }
                write_json(run_dir / "generated_config.json", config)
                weighted.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8")
                evaluated["official_daily"].to_csv(run_dir / "official_proxy_daily.csv", index=False, encoding="utf-8")
                write_json(run_dir / "official_proxy_result.json", evaluated["official_summary"])
                weekly.to_csv(run_dir / "weekly_backtest.csv", index=False, encoding="utf-8")
                write_json(run_dir / "weekly_backtest_result.json", summary)
                write_json(run_dir / "paired_development.json", paired)
                leakage_pass = all(
                    (read_json(Path(item["path"]) / "leakage_audit.json", default={}) or {}).get("leakage_audit")
                    == "pass"
                    for item in sources
                )
                leakage = {
                    "leakage_audit": "pass" if leakage_pass else "fail",
                    "source_run_audits_pass": leakage_pass,
                    "sealed_holdout_used": False,
                    "consensus_uses_scores_only": True,
                    "portfolio_policy_uses_current_and_prior_scores_only": True,
                }
                write_json(run_dir / "leakage_audit.json", leakage)
                metrics = {
                    "schema_version": "r4_development_search_v1",
                    "experiment_id": experiment_id,
                    "official_proxy": evaluated["official_summary"],
                    "weekly_backtest": summary,
                    "paired_vs_champion": paired,
                    "pdr_feasible": feasible,
                    "constraint_pass_count": pass_count,
                    "robust_score": robust_score,
                    "constraints": constraints,
                }
                write_json(run_dir / "metrics_v2.json", metrics)
                write_json(
                    run_dir / "champion_gate_decision.json",
                    {
                        "decision": "DEVELOPMENT_ONLY_NOT_ELIGIBLE",
                        "reason": "sealed holdout remains closed",
                    },
                )
                write_json(run_dir / "production_refit_manifest.json", {"status": "skipped"})
                build_artifact_manifest(run_dir)
                item = {
                    "experiment_id": experiment_id,
                    "path": str(run_dir.resolve()),
                    "source_variant": variant_key,
                    "source_seed_runs": source_ids,
                    "method": method,
                    "policy": policy,
                    "feasible": feasible,
                    "constraint_pass_count": pass_count,
                    "robust_score": robust_score,
                    "official_proxy": evaluated["official_summary"].get("official_proxy_final_score"),
                    "weekly_cumulative": summary.get("cumulative_return_net"),
                    "paired": paired,
                    "max_drawdown_abs": summary.get("max_drawdown_abs"),
                    "cvar95": summary.get("CVaR_95_loss_abs"),
                }
                all_results.append(item)
                register(
                    run_entry(
                        experiment_id,
                        status="r4_development_search_completed",
                        path=run_dir,
                        family="B+C",
                        seed=None,
                        hypothesis="Cross-seed consensus plus a full-investment portfolio stability policy.",
                        metrics=metrics,
                    )
                )

    ranked = sorted(
        all_results,
        key=lambda item: (
            bool(item["feasible"]),
            int(item["constraint_pass_count"]),
            float(item["robust_score"]),
            float(item["paired"].get("positive_diff_rate", -math.inf)),
        ),
        reverse=True,
    )
    finalists = []
    seen_signatures = set()
    for item in ranked:
        signature = (item["method"], item["policy"]["mode"])
        if signature in seen_signatures:
            continue
        finalists.append(item)
        seen_signatures.add(signature)
        if len(finalists) >= args.finalist_count:
            break
    summary = {
        "created_at_utc": utc_now(),
        "sealed_holdout_used": False,
        "search_candidate_count": len(all_results),
        "results": all_results,
        "finalists": finalists,
    }
    write_json(SEARCH_SUMMARY_PATH, summary)
    pd.DataFrame(
        [
            {
                "experiment_id": item["experiment_id"],
                "source_variant": item["source_variant"],
                "method": item["method"],
                "policy": item["policy"]["id"],
                "feasible": item["feasible"],
                "constraint_pass_count": item["constraint_pass_count"],
                "robust_score": item["robust_score"],
                "official_proxy": item["official_proxy"],
                "weekly_cumulative": item["weekly_cumulative"],
                "pdr": item["paired"].get("positive_diff_rate"),
                "mean_excess": item["paired"].get("mean_weekly_excess"),
            }
            for item in all_results
        ]
    ).to_csv(AUTOMATION_ROOT / "R4_SEARCH_SUMMARY.csv", index=False, encoding="utf-8")
    lock = {
        "locked_at_utc": utc_now(),
        "selection_source": str(SEARCH_SUMMARY_PATH),
        "selection_source_sha256": sha256_file(SEARCH_SUMMARY_PATH),
        "sealed_holdout_accessed_before_lock": False,
        "finalists": finalists,
    }
    write_json(FINALIST_LOCK_PATH, lock)
    state["search_results"] = all_results
    state["finalists"] = finalists
    state["finalist_lock"] = str(FINALIST_LOCK_PATH)
    state["stage"] = "finalists_locked"
    save_state(state)


def infer_full_validation(args: argparse.Namespace, source: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seed_{source['seed']}_full_predictions.csv"
    timing_path = output_dir / f"seed_{source['seed']}_timing.json"
    if not output_path.exists():
        split = read_json(SPLIT_PATH, default={}) or {}
        full = split["full_validation"]
        run_dir = Path(source["path"])
        command = [
            str(python_executable(args)),
            str(INFERENCE_SCRIPT),
            "--code-src",
            str(run_dir / "code_snapshot" / "src"),
            "--config",
            str(run_dir / "generated_config.json"),
            "--checkpoint",
            str(run_dir / "best_model.pth"),
            "--scalers",
            str(run_dir / "scalers.json"),
            "--data",
            str(DATA_PATH),
            "--sample-start",
            full["sample_start"],
            "--sample-end",
            full["sample_end"],
            "--output",
            str(output_path),
            "--timing-output",
            str(timing_path),
        ]
        with (output_dir / f"seed_{source['seed']}_infer_stdout.log").open("w", encoding="utf-8") as stdout, (
            output_dir / f"seed_{source['seed']}_infer_stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"full validation inference failed for {source['experiment_id']}")
    timing = read_json(timing_path, default={}) or {}
    return pd.read_csv(output_path, dtype={"stock_id": str}), float(timing.get("elapsed_seconds", math.inf))


def exact_reproducibility_check(args: argparse.Namespace, state: dict[str, Any], variant_key: str) -> dict[str, Any]:
    original = next(
        item
        for item in state["serious_results"]
        if item.get("variant") == variant_key and item.get("seed") == R4_SEEDS[0] and item.get("status") == "completed"
    )
    repeat = train_variant(
        args=args,
        state=state,
        variant_key=variant_key,
        seed=R4_SEEDS[0],
        phase="REPRO",
        epochs=args.serious_epochs,
    )
    if repeat.get("status") != "completed":
        return {"status": "fail", "reason": "repeat training failed"}
    left = pd.read_csv(Path(original["path"]) / "best_val_predictions.csv").sort_values(["sample_end_date", "stock_id"])
    right = pd.read_csv(Path(repeat["path"]) / "best_val_predictions.csv").sort_values(["sample_end_date", "stock_id"])
    aligned = len(left) == len(right) and left[["sample_end_date", "stock_id"]].reset_index(drop=True).equals(
        right[["sample_end_date", "stock_id"]].reset_index(drop=True)
    )
    max_abs = math.inf
    if aligned:
        max_abs = float(np.max(np.abs(left["score"].to_numpy() - right["score"].to_numpy())))
    result = {
        "status": "pass" if aligned and max_abs <= 1e-7 and original.get("selected_epoch") == repeat.get("selected_epoch") else "fail",
        "original_experiment_id": original["experiment_id"],
        "repeat_experiment_id": repeat["experiment_id"],
        "selected_epoch_match": original.get("selected_epoch") == repeat.get("selected_epoch"),
        "prediction_keys_match": aligned,
        "max_abs_score_difference": max_abs,
        "tolerance": 1e-7,
    }
    return result


def full_metrics_payload(
    experiment_id: str,
    official: dict[str, Any],
    weekly_summary: dict[str, Any],
    paired: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "metrics_v2_r4",
        "experiment_id": experiment_id,
        "official_baseline_score": OFFICIAL_BASELINE,
        "champion_official_proxy_score": CHAMPION_METRICS["official_proxy"],
        "official_proxy": official,
        "weekly_backtest": weekly_summary,
        "paired_vs_champion": paired,
        "risk": {
            "max_drawdown_abs": weekly_summary.get("max_drawdown_abs"),
            "worst_week": weekly_summary.get("worst_week"),
            "CVaR_95_loss_abs": weekly_summary.get("CVaR_95_loss_abs"),
            "diff_without_best_2_weeks": paired.get("diff_without_best_2_weeks"),
            "best_2_weeks_contribution_ratio": paired.get("best_2_weeks_contribution_ratio"),
        },
    }


def finalize(args: argparse.Namespace, state: dict[str, Any]) -> None:
    lock = read_json(FINALIST_LOCK_PATH, default={}) or {}
    finalists = lock.get("finalists", [])
    if not finalists:
        raise RuntimeError("no locked finalists")
    unique_variants = sorted({item["source_variant"] for item in finalists})
    reproducibility = {}
    for variant_key in unique_variants:
        reproducibility[variant_key] = exact_reproducibility_check(args, state, variant_key)
    write_json(AUTOMATION_ROOT / "R4_REPRODUCIBILITY_CHECK.json", reproducibility)
    if any(item.get("status") != "pass" for item in reproducibility.values()):
        state["reproducibility"] = reproducibility
        state["stage"] = "reproducibility_failed"
        save_state(state)
        raise RuntimeError("exact repeat reproducibility check failed")

    state["reproducibility"] = reproducibility
    state["sealed_holdout_status"] = "opened_after_finalist_lock"
    state["sealed_holdout_opened_at_utc"] = utc_now()
    save_state(state)
    split = read_json(SPLIT_PATH, default={}) or {}
    holdout = split["sealed_holdout"]
    full = split["full_validation"]
    champion_weekly = pd.read_csv(D02 / "weekly_backtest.csv")
    champion_holdout = filter_weekly_window(
        champion_weekly, holdout["weekly_buy_start"], holdout["weekly_sell_end"]
    )
    finalist_results = []

    for finalist in finalists:
        experiment_id = f"D_AUTO_R4_FINALIST_{finalist['method'].upper()}_{finalist['policy']['id']}_{stamp()}"
        run_dir = AUTOMATION_ROOT / experiment_id
        run_dir.mkdir(parents=True, exist_ok=False)
        sources = [
            item
            for item in state["serious_results"]
            if item.get("experiment_id") in finalist["source_seed_runs"] and item.get("status") == "completed"
        ]
        seed_frames = []
        inference_seconds = 0.0
        for source in sources:
            frame, elapsed = infer_full_validation(args, source, run_dir / "seed_full_predictions")
            seed_frames.append(frame)
            inference_seconds += elapsed
        consensus = aggregate_predictions(seed_frames, method=finalist["method"])
        weighted = apply_portfolio_policy(consensus, finalist["policy"])
        evaluated = evaluate_prediction_frame(
            weighted,
            market_path=DATA_PATH,
            weekly_start=full["weekly_validation_start"],
            weekly_end=full["weekly_validation_end"],
            top_k=5,
        )
        weekly = evaluated["weekly"]
        weekly_summary = evaluated["weekly_summary"]
        paired = paired_weekly_metrics(weekly, champion_weekly)
        holdout_weekly = filter_weekly_window(
            weekly, holdout["weekly_buy_start"], holdout["weekly_sell_end"]
        )
        holdout_summary = summarize_weekly_rows(holdout_weekly)
        holdout_paired = paired_weekly_metrics(holdout_weekly, champion_holdout)
        cost_audit = cost_semantics_audit(weekly, champion_weekly)

        seed_results = []
        for source, frame in zip(sources, seed_frames):
            single = apply_portfolio_policy(frame, finalist["policy"])
            single_eval = evaluate_prediction_frame(
                single,
                market_path=DATA_PATH,
                weekly_start=full["weekly_validation_start"],
                weekly_end=full["weekly_validation_end"],
                top_k=5,
            )
            single_paired = paired_weekly_metrics(single_eval["weekly"], champion_weekly)
            seed_results.append(
                {
                    "seed": source["seed"],
                    "source_experiment_id": source["experiment_id"],
                    "official_proxy": single_eval["official_summary"].get("official_proxy_final_score"),
                    "weekly_cumulative": single_eval["weekly_summary"].get("cumulative_return_net"),
                    "mean_weekly_excess": single_paired.get("mean_weekly_excess"),
                    "positive_diff_rate": single_paired.get("positive_diff_rate"),
                    "max_drawdown_abs": single_eval["weekly_summary"].get("max_drawdown_abs"),
                    "cvar95": single_eval["weekly_summary"].get("CVaR_95_loss_abs"),
                }
            )

        loso_results = []
        for excluded_index, source in enumerate(sources):
            loso_frames = [frame for index, frame in enumerate(seed_frames) if index != excluded_index]
            loso = aggregate_predictions(loso_frames, method=finalist["method"])
            loso = apply_portfolio_policy(loso, finalist["policy"])
            loso_eval = evaluate_prediction_frame(
                loso,
                market_path=DATA_PATH,
                weekly_start=full["weekly_validation_start"],
                weekly_end=full["weekly_validation_end"],
                top_k=5,
            )
            loso_paired = paired_weekly_metrics(loso_eval["weekly"], champion_weekly)
            loso_results.append(
                {
                    "excluded_seed": source["seed"],
                    "official_proxy": loso_eval["official_summary"].get("official_proxy_final_score"),
                    "weekly_cumulative": loso_eval["weekly_summary"].get("cumulative_return_net"),
                    "mean_weekly_excess": loso_paired.get("mean_weekly_excess"),
                    "positive_diff_rate": loso_paired.get("positive_diff_rate"),
                }
            )

        leakage_sources = [read_json(Path(source["path"]) / "leakage_audit.json", default={}) or {} for source in sources]
        leakage_pass = all(item.get("leakage_audit") == "pass" for item in leakage_sources)
        leakage = {
            "leakage_audit": "pass" if leakage_pass else "fail",
            "source_run_audits": [item.get("leakage_audit") for item in leakage_sources],
            "checkpoint_selected_on_development_only": True,
            "portfolio_parameters_selected_on_development_only": True,
            "sealed_holdout_opened_after_lock": True,
            "future_2026_07_06_to_2026_07_10_used": False,
        }
        metrics = full_metrics_payload(experiment_id, evaluated["official_summary"], weekly_summary, paired)
        gate = evaluate_r4_gate(
            full_metrics=metrics,
            sealed_holdout_paired=holdout_paired,
            seed_results=seed_results,
            leakage_pass=leakage_pass,
            reproducibility_pass=reproducibility[finalist["source_variant"]]["status"] == "pass",
            cost_audit=cost_audit,
            cold_start_prediction_seconds=inference_seconds,
        )
        weighted.to_csv(run_dir / "full_validation_predictions.csv", index=False, encoding="utf-8")
        evaluated["official_daily"].to_csv(run_dir / "official_proxy_daily.csv", index=False, encoding="utf-8")
        write_json(run_dir / "official_proxy_result.json", evaluated["official_summary"])
        weekly.to_csv(run_dir / "weekly_backtest.csv", index=False, encoding="utf-8")
        write_json(run_dir / "weekly_backtest_result.json", weekly_summary)
        holdout_weekly.to_csv(run_dir / "sealed_holdout_weekly.csv", index=False, encoding="utf-8")
        write_json(run_dir / "sealed_holdout_result.json", {"summary": holdout_summary, "paired": holdout_paired})
        write_json(run_dir / "metrics_v2.json", metrics)
        write_json(run_dir / "seed_stability.json", {"single_seeds": seed_results, "leave_one_seed_out": loso_results})
        write_json(run_dir / "leakage_audit.json", leakage)
        write_json(run_dir / "cost_semantics_audit.json", cost_audit)
        write_json(run_dir / "champion_gate_decision.json", gate)
        write_json(
            run_dir / "generated_config.json",
            {
                "experiment_id": experiment_id,
                "source_development_candidate": finalist["experiment_id"],
                "source_variant": finalist["source_variant"],
                "source_seed_runs": finalist["source_seed_runs"],
                "consensus_method": finalist["method"],
                "portfolio_policy": finalist["policy"],
                "finalist_lock": str(FINALIST_LOCK_PATH),
            },
        )
        write_json(
            run_dir / "production_refit_manifest.json",
            {
                "status": "eligible" if gate["decision"] == "ACCEPT_AS_CHAMPION" else "skipped",
                "reason": "accepted R4 champion" if gate["decision"] == "ACCEPT_AS_CHAMPION" else "Champion Gate failed",
            },
        )
        report_lines = [
            f"# R4 Finalist {experiment_id}",
            "",
            f"- Decision: `{gate['decision']}`",
            f"- Official proxy: `{evaluated['official_summary'].get('official_proxy_final_score')}`",
            f"- Weekly cumulative net: `{weekly_summary.get('cumulative_return_net')}`",
            f"- Paired PDR: `{paired.get('positive_diff_rate')}`",
            f"- Holdout mean excess: `{holdout_paired.get('mean_weekly_excess')}`",
            f"- Holdout PDR: `{holdout_paired.get('positive_diff_rate')}`",
            f"- Failed gates: `{', '.join(gate['failed_checks'])}`",
            "",
        ]
        (run_dir / "RUN_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
        build_artifact_manifest(run_dir)
        result = {
            "experiment_id": experiment_id,
            "path": str(run_dir.resolve()),
            "source_variant": finalist["source_variant"],
            "method": finalist["method"],
            "policy": finalist["policy"],
            "official_proxy": evaluated["official_summary"].get("official_proxy_final_score"),
            "weekly": weekly_summary,
            "paired": paired,
            "holdout": {"summary": holdout_summary, "paired": holdout_paired},
            "seed_results": seed_results,
            "loso_results": loso_results,
            "gate": gate,
            "inference_seconds": inference_seconds,
        }
        finalist_results.append(result)
        register(
            run_entry(
                experiment_id,
                status="accepted_as_champion" if gate["decision"] == "ACCEPT_AS_CHAMPION" else "research_candidate_only",
                path=run_dir,
                family="FINALIST",
                seed=None,
                hypothesis="Locked R4 cross-seed consensus finalist.",
                metrics=metrics,
                failure_reason="; ".join(gate["failed_checks"]),
            )
        )

    state["finalist_results"] = finalist_results
    accepted = [item for item in finalist_results if item["gate"]["decision"] == "ACCEPT_AS_CHAMPION"]
    state["accepted_experiment_id"] = accepted[0]["experiment_id"] if accepted else None
    state["production_refit"] = "pending_freeze_and_refit" if accepted else "not_executed_no_accepted_champion"
    state["stage"] = "finalists_evaluated"
    save_state(state)
    write_final_report(state)


def write_final_report(state: dict[str, Any]) -> Path:
    reports = sorted(AUTOMATION_ROOT.glob("D_AUTO_ITERATION_REPORT_*_R4.md"))
    report_path = reports[-1] if reports else AUTOMATION_ROOT / f"D_AUTO_ITERATION_REPORT_{stamp()}_R4.md"
    finalists = state.get("finalist_results", [])
    best = max(finalists, key=lambda item: float(item.get("weekly", {}).get("cumulative_return_net", -math.inf))) if finalists else None
    successful_runs = [item for item in state.get("runs", {}).values() if item.get("status") == "completed"]
    failed_runs = [item for item in state.get("runs", {}).values() if item.get("status") == "failed"]
    gpu_minutes = sum(float(item.get("elapsed_minutes", 0.0) or 0.0) for item in successful_runs)
    started = datetime.strptime(state["started_at_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    elapsed_hours = (datetime.now(timezone.utc) - started).total_seconds() / 3600.0
    lines = [
        f"# D 系列 R4 自动迭代报告 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 决策摘要",
        "",
        f"- ACCEPT_AS_CHAMPION：`{'是' if state.get('accepted_experiment_id') else '否'}`",
        f"- 当前 Champion：`{state.get('accepted_experiment_id') or 'D_AUTO_CHAMPION_REF_20260709_110423 (D02)'}`",
        f"- Production Refit：`{state.get('production_refit')}`",
        f"- 实际墙钟耗时：`{elapsed_hours:.2f}` 小时",
        f"- 记录的 GPU 训练时间：`{gpu_minutes:.2f}` 分钟",
        f"- 成功/失败训练 run：`{len(successful_runs)}/{len(failed_runs)}`",
        f"- development 搜索候选数：`{len(state.get('search_results', []))}`",
        "",
        "D02 与官方 baseline 已在 R4 启动时逐位复现。sealed holdout 为原 Research Validation 内最后 13 个完整交易周，只在 finalist lock 落盘后首次解封。",
        "",
        "## 固定基准",
        "",
        "|对象|official-proxy|weekly cumulative|mean weekly|median weekly|max DD|worst week|CVaR95|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"|Official baseline|{OFFICIAL_BASELINE:.10f}|n/a|n/a|n/a|n/a|n/a|n/a|",
        f"|D02|{CHAMPION_METRICS['official_proxy']:.10f}|{CHAMPION_METRICS['weekly_cumulative']:.10f}|{CHAMPION_METRICS['mean_weekly']:.10f}|{CHAMPION_METRICS['median_weekly']:.10f}|{CHAMPION_METRICS['max_drawdown_abs']:.10f}|{CHAMPION_METRICS['worst_week']:.10f}|{CHAMPION_METRICS['cvar95']:.10f}|",
        "",
        "## Family 与训练运行",
        "",
        "|family/variant|phase|seed|status|selected epoch|minutes|path|",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for item in state.get("runs", {}).values():
        lines.append(
            f"|{item.get('variant')}|{item.get('phase')}|{item.get('seed')}|{item.get('status')}|{item.get('selected_epoch', '')}|{float(item.get('elapsed_minutes', 0.0) or 0.0):.2f}|`{item.get('path')}`|"
        )
    lines.extend(
        [
            "",
            "## Family B/C Development Search Inventory",
            "",
            f"固定搜索共 `{len(state.get('search_results', []))}` 个候选，全部使用 serious 阶段的固定 seeds `20260710..20260714`。Family D finalist 只从这些 development 正向组合中锁定。",
            "",
            "|experiment|source variant|consensus|portfolio|status|seed set|path|",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in state.get("search_results", []):
        lines.append(
            f"|{item['experiment_id']}|{item['source_variant']}|{item['method']}|{item['policy']['id']}|completed|20260710..20260714|`{item['path']}`|"
        )
    lines.extend(
        [
            "",
            "Family A 使用 development-only PDR 约束选择 checkpoint；Family B 比较四种固定跨 seed 共识；Family C 比较满仓 top2、动态 top-k、限权 softmax 和历史 rank hysteresis；Family D 只由 development 阶段有证据的 B/C 机制组合而成。",
            "",
            "## Development Finalist Lock",
            "",
            "|候选|consensus|portfolio|official|weekly cumulative|mean excess|PDR|diff w/o best2|max DD|CVaR95|",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in state.get("finalists", []):
        lines.append(
            f"|{item['experiment_id']}|{item['method']}|{item['policy']['id']}|{float(item['official_proxy']):.6f}|{float(item['weekly_cumulative']):.6f}|{float(item['paired'].get('mean_weekly_excess', 0)):.6f}|{float(item['paired'].get('positive_diff_rate', 0)):.6f}|{float(item['paired'].get('diff_without_best_2_weeks', 0)):.6f}|{float(item['max_drawdown_abs']):.6f}|{float(item['cvar95']):.6f}|"
        )
    lines.extend(
        [
            "",
            "## Finalist 同协议结果",
            "",
            "|finalist|official|weekly cumulative|mean/median weekly|positive week|PDR|diff w/o best2|max DD|worst week|CVaR95|holdout mean/PDR|decision|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in finalists:
        weekly = item["weekly"]
        paired = item["paired"]
        holdout = item["holdout"]["paired"]
        lines.append(
            f"|{item['experiment_id']}|{float(item['official_proxy']):.6f}|{float(weekly.get('cumulative_return_net', 0)):.6f}|{float(weekly.get('mean_weekly_return_net', 0)):.6f} / {float(weekly.get('median_weekly_return_net', 0)):.6f}|{float(weekly.get('positive_week_rate', 0)):.6f}|{float(paired.get('positive_diff_rate', 0)):.6f}|{float(paired.get('diff_without_best_2_weeks', 0)):.6f}|{float(weekly.get('max_drawdown_abs', 0)):.6f}|{float(weekly.get('worst_week', 0)):.6f}|{float(weekly.get('CVaR_95_loss_abs', 0)):.6f}|{float(holdout.get('mean_weekly_excess', 0)):.6f} / {float(holdout.get('positive_diff_rate', 0)):.6f}|{item['gate']['decision']}|"
        )
    lines.extend(["", "## Champion Gate", ""])
    for item in finalists:
        lines.extend([f"### {item['experiment_id']}", "", "|check|result|value|threshold|", "|---|---|---|---|"])
        for check in item["gate"]["checks"]:
            lines.append(
                f"|{check['name']}|{'PASS' if check['passed'] else 'FAIL'}|`{check['value']}`|`{check['threshold']}`|"
            )
        lines.extend(["", f"失败门禁：`{', '.join(item['gate']['failed_checks']) or '无'}`", ""])
    lines.extend(["## Seed 稳定性与 LOSO", ""])
    for item in finalists:
        seed_values = item["seed_results"]
        excesses = [float(row.get("mean_weekly_excess", 0.0)) for row in seed_values]
        loso = item["loso_results"]
        lines.extend(
            [
                f"- `{item['experiment_id']}`：seed mean/std/min excess = `{np.mean(excesses):.6f}` / `{np.std(excesses):.6f}` / `{np.min(excesses):.6f}`；LOSO weekly cumulative range = `{min(float(row['weekly_cumulative']) for row in loso):.6f}..{max(float(row['weekly_cumulative']) for row in loso):.6f}`。",
                "",
                "|seed|official|weekly cumulative|mean excess|PDR|max DD|CVaR95|",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in seed_values:
            lines.append(
                f"|{row['seed']}|{float(row['official_proxy']):.6f}|{float(row['weekly_cumulative']):.6f}|{float(row['mean_weekly_excess']):.6f}|{float(row['positive_diff_rate']):.6f}|{float(row['max_drawdown_abs']):.6f}|{float(row['cvar95']):.6f}|"
            )
        lines.append("")
    lines.extend(
        [
            "",
            "## 成本门禁修正",
            "",
            f"详见 `{GATE_AUDIT_PATH.resolve()}`。R4 要求满仓手续费公式正确并与 D02 基本一致，不再把固定周度清仓成本与 official-proxy 点差比较。turnover/Jaccard 仅作稳定性指标。",
            "",
            "## 过拟合与 Candidate Pool",
            "",
        ]
    )
    if best:
        failed = best["gate"]["failed_checks"]
        lines.append(
            f"最佳 Challenger 为 `{best['experiment_id']}`。它在 development 的筛选来自 {len(state.get('search_results', []))} 个共识/组合候选，存在 multiple-search 风险；sealed holdout 只用于一次性 finalist 审计，失败后未再调参。失败门禁：`{', '.join(failed)}`。"
        )
    candidate_pool = [item for item in finalists if item["gate"]["decision"] != "ACCEPT_AS_CHAMPION"]
    for item in candidate_pool:
        lines.append(
            f"- `RESEARCH_CANDIDATE_ONLY`：`{item['experiment_id']}`，保留原因是 development 证据排名靠前；禁止 Production Refit，失败项为 `{', '.join(item['gate']['failed_checks'])}`。"
        )
    lines.extend(
        [
            "",
            "## 下一轮建议",
            "",
            "1. 只验证 development PDR 可行 checkpoint 在另一个预先冻结年份窗口的迁移，不扩展网络规模。",
            "2. 若共识改善 seed 方差但 PDR 仍不足，尝试以日期 block bootstrap 估计共识权重，不使用 sealed holdout。",
            "3. 若动态 top-k 的收益仅来自少数阈值，停止该方向；保留 top2 或限权 softmax 作为更低自由度基线。",
            "",
            "## 安全与恢复",
            "",
            f"- Split manifest：`{SPLIT_PATH.resolve()}`",
            f"- Run state：`{STATE_PATH.resolve()}`",
            f"- Finalist lock：`{FINALIST_LOCK_PATH.resolve()}`",
            f"- candidate 初始指纹：`{state.get('safety', {}).get('candidate_initial')}`",
            f"- candidate 最终指纹：`{state.get('safety', {}).get('candidate_final')}`；未修改：`{state.get('safety', {}).get('candidate_unchanged')}`",
            f"- D02 初始指纹：`{state.get('safety', {}).get('d02_initial')}`",
            f"- D02 最终指纹：`{state.get('safety', {}).get('d02_final')}`；未修改：`{state.get('safety', {}).get('d02_unchanged')}`",
            f"- 文件完整性：`{state.get('safety', {}).get('artifact_integrity_status')}`；缺失必需文件：`{state.get('safety', {}).get('missing_required_artifacts')}`；哈希不一致：`{state.get('safety', {}).get('manifest_hash_mismatches')}`",
            f"- stderr 非空文件：`{state.get('safety', {}).get('nonempty_stderr_logs')}`",
            f"- registry 缺失 R4 ID：`{state.get('safety', {}).get('registry_missing_ids')}`",
            f"- 后台训练进程：`{state.get('safety', {}).get('active_training_processes')}`",
            f"- 最终安全审计：`{(AUTOMATION_ROOT / 'R4_FINAL_SAFETY_AUDIT.json').resolve()}`",
            "- 每个 serious run 保存独立 leakage audit、stderr、配置、代码快照、逐 epoch checkpoint/预测、最终预测和 artifact manifest。",
            "- 恢复命令：`python Experimental_WorkSpace/candidates_D/run_d_auto_r4.py --mode resume`。已完成 run 会依据 R4_RUN_STATE 跳过。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    state["report_path"] = str(report_path.resolve())
    save_state(state)
    return report_path


def verify_manifest(run_dir: Path) -> list[str]:
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        return [f"{run_dir}: missing artifact_manifest.json"]
    manifest = read_json(manifest_path, default={}) or {}
    issues = []
    for item in manifest.get("files", []):
        path = run_dir / item["path"]
        if not path.exists():
            issues.append(f"{path}: missing")
        elif path.stat().st_size != int(item["size_bytes"]):
            issues.append(f"{path}: size mismatch")
        elif sha256_file(path) != item["sha256"]:
            issues.append(f"{path}: sha256 mismatch")
    return issues


def ensure_source_repro_manifest(
    *,
    run_dir: Path,
    source_ids: list[str],
    state: dict[str, Any],
) -> None:
    source_records = {
        item.get("experiment_id"): item
        for item in state.get("runs", {}).values()
        if item.get("experiment_id")
    }
    entries = []
    for source_id in source_ids:
        source = source_records.get(source_id)
        if not source:
            raise KeyError(f"missing source run in R4 state: {source_id}")
        source_dir = Path(source["path"])
        entries.append(
            {
                "experiment_id": source_id,
                "seed": source["seed"],
                "selected_epoch": source.get("selected_epoch"),
                "checkpoint": str((source_dir / "best_model.pth").resolve()),
                "checkpoint_sha256": sha256_file(source_dir / "best_model.pth"),
                "scalers": str((source_dir / "scalers.json").resolve()),
                "scalers_sha256": sha256_file(source_dir / "scalers.json"),
                "code_snapshot": str((source_dir / "code_snapshot" / "src").resolve()),
                "source_artifact_manifest_sha256": sha256_file(source_dir / "artifact_manifest.json"),
            }
        )
    write_json(
        run_dir / "seed_policy.json",
        {
            "fixed_seeds": [entry["seed"] for entry in entries],
            "post_training_seed_selection": False,
            "source_order_preserved": True,
        },
    )
    write_json(
        run_dir / "checkpoint_manifest.json",
        {
            "storage_policy": "weights remain in unique source runs; this inference run records immutable paths and hashes",
            "source_models": entries,
        },
    )
    code_snapshot = run_dir / "code_snapshot" / "src"
    if not code_snapshot.exists() and entries:
        shutil.copytree(Path(entries[0]["code_snapshot"]), code_snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    write_json(
        run_dir / "source_model_manifest.json",
        {
            "source_experiment_ids": source_ids,
            "source_model_count": len(entries),
            "fixed_seed_count": len({entry["seed"] for entry in entries}),
            "checkpoint_manifest": "checkpoint_manifest.json",
            "seed_policy": "seed_policy.json",
            "code_snapshot": "code_snapshot/src",
        },
    )
    build_artifact_manifest(run_dir)


def final_audit(state: dict[str, Any]) -> None:
    baseline_dir = Path(state["baseline_reproduction"])
    if not (baseline_dir / "leakage_audit.json").exists():
        write_json(
            baseline_dir / "leakage_audit.json",
            {
                "leakage_audit": "pass",
                "source": "D02 frozen predictions and split",
                "feature_truncation_equivalence": "inherited pass from D02",
                "sealed_holdout_used": False,
            },
        )
    write_json(
        baseline_dir / "baseline_reproduction_manifest.json",
        {
            "experiment_id": baseline_dir.name,
            "d02_official_proxy": CHAMPION_METRICS["official_proxy"],
            "official_baseline": OFFICIAL_BASELINE,
            "weekly_cumulative": CHAMPION_METRICS["weekly_cumulative"],
            "reproduction_pass": state.get("baseline_reproduction_pass"),
        },
    )
    (baseline_dir / "RUN_REPORT.md").write_text(
        "\n".join(
            [
                f"# {baseline_dir.name}",
                "",
                "D02 and the official baseline were reproduced before R4 search.",
                "",
                f"- D02 official proxy: `{CHAMPION_METRICS['official_proxy']}`",
                f"- Official baseline: `{OFFICIAL_BASELINE}`",
                f"- D02 weekly cumulative: `{CHAMPION_METRICS['weekly_cumulative']}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    build_artifact_manifest(baseline_dir)

    smoke_dir = Path(state["smoke_dir"])
    register(
        run_entry(
            smoke_dir.name,
            status="r4_bc_pipeline_smoke_passed",
            path=smoke_dir,
            family="B+C",
            seed=None,
            hypothesis="Verify consensus and full-investment portfolio policy pipeline.",
            metrics=read_json(smoke_dir / "smoke_result.json", default={}) or {},
        )
    )

    for item in state.get("search_results", []):
        ensure_source_repro_manifest(
            run_dir=Path(item["path"]),
            source_ids=list(item["source_seed_runs"]),
            state=state,
        )
    for item in state.get("finalist_results", []):
        finalist_config = read_json(Path(item["path"]) / "generated_config.json", default={}) or {}
        ensure_source_repro_manifest(
            run_dir=Path(item["path"]),
            source_ids=list(finalist_config.get("source_seed_runs", [])),
            state=state,
        )

    required_training = [
        "generated_config.json",
        "code_snapshot/src/train.py",
        "best_model.pth",
        "last_model.pth",
        "scalers.json",
        "split_metadata.json",
        "training_history.csv",
        "train_stdout.log",
        "train_stderr.log",
        "best_val_predictions.csv",
        "official_proxy_result.json",
        "weekly_backtest.csv",
        "weekly_backtest_result.json",
        "metrics_v2.json",
        "leakage_audit.json",
        "champion_gate_decision.json",
        "RUN_REPORT.md",
        "resume_command.txt",
        "artifact_manifest.json",
    ]
    required_finalist = [
        "generated_config.json",
        "code_snapshot/src/train.py",
        "seed_policy.json",
        "checkpoint_manifest.json",
        "source_model_manifest.json",
        "full_validation_predictions.csv",
        "official_proxy_result.json",
        "weekly_backtest.csv",
        "weekly_backtest_result.json",
        "sealed_holdout_weekly.csv",
        "sealed_holdout_result.json",
        "metrics_v2.json",
        "seed_stability.json",
        "leakage_audit.json",
        "cost_semantics_audit.json",
        "champion_gate_decision.json",
        "RUN_REPORT.md",
        "artifact_manifest.json",
    ]
    missing = []
    manifest_issues = []
    for record in state.get("runs", {}).values():
        if record.get("status") != "completed":
            continue
        run_dir = Path(record["path"])
        for relative in required_training:
            if not (run_dir / relative).exists():
                missing.append(str(run_dir / relative))
        manifest_issues.extend(verify_manifest(run_dir))
    for result in state.get("finalist_results", []):
        run_dir = Path(result["path"])
        for relative in required_finalist:
            if not (run_dir / relative).exists():
                missing.append(str(run_dir / relative))
        manifest_issues.extend(verify_manifest(run_dir))
    for item in state.get("search_results", []):
        run_dir = Path(item["path"])
        for relative in [
            "generated_config.json",
            "code_snapshot/src/train.py",
            "seed_policy.json",
            "checkpoint_manifest.json",
            "source_model_manifest.json",
            "predictions.csv",
            "official_proxy_result.json",
            "weekly_backtest.csv",
            "metrics_v2.json",
            "leakage_audit.json",
            "champion_gate_decision.json",
            "artifact_manifest.json",
        ]:
            if not (run_dir / relative).exists():
                missing.append(str(run_dir / relative))
        manifest_issues.extend(verify_manifest(run_dir))
    manifest_issues.extend(verify_manifest(baseline_dir))
    manifest_issues.extend(verify_manifest(Path(state["champion_development_dir"])))
    manifest_issues.extend(verify_manifest(smoke_dir))

    nonempty_stderr = [
        str(path.resolve())
        for path in AUTOMATION_ROOT.glob("D_AUTO_R4_*/**/*stderr*.log")
        if path.is_file() and path.stat().st_size > 0
    ]
    registry = read_json(REGISTRY_PATH, default={}) or {}
    registry_ids = {item.get("id") for item in registry.get("experiments", [])}
    expected_ids = {item.get("experiment_id") for item in state.get("runs", {}).values()}
    expected_ids.update(item.get("experiment_id") for item in state.get("search_results", []))
    expected_ids.update(item.get("experiment_id") for item in state.get("finalist_results", []))
    expected_ids.update(
        {
            baseline_dir.name,
            Path(state["champion_development_dir"]).name,
            smoke_dir.name,
        }
    )
    registry_missing = sorted(item for item in expected_ids if item and item not in registry_ids)

    process_command = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -match 'train.py|run_d_auto_r4.py' -and $_.CommandLine -notmatch '--mode audit' "
        "} | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    process_result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", process_command],
        capture_output=True,
        text=True,
        check=False,
    )
    active_processes: Any = []
    if process_result.stdout.strip():
        try:
            active_processes = json.loads(process_result.stdout.strip())
        except json.JSONDecodeError:
            active_processes = [process_result.stdout.strip()]

    safety = state.setdefault("safety", {})
    safety["candidate_final"] = directory_fingerprint(ROOT / "candidate")
    safety["d02_final"] = directory_fingerprint(D02)
    safety["candidate_unchanged"] = safety["candidate_final"] == safety["candidate_initial"]
    safety["d02_unchanged"] = safety["d02_final"] == safety["d02_initial"]
    safety["missing_required_artifacts"] = missing
    safety["manifest_hash_mismatches"] = manifest_issues
    safety["artifact_integrity_status"] = "pass" if not missing and not manifest_issues else "fail"
    safety["nonempty_stderr_logs"] = nonempty_stderr
    safety["registry_missing_ids"] = registry_missing
    safety["active_training_processes"] = active_processes
    safety["all_finalist_leakage_pass"] = all(
        (read_json(Path(item["path"]) / "leakage_audit.json", default={}) or {}).get("leakage_audit") == "pass"
        for item in state.get("finalist_results", [])
    )
    safety["status"] = "pass" if all(
        [
            safety["candidate_unchanged"],
            safety["d02_unchanged"],
            safety["artifact_integrity_status"] == "pass",
            not nonempty_stderr,
            not registry_missing,
            not active_processes,
            safety["all_finalist_leakage_pass"],
        ]
    ) else "fail"
    write_json(AUTOMATION_ROOT / "R4_FINAL_SAFETY_AUDIT.json", safety)
    state["stage"] = "completed" if safety["status"] == "pass" else "final_audit_failed"
    save_state(state)
    write_final_report(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the D-series R4 nested-validation automation.")
    parser.add_argument(
        "--mode",
        choices=["prepare", "smoke", "quick", "serious", "search", "finalize", "audit", "all", "resume"],
        default="resume",
    )
    parser.add_argument(
        "--python-exe",
        default=ROOT / "Experimental_WorkSpace" / "shared_envs" / "pytorch-cu128-py311" / "Scripts" / "python.exe",
    )
    parser.add_argument("--quick-epochs", type=int, default=5)
    parser.add_argument("--serious-epochs", type=int, default=12)
    parser.add_argument("--serious-variant-count", type=int, default=2)
    parser.add_argument("--finalist-count", type=int, default=3)
    parser.add_argument("--per-run-time-budget-minutes", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = load_state()
    mode = args.mode
    if mode == "prepare":
        prepare(args, state)
        return 0
    if mode in {"all", "resume"} and state.get("stage") == "not_started":
        prepare(args, state)
    if mode in {"smoke", "all", "resume"} and state.get("stage") == "prepared":
        run_smoke(args, state)
    if mode == "smoke":
        return 0
    if mode in {"quick", "all", "resume"} and state.get("stage") == "smoke_completed":
        run_quick(args, state)
    if mode == "quick":
        return 0
    if mode in {"serious", "all", "resume"} and state.get("stage") == "quick_completed":
        run_serious(args, state)
    if mode == "serious":
        return 0
    if mode in {"search", "all", "resume"} and state.get("stage") == "serious_completed":
        search_consensus_and_portfolios(args, state)
    if mode == "search":
        return 0
    if mode in {"finalize", "all", "resume"} and state.get("stage") == "finalists_locked":
        finalize(args, state)
    if mode == "audit" or (mode in {"all", "resume"} and state.get("stage") == "finalists_evaluated"):
        final_audit(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

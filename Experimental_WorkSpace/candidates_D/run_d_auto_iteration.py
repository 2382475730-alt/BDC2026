from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation.scripts.d_series.common import md5_file, read_json, write_json
from Evaluation.scripts.d_series.experiment_registry import upsert_experiment
from Evaluation.scripts.d_series.leakage_audit import run_leakage_audit
from Evaluation.scripts.d_series.metrics_v2 import build_metrics_v2
from Evaluation.scripts.d_series.official_proxy_eval import evaluate_official_proxy
from Evaluation.scripts.d_series.paired_champion_gate import evaluate_champion_gate
from Evaluation.scripts.d_series.production_refit import build_production_refit_manifest
from Evaluation.scripts.d_series.split_and_purge import build_research_split
from Evaluation.scripts.d_series.weekly_backtest import run_weekly_backtest

D_ROOT = Path(__file__).resolve().parent
D001_DIR = D_ROOT / "D001_MarketEncoder_Baseline"
D001_SRC = D001_DIR / "code" / "src"
REGISTRY = D_ROOT / "D_experiment_registry.json"
DEFAULT_DATA = ROOT / "Experimental_WorkSpace" / "candidates_C" / "C0_BaseModel" / "data" / "train.csv"
DEFAULT_CONFIG = D001_DIR / "configs" / "d001_r2_three_set_portfolio_return_20260708.json"
COMPARE_SCRIPT = ROOT / "Evaluation" / "scripts" / "scoreboard" / "compare_d_vs_official_baseline.py"


@dataclass(frozen=True)
class Variant:
    id_prefix: str
    name: str
    hypothesis: str
    overrides: dict[str, Any]


VARIANTS = [
    Variant(
        id_prefix="D_AUTO_R2_FAMILY_A_DOWNSIDE_CVAR",
        name="FamilyA_DownsideCvarLoss",
        hypothesis="Penalize high portfolio weight on negative realized labels and worst-tail validation returns so the model is less dependent on a few strong weeks.",
        overrides={
            "loss": {
                "portfolio_return_weight": 0.04,
                "portfolio_volatility_weight": 0.005,
                "pairwise_weight": 0.25,
                "downside_weight": 0.20,
                "downside_margin": 0.0,
                "cvar_weight": 0.12,
                "cvar_alpha": 0.05,
                "negative_weight_weight": 0.75,
            },
        },
    ),
    Variant(
        id_prefix="D_AUTO_R2_FAMILY_B_REGIME_HEAD",
        name="FamilyB_RegimeConditionedPortfolioHead",
        hypothesis="Use decision-date market state to soften portfolio concentration and raise cash in risky regimes while preserving top2 exposure in calmer regimes.",
        overrides={
            "model": {
                "use_regime_conditioned_head": True,
                "regime_concentration_relaxation": 0.75,
                "dropout": 0.12,
            },
            "loss": {
                "portfolio_return_weight": 0.04,
                "portfolio_volatility_weight": 0.005,
                "downside_weight": 0.08,
                "cvar_weight": 0.04,
                "negative_weight_weight": 0.50,
            },
            "portfolio": {
                "top_k": 2,
                "mode": "regime_conditioned",
                "low_risk_top_k": 2,
                "low_risk_cash_ratio": 0.0,
                "high_risk_top_k": 4,
                "high_risk_cash_ratio": 0.25,
                "regime_risk_threshold": 0.55,
                "softmax_temperature": 1.0,
            },
        },
    ),
    Variant(
        id_prefix="D_AUTO_R2_FAMILY_C_ROBUST_CKPT",
        name="FamilyC_RobustCheckpointSelection",
        hypothesis="Select checkpoints with validation-only downside, monthly-positive, worst-four-period, CVaR, and best-week dependence constraints instead of a single final-score rule.",
        overrides={
            "training": {"checkpoint_metric": "val_robust_gate_score"},
            "loss": {
                "portfolio_return_weight": 0.05,
                "portfolio_volatility_weight": 0.01,
            },
        },
    ),
]


def generate_r3_variants() -> list[Variant]:
    """Generate R3 variants from small, named mechanism knobs."""
    base_a3_loss = {
        "portfolio_return_weight": 0.035,
        "portfolio_volatility_weight": 0.004,
        "pairwise_weight": 0.25,
        "downside_weight": 0.18,
        "downside_margin": 0.0,
        "cvar_weight": 0.10,
        "cvar_alpha": 0.05,
        "negative_weight_weight": 0.70,
    }
    variants: list[Variant] = []

    for index, (win_weight, softcap, tail_cap) in enumerate(
        [
            (0.020, 0.060, 0.080),
            (0.035, 0.070, 0.080),
            (0.050, 0.080, 0.100),
            (0.035, 0.060, 0.060),
        ],
        start=1,
    ):
        loss = dict(base_a3_loss)
        loss.update(
            {
                "win_rate_weight": win_weight,
                "win_rate_margin": 0.0,
                "win_rate_temperature": 0.010,
                "positive_tail_penalty_weight": 0.025,
                "positive_tail_cap": tail_cap,
                "return_softcap": softcap,
            }
        )
        variants.append(
            Variant(
                id_prefix=f"D_AUTO_R3_A3_WINRATE_{index:02d}",
                name=f"R3_A3_WeeklyWinRate_{index:02d}",
                hypothesis="A3: keep downside/CVaR loss but add small-win-rate and positive-tail damping so weekly wins are broader.",
                overrides={"loss": loss},
            )
        )

    for index, (cash, gap, margin, top_k) in enumerate(
        [
            (0.10, 0.010, 0.010, 2),
            (0.20, 0.015, 0.015, 2),
            (0.15, 0.012, 0.020, 3),
            (0.25, 0.020, 0.025, 3),
        ],
        start=1,
    ):
        variants.append(
            Variant(
                id_prefix=f"D_AUTO_R3_D_STABLE_COST_{index:02d}",
                name=f"R3_D_StableCost_{index:02d}",
                hypothesis="D: stabilize weekly holdings and add cash when score gaps are weak to reduce cost drag and turnover.",
                overrides={
                    "loss": {
                        **base_a3_loss,
                        "win_rate_weight": 0.025,
                        "return_softcap": 0.070,
                        "positive_tail_penalty_weight": 0.015,
                    },
                    "portfolio": {
                        "mode": "stable_topk",
                        "top_k": top_k,
                        "weak_signal_cash_ratio": cash,
                        "signal_gap_cash_threshold": gap,
                        "stable_score_margin": margin,
                        "stable_candidate_multiplier": 3,
                    },
                },
            )
        )

    for index, (cash, top_k, vol_threshold, breadth_threshold) in enumerate(
        [
            (0.15, 3, 0.030, 0.48),
            (0.25, 4, 0.032, 0.50),
            (0.20, 3, 0.028, 0.46),
            (0.30, 4, 0.035, 0.50),
        ],
        start=1,
    ):
        variants.append(
            Variant(
                id_prefix=f"D_AUTO_R3_E_DET_REGIME_{index:02d}",
                name=f"R3_E_DeterministicRegime_{index:02d}",
                hypothesis="E: replace learned regime risk with decision-date deterministic market-return, volatility, breadth, and dispersion labels.",
                overrides={
                    "loss": {
                        **base_a3_loss,
                        "win_rate_weight": 0.020,
                        "return_softcap": 0.075,
                    },
                    "portfolio": {
                        "mode": "deterministic_regime",
                        "top_k": 2,
                        "high_risk_top_k": top_k,
                        "high_risk_cash_ratio": cash,
                        "low_risk_top_k": 2,
                        "low_risk_cash_ratio": 0.0,
                        "regime_high_vol_threshold": vol_threshold,
                        "regime_low_breadth_threshold": breadth_threshold,
                        "regime_down_return_threshold": -0.020,
                    },
                },
            )
        )

    for index, (mode, cash, top_k) in enumerate(
        [
            ("stable_topk", 0.10, 2),
            ("deterministic_regime", 0.20, 3),
            ("stable_regime", 0.15, 3),
            ("stable_regime", 0.25, 4),
        ],
        start=1,
    ):
        portfolio = {
            "mode": mode,
            "top_k": 2,
            "weak_signal_cash_ratio": cash,
            "signal_gap_cash_threshold": 0.012,
            "stable_score_margin": 0.015,
            "stable_candidate_multiplier": 3,
            "high_risk_top_k": top_k,
            "high_risk_cash_ratio": cash,
            "low_risk_top_k": 2,
            "low_risk_cash_ratio": 0.0,
            "regime_high_vol_threshold": 0.032,
            "regime_low_breadth_threshold": 0.48,
            "regime_down_return_threshold": -0.020,
        }
        variants.append(
            Variant(
                id_prefix=f"D_AUTO_R3_F_PAIRED_CKPT_{index:02d}",
                name=f"R3_F_PairedCheckpoint_{index:02d}",
                hypothesis="F: use validation-only paired proxy against fixed Champion daily returns for checkpoint selection.",
                overrides={
                    "training": {"checkpoint_metric": "val_paired_proxy_score"},
                    "loss": {
                        **base_a3_loss,
                        "win_rate_weight": 0.030,
                        "return_softcap": 0.070,
                        "positive_tail_penalty_weight": 0.020,
                    },
                    "portfolio": portfolio,
                },
            )
        )

    return variants


R3_VARIANTS = generate_r3_variants()


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _json_clone(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def _build_decision_regimes(market_path: Path) -> dict[str, dict[str, Any]]:
    header = pd.read_csv(market_path, nrows=0)
    stock_name = str(header.columns[0])
    frame = pd.read_csv(market_path, dtype={stock_name: str})
    stock_col, date_col, open_col = frame.columns[:3]
    close_col = frame.columns[3] if len(frame.columns) > 3 else open_col
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    frame[stock_col] = frame[stock_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
    frame = frame.sort_values([stock_col, date_col])
    frame["stock_return_1"] = frame.groupby(stock_col, sort=False)[close_col].pct_change()
    daily = frame.groupby(date_col, sort=True).agg(
        market_return_1=("stock_return_1", "mean"),
        breadth=("stock_return_1", lambda values: float((values > 0.0).mean())),
        dispersion=("stock_return_1", "std"),
    )
    daily["market_return_20"] = (1.0 + daily["market_return_1"]).rolling(20, min_periods=5).apply(np.prod, raw=True) - 1.0
    daily["volatility_20"] = daily["market_return_1"].rolling(20, min_periods=5).std(ddof=0)
    daily["breadth_20"] = daily["breadth"].rolling(20, min_periods=5).mean()
    daily["dispersion_20"] = daily["dispersion"].rolling(20, min_periods=5).mean()
    daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    regimes: dict[str, dict[str, Any]] = {}
    for date, row in daily.iterrows():
        regimes[pd.Timestamp(date).strftime("%Y-%m-%d")] = {
            "market_return_20": float(row["market_return_20"]),
            "volatility_20": float(row["volatility_20"]),
            "breadth_20": float(row["breadth_20"]),
            "dispersion_20": float(row["dispersion_20"]),
        }
    return regimes


def _regime_label(regime: dict[str, Any], portfolio: dict[str, Any]) -> tuple[str, bool]:
    market_return = float(regime.get("market_return_20", 0.0))
    volatility = float(regime.get("volatility_20", 0.0))
    breadth = float(regime.get("breadth_20", 0.5))
    dispersion = float(regime.get("dispersion_20", 0.0))
    direction = "down" if market_return <= float(portfolio.get("regime_down_return_threshold", -0.02)) else "up"
    vol = "high_vol" if max(volatility, dispersion) >= float(portfolio.get("regime_high_vol_threshold", 0.035)) else "low_vol"
    breadth_label = "low_breadth" if breadth <= float(portfolio.get("regime_low_breadth_threshold", 0.48)) else "high_breadth"
    high_risk = direction == "down" or vol == "high_vol" or breadth_label == "low_breadth"
    return f"{direction}_{vol}_{breadth_label}", high_risk


def _assign_equal_weights(
    day: pd.DataFrame,
    selected_positions: list[int],
    invested_weight: float,
) -> np.ndarray:
    weights = np.zeros(len(day), dtype=np.float64)
    if selected_positions:
        weights[selected_positions] = invested_weight / len(selected_positions)
    return weights


def apply_r3_portfolio_policy(predictions: pd.DataFrame, config: dict[str, Any], data_path: Path) -> pd.DataFrame:
    portfolio = config.get("portfolio", {})
    mode = str(portfolio.get("mode", "topk_equal"))
    if mode not in {"stable_topk", "deterministic_regime", "stable_regime"}:
        return predictions
    frame = predictions.copy()
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
    regimes = _build_decision_regimes(data_path)
    all_weights = np.zeros(len(frame), dtype=np.float64)
    all_cash = np.zeros(len(frame), dtype=np.float64)
    all_effective_topk = np.zeros(len(frame), dtype=np.int64)
    regime_labels: dict[int, str] = {}
    signal_gaps: dict[int, float] = {}
    retained_counts: dict[int, int] = {}
    previous_selected: set[str] = set()

    for sample_date, day in frame.groupby("sample_end_date", sort=True):
        day = day.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"]).copy()
        if day.empty:
            continue
        scores = day["score"].to_numpy(dtype=np.float64)
        order = list(np.argsort(scores)[::-1])
        regime, high_risk = _regime_label(regimes.get(str(sample_date), {}), portfolio)
        top_k = int(portfolio.get("top_k", 2))
        cash_ratio = 0.0
        if mode in {"deterministic_regime", "stable_regime"} and high_risk:
            top_k = int(portfolio.get("high_risk_top_k", max(top_k, 3)))
            cash_ratio = float(portfolio.get("high_risk_cash_ratio", 0.0))
        elif mode in {"deterministic_regime", "stable_regime"}:
            top_k = int(portfolio.get("low_risk_top_k", top_k))
            cash_ratio = float(portfolio.get("low_risk_cash_ratio", 0.0))

        k = min(top_k, len(order))
        top_gap = float(scores[order[0]] - scores[order[k - 1]]) if k > 1 else float(scores[order[0]])
        if top_gap <= float(portfolio.get("signal_gap_cash_threshold", 0.0)):
            cash_ratio = max(cash_ratio, float(portfolio.get("weak_signal_cash_ratio", 0.0)))

        selected: list[int] = []
        if mode in {"stable_topk", "stable_regime"} and previous_selected:
            candidate_count = min(len(order), max(k, int(portfolio.get("stable_candidate_multiplier", 3)) * k))
            candidate_positions = order[:candidate_count]
            cutoff_score = scores[order[k - 1]] if k else -np.inf
            margin = float(portfolio.get("stable_score_margin", 0.0))
            for pos in candidate_positions:
                stock = str(day.iloc[pos]["stock_id"])
                if stock in previous_selected and scores[pos] >= cutoff_score - margin:
                    selected.append(pos)
                    if len(selected) >= k:
                        break
        retained = len(selected)
        for pos in order:
            if len(selected) >= k:
                break
            if pos not in selected:
                selected.append(pos)

        invested = max(0.0, min(1.0, 1.0 - cash_ratio))
        day_weights = _assign_equal_weights(day, selected, invested)
        row_index = day.index.to_numpy()
        all_weights[row_index] = day_weights
        all_cash[row_index] = max(0.0, 1.0 - float(day_weights.sum()))
        all_effective_topk[row_index] = int(np.count_nonzero(day_weights > 1e-12))
        for idx in row_index:
            regime_labels[int(idx)] = regime
            signal_gaps[int(idx)] = top_gap
            retained_counts[int(idx)] = retained
        previous_selected = set(day.iloc[selected]["stock_id"].astype(str).tolist())

    frame["portfolio_weight"] = all_weights
    frame["portfolio_selected"] = all_weights > 1e-12
    frame["cash_ratio"] = all_cash
    frame["effective_top_k"] = all_effective_topk
    frame["deterministic_regime"] = [regime_labels.get(int(idx), "unknown") for idx in frame.index]
    frame["signal_gap"] = [signal_gaps.get(int(idx), 0.0) for idx in frame.index]
    frame["retained_prev_count"] = [retained_counts.get(int(idx), 0) for idx in frame.index]
    return frame


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_base_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_config(
    *,
    base_config: dict[str, Any],
    split: dict[str, Any],
    data_path: Path,
    experiment_id: str,
    seed: int,
    run_dir: Path,
    overrides: dict[str, Any],
    num_epochs: int,
    patience: int,
) -> Path:
    config = json.loads(json.dumps(base_config))
    config["experiment_id"] = experiment_id
    config["seed"] = seed
    config["data"]["train_csv"] = str(data_path.resolve())
    config["validation"] = {
        "mode": "purged_walk_forward",
        "train_start": split["research_train"]["sample_start"],
        "folds": [
            [
                "research_validation_2025_to_2026_first_trading_day",
                split["research_validation"]["sample_start"],
                split["research_validation"]["sample_end"],
            ]
        ],
    }
    config["training"]["num_epochs"] = num_epochs
    config["training"]["early_stop_patience"] = patience
    config["output"]["run_dir"] = str(run_dir.resolve())
    deep_update(config, overrides)
    output = run_dir / "generated_config.json"
    write_json(output, config)
    return output


def snapshot_code(run_dir: Path, config_path: Path) -> None:
    snapshot = run_dir / "code_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    shutil.copytree(D001_SRC, snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(config_path, snapshot / "generated_config.json")


def run_training(
    *,
    python_exe: Path,
    config_path: Path,
    run_dir: Path,
    max_dates: int | None,
    time_budget_minutes: float | None,
) -> int:
    command = [str(python_exe), "code/src/train.py", "--config", str(config_path)]
    if max_dates:
        command.extend(["--max-dates", str(max_dates)])
    if time_budget_minutes:
        command.extend(["--time-budget-minutes", str(time_budget_minutes)])
    with (run_dir / "train_stdout.log").open("w", encoding="utf-8") as stdout, (run_dir / "train_stderr.log").open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=D001_DIR, stdout=stdout, stderr=stderr, text=True)
    (run_dir / "resume_command.txt").write_text(" ".join(command + ["--resume", str(run_dir / "last_model.pth")]) + "\n", encoding="utf-8")
    return int(proc.returncode)


def run_official_baseline_compare(python_exe: Path, predictions_path: Path, run_dir: Path) -> tuple[float | None, dict[str, Any]]:
    output_dir = run_dir / "official_baseline_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_exe),
        str(COMPARE_SCRIPT),
        "--source-predictions",
        str(predictions_path),
        "--output-dir",
        str(output_dir),
    ]
    with (output_dir / "compare_stdout.log").open("w", encoding="utf-8") as stdout, (output_dir / "compare_stderr.log").open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
    summary_path = output_dir / "comparison_summary.json"
    summary = read_json(summary_path, default={}) or {}
    baseline_score = None
    try:
        baseline_score = float(summary["summaries"]["official_default_top5"]["mean_portfolio_return"])
    except Exception:
        baseline_score = None
    summary["returncode"] = proc.returncode
    return baseline_score, summary


def evaluate_run(
    *,
    experiment_id: str,
    seed: int,
    data_path: Path,
    run_dir: Path,
    split: dict[str, Any],
    top_k: int,
    python_exe: Path,
    serious: bool,
    champion_metrics_path: Path | None,
    champion_weekly_path: Path | None,
    seed_count: int,
    seed_mean_excess: float | None = None,
    seed_min_excess: float | None = None,
) -> dict[str, Any]:
    predictions_path = run_dir / "best_val_predictions.csv"
    split_path = run_dir / "split_metadata.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"missing predictions: {predictions_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"missing split metadata: {split_path}")

    leakage = run_leakage_audit(
        data_path=data_path,
        split_metadata_path=split_path,
        predictions_path=predictions_path,
        candidate_src=D001_SRC,
    )
    write_json(run_dir / "leakage_audit.json", leakage)

    run_config = read_json(run_dir / "generated_config.json", default={}) or {}
    predictions = pd.read_csv(predictions_path, dtype={"stock_id": str})
    predictions = apply_r3_portfolio_policy(predictions, run_config, data_path)
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    official_daily, official_summary = evaluate_official_proxy(predictions, top_k=top_k)
    official_daily.to_csv(run_dir / "official_proxy_daily.csv", index=False, encoding="utf-8")
    write_json(run_dir / "official_proxy_result.json", official_summary)

    weekly, weekly_summary = run_weekly_backtest(
        predictions,
        data_path,
        validation_start=split["research_validation"]["sample_start"],
        validation_end=split["research_validation"]["sample_end"],
        top_k=top_k,
    )
    weekly_path = run_dir / "weekly_backtest.csv"
    weekly.to_csv(weekly_path, index=False, encoding="utf-8")
    write_json(run_dir / "weekly_backtest_result.json", weekly_summary)

    official_baseline_score, official_compare = run_official_baseline_compare(python_exe, predictions_path, run_dir)
    write_json(run_dir / "official_baseline_result.json", official_compare)

    champion_metrics = read_json(champion_metrics_path, default={}) if champion_metrics_path else None
    metrics = build_metrics_v2(
        experiment_id=experiment_id,
        seed=seed,
        split_metadata=read_json(split_path, default={}) or {},
        leakage=leakage,
        official_proxy=official_summary,
        weekly_backtest=weekly_summary,
        weekly_csv=weekly_path,
        champion_metrics=champion_metrics,
        champion_weekly_csv=champion_weekly_path,
        official_baseline_score=official_baseline_score,
        reproducibility_check="pass" if seed_count >= 1 else "not_run",
        serious=serious,
    )
    metrics_path = run_dir / "metrics_v2.json"
    write_json(metrics_path, metrics)

    gate = evaluate_champion_gate(
        metrics,
        champion_metrics=champion_metrics,
        seed_count=seed_count,
        seed_mean_excess=seed_mean_excess,
        seed_min_excess=seed_min_excess,
    )
    gate = apply_candidate_pool_decision(
        metrics,
        gate,
        champion_metrics,
        seed_mean_excess=seed_mean_excess,
        seed_min_excess=seed_min_excess,
    )
    gate_path = run_dir / "champion_gate_decision.json"
    write_json(gate_path, gate)
    build_production_refit_manifest(
        experiment_id=experiment_id,
        gate_decision=gate,
        split_metadata=split,
        champion_config=run_dir / "generated_config.json",
        output_path=run_dir / "production_refit_manifest.json",
    )
    write_run_report(run_dir, experiment_id, metrics, gate)
    return {"metrics": metrics, "gate": gate, "metrics_path": str(metrics_path), "weekly_path": str(weekly_path)}


def write_run_report(run_dir: Path, experiment_id: str, metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    official = metrics.get("official_proxy", {})
    weekly = metrics.get("weekly_backtest", {})
    risk = metrics.get("risk", {})
    lines = [
        f"# D-Series Auto Iteration Report: {experiment_id}",
        "",
        f"- Decision: `{gate.get('decision')}`",
        f"- Reason: {gate.get('reason')}",
        f"- Official baseline score: `{metrics.get('official_baseline_score')}`",
        f"- Candidate official-proxy score: `{official.get('official_proxy_final_score')}`",
        f"- Weekly cumulative net return: `{weekly.get('cumulative_return_net')}`",
        f"- Mean weekly net return: `{weekly.get('mean_weekly_return_net')}`",
        f"- Median weekly net return: `{weekly.get('median_weekly_return_net')}`",
        f"- Max drawdown abs: `{risk.get('max_drawdown_abs')}`",
        f"- Worst week: `{risk.get('worst_week')}`",
        f"- CVaR 95 loss abs: `{risk.get('CVaR_95_loss_abs')}`",
        f"- Diff without best 2 weeks: `{risk.get('diff_without_best_2_weeks')}`",
        f"- Best 2 weeks contribution ratio: `{risk.get('best_2_weeks_contribution_ratio')}`",
        "",
        "## Gate Checks",
        "",
        "|check|passed|value|threshold|",
        "|---|---:|---:|---:|",
    ]
    for check in gate.get("checks", []):
        lines.append(f"|{check.get('name')}|{check.get('passed')}|{check.get('value')}|{check.get('threshold')}|")
    (run_dir / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def failed_gate_items(gate: dict[str, Any]) -> list[str]:
    return [
        str(check.get("name"))
        for check in gate.get("checks", [])
        if check.get("hard", True) and not check.get("passed", False)
    ]


def candidate_pool_reason(
    metrics: dict[str, Any],
    gate: dict[str, Any],
    champion_metrics: dict[str, Any] | None,
    *,
    seed_mean_excess: float | None = None,
    seed_min_excess: float | None = None,
) -> dict[str, Any] | None:
    if gate.get("decision") == "ACCEPT_AS_CHAMPION":
        return None
    champion_metrics = champion_metrics or metrics.get("champion_reference") or {}
    champion_weekly = champion_metrics.get("weekly_backtest", {})
    champion_official = champion_metrics.get("official_proxy", {}).get("official_proxy_final_score")
    weekly = metrics.get("weekly_backtest", {})
    official = metrics.get("official_proxy", {})
    paired = metrics.get("paired_vs_champion", {})
    risk = metrics.get("risk", {})
    trading = metrics.get("trading", {})
    reasons: list[str] = []
    next_steps: list[str] = []

    official_score = float(official.get("official_proxy_final_score", 0.0))
    weekly_cum = float(weekly.get("cumulative_return_net", 0.0))
    champion_cum = float(champion_weekly.get("cumulative_return_net", 0.0))
    champion_mdd = float(champion_weekly.get("max_drawdown_abs", math.inf))
    champion_cvar = float(champion_weekly.get("CVaR_95_loss_abs", math.inf))
    pdr = float(paired.get("positive_diff_rate", 0.0))
    cost_drag = float(trading.get("cost_drag", 0.0))
    champion_cost = float(champion_weekly.get("cost_drag", cost_drag))
    turnover = float(trading.get("topk_turnover", 1.0))

    if champion_official is not None and official_score > float(champion_official) and weekly_cum > champion_cum:
        reasons.append("official_proxy_and_weekly_cumulative_above_champion_but_gate_failed")
        next_steps.append("rerun with paired-win-rate and seed-stability constraints")
    if pdr >= 0.48 and risk.get("max_drawdown_abs", math.inf) <= champion_mdd and risk.get("CVaR_95_loss_abs", math.inf) <= champion_cvar:
        reasons.append("positive_diff_rate_near_gate_with_better_risk")
        next_steps.append("continue as weekly-win-rate candidate")
    if seed_mean_excess is not None and seed_mean_excess > 0.0:
        reasons.append("positive_seed_mean_excess_but_seed_min_or_other_gate_failed")
        next_steps.append("stabilize with cross-seed checkpoint filtering")
    if (champion_cost > 0 and cost_drag <= champion_cost * 0.80 and weekly_cum > 0.0) or (turnover < 0.65 and weekly_cum > 0.0):
        reasons.append("improved_cost_or_turnover_but_return_not_champion_ready")
        next_steps.append("combine cost control with stronger A3 loss")

    if not reasons:
        return None
    return {
        "potential_reasons": reasons,
        "failed_gate_items": failed_gate_items(gate),
        "recommended_next_steps": next_steps,
        "suitable_for_continued_training": bool(seed_mean_excess is None or seed_mean_excess > -0.01),
        "suitable_for_ensemble": pdr >= 0.40 or weekly_cum > champion_cum,
        "suitable_as_champion_component": (
            champion_official is not None
            and official_score > float(champion_official)
            and weekly_cum > champion_cum
            and risk.get("CVaR_95_loss_abs", math.inf) <= champion_cvar * 1.10
        ),
    }


def apply_candidate_pool_decision(
    metrics: dict[str, Any],
    gate: dict[str, Any],
    champion_metrics: dict[str, Any] | None,
    *,
    seed_mean_excess: float | None = None,
    seed_min_excess: float | None = None,
) -> dict[str, Any]:
    pool = candidate_pool_reason(
        metrics,
        gate,
        champion_metrics,
        seed_mean_excess=seed_mean_excess,
        seed_min_excess=seed_min_excess,
    )
    if pool is None:
        return gate
    updated = _json_clone(gate)
    updated["decision"] = "PROMOTE_TO_CANDIDATE_POOL"
    updated["candidate_pool"] = pool
    updated["reason"] = "candidate_pool: " + "; ".join(pool["potential_reasons"])
    return updated


def registry_start(
    *,
    experiment_id: str,
    name: str,
    run_dir: Path,
    config_path: Path,
    data_path: Path,
    split: dict[str, Any],
    seed: int,
    hypothesis: str,
    status: str = "running",
) -> None:
    entry = {
        "id": experiment_id,
        "name": name,
        "path": str(run_dir.relative_to(D_ROOT)),
        "status": status,
        "hypothesis": hypothesis,
        "feature_version": "d_market_v1",
        "label_version": "open_t_plus_5_over_open_t_plus_1",
        "training_dates": f"{split['research_train']['sample_start']}..{split['research_train']['sample_end']} purged research train",
        "validation_dates": f"{split['research_validation']['sample_start']}..{split['research_validation']['sample_end']} research validation/backtest",
        "seed": seed,
        "hyperparameters": str(config_path.relative_to(D_ROOT) if config_path.is_relative_to(D_ROOT) else config_path),
        "data_path": str(data_path.resolve()),
        "data_hash": md5_file(data_path),
        "split_metadata": split,
        "artifacts": {
            "run_dir": str(run_dir.relative_to(D_ROOT)),
        },
    }
    upsert_experiment(REGISTRY, entry)


def registry_finish(experiment_id: str, run_dir: Path, result: dict[str, Any] | None, failure_reason: str | None = None) -> None:
    if result is None:
        entry = {
            "id": experiment_id,
            "status": "failed",
            "failure_reason": failure_reason or "unknown failure",
            "artifacts": {"run_dir": str(run_dir.relative_to(D_ROOT))},
        }
    else:
        metrics = result["metrics"]
        gate = result["gate"]
        entry = {
            "id": experiment_id,
            "status": gate.get("decision", "evaluated").lower(),
            "metrics": {
                "official_proxy_final_score": metrics.get("official_proxy", {}).get("official_proxy_final_score"),
                "official_baseline_score": metrics.get("official_baseline_score"),
                "weekly_cumulative_return_net": metrics.get("weekly_backtest", {}).get("cumulative_return_net"),
                "mean_weekly_return_net": metrics.get("weekly_backtest", {}).get("mean_weekly_return_net"),
                "max_drawdown_abs": metrics.get("risk", {}).get("max_drawdown_abs"),
                "CVaR_95_loss_abs": metrics.get("risk", {}).get("CVaR_95_loss_abs"),
            },
            "decision": gate.get("decision"),
            "failure_reason": gate.get("reason") if gate.get("decision") != "ACCEPT_AS_CHAMPION" else "",
            "artifacts": {
                "run_dir": str(run_dir.relative_to(D_ROOT)),
                "metrics_v2": str(Path(result["metrics_path"]).relative_to(D_ROOT)),
                "weekly_backtest": str(Path(result["weekly_path"]).relative_to(D_ROOT)),
                "champion_gate_decision": str((run_dir / "champion_gate_decision.json").relative_to(D_ROOT)),
                "report": str((run_dir / "RUN_REPORT.md").relative_to(D_ROOT)),
            },
        }
    upsert_experiment(REGISTRY, entry)


def run_single(
    *,
    experiment_id: str,
    name: str,
    hypothesis: str,
    overrides: dict[str, Any],
    seed: int,
    split: dict[str, Any],
    data_path: Path,
    python_exe: Path,
    run_root: Path,
    max_dates: int | None,
    num_epochs: int,
    patience: int,
    time_budget_minutes: float | None,
    serious: bool,
    champion_metrics_path: Path | None,
    champion_weekly_path: Path | None,
    seed_count: int = 1,
    seed_mean_excess: float | None = None,
    seed_min_excess: float | None = None,
) -> dict[str, Any] | None:
    run_dir = run_root / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_base_config(DEFAULT_CONFIG)
    effective_overrides = _json_clone(overrides)
    if champion_metrics_path:
        champion_daily_path = champion_metrics_path.resolve().parent / "official_proxy_daily.csv"
        if champion_daily_path.exists():
            deep_update(effective_overrides, {"training": {"champion_daily_path": str(champion_daily_path)}})
    config_path = write_config(
        base_config=base_config,
        split=split,
        data_path=data_path,
        experiment_id=experiment_id,
        seed=seed,
        run_dir=run_dir,
        overrides=effective_overrides,
        num_epochs=num_epochs,
        patience=patience,
    )
    snapshot_code(run_dir, config_path)
    registry_start(
        experiment_id=experiment_id,
        name=name,
        run_dir=run_dir,
        config_path=config_path,
        data_path=data_path,
        split=split,
        seed=seed,
        hypothesis=hypothesis,
    )
    started = time.time()
    returncode = run_training(
        python_exe=python_exe,
        config_path=config_path,
        run_dir=run_dir,
        max_dates=max_dates,
        time_budget_minutes=time_budget_minutes,
    )
    if returncode != 0:
        registry_finish(experiment_id, run_dir, None, failure_reason=f"training_returncode_{returncode}")
        return None
    top_k = int((read_json(config_path, default={}) or {}).get("portfolio", {}).get("top_k", 5))
    try:
        result = evaluate_run(
            experiment_id=experiment_id,
            seed=seed,
            data_path=data_path,
            run_dir=run_dir,
            split=split,
            top_k=top_k,
            python_exe=python_exe,
            serious=serious,
            champion_metrics_path=champion_metrics_path,
            champion_weekly_path=champion_weekly_path,
            seed_count=seed_count,
            seed_mean_excess=seed_mean_excess,
            seed_min_excess=seed_min_excess,
        )
        result["elapsed_minutes_total"] = (time.time() - started) / 60.0
        registry_finish(experiment_id, run_dir, result)
        return result
    except Exception as exc:  # noqa: BLE001 - record failed experiments.
        registry_finish(experiment_id, run_dir, None, failure_reason=f"evaluation_failed: {exc}")
        raise


def run_smoke(args: argparse.Namespace, split: dict[str, Any], run_root: Path) -> list[dict[str, Any] | None]:
    results: list[dict[str, Any] | None] = []
    selected_variants = VARIANTS[args.start_variant_index : args.start_variant_index + args.max_serious_rounds]
    for variant in selected_variants:
        smoke_overrides = json.loads(json.dumps(variant.overrides))
        deep_update(smoke_overrides, {"training": {"batch_size": 2}})
        results.append(
            run_single(
                experiment_id=f"{variant.id_prefix}_SMOKE_{timestamp()}",
                name=f"{variant.name}_Smoke",
                hypothesis=f"Smoke-test {variant.name}: {variant.hypothesis}",
                overrides=smoke_overrides,
                seed=args.seeds[0],
                split=split,
                data_path=args.data_path,
                python_exe=args.python_exe,
                run_root=run_root,
                max_dates=args.smoke_max_dates,
                num_epochs=1,
                patience=1,
                time_budget_minutes=args.smoke_time_budget_minutes,
                serious=False,
                champion_metrics_path=None,
                champion_weekly_path=None,
            )
        )
    return results


def run_auto(args: argparse.Namespace, split: dict[str, Any], run_root: Path) -> None:
    if args.champion_metrics_path and args.champion_weekly_path:
        champion_metrics_path = args.champion_metrics_path.resolve()
        champion_weekly_path = args.champion_weekly_path.resolve()
    else:
        champion_id = f"D_AUTO_CHAMPION_REF_{timestamp()}"
        champion = run_single(
            experiment_id=champion_id,
            name="CurrentChampionProgram_ResearchValidationReference",
            hypothesis="Re-evaluate the current D001 portfolio-return program under the requested 2025 research-validation protocol before testing challengers.",
            overrides={},
            seed=args.seeds[0],
            split=split,
            data_path=args.data_path,
            python_exe=args.python_exe,
            run_root=run_root,
            max_dates=None,
            num_epochs=args.champion_epochs,
            patience=args.champion_patience,
            time_budget_minutes=args.per_run_time_budget_minutes,
            serious=False,
            champion_metrics_path=None,
            champion_weekly_path=None,
        )
        if champion is None:
            return
        champion_metrics_path = Path(champion["metrics_path"])
        champion_weekly_path = Path(champion["weekly_path"])
    consecutive_no_beat = 0

    selected_variants = VARIANTS[args.start_variant_index : args.start_variant_index + args.max_serious_rounds]
    for variant in selected_variants:
        seed_results: list[dict[str, Any]] = []
        seed_excesses: list[float] = []
        for seed in args.seeds[:3]:
            experiment_id = f"{variant.id_prefix}_S{seed}_{timestamp()}"
            result = run_single(
                experiment_id=experiment_id,
                name=variant.name,
                hypothesis=variant.hypothesis,
                overrides=variant.overrides,
                seed=seed,
                split=split,
                data_path=args.data_path,
                python_exe=args.python_exe,
                run_root=run_root,
                max_dates=None,
                num_epochs=args.serious_epochs,
                patience=args.serious_patience,
                time_budget_minutes=args.per_run_time_budget_minutes,
                serious=True,
                champion_metrics_path=champion_metrics_path,
                champion_weekly_path=champion_weekly_path,
                seed_count=1,
            )
            if result is not None:
                seed_results.append(result)
                seed_excesses.append(float(result["metrics"].get("paired_vs_champion", {}).get("mean_weekly_excess", 0.0)))
        if not seed_results:
            consecutive_no_beat += 1
            continue
        seed_mean = float(sum(seed_excesses) / len(seed_excesses))
        seed_min = float(min(seed_excesses))
        best = max(seed_results, key=lambda item: float(item["metrics"].get("weekly_backtest", {}).get("cumulative_return_net", 0.0)))
        best_run_dir = Path(best["metrics_path"]).parent
        updated_gate = evaluate_champion_gate(
            best["metrics"],
            champion_metrics=read_json(champion_metrics_path, default={}) or {},
            seed_count=len(seed_results),
            seed_mean_excess=seed_mean,
            seed_min_excess=seed_min,
        )
        updated_gate = apply_candidate_pool_decision(
            best["metrics"],
            updated_gate,
            read_json(champion_metrics_path, default={}) or {},
            seed_mean_excess=seed_mean,
            seed_min_excess=seed_min,
        )
        write_json(best_run_dir / "champion_gate_decision.json", updated_gate)
        write_run_report(best_run_dir, best["metrics"]["experiment_id"], best["metrics"], updated_gate)
        registry_finish(best["metrics"]["experiment_id"], best_run_dir, {"metrics": best["metrics"], "gate": updated_gate, "metrics_path": best["metrics_path"], "weekly_path": best["weekly_path"]})
        if updated_gate.get("decision") == "ACCEPT_AS_CHAMPION":
            champion_metrics_path = Path(best["metrics_path"])
            champion_weekly_path = Path(best["weekly_path"])
            consecutive_no_beat = 0
        else:
            consecutive_no_beat += 1
        if consecutive_no_beat >= 3:
            break


def quick_screen_passes(metrics: dict[str, Any], champion_metrics: dict[str, Any]) -> bool:
    weekly = metrics.get("weekly_backtest", {})
    official = metrics.get("official_proxy", {})
    paired = metrics.get("paired_vs_champion", {})
    champion_weekly = champion_metrics.get("weekly_backtest", {})
    champion_official = champion_metrics.get("official_proxy", {}).get("official_proxy_final_score", 0.037120)
    return bool(
        metrics.get("leakage_audit") == "pass"
        and (
            float(official.get("official_proxy_final_score", 0.0)) > float(champion_official)
            or float(weekly.get("cumulative_return_net", 0.0)) > float(champion_weekly.get("cumulative_return_net", 2.469685))
        )
        and float(paired.get("positive_diff_rate", 0.0)) >= 0.45
        and float(weekly.get("max_drawdown_abs", math.inf)) <= 0.330575
        and float(weekly.get("CVaR_95_loss_abs", math.inf)) <= 0.122105
        and float(paired.get("diff_without_best_2_weeks", -math.inf)) >= -0.05
    )


def quick_screen_rank(metrics: dict[str, Any], champion_metrics: dict[str, Any]) -> float:
    weekly = metrics.get("weekly_backtest", {})
    official = metrics.get("official_proxy", {})
    paired = metrics.get("paired_vs_champion", {})
    risk = metrics.get("risk", {})
    trading = metrics.get("trading", {})
    champion_weekly = champion_metrics.get("weekly_backtest", {})
    champion_official = float(champion_metrics.get("official_proxy", {}).get("official_proxy_final_score", 0.037120))
    return float(
        8.0 * (official.get("official_proxy_final_score", 0.0) - champion_official)
        + 0.35 * (weekly.get("cumulative_return_net", 0.0) - champion_weekly.get("cumulative_return_net", 0.0))
        + 2.0 * paired.get("mean_weekly_excess", 0.0)
        + 0.25 * paired.get("positive_diff_rate", 0.0)
        + 0.05 * paired.get("diff_without_best_2_weeks", 0.0)
        - 0.20 * max(0.0, risk.get("CVaR_95_loss_abs", 0.0) - champion_weekly.get("CVaR_95_loss_abs", 0.0))
        - 0.02 * trading.get("topk_turnover", 1.0)
    )


def write_candidate_pool_summary(run_root: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(run_root.glob("D_AUTO_R3_*")):
        gate_path = run_dir / "champion_gate_decision.json"
        metrics_path = run_dir / "metrics_v2.json"
        if not gate_path.exists() or not metrics_path.exists():
            continue
        gate = read_json(gate_path, default={}) or {}
        if gate.get("decision") != "PROMOTE_TO_CANDIDATE_POOL":
            continue
        metrics = read_json(metrics_path, default={}) or {}
        weekly = metrics.get("weekly_backtest", {})
        paired = metrics.get("paired_vs_champion", {})
        trading = metrics.get("trading", {})
        pool = gate.get("candidate_pool", {})
        rows.append(
            {
                "experiment_id": metrics.get("experiment_id", run_dir.name),
                "decision": gate.get("decision"),
                "official_proxy": metrics.get("official_proxy", {}).get("official_proxy_final_score"),
                "weekly_cumulative_net": weekly.get("cumulative_return_net"),
                "mean_weekly_excess": paired.get("mean_weekly_excess"),
                "median_weekly_excess": paired.get("median_weekly_excess"),
                "positive_diff_rate": paired.get("positive_diff_rate"),
                "diff_without_best_2_weeks": paired.get("diff_without_best_2_weeks"),
                "max_drawdown_abs": weekly.get("max_drawdown_abs"),
                "CVaR_95_loss_abs": weekly.get("CVaR_95_loss_abs"),
                "cost_drag": trading.get("cost_drag"),
                "gross_improvement": trading.get("gross_improvement"),
                "topk_turnover": trading.get("topk_turnover"),
                "average_holding_overlap_rate": trading.get("average_holding_overlap_rate"),
                "potential_reasons": "; ".join(pool.get("potential_reasons", [])),
                "failed_gate_items": "; ".join(pool.get("failed_gate_items", [])),
                "recommended_next_steps": "; ".join(pool.get("recommended_next_steps", [])),
                "suitable_for_continued_training": pool.get("suitable_for_continued_training"),
                "suitable_for_ensemble": pool.get("suitable_for_ensemble"),
                "suitable_as_champion_component": pool.get("suitable_as_champion_component"),
                "run_dir": str(run_dir.relative_to(D_ROOT)),
            }
        )
    output = run_root / "R3_candidate_pool_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8")
    return output


def run_r3(args: argparse.Namespace, split: dict[str, Any], run_root: Path) -> None:
    if not args.champion_metrics_path or not args.champion_weekly_path:
        raise ValueError("R3 requires --champion-metrics-path and --champion-weekly-path")
    champion_metrics_path = args.champion_metrics_path.resolve()
    champion_weekly_path = args.champion_weekly_path.resolve()
    champion_metrics = read_json(champion_metrics_path, default={}) or {}
    variants = R3_VARIANTS[args.start_variant_index : args.start_variant_index + args.quick_variant_count]
    quick_results: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None

    for variant in variants:
        experiment_id = f"{variant.id_prefix}_Q_S{args.seeds[0]}_{timestamp()}"
        result = run_single(
            experiment_id=experiment_id,
            name=f"{variant.name}_QuickScreen",
            hypothesis=variant.hypothesis,
            overrides=variant.overrides,
            seed=args.seeds[0],
            split=split,
            data_path=args.data_path,
            python_exe=args.python_exe,
            run_root=run_root,
            max_dates=None,
            num_epochs=args.quick_epochs,
            patience=args.quick_patience,
            time_budget_minutes=args.quick_time_budget_minutes,
            serious=False,
            champion_metrics_path=champion_metrics_path,
            champion_weekly_path=champion_weekly_path,
            seed_count=1,
        )
        if result is None:
            continue
        metrics = result["metrics"]
        passed = quick_screen_passes(metrics, champion_metrics)
        ranked = quick_screen_rank(metrics, champion_metrics)
        quick_results.append({"variant": variant, "result": result, "passed": passed, "rank": ranked})
        if result["gate"].get("decision") == "ACCEPT_AS_CHAMPION":
            accepted = result
            break

    selected = [item for item in quick_results if item["passed"]]
    selected = sorted(selected, key=lambda item: float(item["rank"]), reverse=True)
    if len(selected) < args.serious_top_n:
        seen = {item["variant"].id_prefix for item in selected}
        fillers = [
            item
            for item in sorted(quick_results, key=lambda item: float(item["rank"]), reverse=True)
            if item["variant"].id_prefix not in seen
        ]
        selected.extend(fillers[: max(0, args.serious_top_n - len(selected))])
    selected = selected[: max(args.serious_top_n, min(args.serious_top_max, len(selected)))]
    selected = selected[: args.serious_top_max]

    serious_results: list[dict[str, Any]] = []
    if accepted is None:
        for selected_item in selected:
            variant = selected_item["variant"]
            seed_results: list[dict[str, Any]] = []
            seed_excesses: list[float] = []
            for seed in args.seeds[:3]:
                experiment_id = f"{variant.id_prefix}_SERIOUS_S{seed}_{timestamp()}"
                result = run_single(
                    experiment_id=experiment_id,
                    name=f"{variant.name}_Serious",
                    hypothesis=variant.hypothesis,
                    overrides=variant.overrides,
                    seed=seed,
                    split=split,
                    data_path=args.data_path,
                    python_exe=args.python_exe,
                    run_root=run_root,
                    max_dates=None,
                    num_epochs=args.serious_epochs,
                    patience=args.serious_patience,
                    time_budget_minutes=args.per_run_time_budget_minutes,
                    serious=True,
                    champion_metrics_path=champion_metrics_path,
                    champion_weekly_path=champion_weekly_path,
                    seed_count=1,
                )
                if result is None:
                    continue
                seed_results.append(result)
                seed_excesses.append(float(result["metrics"].get("paired_vs_champion", {}).get("mean_weekly_excess", 0.0)))
            if not seed_results:
                continue
            seed_mean = float(sum(seed_excesses) / len(seed_excesses))
            seed_min = float(min(seed_excesses))
            best = max(seed_results, key=lambda item: float(item["metrics"].get("weekly_backtest", {}).get("cumulative_return_net", 0.0)))
            best_run_dir = Path(best["metrics_path"]).parent
            updated_gate = evaluate_champion_gate(
                best["metrics"],
                champion_metrics=champion_metrics,
                seed_count=len(seed_results),
                seed_mean_excess=seed_mean,
                seed_min_excess=seed_min,
            )
            updated_gate = apply_candidate_pool_decision(
                best["metrics"],
                updated_gate,
                champion_metrics,
                seed_mean_excess=seed_mean,
                seed_min_excess=seed_min,
            )
            write_json(best_run_dir / "champion_gate_decision.json", updated_gate)
            write_run_report(best_run_dir, best["metrics"]["experiment_id"], best["metrics"], updated_gate)
            registry_finish(
                best["metrics"]["experiment_id"],
                best_run_dir,
                {"metrics": best["metrics"], "gate": updated_gate, "metrics_path": best["metrics_path"], "weekly_path": best["weekly_path"]},
            )
            serious_results.append(
                {
                    "variant": variant.id_prefix,
                    "seed_count": len(seed_results),
                    "seed_mean_excess": seed_mean,
                    "seed_min_excess": seed_min,
                    "best_experiment_id": best["metrics"]["experiment_id"],
                    "best_metrics_path": best["metrics_path"],
                    "decision": updated_gate.get("decision"),
                }
            )
            if updated_gate.get("decision") == "ACCEPT_AS_CHAMPION":
                accepted = best
                break

    candidate_pool_path = write_candidate_pool_summary(run_root)
    summary = {
        "mode": "r3_auto",
        "quick_variant_count": len(quick_results),
        "quick_screen_passed_count": sum(1 for item in quick_results if item["passed"]),
        "quick_results": [
            {
                "variant": item["variant"].id_prefix,
                "experiment_id": item["result"]["metrics"]["experiment_id"],
                "passed": item["passed"],
                "rank": item["rank"],
                "decision": item["result"]["gate"].get("decision"),
            }
            for item in quick_results
        ],
        "selected_for_serious": [item["variant"].id_prefix for item in selected],
        "serious_results": serious_results,
        "accepted_experiment_id": None if accepted is None else accepted["metrics"]["experiment_id"],
        "candidate_pool_summary": str(candidate_pool_path.relative_to(D_ROOT)),
    }
    write_json(run_root / f"R3_auto_summary_{timestamp()}.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-series automated model iteration with Champion Gate.")
    default_python = ROOT / "Experimental_WorkSpace" / "shared_envs" / "pytorch-cu128-py311" / "Scripts" / "python.exe"
    parser.add_argument("--mode", choices=["smoke", "auto", "smoke-and-auto", "r3-smoke", "r3-auto"], default="smoke")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--python-exe", type=Path, default=default_python)
    parser.add_argument("--run-root", type=Path, default=D_ROOT / "automation_runs")
    parser.add_argument("--smoke-max-dates", type=int, default=140)
    parser.add_argument("--smoke-time-budget-minutes", type=float, default=20.0)
    parser.add_argument("--per-run-time-budget-minutes", type=float, default=75.0)
    parser.add_argument("--champion-epochs", type=int, default=12)
    parser.add_argument("--champion-patience", type=int, default=4)
    parser.add_argument("--serious-epochs", type=int, default=12)
    parser.add_argument("--serious-patience", type=int, default=4)
    parser.add_argument("--max-serious-rounds", type=int, default=3)
    parser.add_argument("--quick-variant-count", type=int, default=12)
    parser.add_argument("--quick-epochs", type=int, default=5)
    parser.add_argument("--quick-patience", type=int, default=2)
    parser.add_argument("--quick-time-budget-minutes", type=float, default=20.0)
    parser.add_argument("--serious-top-n", type=int, default=3)
    parser.add_argument("--serious-top-max", type=int, default=5)
    parser.add_argument("--start-variant-index", type=int, default=0)
    parser.add_argument("--champion-metrics-path", type=Path, default=None)
    parser.add_argument("--champion-weekly-path", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260708, 20260709, 20260710])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data_path = args.data_path.resolve()
    args.python_exe = args.python_exe.resolve()
    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    split = build_research_split(args.data_path)
    split_path = args.run_root / f"split_metadata_{timestamp()}.json"
    write_json(split_path, split)
    if args.mode in {"smoke", "smoke-and-auto"}:
        run_smoke(args, split, args.run_root)
    if args.mode in {"auto", "smoke-and-auto"}:
        run_auto(args, split, args.run_root)
    if args.mode == "r3-smoke":
        original_variants = list(VARIANTS)
        try:
            globals()["VARIANTS"] = R3_VARIANTS
            run_smoke(args, split, args.run_root)
        finally:
            globals()["VARIANTS"] = original_variants
    if args.mode == "r3-auto":
        run_r3(args, split, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

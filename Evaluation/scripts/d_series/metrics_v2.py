from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import read_json, safe_div, write_json


def _paired_weekly(challenger_weekly: pd.DataFrame, champion_weekly: pd.DataFrame) -> dict[str, Any]:
    if challenger_weekly.empty or champion_weekly.empty:
        return {}
    left = challenger_weekly[challenger_weekly.get("skipped_missing_decision_prediction", 0) == 0].copy()
    right = champion_weekly[champion_weekly.get("skipped_missing_decision_prediction", 0) == 0].copy()
    merged = left.merge(
        right[["week_start", "weekly_return_net"]],
        on="week_start",
        how="inner",
        suffixes=("_challenger", "_champion"),
    )
    if merged.empty:
        return {}
    diff = merged["weekly_return_net_challenger"].astype(float) - merged["weekly_return_net_champion"].astype(float)
    sorted_diff = np.sort(diff.to_numpy(dtype=np.float64))[::-1]
    best2 = float(np.sum(sorted_diff[: min(2, len(sorted_diff))]))
    total_positive = float(np.sum(diff[diff > 0.0]))
    return {
        "paired_week_count": int(len(merged)),
        "mean_weekly_excess": float(diff.mean()),
        "median_weekly_excess": float(diff.median()),
        "positive_diff_rate": float((diff > 0.0).mean()),
        "diff_without_best_2_weeks": float(diff.sum() - best2),
        "best_2_weeks_contribution_ratio": safe_div(best2, total_positive, default=0.0) if total_positive > 0 else 0.0,
        "excess_sum": float(diff.sum()),
    }


def build_metrics_v2(
    *,
    experiment_id: str,
    seed: int | None,
    split_metadata: dict[str, Any],
    leakage: dict[str, Any],
    official_proxy: dict[str, Any],
    weekly_backtest: dict[str, Any],
    weekly_csv: str | Path | None = None,
    champion_metrics: dict[str, Any] | None = None,
    champion_weekly_csv: str | Path | None = None,
    official_baseline_score: float | None = None,
    reproducibility_check: str = "not_run",
    serious: bool = False,
) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    if weekly_csv and champion_weekly_csv and Path(weekly_csv).exists() and Path(champion_weekly_csv).exists():
        paired = _paired_weekly(pd.read_csv(weekly_csv), pd.read_csv(champion_weekly_csv))
    official_score = float(official_proxy.get("official_proxy_final_score", 0.0))
    gross_improvement = None
    gross_improvement_vs_official_baseline = None
    if official_baseline_score is not None:
        gross_improvement_vs_official_baseline = official_score - float(official_baseline_score)
        gross_improvement = gross_improvement_vs_official_baseline
    champion_official_score = None
    if champion_metrics:
        try:
            champion_official_score = float(champion_metrics.get("official_proxy", {}).get("official_proxy_final_score"))
            gross_improvement = official_score - champion_official_score
        except (TypeError, ValueError):
            champion_official_score = None
    return {
        "schema_version": "metrics_v2",
        "experiment_id": experiment_id,
        "seed": seed,
        "serious_challenger": serious,
        "split": split_metadata,
        "leakage_audit": leakage.get("leakage_audit", "unknown"),
        "reproducibility_check": reproducibility_check,
        "official_baseline_score": official_baseline_score,
        "champion_official_proxy_score": champion_official_score,
        "official_proxy": official_proxy,
        "weekly_backtest": weekly_backtest,
        "risk": {
            "max_drawdown_abs": weekly_backtest.get("max_drawdown_abs", 0.0),
            "worst_week": weekly_backtest.get("worst_week", 0.0),
            "CVaR_95_loss_abs": weekly_backtest.get("CVaR_95_loss_abs", 0.0),
            "worst_4_week_cumulative_return": weekly_backtest.get("worst_4_week_cumulative_return", 0.0),
            "diff_without_best_2_weeks": paired.get("diff_without_best_2_weeks"),
            "best_2_weeks_contribution_ratio": paired.get("best_2_weeks_contribution_ratio"),
        },
        "paired_vs_champion": paired,
        "champion_reference": champion_metrics or {},
        "trading": {
            "cost_drag": weekly_backtest.get("cost_drag", 0.0),
            "gross_improvement": gross_improvement,
            "gross_improvement_vs_champion": (
                None if champion_official_score is None else official_score - champion_official_score
            ),
            "gross_improvement_vs_official_baseline": gross_improvement_vs_official_baseline,
            "cost_drag_to_gross_improvement": safe_div(float(weekly_backtest.get("cost_drag", 0.0)), gross_improvement or 0.0),
            "topk_turnover": weekly_backtest.get("topk_turnover", 0.0),
            "average_holding_overlap_rate": weekly_backtest.get("average_holding_overlap_rate", 0.0),
            "average_weight_sum": weekly_backtest.get("average_weight_sum", 0.0),
            "cash_ratio": weekly_backtest.get("cash_ratio", 0.0),
            "net_improvement_vs_champion": (
                None
                if not champion_metrics
                else weekly_backtest.get("cumulative_return_net", 0.0)
                - champion_metrics.get("weekly_backtest", {}).get("cumulative_return_net", 0.0)
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D-series metrics_v2.json.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-metadata", required=True)
    parser.add_argument("--leakage", required=True)
    parser.add_argument("--official-proxy", required=True)
    parser.add_argument("--weekly-backtest", required=True)
    parser.add_argument("--weekly-csv", default="")
    parser.add_argument("--champion-metrics", default="")
    parser.add_argument("--champion-weekly-csv", default="")
    parser.add_argument("--official-baseline-score", type=float, default=None)
    parser.add_argument("--reproducibility-check", default="not_run")
    parser.add_argument("--serious", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_metrics_v2(
        experiment_id=args.experiment_id,
        seed=args.seed,
        split_metadata=read_json(args.split_metadata, default={}) or {},
        leakage=read_json(args.leakage, default={}) or {},
        official_proxy=read_json(args.official_proxy, default={}) or {},
        weekly_backtest=read_json(args.weekly_backtest, default={}) or {},
        weekly_csv=args.weekly_csv or None,
        champion_metrics=read_json(args.champion_metrics, default={}) if args.champion_metrics else None,
        champion_weekly_csv=args.champion_weekly_csv or None,
        official_baseline_score=args.official_baseline_score,
        reproducibility_check=args.reproducibility_check,
        serious=args.serious,
    )
    write_json(args.output, payload)


if __name__ == "__main__":
    main()

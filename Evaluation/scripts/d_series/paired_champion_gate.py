from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import read_json, write_json


def evaluate_champion_gate(
    metrics: dict[str, Any],
    *,
    champion_metrics: dict[str, Any] | None = None,
    seed_count: int = 1,
    seed_mean_excess: float | None = None,
    seed_min_excess: float | None = None,
) -> dict[str, Any]:
    champion_metrics = champion_metrics or metrics.get("champion_reference") or {}
    official_proxy = metrics.get("official_proxy", {})
    weekly = metrics.get("weekly_backtest", {})
    paired = metrics.get("paired_vs_champion", {})
    trading = metrics.get("trading", {})
    risk = metrics.get("risk", {})
    champion_weekly = champion_metrics.get("weekly_backtest", {})

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, value: Any = None, threshold: Any = None, hard: bool = True) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value, "threshold": threshold, "hard": hard})

    add("leakage_audit_pass", metrics.get("leakage_audit") == "pass", metrics.get("leakage_audit"))
    add("reproducibility_check_pass", metrics.get("reproducibility_check") == "pass", metrics.get("reproducibility_check"))
    add("same_protocol_declared", True, "fixed by orchestrator")
    baseline_score = metrics.get("official_baseline_score")
    candidate_score = official_proxy.get("official_proxy_final_score", 0.0)
    add("exceeds_official_baseline", baseline_score is not None and candidate_score > baseline_score, candidate_score, baseline_score)
    champion_official_score = metrics.get("champion_official_proxy_score")
    if champion_official_score is None:
        try:
            champion_official_score = champion_metrics.get("official_proxy", {}).get("official_proxy_final_score")
        except AttributeError:
            champion_official_score = None
    add(
        "official_proxy_gt_champion",
        champion_official_score is not None and candidate_score > champion_official_score,
        candidate_score,
        champion_official_score,
    )

    champion_return = champion_weekly.get("cumulative_return_net", float("-inf"))
    add("net_cumulative_return_gt_champion", weekly.get("cumulative_return_net", 0.0) > champion_return, weekly.get("cumulative_return_net"), champion_return)
    add("mean_weekly_excess", paired.get("mean_weekly_excess", 0.0) >= 0.0010, paired.get("mean_weekly_excess"), 0.0010)
    add("median_weekly_excess", paired.get("median_weekly_excess", 0.0) >= 0.0, paired.get("median_weekly_excess"), 0.0)
    add("positive_diff_rate", paired.get("positive_diff_rate", 0.0) >= 0.52, paired.get("positive_diff_rate"), 0.52)

    champion_mdd = champion_weekly.get("max_drawdown_abs", float("inf"))
    champion_cvar = champion_weekly.get("CVaR_95_loss_abs", float("inf"))
    add("max_drawdown_abs", weekly.get("max_drawdown_abs", 0.0) <= champion_mdd * 1.10, weekly.get("max_drawdown_abs"), champion_mdd * 1.10)
    add("CVaR_95_loss_abs", weekly.get("CVaR_95_loss_abs", 0.0) <= champion_cvar * 1.10, weekly.get("CVaR_95_loss_abs"), champion_cvar * 1.10)
    add("diff_without_best_2_weeks", risk.get("diff_without_best_2_weeks", -1.0) is not None and risk.get("diff_without_best_2_weeks", -1.0) >= 0.0, risk.get("diff_without_best_2_weeks"), 0.0)
    best2_ok = risk.get("best_2_weeks_contribution_ratio", 1.0) is not None and risk.get("best_2_weeks_contribution_ratio", 1.0) <= 0.50
    add("best_2_weeks_contribution_ratio", best2_ok, risk.get("best_2_weeks_contribution_ratio"), 0.50)

    serious = bool(metrics.get("serious_challenger"))
    add("serious_challenger_3_seeds", serious and seed_count >= 3, seed_count, 3)
    add("seed_mean_excess_positive", seed_mean_excess is not None and seed_mean_excess > 0.0, seed_mean_excess, 0.0)
    add("seed_min_excess_not_materially_bad", seed_min_excess is not None and seed_min_excess > -0.005, seed_min_excess, -0.005)
    gross_improvement = trading.get("gross_improvement")
    cost_drag = trading.get("cost_drag", 0.0)
    add("cost_drag_le_half_gross_improvement", gross_improvement is not None and gross_improvement > 0 and cost_drag <= gross_improvement * 0.50, cost_drag, None if gross_improvement is None else gross_improvement * 0.50)

    failed_hard = [check for check in checks if check["hard"] and not check["passed"]]
    failed_names = [check["name"] for check in failed_hard]
    if not failed_hard:
        decision = "ACCEPT_AS_CHAMPION"
        reason = "all Champion Gate checks passed"
    elif failed_names == ["best_2_weeks_contribution_ratio"]:
        decision = "PROMOTE_TO_CANDIDATE_POOL"
        reason = "strong result but too dependent on best two weeks"
    else:
        decision = "REJECT"
        reason = "; ".join(failed_names)
    return {
        "decision": decision,
        "reason": reason,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply D-series paired Champion Gate.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--champion-metrics", default="")
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--seed-mean-excess", type=float, default=None)
    parser.add_argument("--seed-min-excess", type=float, default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json(
        args.output,
        evaluate_champion_gate(
            read_json(args.metrics, default={}) or {},
            champion_metrics=read_json(args.champion_metrics, default={}) if args.champion_metrics else None,
            seed_count=args.seed_count,
            seed_mean_excess=args.seed_mean_excess,
            seed_min_excess=args.seed_min_excess,
        ),
    )


if __name__ == "__main__":
    main()

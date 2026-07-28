from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Evaluation.scripts.d_series.r4_protocol import (
    aggregate_predictions,
    apply_portfolio_policy,
    cost_semantics_audit,
    evaluate_r4_gate,
    paired_weekly_metrics,
)


def prediction_frame(offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for day_index, date in enumerate(["2025-01-03", "2025-01-10", "2025-01-17"]):
        for stock_index in range(6):
            rows.append(
                {
                    "sample_end_date": date,
                    "stock_id": f"{stock_index + 1:06d}",
                    "score": float(6 - stock_index + offset + day_index * 0.01),
                    "label": float(0.03 - stock_index * 0.01),
                }
            )
    return pd.DataFrame(rows)


def weekly_frame(returns: list[float], weight_sum: float = 1.0) -> pd.DataFrame:
    rows = []
    for index, value in enumerate(returns):
        cost = 0.0015 * weight_sum
        rows.append(
            {
                "week_start": f"2025-01-{index + 6:02d}",
                "buy_date": f"2025-01-{index + 6:02d}",
                "sell_date": f"2025-01-{index + 6:02d}",
                "weekly_return_gross": value + cost,
                "weekly_return_net": value,
                "cost": cost,
                "weight_sum": weight_sum,
                "selected_count": 2,
                "skipped_missing_decision_prediction": 0,
            }
        )
    return pd.DataFrame(rows)


class R4ProtocolTests(unittest.TestCase):
    def test_all_portfolio_policies_are_fully_invested_and_bounded(self) -> None:
        source = prediction_frame()
        policies = [
            {"mode": "top2_equal"},
            {"mode": "dynamic_gap", "top2_gap_threshold": 0.2, "top3_gap_threshold": 0.1},
            {"mode": "temperature_softmax", "top_k": 3, "temperature": 1.0, "max_weight": 0.5},
            {"mode": "rank_hysteresis", "top_k": 2, "candidate_pool": 5, "margin": 0.1},
        ]
        for policy in policies:
            weighted = apply_portfolio_policy(source, policy)
            sums = weighted.groupby("sample_end_date")["portfolio_weight"].sum()
            counts = weighted.groupby("sample_end_date")["portfolio_weight"].apply(lambda values: int((values > 0).sum()))
            self.assertTrue(np.allclose(sums.to_numpy(), 1.0, atol=1e-10), policy)
            self.assertTrue((counts <= 5).all(), policy)
            self.assertLessEqual(float(weighted["portfolio_weight"].max()), 0.5 + 1e-10)

    def test_all_consensus_methods_preserve_alignment(self) -> None:
        frames = [prediction_frame(offset=index * 0.1) for index in range(5)]
        for method in [
            "mean_percentile_rank",
            "trimmed_mean_standardized",
            "borda_rank_average",
            "disagreement_penalized_consensus",
        ]:
            consensus = aggregate_predictions(frames, method=method)
            self.assertEqual(len(consensus), len(frames[0]))
            self.assertTrue(np.isfinite(consensus["score"]).all())
            self.assertTrue((consensus["ensemble_size"] == 5).all())

    def test_cost_semantics_accepts_equal_full_investment_and_rejects_cash(self) -> None:
        champion = weekly_frame([0.01, -0.02, 0.03])
        challenger = weekly_frame([0.02, -0.01, 0.04])
        self.assertEqual(cost_semantics_audit(challenger, champion)["status"], "pass")
        cash = weekly_frame([0.02, -0.01, 0.04], weight_sum=0.9)
        self.assertEqual(cost_semantics_audit(cash, champion)["status"], "fail")

    def test_paired_metrics_and_gate_thresholds(self) -> None:
        champion_weekly = weekly_frame([0.01] * 13)
        challenger_weekly = weekly_frame([0.012] * 13)
        holdout = paired_weekly_metrics(challenger_weekly, champion_weekly)
        full_metrics = {
            "official_proxy": {"official_proxy_final_score": 0.04},
            "weekly_backtest": {
                "cumulative_return_net": 2.6,
                "max_drawdown_abs": 0.29,
                "CVaR_95_loss_abs": 0.10,
                "worst_week": -0.13,
                "number_of_selected_stocks": 2.0,
                "average_weight_sum": 1.0,
            },
            "paired_vs_champion": {
                "positive_diff_rate": 0.60,
                "mean_weekly_excess": 0.002,
                "diff_without_best_2_weeks": 0.02,
            },
        }
        seed_results = [
            {"official_proxy": 0.04, "weekly_cumulative": 2.6, "mean_weekly_excess": 0.002},
            {"official_proxy": 0.039, "weekly_cumulative": 2.55, "mean_weekly_excess": 0.001},
            {"official_proxy": 0.038, "weekly_cumulative": 2.5, "mean_weekly_excess": 0.001},
        ]
        cost = cost_semantics_audit(weekly_frame([0.01] * 13), weekly_frame([0.01] * 13))
        gate = evaluate_r4_gate(
            full_metrics=full_metrics,
            sealed_holdout_paired=holdout,
            seed_results=seed_results,
            leakage_pass=True,
            reproducibility_pass=True,
            cost_audit=cost,
            cold_start_prediction_seconds=30.0,
        )
        self.assertEqual(gate["decision"], "ACCEPT_AS_CHAMPION")


if __name__ == "__main__":
    unittest.main()

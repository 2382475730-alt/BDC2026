from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Evaluation.scripts.d_series import defensive_combo_v2 as v2
from Evaluation.scripts.d_series.defensive_combo_v1 import MarketData, Policy


class StubMarket:
    def snapshot(
        self, decision_date: pd.Timestamp, stocks: list[str]
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict[str, float]]:
        del decision_date
        count = len(stocks)
        if count <= 20:
            values = np.full((count, count), 0.90)
        else:
            values = np.zeros((count, count))
            for left in range(count):
                for right in range(count):
                    if left % 3 == right % 3:
                        values[left, right] = 0.90
        np.fill_diagonal(values, 1.0)
        correlation = pd.DataFrame(values, index=stocks, columns=stocks)
        volatility = pd.Series(
            np.linspace(0.01, 0.03, count), index=stocks
        )
        reversal = pd.Series(np.linspace(0.0, 1.0, count), index=stocks)
        raw_state = {field: 0.0 for field in v2.v1.STATE_FIELDS}
        return volatility, reversal, correlation, raw_state


class DefensiveComboV2Tests(unittest.TestCase):
    def test_complete_link_clusters_do_not_chain_weak_endpoints(self) -> None:
        stocks = ["000001", "000002", "000003"]
        correlation = pd.DataFrame(
            [
                [1.0, 0.90, 0.40],
                [0.90, 1.0, 0.90],
                [0.40, 0.90, 1.0],
            ],
            index=stocks,
            columns=stocks,
        )
        clusters = v2.complete_clusters(correlation)
        groups: dict[str, list[str]] = {}
        for stock, cluster in clusters.items():
            groups.setdefault(cluster, []).append(stock)
        for members in groups.values():
            for left in members:
                for right in members:
                    self.assertGreaterEqual(
                        correlation.loc[left, right],
                        v2.v1.CORRELATION_CLUSTER_CUT,
                    )
        self.assertNotEqual(clusters["000001"], clusters["000003"])

    def test_complete_clusters_are_order_invariant(self) -> None:
        stocks = ["000003", "000001", "000004", "000002"]
        values = np.array(
            [
                [1.0, 0.1, 0.8, 0.1],
                [0.1, 1.0, 0.1, 0.8],
                [0.8, 0.1, 1.0, 0.1],
                [0.1, 0.8, 0.1, 1.0],
            ]
        )
        correlation = pd.DataFrame(values, index=stocks, columns=stocks)
        first = v2.complete_clusters(correlation)
        ordered = sorted(stocks)
        second = v2.complete_clusters(
            correlation.reindex(index=ordered, columns=ordered)
        )
        self.assertEqual(first, second)

    def test_candidate_pool_steps_are_deterministic(self) -> None:
        self.assertEqual(v2.candidate_pool_sizes(300), [20, 50, 100, 300])
        self.assertEqual(v2.candidate_pool_sizes(80), [20, 50, 80])
        self.assertEqual(v2.candidate_pool_sizes(20), [20])

    def test_context_expands_from_top20_to_top50(self) -> None:
        stocks = [f"{index:06d}" for index in range(1, 61)]
        day = pd.DataFrame(
            {
                "marker": "TEST",
                "sample_end_date": pd.Timestamp("2025-01-03"),
                "stock_id": stocks,
                "score": np.linspace(2.0, 1.0, len(stocks)),
            }
        )
        history = {field: [] for field in v2.v1.STATE_FIELDS}
        context = v2.build_day_context(
            pd.Timestamp("2025-01-03"), day, StubMarket(), history
        )
        self.assertEqual(context.pool_size_used, 50)
        self.assertGreaterEqual(context.feasible_capacity, 5)

    def test_defensive_selection_is_full_and_deterministic(self) -> None:
        stocks = [f"{index:06d}" for index in range(1, 61)]
        day = pd.DataFrame(
            {
                "marker": "TEST",
                "sample_end_date": pd.Timestamp("2025-01-03"),
                "stock_id": stocks,
                "score": np.linspace(2.0, 1.0, len(stocks)),
            }
        )
        history = {field: [] for field in v2.v1.STATE_FIELDS}
        context = v2.build_day_context(
            pd.Timestamp("2025-01-03"), day, StubMarket(), history
        )
        policy = Policy("TOP5_RANK_30_25_20_15_10", 0.1, 0.1, 0.05, 0.1)
        first = v2.select_defensive(context, policy, {})
        second = v2.select_defensive(context, policy, {})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertAlmostEqual(sum(item["weight"] for item in first), 1.0)
        counts = pd.Series([item["cluster_id"] for item in first]).value_counts()
        self.assertLessEqual(int(counts.max()), 2)

    def test_future_market_rows_cannot_change_lag1_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "market.csv"
            dates = pd.bdate_range("2024-01-02", periods=90)
            stocks = ["000001", "000002", "000003"]
            base = pd.DataFrame(
                [
                    {
                        "股票代码": stock,
                        "日期": date,
                        "收盘": 10.0
                        + stock_index
                        + day_index * (0.01 + stock_index * 0.002),
                    }
                    for day_index, date in enumerate(dates)
                    for stock_index, stock in enumerate(stocks)
                ]
            )
            decision_date = dates[75]
            base.to_csv(path, index=False)
            before = MarketData(path).snapshot(decision_date, stocks)
            future = pd.DataFrame(
                [
                    {
                        "股票代码": stock,
                        "日期": date,
                        "收盘": 1000.0 + stock_index * 100.0 + day_index * 50.0,
                    }
                    for day_index, date in enumerate(dates[76:])
                    for stock_index, stock in enumerate(stocks)
                ]
            )
            pd.concat(
                [base.loc[base["日期"] <= dates[75]], future],
                ignore_index=True,
            ).to_csv(path, index=False)
            after = MarketData(path).snapshot(decision_date, stocks)
        pd.testing.assert_series_equal(before[0], after[0])
        pd.testing.assert_series_equal(before[1], after[1])
        pd.testing.assert_frame_equal(before[2], after[2])
        self.assertEqual(before[3], after[3])

    def test_rank_projection_hash_ignores_label_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            frame = pd.DataFrame(
                {
                    "marker": ["TEST", "TEST"],
                    "sample_end_date": ["2025-01-03", "2025-01-03"],
                    "stock_id": ["000001", "000002"],
                    "score": [0.9, 0.8],
                    "label": [0.1, -0.1],
                }
            )
            frame.to_csv(path, index=False)
            first = v2.sha256_bytes(v2.rank_projection_bytes(path))
            frame["label"] = [99.0, -99.0]
            frame.to_csv(path, index=False)
            second = v2.sha256_bytes(v2.rank_projection_bytes(path))
        self.assertEqual(first, second)

    def test_global_policy_tie_prefers_equal_then_lower_id(self) -> None:
        rows = []
        for policy_id, equal_weight in (
            ("FX_V0.000_C0.000_T0.000_R0.000", False),
            ("EQ_V0.100_C0.100_T0.000_R0.000", True),
            ("EQ_V0.000_C0.000_T0.000_R0.000", True),
        ):
            rows.append(
                {
                    "policy_id": policy_id,
                    "candidate_positive_rate": 0.6,
                    "candidate_median": 0.01,
                    "candidate_cvar20": -0.01,
                    "candidate_worst": -0.02,
                    "excess_sum_remove_best2": 0.03,
                    "mean_cross_week_turnover": 0.5,
                    "mean_pair_correlation": 0.2,
                    "total_penalty": 0.0,
                    "equal_weight_template": equal_weight,
                    "eligible": True,
                }
            )
        winner = v2.choose_global_policy(pd.DataFrame(rows))
        self.assertEqual(
            winner["policy_id"], "EQ_V0.000_C0.000_T0.000_R0.000"
        )


if __name__ == "__main__":
    unittest.main()

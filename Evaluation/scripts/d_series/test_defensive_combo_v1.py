from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Evaluation.scripts.d_series.defensive_combo_v1 import (
    DayContext,
    MarketData,
    Policy,
    canonical_clusters,
    decision_dates,
    moving_block_bootstrap_lower,
    select_defensive,
    sha256_file,
    transition_turnover,
)


class DefensiveComboV1Tests(unittest.TestCase):
    def context(self) -> DayContext:
        stocks = [f"{index:06d}" for index in range(1, 21)]
        candidates = pd.DataFrame(
            {
                "stock_id": stocks,
                "score": np.linspace(2.0, 1.0, 20),
                "d_rank_score": np.linspace(1.0, 0.0, 20),
                "volatility_rank": np.linspace(0.0, 1.0, 20),
                "reversal_risk_rank": np.linspace(1.0, 0.0, 20),
                "cluster_id": [f"C{index // 2}" for index in range(20)],
            }
        )
        correlation = pd.DataFrame(
            np.eye(20), index=stocks, columns=stocks
        )
        return DayContext(
            decision_date=pd.Timestamp("2025-01-03"),
            candidates=candidates,
            correlation=correlation,
            raw_state={},
            regime_multiplier=1.0,
        )

    def test_five_names_full_investment_and_cluster_cap(self) -> None:
        policy = Policy("TOP5_EQUAL", 0.1, 0.1, 0.05, 0.1)
        selected = select_defensive(self.context(), policy, {})
        self.assertEqual(len(selected), 5)
        self.assertAlmostEqual(sum(item["weight"] for item in selected), 1.0)
        self.assertLessEqual(max(item["weight"] for item in selected), 0.30)
        counts = pd.Series([item["cluster_id"] for item in selected]).value_counts()
        self.assertLessEqual(int(counts.max()), 2)

    def test_selection_is_deterministic(self) -> None:
        policy = Policy(
            "TOP5_RANK_30_25_20_15_10", 0.15, 0.15, 0.05, 0.10
        )
        first = select_defensive(self.context(), policy, {"000004": 0.2})
        second = select_defensive(self.context(), policy, {"000004": 0.2})
        self.assertEqual(first, second)

    def test_turnover_uses_only_previous_weights(self) -> None:
        previous = {"000001": 0.5, "000002": 0.5}
        current = {"000001": 0.2, "000003": 0.8}
        self.assertAlmostEqual(transition_turnover(current, previous), 0.8)
        self.assertAlmostEqual(transition_turnover(current, {}), 1.0)

    def test_decision_dates_start_first_then_every_fifth(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=13)
        frame = pd.DataFrame({"sample_end_date": dates})
        selected = decision_dates(frame)
        self.assertEqual(selected, [dates[0], dates[5], dates[10]])

    def test_canonical_clusters_are_stable(self) -> None:
        stocks = ["000001", "000002", "000003"]
        corr = pd.DataFrame(
            [[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]],
            index=stocks,
            columns=stocks,
        )
        clusters = canonical_clusters(corr)
        self.assertEqual(clusters["000001"], clusters["000002"])
        self.assertNotEqual(clusters["000001"], clusters["000003"])
        self.assertEqual(clusters["000001"], "000001")

    def test_bootstrap_is_deterministic(self) -> None:
        values = np.linspace(-0.02, 0.03, 20)
        first = moving_block_bootstrap_lower(values, reps=200, seed=17)
        second = moving_block_bootstrap_lower(values, reps=200, seed=17)
        self.assertEqual(first, second)

    def test_file_hash_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.txt"
            path.write_text("one", encoding="utf-8")
            first = sha256_file(path)
            path.write_text("two", encoding="utf-8")
            second = sha256_file(path)
        self.assertNotEqual(first, second)

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


if __name__ == "__main__":
    unittest.main()

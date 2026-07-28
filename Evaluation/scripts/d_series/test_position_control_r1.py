from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Evaluation.scripts.d_series.position_control_r1 import (
    CONTROLLER_FEATURES,
    CausalMarketFeatures,
    apply_static_policy,
    block_bootstrap_probability,
    bounded_simplex_projection,
    controller_feature_row,
    paired_position_metrics,
    robust_zscore,
)


class PositionControlR1Tests(unittest.TestCase):
    def _feature_bundle(self) -> CausalMarketFeatures:
        date = pd.Timestamp("2024-01-02")
        stocks = [f"{index:06d}" for index in range(1, 7)]
        stock_features = pd.DataFrame(
            {
                "date": [date] * len(stocks),
                "stock_id": stocks,
                "vol20": np.linspace(0.01, 0.06, len(stocks)),
                "vol60": np.linspace(0.02, 0.07, len(stocks)),
            }
        )
        returns = {
            stock: pd.Series(np.linspace(-0.01, 0.01, 60) + index * 1e-4, index=pd.bdate_range(end=date, periods=60))
            for index, stock in enumerate(stocks)
        }
        return CausalMarketFeatures(
            stock_features=stock_features,
            market_features=pd.DataFrame({"date": [date], "market_vol20": [0.012]}),
            returns_by_stock=returns,
        )

    def _predictions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_end_date": ["2024-01-02"] * 6,
                "stock_id": [f"{index:06d}" for index in range(1, 7)],
                "score": [3.0, 2.0, 1.5, 1.0, 0.5, -1.0],
                "label": [0.01, 0.02, -0.01, 0.03, 0.00, -0.02],
            }
        )

    def test_robust_zscore_is_clipped(self) -> None:
        result = robust_zscore([0.0, 0.1, 0.2, 1000.0])
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(float(result.max()), 5.0)
        self.assertGreaterEqual(float(result.min()), -5.0)

    def test_bounded_simplex_projection(self) -> None:
        result = bounded_simplex_projection([10.0, 1.0], 0.4, 0.6)
        self.assertAlmostEqual(float(result.sum()), 1.0, places=12)
        self.assertTrue((result >= 0.4 - 1e-12).all())
        self.assertTrue((result <= 0.6 + 1e-12).all())
        result5 = bounded_simplex_projection([10.0, 2.0, 1.0, 0.5, 0.25], 0.1, 0.3)
        self.assertAlmostEqual(float(result5.sum()), 1.0, places=12)
        self.assertTrue((result5 >= 0.1 - 1e-12).all())
        self.assertTrue((result5 <= 0.3 + 1e-12).all())

    def test_all_static_policies_are_fully_invested_and_bounded(self) -> None:
        features = self._feature_bundle()
        predictions = self._predictions()
        expected_counts = {"P0": 2, "P1": 2, "P2": 2, "P3": 5, "P4": 5}
        bounds = {"P0": (0.5, 0.5), "P1": (0.4, 0.6), "P2": (0.4, 0.6), "P3": (0.2, 0.2), "P4": (0.1, 0.3)}
        for policy_id in expected_counts:
            weighted = apply_static_policy(predictions, policy_id, features)
            selected = weighted[weighted["portfolio_weight"] > 1e-12]
            self.assertEqual(len(selected), expected_counts[policy_id])
            self.assertAlmostEqual(float(selected["portfolio_weight"].sum()), 1.0, places=12)
            self.assertGreaterEqual(float(selected["portfolio_weight"].min()), bounds[policy_id][0] - 1e-12)
            self.assertLessEqual(float(selected["portfolio_weight"].max()), bounds[policy_id][1] + 1e-12)

    def test_controller_feature_order_and_finiteness(self) -> None:
        row = controller_feature_row(self._predictions(), self._feature_bundle())
        self.assertEqual(tuple(row), CONTROLLER_FEATURES)
        self.assertTrue(np.isfinite(np.asarray(list(row.values()))).all())

    def test_paired_tie_metrics(self) -> None:
        base = pd.DataFrame(
            {
                "week_start": ["a", "b", "c"],
                "weekly_return_gross": [0.01, 0.02, 0.03],
                "weekly_return_net": [0.0085, 0.0185, 0.0285],
                "cost": [0.0015] * 3,
            }
        )
        challenger = base.copy()
        challenger["weekly_return_gross"] += [0.01, 0.0, -0.005]
        challenger["weekly_return_net"] += [0.01, 0.0, -0.005]
        metrics = paired_position_metrics(challenger, base)
        self.assertEqual(metrics["wins"], 1)
        self.assertEqual(metrics["ties"], 1)
        self.assertEqual(metrics["losses"], 1)
        self.assertAlmostEqual(metrics["tie_adjusted_PDR"], 0.5)
        self.assertAlmostEqual(metrics["incremental_cost_drag"], 0.0)

    def test_block_bootstrap_is_deterministic(self) -> None:
        first = block_bootstrap_probability([0.01, 0.02, -0.01, 0.03, 0.0], samples=100, seed=7)
        second = block_bootstrap_probability([0.01, 0.02, -0.01, 0.03, 0.0], samples=100, seed=7)
        self.assertEqual(first, second)

    def test_causal_market_features_truncate_equivalently(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=90)
        rows = []
        for stock_index, stock in enumerate(("000001", "000002", "000003")):
            for index, date in enumerate(dates):
                close = 10.0 + stock_index + 0.02 * index + 0.1 * np.sin(index / 5.0 + stock_index)
                rows.append(
                    {
                        "股票代码": stock,
                        "日期": date.strftime("%Y-%m-%d"),
                        "开盘": close * 0.999,
                        "收盘": close,
                        "最高": close * 1.01,
                        "最低": close * 0.99,
                        "成交量": 100000 + index,
                        "成交额": 1000000 + index,
                        "换手率": 0.01,
                    }
                )
        frame = pd.DataFrame(rows)
        decision = dates[69]
        with tempfile.TemporaryDirectory() as directory:
            full_path = Path(directory) / "full.csv"
            truncated_path = Path(directory) / "truncated.csv"
            frame.to_csv(full_path, index=False, encoding="utf-8")
            frame[pd.to_datetime(frame["日期"]) <= decision].to_csv(truncated_path, index=False, encoding="utf-8")
            full = CausalMarketFeatures.from_market_path(full_path)
            truncated = CausalMarketFeatures.from_market_path(truncated_path)
            for stock in ("000001", "000002", "000003"):
                self.assertTrue(np.allclose(full.stock_volatility(decision, stock), truncated.stock_volatility(decision, stock)))
            self.assertAlmostEqual(full.market_vol20(decision), truncated.market_vol20(decision), places=15)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from core import (  # noqa: E402
    GRID_LAMBDAS,
    build_causal_factor_table,
    calibration_choice,
    combine_alpha_scores,
    cross_section_rank,
    policy_weights,
)


def synthetic_panel(stock_count: int = 25, periods: int = 110) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows: list[dict[str, object]] = []
    for stock_index in range(stock_count):
        stock = f"{stock_index + 1:06d}"
        base = 10.0 + stock_index * 0.2
        for date_index, date in enumerate(dates):
            trend = 1.0 + 0.001 * date_index + 0.0002 * stock_index * date_index
            close = base * trend * (1.0 + 0.002 * np.sin(date_index + stock_index))
            open_price = close * (1.0 - 0.001 * np.cos(date_index))
            high = max(open_price, close) * 1.01
            low = min(open_price, close) * 0.99
            rows.append(
                {
                    "stock_id": stock,
                    "date": date,
                    "open": open_price,
                    "close": close,
                    "high": high,
                    "low": low,
                    "volume": 1_000_000 + stock_index * 10_000 + date_index * 1_000,
                    "amount": close
                    * (1_000_000 + stock_index * 10_000 + date_index * 1_000),
                }
            )
    return pd.DataFrame(rows)


class RankAndGridTests(unittest.TestCase):
    def test_grid_is_exactly_the_preregistered_36_points(self) -> None:
        self.assertEqual(36, len(GRID_LAMBDAS))
        self.assertEqual(
            {0.0, 0.05, 0.10, 0.15},
            {item[0] for item in GRID_LAMBDAS},
        )
        self.assertEqual({0.0, 0.05, 0.10}, {item[1] for item in GRID_LAMBDAS})
        self.assertEqual({0.0, 0.05, 0.10}, {item[2] for item in GRID_LAMBDAS})

    def test_cross_section_rank_is_deterministic_and_clipped(self) -> None:
        values = pd.Series([5.0, 5.0, 1.0, 9.0], index=["2", "1", "4", "3"])
        ranked = cross_section_rank(values)
        self.assertTrue((ranked.abs() <= 0.95 + 1e-12).all())
        self.assertAlmostEqual(float(ranked.loc["1"]), float(ranked.loc["2"]))
        self.assertGreater(float(ranked.loc["3"]), float(ranked.loc["1"]))
        self.assertGreater(float(ranked.loc["1"]), float(ranked.loc["4"]))

    def test_alpha_formula_has_frozen_signs(self) -> None:
        scores = pd.DataFrame(
            {
                "decision_date": ["2024-05-01"] * 3,
                "stock_id": ["000001", "000002", "000003"],
                "score": [3.0, 2.0, 1.0],
            }
        )
        factors = pd.DataFrame(
            {
                "decision_date": ["2024-05-01"] * 3,
                "stock_id": ["000001", "000002", "000003"],
                "momentum_rank": [0.0, 0.95, -0.95],
                "volume_rank": [0.0, 0.95, -0.95],
                "reversal_rank": [0.95, -0.95, 0.0],
            }
        )
        result = combine_alpha_scores(scores, factors, (0.10, 0.10, 0.10))
        by_stock = result.set_index("stock_id")
        expected = (
            by_stock["d_rank_score"]
            + 0.10 * by_stock["momentum_rank"]
            + 0.10 * by_stock["volume_rank"]
            - 0.10 * by_stock["reversal_rank"]
        )
        np.testing.assert_allclose(by_stock["alpha_score"], expected)

    def test_fixed_portfolios_are_full_investment_and_below_hhi_gate(self) -> None:
        equal = policy_weights("TOP5_EQUAL")
        rank_fixed = policy_weights("TOP5_FIXED_30_25_20_15_10")
        self.assertAlmostEqual(1.0, float(equal.sum()))
        self.assertAlmostEqual(1.0, float(rank_fixed.sum()))
        self.assertAlmostEqual(0.20, float(np.square(equal).sum()))
        self.assertAlmostEqual(0.225, float(np.square(rank_fixed).sum()))
        self.assertLessEqual(float(np.square(equal).sum()), 0.30)
        self.assertLessEqual(float(np.square(rank_fixed).sum()), 0.30)


class FactorCausalityTests(unittest.TestCase):
    def test_factors_are_invariant_to_rows_after_decision_date(self) -> None:
        raw = synthetic_panel()
        decision = pd.Timestamp("2024-05-17")
        prefix = raw.loc[raw["date"] < decision].copy()
        future = raw.copy()
        future.loc[future["date"] > decision, "close"] *= 1000.0
        future.loc[future["date"] > decision, "volume"] *= 1000.0
        left = build_causal_factor_table(prefix, [decision])
        right = build_causal_factor_table(future, [decision])
        columns = [
            "decision_date",
            "stock_id",
            "momentum_rank",
            "volume_rank",
            "reversal_rank",
            "factor_asof_date",
        ]
        pd.testing.assert_frame_equal(
            left[columns].reset_index(drop=True),
            right[columns].reset_index(drop=True),
            check_exact=True,
        )
        self.assertTrue(
            (pd.to_datetime(left["factor_asof_date"]) < decision).all()
        )

    def test_factor_builder_rejects_too_few_stocks(self) -> None:
        raw = synthetic_panel(stock_count=9)
        with self.assertRaisesRegex(ValueError, "at least 10"):
            build_causal_factor_table(raw, [pd.Timestamp("2024-05-17")])


class CalibrationTests(unittest.TestCase):
    def test_calibration_rejects_when_nonzero_grid_does_not_beat_zero(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "lambda_momentum": 0.0,
                    "lambda_volume": 0.0,
                    "lambda_reversal": 0.0,
                    "stability_score": 0.02,
                    "mean_excess": 0.0,
                    "median_excess": 0.0,
                },
                {
                    "lambda_momentum": 0.05,
                    "lambda_volume": 0.0,
                    "lambda_reversal": 0.0,
                    "stability_score": 0.01,
                    "mean_excess": 0.01,
                    "median_excess": 0.01,
                },
            ]
        )
        result = calibration_choice(metrics)
        self.assertEqual("REJECTED_AT_CALIBRATION_GATE", result["status"])

    def test_calibration_uses_smaller_lambda_sum_as_tie_breaker(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "lambda_momentum": 0.0,
                    "lambda_volume": 0.0,
                    "lambda_reversal": 0.0,
                    "stability_score": 0.0,
                    "mean_excess": 0.0,
                    "median_excess": 0.0,
                },
                {
                    "lambda_momentum": 0.10,
                    "lambda_volume": 0.05,
                    "lambda_reversal": 0.0,
                    "stability_score": 0.02,
                    "mean_excess": 0.01,
                    "median_excess": 0.01,
                },
                {
                    "lambda_momentum": 0.05,
                    "lambda_volume": 0.05,
                    "lambda_reversal": 0.0,
                    "stability_score": 0.02,
                    "mean_excess": 0.01,
                    "median_excess": 0.01,
                },
            ]
        )
        result = calibration_choice(metrics)
        self.assertEqual("PASS", result["status"])
        self.assertEqual(
            {
                "lambda_momentum": 0.05,
                "lambda_volume": 0.05,
                "lambda_reversal": 0.0,
            },
            result["selected_lambdas"],
        )


if __name__ == "__main__":
    unittest.main()

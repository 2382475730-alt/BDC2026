from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from pipeline import (  # noqa: E402
    build_forward_outcomes,
    build_grid_holdings,
    calibration_metrics,
    five_day_decision_dates,
    score_grid_locked,
)
from workflow import write_prediction_lock  # noqa: E402


class PipelineTests(unittest.TestCase):
    def score_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for date in pd.bdate_range("2025-01-02", periods=11):
            for stock_index in range(10):
                rows.append(
                    {
                        "marker": "TEST",
                        "experiment_id": "X",
                        "decision_date": date,
                        "stock_id": f"{stock_index + 1:06d}",
                        "score": float(10 - stock_index),
                        "seed": 20260726,
                        "fold_id": "2025Q1",
                        "phase": "inner",
                    }
                )
        return pd.DataFrame(rows)

    def factor_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for date in pd.bdate_range("2025-01-02", periods=11):
            for stock_index in range(10):
                rows.append(
                    {
                        "decision_date": date,
                        "stock_id": f"{stock_index + 1:06d}",
                        "momentum_rank": (stock_index - 5) / 5,
                        "volume_rank": (5 - stock_index) / 5,
                        "reversal_rank": 0.0,
                        "factor_complete": True,
                    }
                )
        return pd.DataFrame(rows)

    def test_decision_dates_take_every_fifth_available_date(self) -> None:
        dates = five_day_decision_dates(self.score_frame()["decision_date"])
        self.assertEqual(
            ["2025-01-02", "2025-01-09", "2025-01-16"],
            [pd.Timestamp(value).strftime("%Y-%m-%d") for value in dates],
        )

    def test_grid_holdings_are_locked_to_three_frozen_policies(self) -> None:
        holdings = build_grid_holdings(self.score_frame(), self.factor_frame())
        policies = set(holdings["policy_id"])
        self.assertEqual(
            {
                "TOP2_EQUAL",
                "TOP5_EQUAL",
                "TOP5_FIXED_30_25_20_15_10",
            },
            policies,
        )
        top5 = holdings.loc[holdings["policy_id"] == "TOP5_EQUAL"]
        expected_rows = 36 * 3 * 5
        self.assertEqual(expected_rows, len(top5))
        sums = holdings.groupby(
            [
                "seed",
                "fold_id",
                "phase",
                "decision_date",
                "lambda_momentum",
                "lambda_volume",
                "lambda_reversal",
                "policy_id",
            ]
        )["weight"].sum()
        self.assertTrue(np.allclose(sums.to_numpy(), 1.0))

    def test_forward_outcomes_use_open_t_plus_1_to_t_plus_5(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=8)
        raw = pd.DataFrame(
            {
                "stock": ["000001"] * len(dates),
                "date": dates,
                "open": np.arange(10.0, 18.0),
            }
        )
        outcomes = build_forward_outcomes(
            raw,
            ["2025-01-02"],
            columns={"stock": "stock", "date": "date", "open": "open"},
        )
        self.assertAlmostEqual(15.0 / 11.0 - 1.0, outcomes.loc[0, "forward_return"])

    def test_locked_grid_scoring_and_calibration_metrics(self) -> None:
        holdings = build_grid_holdings(self.score_frame(), self.factor_frame())
        decision_dates = five_day_decision_dates(self.score_frame()["decision_date"])
        outcomes = pd.DataFrame(
            [
                {
                    "decision_date": date,
                    "stock_id": f"{stock_index + 1:06d}",
                    "forward_return": 0.01 * stock_index,
                }
                for date in decision_dates
                for stock_index in range(10)
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            holdings_path = root / "holdings.csv"
            holdings.to_csv(holdings_path, index=False)
            lock_path = root / "lock.json"
            write_prediction_lock([holdings_path], lock_path, {"marker": "TEST"})
            scored = score_grid_locked(
                holdings_path,
                outcomes,
                lock_path,
                cost_bps_per_side=10,
            )
        metrics = calibration_metrics(scored)
        self.assertEqual(36, len(metrics))
        zero = metrics.loc[
            (metrics["lambda_momentum"] == 0.0)
            & (metrics["lambda_volume"] == 0.0)
            & (metrics["lambda_reversal"] == 0.0)
        ]
        self.assertAlmostEqual(0.0, float(zero.iloc[0]["mean_excess"]))

    def test_locked_grid_scoring_rejects_permanently_exposed_dates(self) -> None:
        holdings = pd.DataFrame(
            {
                "seed": [20260726],
                "fold_id": ["EXPOSED"],
                "phase": ["outer"],
                "decision_date": ["2026-07-20"],
                "stock_id": ["000001"],
                "weight": [1.0],
                "lambda_momentum": [0.0],
                "lambda_volume": [0.0],
                "lambda_reversal": [0.0],
                "policy_id": ["TOP5_EQUAL"],
            }
        )
        outcomes = pd.DataFrame(
            {
                "decision_date": ["2026-07-20"],
                "stock_id": ["000001"],
                "forward_return": [0.01],
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            holdings_path = root / "holdings.csv"
            holdings.to_csv(holdings_path, index=False)
            lock_path = root / "lock.json"
            write_prediction_lock([holdings_path], lock_path, {"marker": "TEST"})
            with self.assertRaisesRegex(ValueError, "permanently exposed"):
                score_grid_locked(
                    holdings_path,
                    outcomes,
                    lock_path,
                    cost_bps_per_side=10,
                )


if __name__ == "__main__":
    unittest.main()

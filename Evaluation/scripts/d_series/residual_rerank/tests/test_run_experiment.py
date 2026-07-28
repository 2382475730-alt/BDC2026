from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from run_experiment import (  # noqa: E402
    attribution_gate,
    build_score_jobs,
    historical_score_universe_check,
    select_locked_outer_holdings,
)


class ExperimentRunnerTests(unittest.TestCase):
    def test_score_jobs_are_exactly_two_calibration_and_eight_outer(self) -> None:
        jobs = build_score_jobs()
        self.assertEqual(10, len(jobs))
        self.assertEqual(2, sum(job["phase"] == "inner" for job in jobs))
        self.assertEqual(8, sum(job["phase"] == "outer" for job in jobs))
        self.assertEqual({20260726, 20260727}, {job["seed"] for job in jobs})
        self.assertFalse(
            any(
                job["start"] <= "2026-07-24" and job["end"] >= "2026-07-06"
                for job in jobs
            )
        )

    def test_selected_outer_holdings_keep_only_preregistered_comparators(self) -> None:
        rows = []
        for policy in (
            "TOP2_EQUAL",
            "TOP5_EQUAL",
            "TOP5_FIXED_30_25_20_15_10",
        ):
            for lambdas in ((0.0, 0.0, 0.0), (0.1, 0.05, 0.0), (0.15, 0.1, 0.1)):
                rows.append(
                    {
                        "seed": 20260726,
                        "fold_id": "2025Q1",
                        "phase": "outer",
                        "decision_date": "2025-01-03",
                        "lambda_momentum": lambdas[0],
                        "lambda_volume": lambdas[1],
                        "lambda_reversal": lambdas[2],
                        "policy_id": policy,
                        "portfolio_rank": 1,
                        "stock_id": "000001",
                        "weight": 1.0,
                    }
                )
        selected = select_locked_outer_holdings(
            pd.DataFrame(rows),
            {
                "lambda_momentum": 0.1,
                "lambda_volume": 0.05,
                "lambda_reversal": 0.0,
            },
        )
        identities = set(
            zip(
                selected["policy_id"],
                selected["lambda_momentum"],
                selected["lambda_volume"],
                selected["lambda_reversal"],
            )
        )
        self.assertEqual(
            {
                ("TOP2_EQUAL", 0.0, 0.0, 0.0),
                ("TOP5_EQUAL", 0.0, 0.0, 0.0),
                ("TOP5_EQUAL", 0.1, 0.05, 0.0),
                ("TOP5_FIXED_30_25_20_15_10", 0.0, 0.0, 0.0),
                ("TOP5_FIXED_30_25_20_15_10", 0.1, 0.05, 0.0),
            },
            identities,
        )

    def test_attribution_gate_uses_only_frozen_four_checks(self) -> None:
        result = attribution_gate(
            {
                "checks": {
                    "positive_folds_at_least_3_of_4": True,
                    "both_seed_means_positive": True,
                    "strict_weekly_win_rate_at_least_55pct": True,
                    "excess_positive_after_best_two_removed": True,
                    "cvar20_not_worse_than_control": False,
                }
            }
        )
        self.assertTrue(result["passed"])
        self.assertEqual(4, len(result["checks"]))

    def test_historical_full_available_universe_allows_297_but_not_duplicates(self) -> None:
        valid = pd.DataFrame(
            {
                "decision_date": ["2024-10-08"] * 297,
                "stock_id": [f"{value:06d}" for value in range(297)],
            }
        )
        self.assertTrue(historical_score_universe_check(valid)["passed"])
        duplicate = valid.copy()
        duplicate.loc[296, "stock_id"] = "000000"
        self.assertFalse(historical_score_universe_check(duplicate)["passed"])
        self.assertFalse(
            historical_score_universe_check(valid.iloc[:289].copy())["passed"]
        )


if __name__ == "__main__":
    unittest.main()

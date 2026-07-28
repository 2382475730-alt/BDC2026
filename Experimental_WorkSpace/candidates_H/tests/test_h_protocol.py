from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC))

from protocol import (  # noqa: E402
    ALLOWED_RESIDUAL_LAMBDAS,
    DEVELOPMENT_FOLDS,
    apply_portfolio_policy,
    combine_residual_rank,
    compute_weekly_metrics,
    filter_development_dates,
    five_trading_day_portfolios,
    paired_quick_metrics,
    quick_gate,
    should_evaluate_validation,
    validate_fold_plan,
)


class FoldProtocolTests(unittest.TestCase):
    def test_inner_windows_precede_outer_and_never_enter_sealed_periods(self):
        validate_fold_plan(DEVELOPMENT_FOLDS)
        for fold in DEVELOPMENT_FOLDS:
            self.assertLess(pd.Timestamp(fold.inner_end), pd.Timestamp(fold.outer_start))
            self.assertLess(pd.Timestamp(fold.train_end), pd.Timestamp(fold.inner_start))
            self.assertNotEqual(pd.Timestamp(fold.outer_start).quarter, 4)
        fold4 = DEVELOPMENT_FOLDS[-1]
        self.assertLessEqual(pd.Timestamp(fold4.inner_end), pd.Timestamp("2025-09-30"))

    def test_outer_validation_happens_once_after_final_epoch(self):
        schedule = [
            should_evaluate_validation(epoch, selected_epochs=3, outer_once=True)
            for epoch in (1, 2, 3)
        ]
        self.assertEqual(schedule, [False, False, True])
        self.assertTrue(
            all(
                should_evaluate_validation(epoch, 3, outer_once=False)
                for epoch in (1, 2, 3)
            )
        )

    def test_materialization_excludes_both_sealed_label_ranges(self):
        dates = pd.date_range("2025-09-29", "2026-04-07", freq="B")
        allowed = filter_development_dates(dates, end="2026-04-07")
        self.assertNotIn(pd.Timestamp("2025-10-01"), allowed)
        self.assertNotIn(pd.Timestamp("2025-12-31"), allowed)
        self.assertNotIn(pd.Timestamp("2026-04-03"), allowed)
        self.assertIn(pd.Timestamp("2026-01-05"), allowed)


class PortfolioPolicyTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "stock_id": list("ABCDEF"),
                "score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "label": [0.06, 0.05, 0.04, 0.03, 0.02, -0.20],
            }
        )

    def test_equal_and_rank_policies_have_valid_weights_and_hhi(self):
        for k in (2, 3, 4, 5):
            equal = apply_portfolio_policy(self.frame, k=k, weighting="equal")
            rank = apply_portfolio_policy(self.frame, k=k, weighting="rank")
            self.assertAlmostEqual(equal["weight"].sum(), 1.0)
            self.assertAlmostEqual(rank["weight"].sum(), 1.0)
            self.assertLessEqual(float((equal["weight"] ** 2).sum()), 0.5)
            self.assertLessEqual(float((rank["weight"] ** 2).sum()), 0.56)

    def test_fixed_rank_policy_uses_preregistered_weights(self):
        fixed = apply_portfolio_policy(
            self.frame,
            k=5,
            weighting="fixed_30_25_20_15_10",
        )
        np.testing.assert_allclose(
            fixed["weight"].to_numpy(),
            [0.30, 0.25, 0.20, 0.15, 0.10],
        )

    def test_five_day_portfolios_are_nonoverlapping(self):
        rows = []
        dates = pd.date_range("2025-01-02", periods=12, freq="B")
        for date in dates:
            for index in range(6):
                rows.append(
                    {
                        "sample_end_date": date,
                        "stock_id": f"S{index}",
                        "score": 6.0 - index,
                        "label": 0.01 * (5 - index),
                    }
                )
        result = five_trading_day_portfolios(pd.DataFrame(rows), k=5)
        self.assertEqual(
            list(pd.to_datetime(result["decision_date"])),
            [dates[0], dates[5], dates[10]],
        )
        np.testing.assert_allclose(result["hhi"], 0.20)

    def test_residual_combiner_is_clipped_and_lambda_grid_is_frozen(self):
        self.assertEqual(ALLOWED_RESIDUAL_LAMBDAS, (0.0, 0.05, 0.10))
        base = np.array([0.2, -0.4])
        aux = np.array([9.0, -9.0])
        result = combine_residual_rank(base, aux, 0.10)
        np.testing.assert_allclose(result, [0.3, -0.5])
        with self.assertRaises(ValueError):
            combine_residual_rank(base, aux, 0.07)


class StabilityMetricTests(unittest.TestCase):
    def test_weekly_metrics_include_robustness_diagnostics(self):
        values = [0.03, -0.02, 0.01, 0.05, -0.01]
        metrics = compute_weekly_metrics(values)
        for key in (
            "mean",
            "median",
            "hit_rate",
            "std",
            "q10",
            "cvar20",
            "worst",
            "sum_remove_best1",
            "sum_remove_best2",
            "best_positive_contribution",
        ):
            self.assertIn(key, metrics)
        self.assertAlmostEqual(metrics["worst"], -0.02)
        self.assertLessEqual(metrics["best_positive_contribution"], 1.0)

    def test_quick_gate_enforces_every_preregistered_threshold(self):
        passing = {
            "positive_folds": 3,
            "recent_fold_excess": 0.01,
            "seed_mean_excess": [0.01, 0.02],
            "strict_weekly_win_rate": 0.60,
            "excess_sum_remove_best2": 0.001,
            "candidate_cvar20": -0.03,
            "control_cvar20": -0.04,
            "hhi": 0.20,
            "best_positive_contribution": 0.15,
        }
        result = quick_gate(passing)
        self.assertTrue(result.passed)
        for key in passing:
            broken = dict(passing)
            if key == "positive_folds":
                broken[key] = 2
            elif key == "recent_fold_excess":
                broken[key] = 0.0
            elif key == "seed_mean_excess":
                broken[key] = [0.01, -0.01]
            elif key == "strict_weekly_win_rate":
                broken[key] = 0.54
            elif key == "excess_sum_remove_best2":
                broken[key] = 0.0
            elif key == "candidate_cvar20":
                broken[key] = -0.05
            elif key == "control_cvar20":
                broken[key] = -0.02
            elif key == "hhi":
                broken[key] = 0.31
            elif key == "best_positive_contribution":
                broken[key] = 0.21
            self.assertFalse(quick_gate(broken).passed, key)

    def test_paired_quick_metrics_match_seed_fold_and_date(self):
        rows = []
        for seed in (1, 2):
            for fold_index, fold_id in enumerate(("F1", "F2", "F3", "F4")):
                for week in range(5):
                    rows.append(
                        {
                            "seed": seed,
                            "fold_id": fold_id,
                            "decision_date": f"2025-{fold_index + 1:02d}-{week + 1:02d}",
                            "portfolio_return": 0.01,
                            "hhi": 0.20,
                        }
                    )
        control = pd.DataFrame(rows)
        candidate = control.copy()
        candidate["portfolio_return"] += 0.001
        metrics, paired = paired_quick_metrics(control, candidate)
        self.assertEqual(len(paired), len(control))
        self.assertEqual(metrics["positive_folds"], 4)
        np.testing.assert_allclose(metrics["seed_mean_excess"], [0.001, 0.001])
        self.assertEqual(metrics["strict_weekly_win_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Evaluation.scripts.walk_forward.tri_candidate_recent_rolling import (
    COL_CODE,
    COL_DATE,
    COL_OPEN,
    load_market_through_date,
    max_drawdown,
    portfolio_turnover,
    portfolio_week_score,
    provisional_labels,
    validate_result,
)


class TriCandidateRecentRollingTests(unittest.TestCase):
    def test_date_gate_handles_stock_major_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "market.csv"
            pd.DataFrame(
                {
                    COL_CODE: ["000001", "000001", "000002", "000002"],
                    COL_DATE: ["2026-07-10", "2026-07-13", "2026-07-10", "2026-07-13"],
                    COL_OPEN: [10.0, 99.0, 20.0, 88.0],
                }
            ).to_csv(path, index=False, encoding="utf-8-sig")
            gated = load_market_through_date(path, cutoff="2026-07-10")
            self.assertEqual(len(gated), 2)
            self.assertEqual(gated[COL_OPEN].tolist(), [10.0, 20.0])
            self.assertEqual(str(gated[COL_DATE].max().date()), "2026-07-10")

    def test_official_return_and_cost_sensitivity(self) -> None:
        market = pd.DataFrame(
            {
                COL_CODE: ["000001", "000002", "000001", "000002"],
                COL_DATE: pd.to_datetime(["2026-07-06", "2026-07-06", "2026-07-10", "2026-07-10"]),
                COL_OPEN: [10.0, 20.0, 11.0, 18.0],
            }
        )
        result = pd.DataFrame({"stock_id": ["000001", "000002"], "weight": [0.5, 0.25]})
        score, details = portfolio_week_score(
            result,
            market,
            buy_date="2026-07-06",
            sell_date="2026-07-10",
            estimated_codes={"000002"},
        )
        self.assertAlmostEqual(score["weekly_return_gross"], 0.025)
        self.assertGreater(score["weekly_return_net_0bps"], score["weekly_return_net_10bps"])
        self.assertGreater(score["weekly_return_net_10bps"], score["weekly_return_net_20bps"])
        self.assertAlmostEqual(score["estimated_feature_exposure_weight"], 0.25)
        self.assertEqual(len(details), 2)

    def test_result_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.csv"
            pd.DataFrame({"stock_id": ["000001", "000002"], "weight": [0.5, 0.5]}).to_csv(path, index=False)
            parsed = validate_result(path, {"000001", "000002"})
            self.assertEqual(parsed["stock_id"].tolist(), ["000001", "000002"])

    def test_drawdown(self) -> None:
        value = max_drawdown(np.asarray([0.10, -0.20], dtype=float))
        self.assertAlmostEqual(value, 0.20)

    def test_weighted_portfolio_turnover(self) -> None:
        previous = {"000001": 0.5, "000002": 0.5}
        current = {"000002": 0.25, "000003": 0.75}
        self.assertAlmostEqual(portfolio_turnover(previous, current), 0.75)

    def test_provisional_label(self) -> None:
        summaries = {
            "THU-BDC2026-main": {
                "mean_weekly_return_net_10bps": 0.01,
                "cumulative_return_gross": 0.02,
                "worst_week_net_10bps": -0.01,
                "all_rounds_valid": True,
            },
            "Candidate-C": {
                "mean_weekly_return_net_10bps": 0.02,
                "cumulative_return_gross": 0.04,
                "worst_week_net_10bps": -0.015,
                "all_rounds_valid": True,
            },
            "Candidate-D": {
                "mean_weekly_return_net_10bps": 0.005,
                "cumulative_return_gross": 0.01,
                "worst_week_net_10bps": -0.005,
                "all_rounds_valid": True,
            },
        }
        labels = provisional_labels(summaries)
        self.assertEqual(labels["Candidate-C"], "临时领先")
        self.assertEqual(labels["Candidate-D"], "风险较低但收益未领先")


if __name__ == "__main__":
    unittest.main()

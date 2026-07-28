from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Experimental_WorkSpace.candidates_D.run_d_defensive_position_r3c import (
    attach_projected_softmax,
    candidate_grid,
    config_hash,
)


class _RiskStub:
    def state_row(self, date: str) -> dict[str, float]:
        return {"market_stress": 0.5, "market_breadth_20": 0.5}


class DefensivePositionR3CTests(unittest.TestCase):
    def test_grid_is_fixed_and_unique(self) -> None:
        configs = candidate_grid()
        identifiers = [config["id"] for config in configs]
        self.assertEqual(len(configs), 12)
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(len(config_hash()), 64)

    def test_projected_softmax_is_exact_five_full_investment(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_end_date": ["2025-01-03"] * 10,
                "stock_id": [f"{index + 1:06d}" for index in range(10)],
                "score": np.linspace(1.0, 0.1, 10),
                "label": np.linspace(0.02, -0.01, 10),
            }
        )
        config = next(item for item in candidate_grid() if item["weight_mode"] == "projected_softmax")
        weighted, decisions = attach_projected_softmax(frame, config, _RiskStub())
        selected = weighted[weighted["portfolio_weight"] > 1e-12]
        self.assertEqual(len(selected), 5)
        self.assertAlmostEqual(float(selected["portfolio_weight"].sum()), 1.0, places=12)
        self.assertGreaterEqual(float(selected["portfolio_weight"].min()), 0.10 - 1e-12)
        self.assertLessEqual(float(selected["portfolio_weight"].max()), 0.30 + 1e-12)
        self.assertLessEqual(float(np.square(selected["portfolio_weight"]).sum()), 0.24 + 1e-12)
        self.assertEqual(int(decisions.iloc[0]["selected_count"]), 5)

    def test_softmax_is_deterministic_under_row_shuffle(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_end_date": ["2025-01-03"] * 10,
                "stock_id": [f"{index + 1:06d}" for index in range(10)],
                "score": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                "label": np.zeros(10),
            }
        )
        config = next(item for item in candidate_grid() if item["weight_mode"] == "projected_softmax")
        left, _ = attach_projected_softmax(frame, config, _RiskStub())
        right, _ = attach_projected_softmax(
            frame.sample(frac=1.0, random_state=7).reset_index(drop=True), config, _RiskStub()
        )
        left = left.sort_values("stock_id")["portfolio_weight"].to_numpy(dtype=float)
        right = right.sort_values("stock_id")["portfolio_weight"].to_numpy(dtype=float)
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

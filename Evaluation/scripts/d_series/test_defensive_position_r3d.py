from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Experimental_WorkSpace.candidates_D.run_d_defensive_position_r3d import (
    attach_weights,
    candidate_grid,
    config_hash,
)


class DefensivePositionR3DTests(unittest.TestCase):
    @staticmethod
    def _predictions() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_end_date": ["2025-10-10"] * 10,
                "stock_id": [f"{index + 1:06d}" for index in range(10)],
                "score": np.linspace(2.0, -1.0, 10),
                "label": np.linspace(0.03, -0.02, 10),
            }
        )

    def test_grid_is_fixed_and_unique(self) -> None:
        configs = candidate_grid()
        self.assertEqual(len(configs), 6)
        self.assertEqual(len({config["id"] for config in configs}), 6)
        self.assertEqual(len(config_hash()), 64)

    def test_every_policy_obeys_exact_five_contract(self) -> None:
        for config in candidate_grid():
            weighted, decisions = attach_weights(self._predictions(), config)
            selected = weighted[weighted["portfolio_weight"] > 1e-12]
            self.assertEqual(len(selected), 5, msg=config["id"])
            self.assertAlmostEqual(float(selected["portfolio_weight"].sum()), 1.0, places=12)
            self.assertGreaterEqual(float(selected["portfolio_weight"].min()), 0.10 - 1e-12)
            self.assertLessEqual(float(selected["portfolio_weight"].max()), 0.30 + 1e-12)
            self.assertLessEqual(float(np.square(selected["portfolio_weight"]).sum()), 0.24 + 1e-12)
            self.assertEqual(int(decisions.iloc[0]["selected_count"]), 5)

    def test_policy_is_deterministic_under_row_shuffle(self) -> None:
        config = next(item for item in candidate_grid() if item["id"] == "R3D_SOFTMAX_T100")
        left, _ = attach_weights(self._predictions(), config)
        right, _ = attach_weights(
            self._predictions().sample(frac=1.0, random_state=17).reset_index(drop=True), config
        )
        left_values = left.sort_values("stock_id")["portfolio_weight"].to_numpy(dtype=float)
        right_values = right.sort_values("stock_id")["portfolio_weight"].to_numpy(dtype=float)
        np.testing.assert_allclose(left_values, right_values, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

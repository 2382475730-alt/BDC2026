from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from candidate_full_scores import canonical_full_score_frame  # noqa: E402


class CandidateFullScoreTests(unittest.TestCase):
    def test_full_score_export_is_score_only_and_deterministically_sorted(self) -> None:
        frame = canonical_full_score_frame(
            decision_date="2026-07-24",
            stock_codes=["000002", "000001"],
            scores=[0.25, 0.75],
            experiment_id="TEST_X",
        )
        self.assertEqual(
            ["marker", "experiment_id", "decision_date", "stock_id", "score"],
            list(frame.columns),
        )
        self.assertEqual(["000001", "000002"], frame["stock_id"].tolist())
        self.assertNotIn("weight", frame.columns)
        self.assertNotIn("forward_return", frame.columns)

    def test_duplicate_stock_codes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            canonical_full_score_frame(
                decision_date=pd.Timestamp("2026-07-24"),
                stock_codes=["000001", "000001"],
                scores=[0.1, 0.2],
                experiment_id="TEST_X",
            )


if __name__ == "__main__":
    unittest.main()

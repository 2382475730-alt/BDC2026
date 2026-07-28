from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from checkpoint_scores import (  # noqa: E402
    assert_byte_identical,
    canonical_score_frame,
    decision_dates_from_range,
)


class CheckpointScoreHelpersTests(unittest.TestCase):
    def test_date_range_is_sorted_and_closed(self) -> None:
        raw = pd.DataFrame(
            {
                "date": [
                    "2025-01-07",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-02",
                ]
            }
        )
        dates = decision_dates_from_range(
            raw,
            date_column="date",
            start="2025-01-03",
            end="2025-01-06",
        )
        self.assertEqual(["2025-01-03", "2025-01-06"], dates)

    def test_canonical_scores_never_emit_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_end_date": ["2025-01-03", "2025-01-03"],
                "stock_id": ["2", "1"],
                "score": [0.1, 0.2],
                "label": [99.0, -99.0],
            }
        )
        canonical = canonical_score_frame(frame, marker="TEST", experiment_id="X")
        self.assertEqual(
            ["marker", "experiment_id", "decision_date", "stock_id", "score"],
            list(canonical.columns),
        )
        self.assertEqual(["000001", "000002"], canonical["stock_id"].tolist())
        self.assertNotIn("label", canonical.columns)

    def test_byte_identity_check_rejects_any_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left = root / "left.csv"
            right = root / "right.csv"
            left.write_bytes(b"a,b\n1,2\n")
            right.write_bytes(b"a,b\n1,2\n")
            result = assert_byte_identical(left, right)
            self.assertTrue(result["identical"])
            right.write_bytes(b"a,b\n1,3\n")
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                assert_byte_identical(left, right)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from workflow import (  # noqa: E402
    compare_fingerprints,
    fingerprint_tree,
    quick_gate,
    score_locked_portfolios,
    write_prediction_lock,
)


class PredictionLockTests(unittest.TestCase):
    def test_scorer_refuses_missing_or_stale_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            predictions = root / "portfolios.csv"
            pd.DataFrame(
                [
                    {
                        "decision_date": "2025-01-03",
                        "stock_id": "000001",
                        "weight": 1.0,
                    }
                ]
            ).to_csv(predictions, index=False)
            outcomes = pd.DataFrame(
                [
                    {
                        "decision_date": "2025-01-03",
                        "stock_id": "000001",
                        "forward_return": 0.02,
                    }
                ]
            )
            with self.assertRaisesRegex(ValueError, "lock"):
                score_locked_portfolios(
                    predictions,
                    outcomes,
                    root / "missing-lock.json",
                    cost_bps_per_side=10,
                )
            lock = root / "prediction-lock.json"
            write_prediction_lock([predictions], lock, {"marker": "TEST"})
            predictions.write_text(
                predictions.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                score_locked_portfolios(
                    predictions,
                    outcomes,
                    lock,
                    cost_bps_per_side=10,
                )

    def test_scorer_rejects_permanently_exposed_diagnostic_dates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            predictions = root / "portfolios.csv"
            pd.DataFrame(
                [
                    {
                        "decision_date": "2026-07-10",
                        "stock_id": "000001",
                        "weight": 1.0,
                    }
                ]
            ).to_csv(predictions, index=False)
            lock = root / "prediction-lock.json"
            write_prediction_lock([predictions], lock, {"marker": "TEST"})
            outcomes = pd.DataFrame(
                [
                    {
                        "decision_date": "2026-07-10",
                        "stock_id": "000001",
                        "forward_return": 0.02,
                    }
                ]
            )
            with self.assertRaisesRegex(ValueError, "exposed"):
                score_locked_portfolios(
                    predictions,
                    outcomes,
                    lock,
                    cost_bps_per_side=10,
                )

    def test_full_investment_cost_is_applied_equally(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            predictions = root / "portfolios.csv"
            pd.DataFrame(
                [
                    {
                        "decision_date": "2025-01-03",
                        "stock_id": f"{index + 1:06d}",
                        "weight": 0.2,
                    }
                    for index in range(5)
                ]
            ).to_csv(predictions, index=False)
            lock = root / "prediction-lock.json"
            write_prediction_lock([predictions], lock, {"marker": "TEST"})
            outcomes = pd.DataFrame(
                [
                    {
                        "decision_date": "2025-01-03",
                        "stock_id": f"{index + 1:06d}",
                        "forward_return": 0.01 * (index + 1),
                    }
                    for index in range(5)
                ]
            )
            scored = score_locked_portfolios(
                predictions,
                outcomes,
                lock,
                cost_bps_per_side=10,
            )
            self.assertAlmostEqual(0.03, float(scored.loc[0, "gross_return"]))
            self.assertAlmostEqual(0.002, float(scored.loc[0, "cost"]))
            self.assertAlmostEqual(0.028, float(scored.loc[0, "net_return"]))


class GateTests(unittest.TestCase):
    def passing_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for seed in (20260726, 20260727):
            for fold_index, fold in enumerate(
                ("2025Q1", "2025Q2", "2025Q3", "2026Q1")
            ):
                for week in range(12):
                    control = -0.002 + 0.0001 * week
                    excess = 0.003 + 0.0001 * fold_index
                    candidate = control + excess
                    rows.append(
                        {
                            "seed": seed,
                            "fold_id": fold,
                            "decision_date": f"2025-{fold_index + 1:02d}-{week + 1:02d}",
                            "control_net_return": control,
                            "candidate_net_return": candidate,
                            "control_hhi": 0.5,
                            "candidate_hhi": 0.2,
                        }
                    )
        return pd.DataFrame(rows)

    def test_quick_gate_enforces_all_preregistered_checks(self) -> None:
        result = quick_gate(self.passing_frame())
        self.assertTrue(result["passed"])
        self.assertEqual([], result["failed_checks"])
        self.assertGreaterEqual(result["metrics"]["strict_weekly_win_rate"], 0.55)

    def test_quick_gate_fails_leave_best_two_and_seed_direction(self) -> None:
        frame = self.passing_frame()
        frame.loc[frame["seed"] == 20260726, "candidate_net_return"] = (
            frame.loc[frame["seed"] == 20260726, "control_net_return"] - 0.001
        )
        result = quick_gate(frame)
        self.assertFalse(result["passed"])
        self.assertIn("both_seed_means_positive", result["failed_checks"])


class FingerprintTests(unittest.TestCase):
    def test_tree_fingerprint_detects_content_and_path_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.txt").write_text("alpha", encoding="utf-8")
            before = fingerprint_tree(root)
            same = fingerprint_tree(root)
            self.assertTrue(compare_fingerprints(before, same)["identical"])
            (root / "a.txt").write_text("beta", encoding="utf-8")
            after = fingerprint_tree(root)
            result = compare_fingerprints(before, after)
            self.assertFalse(result["identical"])
            self.assertEqual(["a.txt"], result["changed_paths"])

    def test_lock_is_machine_readable_and_contains_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            item = root / "artifact.txt"
            item.write_text("locked", encoding="utf-8")
            lock = root / "lock.json"
            write_prediction_lock([item], lock, {"marker": "TEST"})
            payload = json.loads(lock.read_text(encoding="utf-8"))
            self.assertEqual("TEST", payload["metadata"]["marker"])
            self.assertEqual(64, len(payload["files"][0]["sha256"]))


if __name__ == "__main__":
    unittest.main()

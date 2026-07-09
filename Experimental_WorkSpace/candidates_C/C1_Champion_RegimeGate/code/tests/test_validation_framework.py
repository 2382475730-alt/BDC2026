from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))


class WalkForwardFoldTests(unittest.TestCase):
    def test_build_walk_forward_folds_uses_sample_end_dates_and_purge(self) -> None:
        from validation import build_walk_forward_folds

        trading_days = pd.bdate_range("2023-12-20", "2024-01-31")
        rows = [
            {"股票代码": f"{stock_id:06d}", "日期": day.strftime("%Y-%m-%d")}
            for day in trading_days
            for stock_id in range(1, 4)
        ]
        df = pd.DataFrame(rows)

        folds = build_walk_forward_folds(
            df,
            sequence_length=3,
            horizon=2,
            fold_ranges=[("toy", "2024-01-02", "2024-01-10")],
        )

        self.assertEqual(len(folds), 1)
        fold = folds[0]
        self.assertEqual(fold.name, "toy")
        self.assertEqual(fold.val_start, pd.Timestamp("2024-01-02"))
        self.assertEqual(fold.val_end, pd.Timestamp("2024-01-10"))
        self.assertTrue(fold.checks["no_train_val_overlap"])
        self.assertTrue(fold.checks["train_label_end_before_val_start"])
        self.assertTrue(fold.checks["val_labels_complete"])
        self.assertTrue(fold.checks["val_history_complete"])
        self.assertLess(fold.max_train_label_end_date, fold.val_start)
        self.assertNotIn(pd.Timestamp("2023-12-29"), fold.train_sample_dates)
        self.assertEqual(fold.train_row_count, len(fold.train_sample_dates) * 3)
        self.assertEqual(fold.val_row_count, len(fold.val_sample_dates) * 3)


class DatasetControlTests(unittest.TestCase):
    def test_vectorized_dataset_filters_exact_sample_dates_and_returns_weights(self) -> None:
        from utils import create_ranking_dataset_vectorized

        dates = pd.bdate_range("2024-01-02", periods=11)
        rows = []
        for date_index, day in enumerate(dates):
            for stock_id in range(10):
                rows.append(
                    {
                        "instrument": stock_id,
                        "日期": day.strftime("%Y-%m-%d"),
                        "f0": float(date_index + stock_id / 100.0),
                        "label": float(stock_id - 5),
                    }
                )
        df = pd.DataFrame(rows)
        requested_dates = [dates[3], dates[4]]

        result = create_ranking_dataset_vectorized(
            df,
            ["f0"],
            sequence_length=3,
            sample_end_dates=requested_dates,
            duplicate_recent_samples=False,
            sample_weight_mode="recent",
            recent_weight_start_date=dates[4],
            recent_sample_weight=2.0,
            return_metadata=True,
        )

        sequences, targets, relevance, stock_indices, metadata = result
        self.assertEqual(len(sequences), 2)
        self.assertEqual(len(targets), 2)
        self.assertEqual(len(relevance), 2)
        self.assertEqual(len(stock_indices), 2)
        self.assertEqual(metadata["sample_dates"], [pd.Timestamp(dates[3]), pd.Timestamp(dates[4])])
        self.assertEqual(metadata["sample_weights"], [1.0, 2.0])


class MetricTests(unittest.TestCase):
    def test_metrics_separate_base_and_rerank_and_report_gain(self) -> None:
        from validation_metrics import summarize_score_columns

        rows = []
        for day in pd.to_datetime(["2024-01-02", "2024-01-03"]):
            for stock_id, label in enumerate([0.50, 0.40, 0.30, 0.20, 0.10, -0.20]):
                rows.append(
                    {
                        "sample_end_date": day,
                        "stock_id": f"{stock_id:06d}",
                        "raw_label": label,
                        "relevance": label,
                        "base_score": label,
                        "rerank_score": -label,
                    }
                )
        predictions = pd.DataFrame(rows)

        summary = summarize_score_columns(
            predictions,
            date_col="sample_end_date",
            label_col="raw_label",
            relevance_col="relevance",
            score_columns={"base": "base_score", "rerank": "rerank_score"},
            top_ks=(5,),
            num_windows=2,
        )

        self.assertIn("base", summary["score_metrics"])
        self.assertIn("rerank", summary["score_metrics"])
        self.assertGreater(summary["score_metrics"]["base"]["rankic_mean"], 0.99)
        self.assertLess(summary["score_metrics"]["rerank"]["rankic_mean"], -0.99)
        self.assertLess(summary["rerank_gain"]["cv_robust_score"], 0.0)


if __name__ == "__main__":
    unittest.main()

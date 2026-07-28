from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC))

from loss import (  # noqa: E402
    PairwiseBoundaryLoss,
    PortfolioAwareRankingLoss,
    PredictedBoundaryLoss,
)
from state_cache import (  # noqa: E402
    cache_key,
    load_training_state_cache,
    save_training_state_cache,
)


class PredictedBoundaryLossTests(unittest.TestCase):
    def test_high_scoring_false_positive_is_a_hard_negative(self):
        labels = torch.tensor(
            [[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04]],
            dtype=torch.float32,
        )
        good_scores = torch.tensor(
            [[0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]],
            dtype=torch.float32,
        )
        bad_scores = torch.tensor(
            [[0.70, 0.60, 0.50, 0.40, 0.30, 0.80, 0.10]],
            dtype=torch.float32,
        )
        loss = PredictedBoundaryLoss(margin=0.05, top_k=5, hard_negative_k=15)
        self.assertGreater(float(loss(bad_scores, labels)), float(loss(good_scores, labels)))

    def test_predicted_boundary_catches_near_boundary_error_missed_by_bottom20(self):
        labels = torch.arange(30, 0, -1, dtype=torch.float32).unsqueeze(0)
        scores = labels.clone()
        scores[0, 5] = 100.0
        mask = torch.ones_like(labels, dtype=torch.bool)
        old_loss = PairwiseBoundaryLoss(margin=0.05, top_k=5, bottom_k=20)
        new_loss = PredictedBoundaryLoss(margin=0.05, top_k=5, hard_negative_k=15)
        self.assertEqual(float(old_loss(scores, labels, mask)), 0.0)
        self.assertGreater(float(new_loss(scores, labels, mask)), 0.0)

    def test_masked_stocks_cannot_be_selected_as_hard_negatives(self):
        labels = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1, -1.0]])
        scores = torch.tensor([[0.6, 0.5, 0.4, 0.3, 0.2, 99.0]], requires_grad=True)
        mask = torch.tensor([[True, True, True, True, True, False]])
        loss = PredictedBoundaryLoss()(scores, labels, mask)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(scores.grad[0, 5]), 0.0)


class H5DiagnosticTests(unittest.TestCase):
    def test_objective_mode_exposes_raw_weighted_and_score_gradient_diagnostics(self):
        criterion = PortfolioAwareRankingLoss(
            objective_mode="predicted_boundary_stability",
            ranking_weight=1.0,
            pairwise_weight=0.25,
            topk_return_weight=0.05,
            downside_weight=0.08,
            cvar_weight=0.04,
            positive_tail_weight=0.02,
        )
        scores = torch.tensor(
            [[0.9, 0.8, 0.7, 0.6, 0.4, 0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        outputs = {
            "ranking_scores": scores,
            "portfolio_logits": scores,
        }
        labels = torch.tensor([[0.06, 0.05, 0.04, 0.03, 0.02, -0.01]])
        relevance = labels.clone()
        mask = torch.ones_like(labels, dtype=torch.bool)
        breakdown = criterion(outputs, labels, relevance, mask)
        diagnostics = criterion.diagnostic_score_gradients(
            outputs,
            labels,
            relevance,
            mask,
        )
        breakdown.total.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(
            set(diagnostics),
            {
                "raw",
                "weighted",
                "score_gradient_l2",
            },
        )
        self.assertEqual(
            set(diagnostics["raw"]),
            {
                "ranking",
                "pairwise",
                "topk_return",
                "downside",
                "cvar20",
                "positive_tail",
            },
        )
        self.assertAlmostEqual(
            diagnostics["weighted"]["pairwise"],
            0.25 * diagnostics["raw"]["pairwise"],
            places=7,
        )
        self.assertGreater(diagnostics["score_gradient_l2"]["pairwise"], 0.0)


class TrainingStateCacheTests(unittest.TestCase):
    def setUp(self):
        self.identity = {
            "schema_version": 1,
            "data_sha256": "a" * 64,
            "feature_spec": {"sequence_length": 20, "enable_futures": False},
            "feature_source_sha256": {"dataset.py": "b" * 64},
            "sample_date_whitelist": ["2025-01-02", "2025-01-03"],
            "split": {
                "phase": "inner",
                "fold_id": "2025Q2",
                "training_end": "2024-12-24",
                "validation_start": "2025-01-03",
                "validation_end": "2025-03-24",
            },
            "scaler_fit_scope": "fold_training_samples_only",
        }

    def test_key_changes_with_whitelist_or_training_cutoff(self):
        original = cache_key(self.identity)
        changed_dates = dict(self.identity)
        changed_dates["sample_date_whitelist"] = ["2025-01-02"]
        changed_split = dict(self.identity)
        changed_split["split"] = dict(self.identity["split"], training_end="2024-12-23")
        self.assertNotEqual(original, cache_key(changed_dates))
        self.assertNotEqual(original, cache_key(changed_split))

    def test_round_trip_is_exact_and_manifest_is_verified(self):
        state = {
            "array": np.arange(12, dtype=np.float32).reshape(3, 4),
            "metadata": {"fold": "2025Q2", "cache_hit": False},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            key = cache_key(self.identity)
            manifest = save_training_state_cache(root, key, self.identity, state)
            loaded, audit = load_training_state_cache(root, key, self.identity)
            np.testing.assert_array_equal(loaded["array"], state["array"])
            self.assertEqual(loaded["metadata"], state["metadata"])
            self.assertEqual(manifest["blob_sha256"], audit["blob_sha256"])
            self.assertTrue(audit["verified"])

    def test_identity_mismatch_is_a_cache_miss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            key = cache_key(self.identity)
            save_training_state_cache(root, key, self.identity, {"value": 1})
            changed = dict(self.identity, data_sha256="c" * 64)
            loaded, audit = load_training_state_cache(root, key, changed)
            self.assertIsNone(loaded)
            self.assertEqual(audit["status"], "IDENTITY_MISMATCH")


if __name__ == "__main__":
    unittest.main()

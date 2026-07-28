from pathlib import Path
import sys
import unittest

import torch


SRC = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC))

from loss import (  # noqa: E402
    HStabilityLoss,
    differentiable_topk_weights,
)


class DifferentiableTopKWeightsTests(unittest.TestCase):
    def test_forward_is_exact_hard_top5_equal_and_backward_is_finite(self):
        scores = torch.tensor(
            [[0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6]],
            dtype=torch.float32,
            requires_grad=True,
        )
        weights = differentiable_topk_weights(scores, k=5, temperature=0.2)
        expected = torch.tensor(
            [[0.0, 0.2, 0.0, 0.2, 0.2, 0.0, 0.2]],
            dtype=torch.float32,
        )
        expected[0, 5] = 0.2
        self.assertTrue(torch.equal(weights.detach(), expected))
        self.assertAlmostEqual(float(weights.detach().sum()), 1.0, places=6)
        (weights * torch.arange(7, dtype=torch.float32)).sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_mask_and_small_cross_section_are_supported(self):
        scores = torch.tensor([[3.0, 2.0, 1.0, -1.0]])
        mask = torch.tensor([[True, False, True, False]])
        weights = differentiable_topk_weights(
            scores, k=5, temperature=0.1, mask=mask
        )
        self.assertTrue(
            torch.equal(weights, torch.tensor([[0.5, 0.0, 0.5, 0.0]]))
        )


class HStabilityLossTests(unittest.TestCase):
    def setUp(self):
        self.loss = HStabilityLoss(
            ranking_weight=1.0,
            pairwise_weight=0.25,
            pairwise_margin=0.05,
            topk_return_weight=0.05,
            downside_weight=0.08,
            downside_temperature=0.02,
            cvar_weight=0.04,
            cvar_alpha=0.20,
            positive_tail_weight=0.02,
            positive_tail_cap=0.08,
            topk=5,
            topk_temperature=0.10,
        )

    def test_combined_loss_has_finite_gradient_and_full_breakdown(self):
        scores = torch.tensor(
            [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]],
            dtype=torch.float32,
            requires_grad=True,
        )
        labels = torch.tensor(
            [[0.04, 0.03, 0.02, 0.01, -0.01, -0.02]],
            dtype=torch.float32,
        )
        total, breakdown = self.loss(scores, labels)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(
            set(breakdown),
            {
                "ranking",
                "pairwise",
                "topk_return",
                "downside",
                "cvar20",
                "positive_tail",
                "total",
            },
        )

    def test_topk_component_uses_ranking_scores(self):
        ranking_scores = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, -5.0]])
        labels = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, -1.0]])
        _, good = self.loss(ranking_scores, labels)
        _, bad = self.loss(-ranking_scores, labels)
        self.assertLess(good["topk_return"], bad["topk_return"])

    def test_downside_and_cvar_penalize_negative_selected_returns(self):
        scores = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
        positive = torch.full_like(scores, 0.02)
        negative = torch.full_like(scores, -0.02)
        _, pos = self.loss(scores, positive)
        _, neg = self.loss(scores, negative)
        self.assertGreater(neg["downside"], pos["downside"])
        self.assertGreater(neg["cvar20"], pos["cvar20"])

    def test_positive_tail_penalty_activates_on_spike(self):
        scores = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
        ordinary = torch.full_like(scores, 0.04)
        spike = torch.full_like(scores, 0.20)
        _, ordinary_parts = self.loss(scores, ordinary)
        _, spike_parts = self.loss(scores, spike)
        self.assertEqual(float(ordinary_parts["positive_tail"]), 0.0)
        self.assertGreater(float(spike_parts["positive_tail"]), 0.0)


if __name__ == "__main__":
    unittest.main()

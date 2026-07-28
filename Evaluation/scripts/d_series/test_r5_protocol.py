from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    ROOT
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "D01_FrozenResearchChampion_20260709"
    / "code_snapshot"
    / "src"
)
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from evaluator import weights_from_scores  # noqa: E402
from loss import PortfolioAwareRankingLoss  # noqa: E402
from model import MarketIntelligenceRanker  # noqa: E402

try:
    from featurework import assert_feature_truncation_equivalence  # type: ignore[attr-defined]  # noqa: E402

    R5_TRUNCATION_SOURCE_RETAINED = True
except ImportError:
    R5_TRUNCATION_SOURCE_RETAINED = False

    def assert_feature_truncation_equivalence(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("R5 run-local truncation-audit source was retired during cleanup")

try:
    from loss import differentiable_topk_weights  # type: ignore[attr-defined]  # noqa: E402

    R5_OBJECTIVE_SOURCE_RETAINED = True
except ImportError:
    R5_OBJECTIVE_SOURCE_RETAINED = False

    def differentiable_topk_weights(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("R5 run-local objective source was retired during cleanup")


def primary_scores(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stable ranking-score contract retained independently of run snapshots."""

    return outputs["ranking_scores"]

from Evaluation.scripts.d_series.r5_protocol import (  # noqa: E402
    assert_sealed_holdout_access,
    read_json,
)


def model_config(use_factor_encoder: bool) -> dict[str, object]:
    return {
        "d_model": 16,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 32,
        "dropout": 0.0,
        "use_factor_encoder": use_factor_encoder,
    }


def model_batch(factors: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "stock_sequence": torch.randn(2, 5, 6, 3),
        "market_features": torch.randn(2, 4),
        "factor_features": torch.randn(2, 5, 8) if factors is None else factors,
        "mask": torch.tensor([[True, True, True, True, True], [True, True, True, False, False]]),
    }


class R5ObjectiveTests(unittest.TestCase):
    @unittest.skipUnless(
        R5_OBJECTIVE_SOURCE_RETAINED,
        "R5 run-local top-k objective source was intentionally retired; locked reports preserve its result.",
    )
    def test_topk_aligned_loss_uses_primary_scores_with_finite_gradient(self) -> None:
        scores = torch.tensor(
            [[0.10, 0.08, 0.07, 0.01, -0.02], [0.03, 0.02, -0.01, -1e9, -1e9]],
            dtype=torch.float32,
            requires_grad=True,
        )
        portfolio_logits = torch.zeros_like(scores, requires_grad=True)
        mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
        labels = torch.tensor(
            [[0.08, -0.04, 0.03, -0.02, 0.01], [0.02, -0.03, 0.01, 0.0, 0.0]],
            dtype=torch.float32,
        )
        relevance = labels.clone()
        outputs = {"ranking_scores": scores, "portfolio_logits": portfolio_logits}
        criterion = PortfolioAwareRankingLoss(
            objective_mode="topk_aligned",
            pairwise_weight=0.25,
            portfolio_return_weight=0.0,
            topk_k=2,
            topk_temperature=0.05,
            topk_return_weight=0.05,
            robust_penalty_weight=0.01,
        )
        breakdown = criterion(outputs, labels, relevance, mask)
        breakdown.total.backward()
        self.assertIs(primary_scores(outputs), scores)
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)
        self.assertIsNone(portfolio_logits.grad)
        self.assertEqual(breakdown.objective_mode, "topk_aligned")

    @unittest.skipUnless(
        R5_OBJECTIVE_SOURCE_RETAINED,
        "R5 run-local top-k objective source was intentionally retired; locked reports preserve its result.",
    )
    def test_soft_topk_mask_and_small_cross_section(self) -> None:
        scores = torch.tensor([[3.0, 2.0, 1.0, 0.0], [1.0, 0.0, -9.0, -9.0]])
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        weights = differentiable_topk_weights(scores, mask, top_k=5, temperature=0.1)
        np.testing.assert_allclose(weights.sum(dim=1).numpy(), np.ones(2), atol=1e-7)
        self.assertTrue(torch.equal(weights[1, 2:], torch.zeros(2)))
        np.testing.assert_allclose(weights[1, :2].numpy(), np.asarray([0.5, 0.5]), atol=1e-7)

    def test_factor_encoder_off_ignores_factor_inputs(self) -> None:
        torch.manual_seed(11)
        model = MarketIntelligenceRanker(3, 4, 8, 6, model_config(False)).eval()
        first = model_batch(torch.zeros(2, 5, 8))
        second = {**first, "factor_features": torch.ones(2, 5, 8) * 100.0}
        with torch.no_grad():
            left = primary_scores(model(first))
            right = primary_scores(model(second))
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)

    def test_factor_encoder_on_changes_scores_and_receives_gradient(self) -> None:
        torch.manual_seed(11)
        model = MarketIntelligenceRanker(3, 4, 8, 6, model_config(True)).eval()
        factors = torch.randn(2, 5, 8, requires_grad=True)
        batch = model_batch(factors)
        scores = primary_scores(model(batch))
        scores[batch["mask"]].sum().backward()
        self.assertIsNotNone(factors.grad)
        self.assertGreater(float(factors.grad.abs().sum()), 0.0)
        with torch.no_grad():
            changed = primary_scores(model({**batch, "factor_features": factors.detach() + 1.0}))
        self.assertGreater(float((scores.detach()[batch["mask"]] - changed[batch["mask"]]).abs().sum()), 0.0)

    @unittest.skipUnless(
        R5_OBJECTIVE_SOURCE_RETAINED,
        "R5 run-local capped-portfolio source was intentionally retired; the locked audit remains.",
    )
    def test_competition_portfolios_are_full_investment_and_max_five(self) -> None:
        scores = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
        top2 = weights_from_scores(scores, top_k=2, mode="topk_equal")
        top5 = weights_from_scores(scores, top_k=5, mode="capped_softmax", max_weight=0.35)
        self.assertAlmostEqual(float(top2.sum()), 1.0, places=12)
        self.assertEqual(int(np.count_nonzero(top2)), 2)
        self.assertAlmostEqual(float(top5.sum()), 1.0, delta=1e-10)
        self.assertLessEqual(int(np.count_nonzero(top5)), 5)
        self.assertLessEqual(float(top5.max()), 0.35 + 1e-10)


class R5LeakageAndSplitTests(unittest.TestCase):
    @staticmethod
    def synthetic_market() -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=75)
        rows = []
        for stock_index in range(6):
            for date_index, date in enumerate(dates):
                base = 10.0 + stock_index + 0.02 * date_index
                rows.append(
                    {
                        "股票代码": f"{stock_index + 1:06d}",
                        "日期": date,
                        "开盘": base,
                        "收盘": base * (1.0 + 0.001 * np.sin(date_index + stock_index)),
                        "最高": base * 1.01,
                        "最低": base * 0.99,
                        "成交量": 1_000_000 + 1000 * date_index + stock_index,
                        "成交额": base * (1_000_000 + 1000 * date_index + stock_index),
                        "振幅": 2.0,
                        "涨跌额": 0.01,
                        "换手率": 1.0 + 0.01 * stock_index,
                        "涨跌幅": 0.1,
                    }
                )
        return pd.DataFrame(rows)

    @unittest.skipUnless(
        R5_TRUNCATION_SOURCE_RETAINED,
        "R5 run-local truncation-audit source was intentionally retired; the locked audit remains.",
    )
    def test_feature_truncation_equivalence(self) -> None:
        market = self.synthetic_market()
        sample_dates = [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-15")]
        self.assertTrue(assert_feature_truncation_equivalence(market, sample_dates))

    def test_locked_split_has_purge_embargo_and_four_folds(self) -> None:
        manifest = read_json(
            ROOT / "Experimental_WorkSpace" / "candidates_D" / "automation_runs" / "R5_SPLIT_MANIFEST.json",
            default={},
        )
        self.assertTrue(all(manifest["checks"].values()))
        folds = manifest["rolling_origin_development"]["folds"]
        self.assertEqual(len(folds), 4)
        self.assertTrue(all(8 <= fold["outer_evaluation"]["complete_week_count"] <= 13 for fold in folds))
        self.assertEqual(manifest["recent_embargo"]["complete_week_count"], 1)
        self.assertEqual(manifest["recent_sealed_holdout"]["complete_week_count"], 13)

    def test_sealed_holdout_is_inaccessible_before_finalist_lock(self) -> None:
        manifest = read_json(
            ROOT / "Experimental_WorkSpace" / "candidates_D" / "automation_runs" / "R5_SPLIT_MANIFEST.json",
            default={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "R5_FINALIST_LOCK.json"
            with self.assertRaises(PermissionError):
                assert_sealed_holdout_access(
                    split_manifest=manifest,
                    finalist_lock_path=lock_path,
                    run_state={"sealed_holdout_status": "sealed", "sealed_holdout_unseal_count": 0},
                )
            lock_path.write_text(json.dumps({"locked_finalists": ["x"]}), encoding="utf-8")
            with self.assertRaises(PermissionError):
                assert_sealed_holdout_access(
                    split_manifest=manifest,
                    finalist_lock_path=lock_path,
                    run_state={"sealed_holdout_status": "sealed", "sealed_holdout_unseal_count": 0},
                )


if __name__ == "__main__":
    unittest.main()

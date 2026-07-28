from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC))

from dataset import EFeatureScalers, ESample, collate_e_samples  # noqa: E402
from featurework import (  # noqa: E402
    AMOUNT_COL, CLOSE_COL, DATE_COL, HIGH_COL, LOW_COL, OPEN_COL, STOCK_COL,
    TURNOVER_COL, VOLUME_COL, assert_truncation_equivalence, feature_columns,
    prepare_e_features,
)
from loss import PortfolioAwareRankingLoss, straight_through_topk_weights  # noqa: E402
from model import RecentRegimeMixtureRanker  # noqa: E402
from portfolio import (  # noqa: E402
    assert_portfolio_constraints, portfolio_weights, run_weekly_backtest_exact_cost,
)
from sealed_access import assert_sealed, read_recent_holdout  # noqa: E402


def synthetic_market(days: int = 145, stocks: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=days)
    rows: list[dict[str, object]] = []
    for stock_index in range(stocks):
        for day_index, date in enumerate(dates):
            base = 10.0 + stock_index + 0.01 * day_index + 0.02 * np.sin(day_index / 5 + stock_index)
            rows.append({
                STOCK_COL: f"{600000 + stock_index:06d}", DATE_COL: date,
                OPEN_COL: base, CLOSE_COL: base * (1.0 + 0.001 * ((day_index + stock_index) % 5 - 2)),
                HIGH_COL: base * 1.01, LOW_COL: base * 0.99,
                VOLUME_COL: 1_000_000 + 1000 * day_index + stock_index,
                AMOUNT_COL: base * (1_000_000 + 1000 * day_index), TURNOVER_COL: 0.01 + stock_index * 0.001,
            })
    return pd.DataFrame(rows)


class FeatureAndLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = synthetic_market()

    def test_global_calendar_label_and_missing_target_open(self) -> None:
        raw = self.raw.copy()
        dates = sorted(raw[DATE_COL].unique())
        sample_date = dates[125]
        missing_date = dates[130]
        raw = raw[~((raw[STOCK_COL] == "600000") & (raw[DATE_COL] == missing_date))]
        processed, _, _ = prepare_e_features(raw, "e1_dedup_v1")
        row = processed[(processed[STOCK_COL] == "600000") & (processed[DATE_COL] == sample_date)].iloc[0]
        self.assertTrue(pd.isna(row["label"]))

    def test_full_feature_truncation_equivalence(self) -> None:
        date = sorted(self.raw[DATE_COL].unique())[130]
        self.assertTrue(assert_truncation_equivalence(self.raw, [date], "e1_full_v1"))

    def test_duplicate_alias_removed(self) -> None:
        columns = feature_columns("e1_full_v1")
        self.assertNotIn("open_close_spread", columns.stock)
        self.assertEqual(columns.factor, [])


class ScalerTests(unittest.TestCase):
    def test_finite_train_only_and_range_recorded(self) -> None:
        columns = feature_columns("e1_dedup_v1")
        width = len(columns.stock)
        market_width = len(columns.market)
        samples = [
            ESample(pd.Timestamp("2022-01-04"), pd.Timestamp("2022-01-11"), np.ones((2, market_width)),
                    np.ones((2, 3, width)), np.array([0.1, 0.2]), np.array([-1.0, 1.0]), ["1", "2"]),
            ESample(pd.Timestamp("2022-01-05"), pd.Timestamp("2022-01-12"), np.full((2, market_width), 2.0),
                    np.full((2, 3, width), np.nan), np.array([0.1, 0.2]), np.array([-1.0, 1.0]), ["1", "2"]),
        ]
        scaler = EFeatureScalers.fit(samples, columns, "finite_train_only")
        self.assertEqual(scaler.fit_end_date, "2022-01-05")
        self.assertTrue(all(abs(value - 1.0) < 1e-12 for value in scaler.stock_mean))
        transformed = scaler.transform_sample(samples[1])
        self.assertTrue(np.isfinite(transformed.stock_sequence).all())


class ModelLossTests(unittest.TestCase):
    def _batch(self) -> dict[str, torch.Tensor]:
        return {
            "stock_sequence": torch.randn(2, 6, 10, 8),
            "market_sequence": torch.randn(2, 6, 4),
            "mask": torch.ones(2, 6, dtype=torch.bool),
            "labels": torch.randn(2, 6) * 0.02,
            "relevance": torch.randn(2, 6),
        }

    def _model(self) -> RecentRegimeMixtureRanker:
        return RecentRegimeMixtureRanker(8, 4, 10, 5, {
            "variant": "e3_mixture", "d_model": 16, "nhead": 4, "num_layers": 1,
            "recent_num_layers": 1, "dim_feedforward": 32, "dropout": 0.1, "initial_gate": 0.2,
        })

    def test_score_contract_and_hard_top2_forward(self) -> None:
        torch.manual_seed(3)
        model = self._model().eval()
        batch = self._batch()
        with torch.no_grad():
            output = model(batch)
        self.assertTrue(torch.equal(output["ranking_scores"], output["stable_score"] + output["gate"] * output["recent_residual"]))
        scores = torch.tensor([[1.0, 3.0, 2.0]], requires_grad=True)
        weights = straight_through_topk_weights(scores, torch.ones_like(scores, dtype=torch.bool), 2, 0.1)
        self.assertTrue(torch.allclose(weights, torch.tensor([[0.0, 0.5, 0.5]])))
        weights.sum().backward()
        self.assertIsNotNone(scores.grad)

    def test_exact_checkpoint_resume(self) -> None:
        batch = self._batch()
        criterion = PortfolioAwareRankingLoss(recent_half_life=252, recent_aux_weight=0.15)

        def step(model: RecentRegimeMixtureRanker, optimizer: torch.optim.Optimizer) -> None:
            model.train(); optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = criterion(out, batch["labels"], batch["relevance"], batch["mask"]).total
            loss.backward(); optimizer.step()

        torch.manual_seed(77)
        full = self._model(); full_opt = torch.optim.AdamW(full.parameters(), lr=1e-4)
        step(full, full_opt); saved_rng = torch.get_rng_state().clone(); step(full, full_opt)
        full_state = {key: value.detach().clone() for key, value in full.state_dict().items()}

        torch.manual_seed(77)
        first = self._model(); first_opt = torch.optim.AdamW(first.parameters(), lr=1e-4)
        step(first, first_opt)
        resumed = self._model(); resumed.load_state_dict(first.state_dict())
        resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-4); resumed_opt.load_state_dict(first_opt.state_dict())
        torch.set_rng_state(saved_rng); step(resumed, resumed_opt)
        for key, value in resumed.state_dict().items():
            self.assertTrue(torch.equal(value, full_state[key]), key)


class PortfolioAndSealTests(unittest.TestCase):
    def test_all_main_policies_obey_constraints(self) -> None:
        scores = np.linspace(-1, 1, 8)
        for policy in ("P1_TOP4_BOUNDED_SOFTMAX", "P2_SHRUNK_COV_OPT_TOP5", "P3_CONFIDENCE_EXPOSURE_TOP4"):
            self.assertTrue(assert_portfolio_constraints(portfolio_weights(scores, policy), True))

    def test_actual_sell_notional_cost(self) -> None:
        dates = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"])
        market_rows = []
        for stock in ("600000", "600001"):
            for index, date in enumerate(dates):
                price = 10.0 if index < 5 else 11.0
                market_rows.append({STOCK_COL: stock, DATE_COL: date, OPEN_COL: price, CLOSE_COL: price})
        predictions = pd.DataFrame({
            "sample_end_date": ["2025-01-03", "2025-01-03"], "stock_id": ["600000", "600001"],
            "score": [2.0, 1.0], "portfolio_weight": [0.5, 0.5],
        })
        weekly, _ = run_weekly_backtest_exact_cost(predictions, pd.DataFrame(market_rows), "2025-01-06", "2025-01-10")
        self.assertAlmostEqual(float(weekly.iloc[0]["weekly_return_gross"]), 0.1, places=12)
        self.assertAlmostEqual(float(weekly.iloc[0]["cost"]), 0.0005 + 0.001 * 1.1, places=12)

    def test_sealed_access_denied_then_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "data.csv"; lock = root / "lock.json"; receipt = root / "receipt.json"
            pd.DataFrame({DATE_COL: ["2026-04-03"], STOCK_COL: ["600000"]}).to_csv(source, index=False)
            self.assertTrue(assert_sealed(source, lock, receipt))
            with self.assertRaises(PermissionError):
                read_recent_holdout(source, lock, receipt)
            lock.write_text(json.dumps({
                "status": "FINALIST_LOCKED_BEFORE_HOLDOUT_ACCESS", "config_id": "x", "config_sha256": "x",
                "code_sha256": "x", "seed": 1, "policy": "P0", "locked_at": "now",
            }), encoding="utf-8")
            self.assertEqual(len(read_recent_holdout(source, lock, receipt)), 1)
            with self.assertRaises(PermissionError):
                read_recent_holdout(source, lock, receipt)


if __name__ == "__main__":
    unittest.main()

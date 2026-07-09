from __future__ import annotations

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "code" / "src"
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(TESTS))

from config import load_config
from dataset import DFeatureScalers, build_d_samples, collate_d_samples
from loss import PortfolioAwareRankingLoss
from model import MarketIntelligenceRanker
from test_d001_market_features import synthetic_prices


def test_d001_forward_and_loss_smoke() -> None:
    config = load_config()
    config["features"]["sequence_length"] = 20
    config["data"]["min_stocks_per_day"] = 8
    frame = synthetic_prices(days=80, stocks=10)
    samples, columns = build_d_samples(frame, sequence_length=20, min_stocks_per_day=8)
    scalers = DFeatureScalers.fit(samples[:20])
    batch = collate_d_samples(scalers.transform_samples(samples[:2]))
    model = MarketIntelligenceRanker(
        stock_input_dim=len(columns.stock),
        market_input_dim=len(columns.market),
        factor_input_dim=len(columns.factor),
        sequence_length=20,
        config=config["model"],
    )
    outputs = model(batch)
    assert outputs["ranking_scores"].shape == batch["labels"].shape
    criterion = PortfolioAwareRankingLoss(**config["loss"])
    breakdown = criterion(outputs, batch["labels"], batch["relevance"], batch["mask"])
    assert torch.isfinite(breakdown.total)

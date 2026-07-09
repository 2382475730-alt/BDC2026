from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC))

from dataset import build_d_samples, purged_time_split
from featurework import assert_market_truncation_equivalence


def synthetic_prices(days: int = 90, stocks: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for stock_index in range(stocks):
        base = 10.0 + stock_index
        for day_index, date in enumerate(dates):
            trend = 0.01 * day_index
            seasonal = np.sin(day_index / 5.0 + stock_index) * 0.05
            open_price = base + trend + seasonal
            close = open_price * (1.0 + 0.001 * ((day_index + stock_index) % 7 - 3))
            high = max(open_price, close) * 1.01
            low = min(open_price, close) * 0.99
            volume = 1000000 + 1000 * day_index + 500 * stock_index
            rows.append(
                {
                    "股票代码": f"{stock_index:06d}",
                    "日期": date.strftime("%Y-%m-%d"),
                    "开盘": open_price,
                    "收盘": close,
                    "最高": high,
                    "最低": low,
                    "成交量": volume,
                    "成交额": volume * close,
                    "振幅": (high - low) / open_price,
                    "涨跌额": close - open_price,
                    "换手率": 0.5 + 0.01 * stock_index,
                    "涨跌幅": (close - open_price) / open_price,
                }
            )
    return pd.DataFrame(rows)


def test_market_features_are_truncation_equivalent() -> None:
    frame = synthetic_prices()
    assert assert_market_truncation_equivalence(frame, ["2024-04-01", "2024-04-15"])


def test_purged_split_removes_overlapping_label_windows() -> None:
    frame = synthetic_prices()
    samples, _ = build_d_samples(frame, sequence_length=20, min_stocks_per_day=8)
    train, val = purged_time_split(samples, "2024-04-01", "2024-04-15")
    assert train
    assert val
    assert max(sample.label_end_date for sample in train) < pd.Timestamp("2024-04-01")
    assert all(pd.Timestamp("2024-04-01") <= sample.sample_date <= pd.Timestamp("2024-04-15") for sample in val)

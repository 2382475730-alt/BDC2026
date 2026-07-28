from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC))

from featurework import (  # noqa: E402
    BASE_COLUMNS,
    DATE_COL,
    FUTURES_MARKET_FEATURE_COLUMNS,
    INDUSTRY_STOCK_FEATURE_COLUMNS,
    STOCK_COL,
    prepare_d_features,
)


class FusionFeatureTests(unittest.TestCase):
    def market(self) -> pd.DataFrame:
        rows = []
        dates = pd.bdate_range("2024-01-02", periods=90)
        for stock_index, stock_code in enumerate(["000001", "000002", "000003", "000004"]):
            for day_index, date in enumerate(dates):
                base = 10.0 + stock_index + day_index * (0.01 + stock_index * 0.001)
                rows.append(
                    [
                        stock_code,
                        date,
                        base,
                        base * (1.0 + 0.001 * ((day_index + stock_index) % 3 - 1)),
                        base * 1.01,
                        base * 0.99,
                        100000 + day_index * 100 + stock_index,
                        1000000 + day_index * 1000 + stock_index,
                        1.0 + stock_index * 0.1,
                    ]
                )
        return pd.DataFrame(rows, columns=BASE_COLUMNS)

    def industry(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stock_code": ["000001", "000002", "000003", "000004"],
                "industry": ["A", "A", "B", "B"],
            }
        )

    def futures(self) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=90)
        frame = pd.DataFrame({"date": dates})
        for index, column in enumerate(FUTURES_MARKET_FEATURE_COLUMNS):
            frame[column] = (
                1.0
                if column.endswith("_available")
                else np.linspace(0.0, 0.01 * (index + 1), len(frame))
            )
        return frame

    def test_feature_dimensions_and_finite_values(self) -> None:
        processed, market, columns = prepare_d_features(
            self.market(),
            industry_map=self.industry(),
            futures_frame=self.futures(),
            enable_industry=True,
            enable_futures=True,
        )
        self.assertEqual(len(columns.stock), 29)
        self.assertEqual(len(columns.factor), 8)
        self.assertEqual(len(columns.market), 25)
        latest = processed.sort_values(DATE_COL).iloc[-1]
        self.assertTrue(
            np.isfinite(latest[INDUSTRY_STOCK_FEATURE_COLUMNS].astype(float).fillna(0.0)).all()
        )
        self.assertTrue(np.isfinite(market.iloc[-1][FUTURES_MARKET_FEATURE_COLUMNS]).all())

    def test_future_auxiliary_rows_do_not_change_past_features(self) -> None:
        raw = self.market()
        industry = self.industry()
        futures = self.futures()
        cutoff = pd.Timestamp("2024-03-01")
        full_stock, full_market, columns = prepare_d_features(
            raw,
            industry_map=industry,
            futures_frame=futures,
            enable_industry=True,
            enable_futures=True,
        )
        truncated_stock, truncated_market, _ = prepare_d_features(
            raw[pd.to_datetime(raw[DATE_COL]) <= cutoff],
            industry_map=industry,
            futures_frame=futures[pd.to_datetime(futures["date"]) <= cutoff],
            enable_industry=True,
            enable_futures=True,
        )
        left_stock = full_stock[pd.to_datetime(full_stock[DATE_COL]) == cutoff].sort_values(STOCK_COL)
        right_stock = truncated_stock[pd.to_datetime(truncated_stock[DATE_COL]) == cutoff].sort_values(STOCK_COL)
        np.testing.assert_allclose(
            left_stock[columns.stock].fillna(0.0).to_numpy(dtype=float),
            right_stock[columns.stock].fillna(0.0).to_numpy(dtype=float),
            atol=1e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            full_market.loc[cutoff, columns.market].fillna(0.0).to_numpy(dtype=float),
            truncated_market.loc[cutoff, columns.market].fillna(0.0).to_numpy(dtype=float),
            atol=1e-10,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()

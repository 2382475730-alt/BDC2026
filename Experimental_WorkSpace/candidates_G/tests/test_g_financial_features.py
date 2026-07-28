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
    FINANCIAL_STOCK_FEATURE_COLUMNS,
    FUTURES_MARKET_FEATURE_COLUMNS,
    INDUSTRY_STOCK_FEATURE_COLUMNS,
    STOCK_COL,
    prepare_d_features,
)


class FinancialFusionFeatureTests(unittest.TestCase):
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

    def financial(self) -> pd.DataFrame:
        rows = []
        for stock_index, stock_code in enumerate(
            ["000001", "000002", "000003", "000004"]
        ):
            for report_date, available_date, scale in [
                ("2023-12-31", "2024-02-01", 1.0),
                ("2024-03-31", "2024-04-30", 1.1),
                ("2024-06-30", "2024-07-31", 1.2),
            ]:
                base = scale + stock_index * 0.1
                rows.append(
                    {
                        "stock_code": stock_code,
                        "report_date": report_date,
                        "available_date": available_date,
                        "epsjb": base,
                        "bps": base * 5,
                        "roejq": base * 4,
                        "zzcjll": base * 2,
                        "xsjll": base * 10,
                        "zcfzl": 40 + base,
                        "totaloperaterevetz": base * 3,
                        "parentnetprofittz": base * 4,
                        "jyxjlyysr": base * 5,
                        "ld": base * 1.5,
                        "sd": base,
                    }
                )
        return pd.DataFrame(rows)

    def test_feature_dimensions_and_finite_values(self) -> None:
        processed, market, columns = prepare_d_features(
            self.market(),
            industry_map=self.industry(),
            futures_frame=self.futures(),
            financial_frame=self.financial(),
            enable_industry=True,
            enable_futures=True,
            enable_financial=True,
        )
        self.assertEqual(len(columns.stock), 43)
        self.assertEqual(len(columns.factor), 8)
        self.assertEqual(len(columns.market), 25)
        latest = processed.sort_values(DATE_COL).iloc[-1]
        self.assertTrue(
            np.isfinite(latest[INDUSTRY_STOCK_FEATURE_COLUMNS].astype(float).fillna(0.0)).all()
        )
        self.assertTrue(
            np.isfinite(latest[FINANCIAL_STOCK_FEATURE_COLUMNS].astype(float)).all()
        )
        self.assertTrue(np.isfinite(market.iloc[-1][FUTURES_MARKET_FEATURE_COLUMNS]).all())

    def test_future_auxiliary_rows_do_not_change_past_features(self) -> None:
        raw = self.market()
        industry = self.industry()
        futures = self.futures()
        financial = self.financial()
        cutoff = pd.Timestamp("2024-03-01")
        full_stock, full_market, columns = prepare_d_features(
            raw,
            industry_map=industry,
            futures_frame=futures,
            financial_frame=financial,
            enable_industry=True,
            enable_futures=True,
            enable_financial=True,
        )
        truncated_stock, truncated_market, _ = prepare_d_features(
            raw[pd.to_datetime(raw[DATE_COL]) <= cutoff],
            industry_map=industry,
            futures_frame=futures[pd.to_datetime(futures["date"]) <= cutoff],
            financial_frame=financial[
                pd.to_datetime(financial["available_date"]) <= cutoff
            ],
            enable_industry=True,
            enable_futures=True,
            enable_financial=True,
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

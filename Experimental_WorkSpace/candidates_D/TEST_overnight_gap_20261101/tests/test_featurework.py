import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from featurework import prepare_d_features
from dataset import DFeatureScalers, build_d_samples


def sample_data():
    rows = []
    for stock, base in (("000001", 10.0), ("000002", 30.0)):
        for i, date in enumerate(pd.date_range("2024-01-01", periods=75)):
            close = base + i * 0.2
            opening = close - (0.7 if i % 2 else 0.3)
            rows.append({"股票代码": stock, "日期": date, "开盘": opening,
                         "收盘": close, "最高": close + 1, "最低": opening - 1,
                         "成交量": 1000 + i * 13, "成交额": 10000 + i * i * 7,
                         "换手率": 0.01 + i / 10000})
    return pd.DataFrame(rows).sample(frac=1, random_state=12)


class FeatureTests(unittest.TestCase):
    def test_sample_and_scaler_pipeline_uses_new_schema(self):
        samples, columns = build_d_samples(sample_data(), sequence_length=60, min_stocks_per_day=2)
        self.assertTrue(samples)
        self.assertEqual(samples[0].stock_sequence.shape[-1], 19)
        self.assertEqual(columns.stock[7], "overnight_return")
        scaled = DFeatureScalers.fit(samples).transform_sample(samples[0])
        self.assertTrue(np.isfinite(scaled.stock_sequence).all())

    def test_gap_uses_previous_close_per_stock_and_retains_19_features(self):
        result, _, columns = prepare_d_features(sample_data())
        self.assertEqual(len(columns.stock), 19)
        self.assertEqual(len(set(columns.stock)), 19)
        self.assertNotIn("open_close_spread", columns.stock)
        self.assertIn("overnight_return", columns.stock)
        for _, group in result.groupby("股票代码"):
            group = group.sort_values("日期")
            self.assertTrue(pd.isna(group.iloc[0]["overnight_return"]))
            row, previous = group.iloc[1], group.iloc[0]
            self.assertAlmostEqual(row["overnight_return"], (row["开盘"] - previous["收盘"]) / previous["收盘"])
            self.assertAlmostEqual(row["intraday_return"], (row["收盘"] - row["开盘"]) / row["开盘"])
            self.assertNotAlmostEqual(row["overnight_return"], row["intraday_return"])

    def test_prefix_values_do_not_depend_on_future_rows(self):
        raw = sample_data()
        cutoff = pd.Timestamp("2024-03-01")
        full, _, columns = prepare_d_features(raw)
        prefix, _, _ = prepare_d_features(raw[raw["日期"] <= cutoff])
        full = full[full["日期"] <= cutoff].sort_values(["股票代码", "日期"])
        prefix = prefix.sort_values(["股票代码", "日期"])
        np.testing.assert_allclose(full[columns.stock].to_numpy(), prefix[columns.stock].to_numpy(),
                                   equal_nan=True, rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

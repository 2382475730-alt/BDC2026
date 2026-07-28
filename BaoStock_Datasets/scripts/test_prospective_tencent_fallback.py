from __future__ import annotations

import unittest

import pandas as pd

from BaoStock_Datasets.scripts.download_baostock_expanded_data import TRAIN_COLUMNS
from BaoStock_Datasets.scripts.prospective_tencent_fallback import (
    calibrate_tencent_causal_increment,
)


def base_market(codes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2026-06-26", periods=6, freq="D")
    for code_index, code in enumerate(codes):
        for day_index, date in enumerate(dates):
            close = 10.0 + code_index + day_index * 0.1
            volume = 100_000.0 + code_index * 1_000.0 + day_index * 100.0
            rows.append(
                {
                    "股票代码": code,
                    "日期": date.strftime("%Y-%m-%d"),
                    "开盘": close - 0.1,
                    "收盘": close,
                    "最高": close + 0.2,
                    "最低": close - 0.2,
                    "成交量": volume,
                    "成交额": volume * close * 0.99,
                    "振幅": 4.0,
                    "涨跌额": 0.1,
                    "换手率": volume / 10_000_000.0 * 100.0,
                    "涨跌幅": 1.0,
                }
            )
    return pd.DataFrame(rows, columns=TRAIN_COLUMNS)


def tencent_qfq(base: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code, group in base.groupby("股票代码"):
        volume_divisor = 100.0 if str(code).startswith("688") else 1.0
        for row in group.to_dict("records"):
            rows.append(
                {
                    **row,
                    "开盘": float(row["开盘"]) / 2.0,
                    "收盘": float(row["收盘"]) / 2.0,
                    "最高": float(row["最高"]) / 2.0,
                    "最低": float(row["最低"]) / 2.0,
                    "成交量": float(row["成交量"]) / volume_divisor,
                    "成交额": 0.0,
                    "换手率": 0.0,
                }
            )
        last = group.sort_values("日期").iloc[-1]
        for offset, date in enumerate(
            pd.date_range(
                pd.Timestamp(last["日期"]) + pd.Timedelta(days=1),
                cutoff,
                freq="D",
            ),
            start=1,
        ):
            close = float(last["收盘"]) + offset * 0.1
            volume = (float(last["成交量"]) + offset * 100.0) / volume_divisor
            rows.append(
                {
                    "股票代码": code,
                    "日期": date.strftime("%Y-%m-%d"),
                    "开盘": (close - 0.1) / 2.0,
                    "收盘": close / 2.0,
                    "最高": (close + 0.2) / 2.0,
                    "最低": (close - 0.2) / 2.0,
                    "成交量": volume,
                    "成交额": 0.0,
                    "振幅": 0.0,
                    "涨跌额": 0.0,
                    "换手率": 0.0,
                    "涨跌幅": 0.0,
                }
            )
    return pd.DataFrame(rows, columns=TRAIN_COLUMNS)


class TencentProspectiveFallbackTests(unittest.TestCase):
    def test_calibration_reconstructs_fields_without_replacing_history(self) -> None:
        codes = ["000001", "688008"]
        base = base_market(codes)
        downloaded = tencent_qfq(base, "2026-07-03")
        increment, calibration, report = calibrate_tencent_causal_increment(
            base,
            downloaded,
            cutoff="2026-07-03",
            minimum_calibrated_stocks=2,
            minimum_overlap_rows=5,
            maximum_return_diff_p95=0.0025,
            maximum_scale_cv_p95=0.01,
        )

        self.assertEqual(list(increment.columns), TRAIN_COLUMNS)
        self.assertEqual(increment["日期"].min(), "2026-07-02")
        self.assertEqual(increment["日期"].max(), "2026-07-03")
        self.assertEqual(increment["股票代码"].nunique(), 2)
        self.assertTrue((increment["成交额"] > 0).all())
        self.assertTrue((increment["换手率"] > 0).all())
        self.assertEqual(report["history_overlap_rows_written"], 0)
        self.assertEqual(report["future_rows"], 0)
        self.assertFalse(report["source_mix"])
        star = calibration.loc[calibration["股票代码"] == "688008"].iloc[0]
        self.assertAlmostEqual(float(star["volume_scale"]), 100.0, places=6)

    def test_calibration_rejects_bad_overlap_quality(self) -> None:
        codes = ["000001", "000002"]
        base = base_market(codes)
        downloaded = tencent_qfq(base, "2026-07-03")
        mask = downloaded["股票代码"] == "000002"
        downloaded.loc[mask, "收盘"] = [
            5.0,
            8.0,
            4.0,
            9.0,
            3.0,
            10.0,
            2.0,
            11.0,
        ]
        with self.assertRaisesRegex(ValueError, "return-difference"):
            calibrate_tencent_causal_increment(
                base,
                downloaded,
                cutoff="2026-07-03",
                minimum_calibrated_stocks=2,
                minimum_overlap_rows=5,
                maximum_return_diff_p95=0.0025,
                maximum_scale_cv_p95=0.01,
            )

    def test_calibration_rejects_future_rows(self) -> None:
        base = base_market(["000001", "000002"])
        downloaded = tencent_qfq(base, "2026-07-04")
        with self.assertRaisesRegex(ValueError, "future"):
            calibrate_tencent_causal_increment(
                base,
                downloaded,
                cutoff="2026-07-03",
                minimum_calibrated_stocks=2,
                minimum_overlap_rows=5,
                maximum_return_diff_p95=0.0025,
                maximum_scale_cv_p95=0.01,
            )


if __name__ == "__main__":
    unittest.main()

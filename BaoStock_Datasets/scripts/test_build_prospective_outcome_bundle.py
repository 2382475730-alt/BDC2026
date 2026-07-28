from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

import BaoStock_Datasets.scripts.build_prospective_outcome_bundle as outcome_builder
from BaoStock_Datasets.scripts.build_prospective_outcome_bundle import (
    assert_scoring_time,
    validate_outcome_market,
)
from BaoStock_Datasets.scripts.download_baostock_expanded_data import DownloadReport


CN = ZoneInfo("Asia/Shanghai")
TRAIN_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]


def stock_market(dates: list[str], codes: list[str]) -> pd.DataFrame:
    rows = []
    for date in dates:
        for code in codes:
            rows.append(
                {
                    "股票代码": code,
                    "日期": date,
                    "开盘": 10.0,
                    "收盘": 10.2,
                    "最高": 10.5,
                    "最低": 9.8,
                    "成交量": 1000.0,
                    "成交额": 10000.0,
                    "振幅": 7.0,
                    "涨跌额": 0.2,
                    "换手率": 1.0,
                    "涨跌幅": 2.0,
                }
            )
    return pd.DataFrame(rows, columns=TRAIN_COLUMNS)


class ProspectiveOutcomeBundleTests(unittest.TestCase):
    def test_scoring_time_gate_opens_only_after_sell_open(self) -> None:
        with self.assertRaisesRegex(ValueError, "time gate"):
            assert_scoring_time(
                "2026-08-07",
                datetime(2026, 8, 7, 9, 30, tzinfo=CN),
            )
        report = assert_scoring_time(
            "2026-08-07",
            datetime(2026, 8, 7, 9, 31, tzinfo=CN),
        )
        self.assertEqual(report["status"], "PASS")

    def test_market_validation_rejects_later_rows_and_universe_drift(self) -> None:
        universe = {"000001", "000002"}
        with self.assertRaisesRegex(ValueError, "later-window"):
            validate_outcome_market(
                stock_market(
                    ["2026-08-03", "2026-08-07", "2026-08-10"],
                    sorted(universe),
                ),
                buy_date="2026-08-03",
                sell_date="2026-08-07",
                universe=universe,
                minimum_endpoint_stocks=2,
            )
        with self.assertRaisesRegex(ValueError, "universe"):
            validate_outcome_market(
                stock_market(
                    ["2026-08-03", "2026-08-07"],
                    ["000001", "999999"],
                ),
                buy_date="2026-08-03",
                sell_date="2026-08-07",
                universe=universe,
                minimum_endpoint_stocks=1,
            )

    def test_market_validation_requires_both_endpoints_and_coverage(self) -> None:
        universe = {"000001", "000002"}
        with self.assertRaisesRegex(ValueError, "sell endpoint"):
            validate_outcome_market(
                stock_market(["2026-08-03"], sorted(universe)),
                buy_date="2026-08-03",
                sell_date="2026-08-07",
                universe=universe,
                minimum_endpoint_stocks=2,
            )
        sparse = stock_market(
            ["2026-08-03", "2026-08-07"],
            ["000001"],
        )
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_outcome_market(
                sparse,
                buy_date="2026-08-03",
                sell_date="2026-08-07",
                universe=universe,
                minimum_endpoint_stocks=2,
            )

    def test_full_bundle_is_external_causal_and_non_overwriting(self) -> None:
        codes = [f"{index:06d}" for index in range(1, 301)]
        prefix = stock_market(["2026-07-31"], codes)
        outcome = stock_market(["2026-08-03", "2026-08-07"], codes)
        reports = [
            DownloadReport(
                symbol=code,
                rows=2,
                start_date="2026-08-03",
                end_date="2026-08-07",
                ok=True,
            )
            for code in codes
        ]
        sse = pd.DataFrame(
            {
                "日期": ["2026-08-03", "2026-08-07"],
                "开盘": [4000.0, 4020.0],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix_path = root / "prefix.csv"
            out_dir = root / "external-vault" / "outcomes" / "P01"
            prefix.to_csv(prefix_path, index=False, encoding="utf-8")
            with (
                mock.patch.object(
                    outcome_builder,
                    "download_universe",
                    return_value=(outcome, reports),
                ) as stock_download,
                mock.patch.object(
                    outcome_builder,
                    "download_sse_open",
                    return_value=sse,
                ) as index_download,
            ):
                manifest = outcome_builder.build_outcome_bundle(
                    window_id="P01",
                    training_prefix=prefix_path,
                    buy_date="2026-08-03",
                    sell_date="2026-08-07",
                    out_dir=out_dir,
                    now=datetime(2026, 8, 7, 9, 35, tzinfo=CN),
                    sleep_sec=0.0,
                    login_retries=1,
                    retry_sec=0.0,
                )
            self.assertFalse(manifest["later_window_query_performed"])
            self.assertEqual(manifest["validation"]["sell_date"], "2026-08-07")
            self.assertTrue((out_dir / "outcome_market.csv").is_file())
            self.assertTrue((out_dir / "sse_open.csv").is_file())
            self.assertEqual(stock_download.call_args.args[1], "2026-08-03")
            self.assertEqual(stock_download.call_args.args[2], "2026-08-07")
            index_download.assert_called_once()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                outcome_builder.build_outcome_bundle(
                    window_id="P01",
                    training_prefix=prefix_path,
                    buy_date="2026-08-03",
                    sell_date="2026-08-07",
                    out_dir=out_dir,
                    now=datetime(2026, 8, 7, 9, 36, tzinfo=CN),
                    sleep_sec=0.0,
                    login_retries=1,
                    retry_sec=0.0,
                )

    def test_outcome_switches_stocks_and_sse_together_to_tencent(self) -> None:
        codes = [f"{index:06d}" for index in range(1, 301)]
        prefix = stock_market(["2026-07-31"], codes)
        outcome = stock_market(["2026-08-03", "2026-08-07"], codes)
        reports = [
            {
                "symbol": code,
                "source": "akshare.stock_zh_a_hist_tx:qfq",
                "rows": 2,
                "ok": True,
                "error": "",
            }
            for code in codes
        ]
        sse = pd.DataFrame(
            {
                "日期": ["2026-08-03", "2026-08-07"],
                "开盘": [4000.0, 4020.0],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix_path = root / "prefix.csv"
            out_dir = root / "external-vault" / "outcomes" / "P01"
            prefix.to_csv(prefix_path, index=False, encoding="utf-8")
            with (
                mock.patch.object(
                    outcome_builder,
                    "download_universe",
                    return_value=(outcome, []),
                ),
                mock.patch.object(
                    outcome_builder,
                    "download_sse_open",
                    side_effect=RuntimeError("BaoStock unavailable"),
                ),
                mock.patch.object(
                    outcome_builder,
                    "download_tencent_outcome_market",
                    return_value=(outcome, reports),
                ) as stock_fallback,
                mock.patch.object(
                    outcome_builder,
                    "download_tencent_sse_open",
                    return_value=sse,
                ) as sse_fallback,
            ):
                manifest = outcome_builder.build_outcome_bundle(
                    window_id="P01",
                    training_prefix=prefix_path,
                    buy_date="2026-08-03",
                    sell_date="2026-08-07",
                    out_dir=out_dir,
                    now=datetime(2026, 8, 7, 9, 35, tzinfo=CN),
                    sleep_sec=0.0,
                    login_retries=1,
                    retry_sec=0.0,
                )
        self.assertEqual(manifest["source"], "Tencent")
        self.assertEqual(manifest["stock_adjustflag"], "qfq")
        self.assertFalse(manifest["source_mix"])
        self.assertEqual(manifest["attempts"][0]["status"], "FAILED")
        self.assertEqual(manifest["attempts"][1]["status"], "PASS")
        stock_fallback.assert_called_once()
        sse_fallback.assert_called_once_with(
            "2026-08-03",
            "2026-08-07",
            1,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

import BaoStock_Datasets.scripts.build_prospective_causal_prefix as prefix_builder
from BaoStock_Datasets.scripts.build_prospective_causal_prefix import (
    assert_acquisition_time,
    causal_query_bounds,
    validate_and_merge_prefix,
)
from BaoStock_Datasets.scripts.download_baostock_expanded_data import DownloadReport


CN = ZoneInfo("Asia/Shanghai")
COLUMNS = [
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


def market_rows(dates: list[str], codes: list[str]) -> pd.DataFrame:
    rows = []
    for date in dates:
        for code in codes:
            rows.append(
                {
                    "股票代码": code,
                    "日期": date,
                    "开盘": 10.0,
                    "收盘": 10.5,
                    "最高": 10.8,
                    "最低": 9.8,
                    "成交量": 1000.0,
                    "成交额": 10000.0,
                    "振幅": 10.0,
                    "涨跌额": 0.5,
                    "换手率": 1.0,
                    "涨跌幅": 5.0,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


class ProspectiveCausalPrefixTests(unittest.TestCase):
    def test_query_bounds_begin_after_base_and_end_at_decision(self) -> None:
        base = market_rows(["2026-07-02", "2026-07-03"], ["000001", "000002"])
        bounds = causal_query_bounds(base, "2026-07-31")
        self.assertEqual(bounds["start_date"], "2026-07-04")
        self.assertEqual(bounds["end_date"], "2026-07-31")

    def test_acquisition_time_gate_rejects_preclose_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "time gate"):
            assert_acquisition_time(
                "2026-07-31",
                datetime(2026, 7, 31, 17, 59, tzinfo=CN),
            )
        report = assert_acquisition_time(
            "2026-07-31",
            datetime(2026, 7, 31, 18, 1, tzinfo=CN),
        )
        self.assertEqual(report["status"], "PASS")

    def test_merge_accepts_only_strictly_new_causal_rows(self) -> None:
        codes = ["000001", "000002"]
        base = market_rows(["2026-07-02", "2026-07-03"], codes)
        increment = market_rows(["2026-07-30", "2026-07-31"], codes)
        merged, report = validate_and_merge_prefix(
            base,
            increment,
            cutoff="2026-07-31",
            minimum_cutoff_stocks=2,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["date_end"], "2026-07-31")
        self.assertEqual(report["future_rows"], 0)
        self.assertEqual(len(merged), 8)
        self.assertEqual(list(merged.columns), COLUMNS)

    def test_merge_rejects_future_overlap_and_universe_drift(self) -> None:
        base = market_rows(["2026-07-03"], ["000001", "000002"])
        with self.assertRaisesRegex(ValueError, "future"):
            validate_and_merge_prefix(
                base,
                market_rows(["2026-08-03"], ["000001", "000002"]),
                cutoff="2026-07-31",
                minimum_cutoff_stocks=2,
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_and_merge_prefix(
                base,
                market_rows(["2026-07-03", "2026-07-31"], ["000001", "000002"]),
                cutoff="2026-07-31",
                minimum_cutoff_stocks=2,
            )
        with self.assertRaisesRegex(ValueError, "universe"):
            validate_and_merge_prefix(
                base,
                market_rows(["2026-07-31"], ["000001", "999999"]),
                cutoff="2026-07-31",
                minimum_cutoff_stocks=1,
            )

    def test_merge_rejects_incomplete_cutoff_or_duplicate_rows(self) -> None:
        base = market_rows(["2026-07-03"], ["000001", "000002"])
        with self.assertRaisesRegex(ValueError, "cutoff coverage"):
            validate_and_merge_prefix(
                base,
                market_rows(["2026-07-31"], ["000001"]),
                cutoff="2026-07-31",
                minimum_cutoff_stocks=2,
            )
        duplicated = pd.concat(
            [
                market_rows(["2026-07-31"], ["000001", "000002"]),
                market_rows(["2026-07-31"], ["000001"]),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_and_merge_prefix(
                base,
                duplicated,
                cutoff="2026-07-31",
                minimum_cutoff_stocks=2,
            )

    def test_full_builder_writes_provenance_without_future_query(self) -> None:
        codes = [f"{index:06d}" for index in range(1, 301)]
        base = market_rows(["2026-07-02", "2026-07-03"], codes)
        increment = market_rows(["2026-07-30", "2026-07-31"], codes)
        reports = [
            DownloadReport(
                symbol=code,
                rows=2,
                start_date="2026-07-30",
                end_date="2026-07-31",
                ok=True,
            )
            for code in codes
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = root / "base.csv"
            out_dir = root / "P01"
            base.to_csv(base_path, index=False, encoding="utf-8")
            with mock.patch.object(
                prefix_builder,
                "download_universe",
                return_value=(increment, reports),
            ) as mocked_download:
                manifest = prefix_builder.build_causal_prefix(
                    base_prefix=base_path,
                    cutoff="2026-07-31",
                    out_dir=out_dir,
                    now=datetime(2026, 7, 31, 18, 1, tzinfo=CN),
                    sleep_sec=0.0,
                    login_retries=1,
                    retry_sec=0.0,
                )
            self.assertFalse(manifest["future_query_performed"])
            self.assertEqual(manifest["query_bounds"]["end_date"], "2026-07-31")
            self.assertEqual(manifest["validation"]["cutoff_stocks"], 300)
            self.assertTrue((out_dir / "shared_train_prefix.csv").is_file())
            args = mocked_download.call_args.args
            self.assertEqual(args[1], "2026-07-04")
            self.assertEqual(args[2], "2026-07-31")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                prefix_builder.build_causal_prefix(
                    base_prefix=base_path,
                    cutoff="2026-07-31",
                    out_dir=out_dir,
                    now=datetime(2026, 7, 31, 18, 2, tzinfo=CN),
                    sleep_sec=0.0,
                    login_retries=1,
                    retry_sec=0.0,
                )

    def test_builder_switches_entire_window_to_frozen_tencent_fallback(self) -> None:
        codes = [f"{index:06d}" for index in range(1, 301)]
        base = market_rows(["2026-07-02", "2026-07-03"], codes)
        increment = market_rows(["2026-07-30", "2026-07-31"], codes)
        reports = [
            {
                "symbol": code,
                "source": "akshare.stock_zh_a_hist_tx:qfq",
                "rows": 4,
                "ok": True,
                "error": "",
            }
            for code in codes
        ]
        calibration = pd.DataFrame(
            {
                "股票代码": codes,
                "price_scale": [2.0] * len(codes),
                "volume_scale": [1.0] * len(codes),
            }
        )
        fallback_report = {
            "status": "PASS",
            "source_mix": False,
            "future_rows": 0,
            "history_overlap_rows_written": 0,
            "calibrated_stocks": 300,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = root / "base.csv"
            out_dir = root / "P01"
            base.to_csv(base_path, index=False, encoding="utf-8")
            with (
                mock.patch.object(
                    prefix_builder,
                    "download_universe",
                    side_effect=RuntimeError("BaoStock unavailable"),
                ),
                mock.patch.object(
                    prefix_builder,
                    "build_tencent_causal_increment",
                    return_value=(
                        increment,
                        reports,
                        calibration,
                        fallback_report,
                    ),
                ) as fallback,
            ):
                manifest = prefix_builder.build_causal_prefix(
                    base_prefix=base_path,
                    cutoff="2026-07-31",
                    out_dir=out_dir,
                    now=datetime(2026, 7, 31, 18, 1, tzinfo=CN),
                    sleep_sec=0.0,
                    login_retries=1,
                    retry_sec=0.0,
                )
        self.assertEqual(manifest["source"], "Tencent")
        self.assertEqual(manifest["source_policy"], "baostock_then_tencent_qfq")
        self.assertFalse(manifest["source_mix"])
        self.assertEqual(manifest["adjustflag"], "qfq")
        self.assertEqual(manifest["attempts"][0]["status"], "FAILED")
        self.assertEqual(manifest["attempts"][1]["status"], "PASS")
        fallback.assert_called_once()


if __name__ == "__main__":
    unittest.main()

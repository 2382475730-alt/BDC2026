from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .common import date_strings, md5_file, read_market_csv, sorted_trading_dates, write_json


def _first_on_or_after(dates: list[pd.Timestamp], value: str) -> pd.Timestamp:
    target = pd.Timestamp(value).normalize()
    for date in dates:
        if date >= target:
            return date
    raise ValueError(f"no trading date on or after {value}")


def _year_dates(dates: list[pd.Timestamp], year: int) -> list[pd.Timestamp]:
    return [date for date in dates if date.year == year]


def build_research_split(
    data_path: str | Path,
    *,
    train_year: int = 2022,
    validation_year: int = 2025,
    validation_end_year: int = 2026,
    raw_data_end: str = "2026-07-03",
    sell_offset: int = 5,
) -> dict[str, Any]:
    frame, columns = read_market_csv(data_path, usecols=None)
    dates = sorted_trading_dates(frame, columns.date)
    if not dates:
        raise ValueError("data contains no trading dates")

    train_start = _first_on_or_after(dates, f"{train_year}-01-01")
    validation_dates = _year_dates(dates, validation_year)
    if len(validation_dates) < 2:
        raise ValueError(f"data has fewer than two trading dates in {validation_year}")
    validation_start = validation_dates[1]
    validation_end = _first_on_or_after(dates, f"{validation_end_year}-01-01")

    val_start_index = dates.index(validation_start)
    train_end_index = val_start_index - sell_offset - 1
    if train_end_index < 0:
        raise ValueError("not enough purge gap before validation start")
    train_end = dates[train_end_index]

    available_raw_end = max(date for date in dates if date <= pd.Timestamp(raw_data_end).normalize())
    raw_end_index = dates.index(available_raw_end)
    production_train_index = raw_end_index - sell_offset
    production_train_end = dates[production_train_index] if production_train_index >= 0 else None

    train_dates = [date for date in dates if train_start <= date <= train_end]
    val_dates = [date for date in dates if validation_start <= date <= validation_end]
    payload: dict[str, Any] = {
        "split_version": "research_2022_train_2025_validation_t5_purged_v1",
        "data_path": str(Path(data_path).resolve()),
        "data_md5": md5_file(data_path),
        "date_column": columns.date,
        "stock_column": columns.stock,
        "open_column": columns.open,
        "label": "open(T+5) / open(T+1) - 1",
        "sell_offset": sell_offset,
        "research_train": {
            "sample_start": train_start.strftime("%Y-%m-%d"),
            "sample_end": train_end.strftime("%Y-%m-%d"),
            "sample_count": len(train_dates),
            "max_label_end_before_validation_start": dates[train_end_index + sell_offset].strftime("%Y-%m-%d"),
        },
        "research_validation": {
            "sample_start": validation_start.strftime("%Y-%m-%d"),
            "sample_end": validation_end.strftime("%Y-%m-%d"),
            "sample_count": len(val_dates),
        },
        "production_refit": {
            "raw_data_end": available_raw_end.strftime("%Y-%m-%d"),
            "train_sample_end": production_train_end.strftime("%Y-%m-%d") if production_train_end is not None else None,
            "feature_end_for_inference": available_raw_end.strftime("%Y-%m-%d"),
        },
        "trading_dates": {
            "first": dates[0].strftime("%Y-%m-%d"),
            "last": dates[-1].strftime("%Y-%m-%d"),
            "count": len(dates),
            "research_train_tail": date_strings(train_dates[-5:]),
            "research_validation_head": date_strings(val_dates[:5]),
        },
        "purge_checks": {
            "train_label_end_before_validation_start": dates[train_end_index + sell_offset] < validation_start,
            "date_level_split": True,
            "purge_trading_days": sell_offset,
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D-series research split metadata with T+5 purge.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-data-end", default="2026-07-03")
    parser.add_argument("--sell-offset", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json(args.output, build_research_split(args.data_path, raw_data_end=args.raw_data_end, sell_offset=args.sell_offset))


if __name__ == "__main__":
    main()


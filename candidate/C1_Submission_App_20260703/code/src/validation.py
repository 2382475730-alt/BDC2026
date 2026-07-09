from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


DATE_COL = "日期"

DEFAULT_WALK_FORWARD_FOLDS: tuple[tuple[str, str, str], ...] = (
    ("fold1", "2024-01-02", "2024-06-28"),
    ("fold2", "2024-07-01", "2024-12-31"),
    ("fold3", "2025-01-02", "2025-06-30"),
    ("fold4", "2025-07-01", "2025-12-31"),
    ("fold5", "2026-01-19", "2026-03-12"),
)


@dataclass(frozen=True)
class FoldPlan:
    name: str
    requested_val_start: pd.Timestamp
    requested_val_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    train_sample_dates: list[pd.Timestamp]
    val_sample_dates: list[pd.Timestamp]
    max_train_label_end_date: pd.Timestamp | None
    min_val_label_end_date: pd.Timestamp | None
    max_val_label_end_date: pd.Timestamp | None
    train_row_count: int
    val_row_count: int
    checks: dict[str, bool]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        def fmt(value: pd.Timestamp | None) -> str | None:
            if value is None:
                return None
            return pd.Timestamp(value).strftime("%Y-%m-%d")

        return {
            "name": self.name,
            "requested_val_start": fmt(self.requested_val_start),
            "requested_val_end": fmt(self.requested_val_end),
            "val_start": fmt(self.val_start),
            "val_end": fmt(self.val_end),
            "train_start": fmt(self.train_start),
            "train_end": fmt(self.train_end),
            "train_sample_dates": [fmt(date) for date in self.train_sample_dates],
            "val_sample_dates": [fmt(date) for date in self.val_sample_dates],
            "max_train_label_end_date": fmt(self.max_train_label_end_date),
            "min_val_label_end_date": fmt(self.min_val_label_end_date),
            "max_val_label_end_date": fmt(self.max_val_label_end_date),
            "train_sample_count": len(self.train_sample_dates),
            "val_sample_count": len(self.val_sample_dates),
            "train_row_count": self.train_row_count,
            "val_row_count": self.val_row_count,
            "checks": dict(self.checks),
            "warnings": list(self.warnings),
        }


def normalize_date(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def trading_calendar(df: pd.DataFrame, date_col: str = DATE_COL) -> list[pd.Timestamp]:
    if date_col not in df.columns:
        raise KeyError(f"missing date column: {date_col}")
    dates = pd.to_datetime(df[date_col]).dt.normalize().dropna().unique()
    return sorted(pd.Timestamp(date) for date in dates)


def label_end_dates(
    dates: Sequence[pd.Timestamp],
    horizon: int,
) -> dict[pd.Timestamp, pd.Timestamp]:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return {
        pd.Timestamp(date): pd.Timestamp(dates[index + horizon])
        for index, date in enumerate(dates)
        if index + horizon < len(dates)
    }


def valid_sample_dates(
    dates: Sequence[pd.Timestamp],
    sequence_length: int,
    horizon: int,
) -> list[pd.Timestamp]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return [
        pd.Timestamp(date)
        for index, date in enumerate(dates)
        if index >= sequence_length - 1 and index + horizon < len(dates)
    ]


def row_count_for_dates(
    df: pd.DataFrame,
    sample_dates: Iterable[pd.Timestamp],
    date_col: str = DATE_COL,
) -> int:
    sample_date_set = {normalize_date(date) for date in sample_dates}
    if not sample_date_set:
        return 0
    normalized_dates = pd.to_datetime(df[date_col]).dt.normalize()
    return int(normalized_dates.isin(sample_date_set).sum())


def parse_fold_ranges(
    fold_ranges: Sequence[tuple[str, str, str] | tuple[str, str]] | None,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    source = fold_ranges or DEFAULT_WALK_FORWARD_FOLDS
    parsed: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for index, entry in enumerate(source, start=1):
        if len(entry) == 3:
            name, start, end = entry  # type: ignore[misc]
        elif len(entry) == 2:
            start, end = entry  # type: ignore[misc]
            name = f"fold{index}"
        else:
            raise ValueError(f"invalid fold range: {entry}")
        parsed.append((str(name), normalize_date(start), normalize_date(end)))
    return parsed


def build_walk_forward_folds(
    df: pd.DataFrame,
    sequence_length: int,
    horizon: int = 5,
    fold_ranges: Sequence[tuple[str, str, str] | tuple[str, str]] | None = None,
    date_col: str = DATE_COL,
    min_stocks_per_day: int = 1,
    skip_empty: bool = False,
) -> list[FoldPlan]:
    dates = trading_calendar(df, date_col=date_col)
    if not dates:
        raise ValueError("cannot build folds from an empty calendar")

    labels_by_date = label_end_dates(dates, horizon=horizon)
    valid_dates = valid_sample_dates(dates, sequence_length=sequence_length, horizon=horizon)
    valid_date_set = set(valid_dates)
    counts_by_date = (
        pd.to_datetime(df[date_col])
        .dt.normalize()
        .value_counts()
        .rename_axis("date")
        .to_dict()
    )
    eligible_dates = [
        date
        for date in valid_dates
        if int(counts_by_date.get(pd.Timestamp(date), 0)) >= min_stocks_per_day
    ]

    folds: list[FoldPlan] = []
    for name, requested_start, requested_end in parse_fold_ranges(fold_ranges):
        val_dates = [
            date
            for date in eligible_dates
            if requested_start <= date <= requested_end
        ]
        if not val_dates:
            if skip_empty:
                continue
            raise ValueError(
                f"{name} has no valid validation sample_end_date in "
                f"{requested_start.date()}..{requested_end.date()}"
            )

        val_start = val_dates[0]
        val_end = val_dates[-1]
        train_dates = [
            date
            for date in eligible_dates
            if date < val_start and labels_by_date.get(date, pd.Timestamp.max) < val_start
        ]
        warnings: list[str] = []
        if not train_dates:
            warnings.append("fold has no training sample dates after purge")

        train_label_end_dates = [labels_by_date[date] for date in train_dates if date in labels_by_date]
        val_label_end_dates = [labels_by_date[date] for date in val_dates if date in labels_by_date]
        max_train_label_end = max(train_label_end_dates) if train_label_end_dates else None
        min_val_label_end = min(val_label_end_dates) if val_label_end_dates else None
        max_val_label_end = max(val_label_end_dates) if val_label_end_dates else None

        no_overlap = set(train_dates).isdisjoint(val_dates)
        checks = {
            "no_train_val_overlap": no_overlap,
            "train_label_end_before_val_start": (
                max_train_label_end is None or max_train_label_end < val_start
            ),
            "val_labels_complete": all(date in labels_by_date for date in val_dates),
            "val_history_complete": all(date in valid_date_set for date in val_dates),
            "scaler_fit_scope": True,
        }

        folds.append(
            FoldPlan(
                name=name,
                requested_val_start=requested_start,
                requested_val_end=requested_end,
                val_start=val_start,
                val_end=val_end,
                train_start=train_dates[0] if train_dates else None,
                train_end=train_dates[-1] if train_dates else None,
                train_sample_dates=list(train_dates),
                val_sample_dates=list(val_dates),
                max_train_label_end_date=max_train_label_end,
                min_val_label_end_date=min_val_label_end,
                max_val_label_end_date=max_val_label_end,
                train_row_count=row_count_for_dates(df, train_dates, date_col=date_col),
                val_row_count=row_count_for_dates(df, val_dates, date_col=date_col),
                checks=checks,
                warnings=warnings,
            )
        )
    return folds


def context_bounds_for_sample_dates(
    all_dates: Sequence[pd.Timestamp],
    sample_dates: Sequence[pd.Timestamp],
    sequence_length: int,
    horizon: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not sample_dates:
        raise ValueError("sample_dates cannot be empty")
    index_by_date = {pd.Timestamp(date): index for index, date in enumerate(all_dates)}
    first_index = min(index_by_date[pd.Timestamp(date)] for date in sample_dates)
    last_index = max(index_by_date[pd.Timestamp(date)] for date in sample_dates)
    context_start_index = max(0, first_index - sequence_length + 1)
    context_end_index = min(len(all_dates) - 1, last_index + horizon)
    return all_dates[context_start_index], all_dates[context_end_index]

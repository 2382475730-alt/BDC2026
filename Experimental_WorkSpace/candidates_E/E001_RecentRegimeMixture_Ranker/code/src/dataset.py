from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from featurework import DATE_COL, STOCK_COL, FeatureColumns, prepare_e_features


@dataclass
class ESample:
    sample_date: pd.Timestamp
    label_end_date: pd.Timestamp | None
    market_sequence: np.ndarray
    stock_sequence: np.ndarray
    labels: np.ndarray
    relevance: np.ndarray
    stock_codes: list[str]


class EStockRankingDataset(Dataset):
    def __init__(self, samples: Sequence[ESample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ESample:
        return self.samples[index]


def _online_moments(arrays: Iterable[np.ndarray], width: int, zero_first: bool) -> tuple[list[float], list[float], list[int]]:
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    count = np.zeros(width, dtype=np.int64)
    for values in arrays:
        matrix = np.asarray(values, dtype=np.float64).reshape(-1, width)
        if zero_first:
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            valid = np.ones(matrix.shape, dtype=bool)
        else:
            valid = np.isfinite(matrix)
            matrix = np.where(valid, matrix, 0.0)
        total += matrix.sum(axis=0)
        total_sq += np.square(matrix).sum(axis=0)
        count += valid.sum(axis=0)
    safe_count = np.maximum(count, 1)
    mean = total / safe_count
    variance = np.maximum(total_sq / safe_count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.tolist(), std.tolist(), count.astype(int).tolist()


@dataclass
class EFeatureScalers:
    stock_mean: list[float]
    stock_std: list[float]
    stock_count: list[int]
    market_mean: list[float]
    market_std: list[float]
    market_count: list[int]
    mode: str
    fit_start_date: str
    fit_end_date: str
    fit_sample_count: int
    stock_feature_names: list[str]
    market_feature_names: list[str]

    @classmethod
    def fit(
        cls,
        samples: Sequence[ESample],
        columns: FeatureColumns,
        mode: str = "finite_train_only",
    ) -> "EFeatureScalers":
        if not samples:
            raise ValueError("cannot fit scalers on empty samples")
        if mode not in {"finite_train_only", "legacy_zero_first"}:
            raise ValueError(f"unsupported scaler mode: {mode}")
        zero_first = mode == "legacy_zero_first"
        stock_mean, stock_std, stock_count = _online_moments(
            (sample.stock_sequence for sample in samples), len(columns.stock), zero_first
        )
        market_mean, market_std, market_count = _online_moments(
            (sample.market_sequence for sample in samples), len(columns.market), zero_first
        )
        dates = sorted(pd.Timestamp(sample.sample_date).normalize() for sample in samples)
        return cls(
            stock_mean, stock_std, stock_count,
            market_mean, market_std, market_count,
            mode, dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"),
            len(samples), list(columns.stock), list(columns.market),
        )

    @staticmethod
    def _transform(values: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
        result = (values.astype(np.float32) - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)

    def transform_sample(self, sample: ESample) -> ESample:
        # In-place replacement prevents a second multi-gigabyte sequence copy.
        sample.stock_sequence = self._transform(sample.stock_sequence, self.stock_mean, self.stock_std)
        sample.market_sequence = self._transform(sample.market_sequence, self.market_mean, self.market_std)
        return sample

    def transform_samples(self, samples: Sequence[ESample]) -> list[ESample]:
        return [self.transform_sample(sample) for sample in samples]

    def payload(self) -> dict[str, object]:
        result = self.__dict__.copy()
        canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        result["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
        return result

    def to_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.payload(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "EFeatureScalers":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload.pop("payload_sha256", None)
        return cls(**payload)


def _normalize_dates(values: Iterable[pd.Timestamp | str] | None) -> set[pd.Timestamp] | None:
    return None if values is None else {pd.Timestamp(value).normalize() for value in values}


def build_e_samples(
    raw_df: pd.DataFrame,
    sequence_length: int,
    market_sequence_length: int = 60,
    feature_version: str = "e1_full_v1",
    horizon: int = 5,
    buy_offset: int = 1,
    sell_offset: int = 5,
    min_stocks_per_day: int = 10,
    sample_dates: Iterable[pd.Timestamp | str] | None = None,
    include_labels: bool = True,
) -> tuple[list[ESample], FeatureColumns]:
    processed, market_frame, columns = prepare_e_features(
        raw_df, feature_version=feature_version, horizon=horizon,
        buy_offset=buy_offset, sell_offset=sell_offset,
    )
    allowed = _normalize_dates(sample_dates)
    rows_by_date: dict[pd.Timestamp, list[tuple[str, np.ndarray, float, pd.Timestamp | None]]] = {}
    for stock_code, group in processed.groupby(STOCK_COL, sort=False):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        stock_values = group[columns.stock].to_numpy(dtype=np.float32)
        for position in range(sequence_length - 1, len(group)):
            row = group.iloc[position]
            date = pd.Timestamp(row[DATE_COL]).normalize()
            if allowed is not None and date not in allowed:
                continue
            label, label_end = row.get("label"), row.get("label_end_date")
            if include_labels and (pd.isna(label) or pd.isna(label_end)):
                continue
            rows_by_date.setdefault(date, []).append((
                str(stock_code),
                stock_values[position - sequence_length + 1:position + 1].astype(np.float16),
                float(label) if not pd.isna(label) else 0.0,
                pd.Timestamp(label_end).normalize() if not pd.isna(label_end) else None,
            ))

    market_frame = market_frame.sort_index()
    market_dates = list(market_frame.index)
    market_pos = {pd.Timestamp(value).normalize(): idx for idx, value in enumerate(market_dates)}
    market_values = market_frame[columns.market].to_numpy(dtype=np.float32)
    samples: list[ESample] = []
    for date in sorted(rows_by_date):
        rows = rows_by_date[date]
        if len(rows) < min_stocks_per_day or date not in market_pos:
            continue
        pos = market_pos[date]
        if pos + 1 < market_sequence_length:
            continue
        labels = np.asarray([row[2] for row in rows], dtype=np.float32)
        relevance = np.zeros_like(labels)
        if include_labels:
            relevance = (labels - labels.mean()) / (labels.std() + 1e-8)
        label_ends = [row[3] for row in rows if row[3] is not None]
        samples.append(ESample(
            sample_date=date,
            label_end_date=max(label_ends) if label_ends else None,
            market_sequence=market_values[pos-market_sequence_length+1:pos+1].astype(np.float16),
            stock_sequence=np.stack([row[1] for row in rows]),
            labels=labels,
            relevance=relevance.astype(np.float32),
            stock_codes=[row[0] for row in rows],
        ))
    return samples, columns


def build_inference_sample(
    raw_df: pd.DataFrame,
    sequence_length: int,
    market_sequence_length: int = 60,
    feature_version: str = "e1_full_v1",
    min_stocks_per_day: int = 1,
) -> tuple[ESample, FeatureColumns]:
    latest = pd.to_datetime(raw_df[DATE_COL]).max().normalize()
    samples, columns = build_e_samples(
        raw_df, sequence_length=sequence_length, market_sequence_length=market_sequence_length,
        feature_version=feature_version, min_stocks_per_day=min_stocks_per_day,
        sample_dates=[latest], include_labels=False,
    )
    if not samples:
        raise ValueError(f"could not build inference sample for {latest.date()}")
    return samples[-1], columns


def purged_time_split(
    samples: Sequence[ESample],
    val_start: pd.Timestamp | str,
    val_end: pd.Timestamp | str,
    train_end: pd.Timestamp | str | None = None,
) -> tuple[list[ESample], list[ESample]]:
    start, end = pd.Timestamp(val_start).normalize(), pd.Timestamp(val_end).normalize()
    explicit_train_end = pd.Timestamp(train_end).normalize() if train_end is not None else None
    train: list[ESample] = []
    validation: list[ESample] = []
    for sample in samples:
        date = pd.Timestamp(sample.sample_date).normalize()
        label_end = pd.Timestamp(sample.label_end_date).normalize() if sample.label_end_date is not None else None
        if start <= date <= end:
            validation.append(sample)
        elif date < start and (explicit_train_end is None or date <= explicit_train_end) and label_end is not None and label_end < start:
            train.append(sample)
    return train, validation


def collate_e_samples(batch: Sequence[ESample]) -> dict[str, object]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_stocks = max(sample.stock_sequence.shape[0] for sample in batch)
    seq_len, stock_dim = batch[0].stock_sequence.shape[1:]
    market_len, market_dim = batch[0].market_sequence.shape
    stock_sequence = np.zeros((len(batch), max_stocks, seq_len, stock_dim), dtype=np.float32)
    market_sequence = np.zeros((len(batch), market_len, market_dim), dtype=np.float32)
    labels = np.zeros((len(batch), max_stocks), dtype=np.float32)
    relevance = np.zeros_like(labels)
    mask = np.zeros((len(batch), max_stocks), dtype=bool)
    stock_codes: list[list[str]] = []
    sample_dates: list[str] = []
    for index, sample in enumerate(batch):
        count = sample.stock_sequence.shape[0]
        stock_sequence[index, :count] = sample.stock_sequence
        market_sequence[index] = sample.market_sequence
        labels[index, :count] = sample.labels
        relevance[index, :count] = sample.relevance
        mask[index, :count] = True
        stock_codes.append(sample.stock_codes)
        sample_dates.append(sample.sample_date.strftime("%Y-%m-%d"))
    return {
        "stock_sequence": torch.from_numpy(stock_sequence),
        "market_sequence": torch.from_numpy(market_sequence),
        "market_features": torch.from_numpy(market_sequence[:, -1]),
        "factor_features": torch.zeros((len(batch), max_stocks, 0), dtype=torch.float32),
        "labels": torch.from_numpy(labels), "relevance": torch.from_numpy(relevance),
        "mask": torch.from_numpy(mask), "stock_codes": stock_codes, "sample_dates": sample_dates,
    }


# Compatibility aliases for the copied R5 train/evaluator surface.
DSample = ESample
DStockRankingDataset = EStockRankingDataset
DFeatureScalers = EFeatureScalers
build_d_samples = build_e_samples
collate_d_samples = collate_e_samples

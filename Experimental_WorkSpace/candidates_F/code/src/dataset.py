from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from featurework import DATE_COL, STOCK_COL, FeatureColumns, prepare_d_features


@dataclass(frozen=True)
class DSample:
    sample_date: pd.Timestamp
    label_end_date: pd.Timestamp | None
    market_features: np.ndarray
    factor_features: np.ndarray
    stock_sequence: np.ndarray
    labels: np.ndarray
    relevance: np.ndarray
    stock_codes: list[str]


class DStockRankingDataset(Dataset):
    def __init__(self, samples: Sequence[DSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DSample:
        return self.samples[index]


@dataclass
class DFeatureScalers:
    stock_mean: list[float]
    stock_std: list[float]
    factor_mean: list[float]
    factor_std: list[float]
    market_mean: list[float]
    market_std: list[float]

    @staticmethod
    def _fit_matrix(values: np.ndarray) -> tuple[list[float], list[float]]:
        if values.size == 0:
            return [], []
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return mean.astype(float).tolist(), std.astype(float).tolist()

    @classmethod
    def fit(cls, samples: Sequence[DSample]) -> "DFeatureScalers":
        if not samples:
            raise ValueError("cannot fit scalers on empty samples")
        stock_values = np.concatenate([sample.stock_sequence.reshape(-1, sample.stock_sequence.shape[-1]) for sample in samples])
        factor_values = np.concatenate([sample.factor_features for sample in samples])
        market_values = np.stack([sample.market_features for sample in samples])
        stock_mean, stock_std = cls._fit_matrix(stock_values)
        factor_mean, factor_std = cls._fit_matrix(factor_values)
        market_mean, market_std = cls._fit_matrix(market_values)
        return cls(stock_mean, stock_std, factor_mean, factor_std, market_mean, market_std)

    def transform_sample(self, sample: DSample) -> DSample:
        stock = self._transform(sample.stock_sequence, self.stock_mean, self.stock_std)
        factor = self._transform(sample.factor_features, self.factor_mean, self.factor_std)
        market = self._transform(sample.market_features, self.market_mean, self.market_std)
        return replace(sample, stock_sequence=stock, factor_features=factor, market_features=market)

    def transform_samples(self, samples: Sequence[DSample]) -> list[DSample]:
        return [self.transform_sample(sample) for sample in samples]

    @staticmethod
    def _transform(values: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
        if len(mean) == 0:
            return values.astype(np.float32)
        return ((values - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)).astype(np.float32)

    def to_json(self, path: str | Path) -> None:
        payload = {
            "stock_mean": self.stock_mean,
            "stock_std": self.stock_std,
            "factor_mean": self.factor_mean,
            "factor_std": self.factor_std,
            "market_mean": self.market_mean,
            "market_std": self.market_std,
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "DFeatureScalers":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)


def _normalize_dates(values: Iterable[pd.Timestamp | str] | None) -> set[pd.Timestamp] | None:
    if values is None:
        return None
    return {pd.Timestamp(value).normalize() for value in values}


def _clean_matrix(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def build_d_samples(
    raw_df: pd.DataFrame,
    sequence_length: int,
    horizon: int = 5,
    buy_offset: int = 1,
    sell_offset: int = 5,
    min_stocks_per_day: int = 10,
    sample_dates: Iterable[pd.Timestamp | str] | None = None,
    include_labels: bool = True,
    industry_map: pd.DataFrame | None = None,
    futures_frame: pd.DataFrame | None = None,
    enable_industry: bool = False,
    enable_futures: bool = False,
    industry_missing_label: str = "UNKNOWN",
    futures_lag_days: int = 0,
) -> tuple[list[DSample], FeatureColumns]:
    processed, market_frame, columns = prepare_d_features(
        raw_df,
        horizon=horizon,
        buy_offset=buy_offset,
        sell_offset=sell_offset,
        industry_map=industry_map,
        futures_frame=futures_frame,
        enable_industry=enable_industry,
        enable_futures=enable_futures,
        industry_missing_label=industry_missing_label,
        futures_lag_days=futures_lag_days,
    )
    allowed_dates = _normalize_dates(sample_dates)
    rows_by_date: dict[pd.Timestamp, list[dict[str, object]]] = {}
    max_label_end_by_date: dict[pd.Timestamp, pd.Timestamp] = {}

    for stock_code, group in processed.groupby(STOCK_COL, sort=False):
        group = group.sort_values(DATE_COL).reset_index(drop=True)
        stock_values = _clean_matrix(group[columns.stock].to_numpy())
        factor_values = _clean_matrix(group[columns.factor].to_numpy())
        for position in range(sequence_length - 1, len(group)):
            row = group.iloc[position]
            sample_date = pd.Timestamp(row[DATE_COL]).normalize()
            if allowed_dates is not None and sample_date not in allowed_dates:
                continue
            if sample_date not in market_frame.index:
                continue
            label = row.get("label")
            label_end = row.get("label_end_date")
            if include_labels and (pd.isna(label) or pd.isna(label_end)):
                continue
            stock_sequence = stock_values[position - sequence_length + 1 : position + 1]
            factor = factor_values[position]
            rows_by_date.setdefault(sample_date, []).append(
                {
                    "stock_code": str(stock_code),
                    "sequence": stock_sequence,
                    "factor": factor,
                    "label": float(label) if not pd.isna(label) else 0.0,
                    "label_end": pd.Timestamp(label_end).normalize() if not pd.isna(label_end) else None,
                }
            )
            if not pd.isna(label_end):
                current = max_label_end_by_date.get(sample_date)
                label_end_ts = pd.Timestamp(label_end).normalize()
                if current is None or label_end_ts > current:
                    max_label_end_by_date[sample_date] = label_end_ts

    samples: list[DSample] = []
    for sample_date in sorted(rows_by_date):
        rows = rows_by_date[sample_date]
        if len(rows) < min_stocks_per_day:
            continue
        labels = np.asarray([row["label"] for row in rows], dtype=np.float32)
        if include_labels:
            label_std = float(labels.std())
            relevance = (labels - float(labels.mean())) / (label_std + 1e-8)
        else:
            relevance = np.zeros_like(labels)
        market = _clean_matrix(market_frame.loc[sample_date, columns.market].to_numpy())
        samples.append(
            DSample(
                sample_date=sample_date,
                label_end_date=max_label_end_by_date.get(sample_date),
                market_features=market,
                factor_features=np.stack([row["factor"] for row in rows]).astype(np.float32),
                stock_sequence=np.stack([row["sequence"] for row in rows]).astype(np.float32),
                labels=labels,
                relevance=relevance.astype(np.float32),
                stock_codes=[str(row["stock_code"]) for row in rows],
            )
        )
    return samples, columns


def build_inference_sample(
    raw_df: pd.DataFrame,
    sequence_length: int,
    min_stocks_per_day: int = 1,
    industry_map: pd.DataFrame | None = None,
    futures_frame: pd.DataFrame | None = None,
    enable_industry: bool = False,
    enable_futures: bool = False,
    industry_missing_label: str = "UNKNOWN",
    futures_lag_days: int = 0,
) -> tuple[DSample, FeatureColumns]:
    raw = raw_df.copy()
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL]).dt.normalize()
    latest_date = raw[DATE_COL].max()
    samples, columns = build_d_samples(
        raw,
        sequence_length=sequence_length,
        min_stocks_per_day=min_stocks_per_day,
        sample_dates=[latest_date],
        include_labels=False,
        industry_map=industry_map,
        futures_frame=futures_frame,
        enable_industry=enable_industry,
        enable_futures=enable_futures,
        industry_missing_label=industry_missing_label,
        futures_lag_days=futures_lag_days,
    )
    if not samples:
        raise ValueError(f"could not build inference sample for {latest_date.date()}")
    return samples[-1], columns


def purged_time_split(
    samples: Sequence[DSample],
    val_start: pd.Timestamp | str,
    val_end: pd.Timestamp | str,
) -> tuple[list[DSample], list[DSample]]:
    start = pd.Timestamp(val_start).normalize()
    end = pd.Timestamp(val_end).normalize()
    train: list[DSample] = []
    val: list[DSample] = []
    for sample in samples:
        sample_date = pd.Timestamp(sample.sample_date).normalize()
        label_end = pd.Timestamp(sample.label_end_date).normalize() if sample.label_end_date is not None else None
        if start <= sample_date <= end:
            val.append(sample)
        elif sample_date < start and (label_end is None or label_end < start):
            train.append(sample)
    return train, val


def collate_d_samples(batch: Sequence[DSample]) -> dict[str, object]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_stocks = max(sample.stock_sequence.shape[0] for sample in batch)
    seq_len = batch[0].stock_sequence.shape[1]
    stock_dim = batch[0].stock_sequence.shape[2]
    factor_dim = batch[0].factor_features.shape[1]
    market_dim = batch[0].market_features.shape[0]

    stock_sequence = np.zeros((len(batch), max_stocks, seq_len, stock_dim), dtype=np.float32)
    factor_features = np.zeros((len(batch), max_stocks, factor_dim), dtype=np.float32)
    labels = np.zeros((len(batch), max_stocks), dtype=np.float32)
    relevance = np.zeros((len(batch), max_stocks), dtype=np.float32)
    mask = np.zeros((len(batch), max_stocks), dtype=bool)
    market_features = np.zeros((len(batch), market_dim), dtype=np.float32)
    stock_codes: list[list[str]] = []
    sample_dates: list[str] = []

    for index, sample in enumerate(batch):
        count = sample.stock_sequence.shape[0]
        stock_sequence[index, :count] = sample.stock_sequence
        factor_features[index, :count] = sample.factor_features
        labels[index, :count] = sample.labels
        relevance[index, :count] = sample.relevance
        mask[index, :count] = True
        market_features[index] = sample.market_features
        stock_codes.append(sample.stock_codes)
        sample_dates.append(pd.Timestamp(sample.sample_date).strftime("%Y-%m-%d"))

    return {
        "stock_sequence": torch.from_numpy(stock_sequence),
        "factor_features": torch.from_numpy(factor_features),
        "market_features": torch.from_numpy(market_features),
        "labels": torch.from_numpy(labels),
        "relevance": torch.from_numpy(relevance),
        "mask": torch.from_numpy(mask),
        "stock_codes": stock_codes,
        "sample_dates": sample_dates,
    }

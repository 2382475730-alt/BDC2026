from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketColumns:
    stock: str
    date: str
    open: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def infer_market_columns(frame: pd.DataFrame) -> MarketColumns:
    if len(frame.columns) < 3:
        raise ValueError("market data must contain at least stock, date, and open columns")
    return MarketColumns(stock=str(frame.columns[0]), date=str(frame.columns[1]), open=str(frame.columns[2]))


def read_market_csv(path: str | Path, usecols: list[str] | None = None) -> tuple[pd.DataFrame, MarketColumns]:
    csv_path = Path(path)
    header = pd.read_csv(csv_path, nrows=1)
    columns = infer_market_columns(header)
    frame = pd.read_csv(csv_path, dtype={columns.stock: str}, usecols=usecols)
    if columns.stock in frame.columns:
        frame[columns.stock] = frame[columns.stock].map(normalize_stock_code)
    if columns.date in frame.columns:
        frame[columns.date] = pd.to_datetime(frame[columns.date]).dt.normalize()
    return frame, columns


def normalize_stock_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    lower = text.lower()
    if lower.startswith(("sh.", "sz.")) and len(text) >= 9:
        text = text.split(".", 1)[1]
    if lower.endswith((".sh", ".sz")) and len(text) >= 9:
        text = text.split(".", 1)[0]
    return text.zfill(6) if text.isdigit() else text


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path, default: Any | None = None) -> Any:
    json_path = Path(path)
    if not json_path.exists():
        return default
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def date_strings(values: list[pd.Timestamp]) -> list[str]:
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in values]


def sorted_trading_dates(frame: pd.DataFrame, date_col: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(value).normalize() for value in sorted(pd.to_datetime(frame[date_col]).dropna().unique())]


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if abs(float(denominator)) > 1e-12 else float(default)


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import normalize_stock_code, safe_div, write_json


def _daily_score(day: pd.DataFrame, top_k: int) -> dict[str, Any] | None:
    day = day.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "label"])
    if len(day) < 2:
        return None
    scores = day["score"].to_numpy(dtype=np.float64)
    labels = day["label"].to_numpy(dtype=np.float64)
    k = min(top_k, len(day))
    if "portfolio_weight" in day.columns and day["portfolio_weight"].notna().any():
        weights = day["portfolio_weight"].to_numpy(dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        selected = np.flatnonzero(weights > 1e-12)
        if selected.size == 0:
            selected = np.argsort(scores)[::-1][:k]
            weights = np.zeros_like(scores, dtype=np.float64)
            weights[selected] = 1.0 / k
    else:
        selected = np.argsort(scores)[::-1][:k]
        weights = np.zeros_like(scores, dtype=np.float64)
        weights[selected] = 1.0 / k
    true_top = np.argsort(labels)[::-1][:k]
    ordered_returns = labels[selected]
    true_top_returns = labels[true_top]
    pred_return_sum = float(np.sum(ordered_returns))
    max_return_sum = float(np.sum(true_top_returns))
    random_return_sum = float(k * np.mean(labels))
    denominator = max_return_sum - random_return_sum
    final_score = safe_div(pred_return_sum - random_return_sum, denominator)
    return {
        "stock_count": int(len(day)),
        "selected_count": int(len(selected)),
        "weight_sum": float(np.sum(weights)),
        "cash_ratio": float(max(0.0, 1.0 - np.sum(weights))),
        "portfolio_return": float(np.sum(weights * labels)),
        "top_return_mean": float(np.mean(ordered_returns)),
        "pred_return_sum": pred_return_sum,
        "max_return_sum": max_return_sum,
        "random_return_sum": random_return_sum,
        "final_score": final_score,
        "hit_rate": float(np.mean(ordered_returns > 0.0)),
        "selected_stocks": ",".join(day.iloc[selected]["stock_id"].astype(str).tolist()),
    }


def summarize_daily(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return {
            "sample_count": 0,
            "official_proxy_final_score": 0.0,
            "mean_portfolio_return": 0.0,
            "median_portfolio_return": 0.0,
            "cumulative_return": 0.0,
            "mean_final_score": 0.0,
            "min_final_score": 0.0,
            "hit_rate": 0.0,
        }
    returns = daily["portfolio_return"].astype(float).to_numpy()
    final_scores = daily["final_score"].astype(float).to_numpy()
    return {
        "sample_count": int(len(daily)),
        "date_start": str(daily["sample_end_date"].min()),
        "date_end": str(daily["sample_end_date"].max()),
        "official_proxy_final_score": float(np.mean(returns)),
        "mean_portfolio_return": float(np.mean(returns)),
        "median_portfolio_return": float(np.median(returns)),
        "cumulative_return": float(np.prod(1.0 + returns) - 1.0),
        "mean_final_score": float(np.mean(final_scores)),
        "median_final_score": float(np.median(final_scores)),
        "min_final_score": float(np.min(final_scores)),
        "hit_rate": float(np.mean(returns > 0.0)),
        "average_weight_sum": float(daily["weight_sum"].astype(float).mean()) if "weight_sum" in daily else 1.0,
        "average_cash_ratio": float(daily["cash_ratio"].astype(float).mean()) if "cash_ratio" in daily else 0.0,
        "average_selected_count": float(daily["selected_count"].astype(float).mean()) if "selected_count" in daily else 0.0,
    }


def evaluate_official_proxy(
    predictions: pd.DataFrame,
    *,
    top_k: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"sample_end_date", "stock_id", "score", "label"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")
    frame = predictions.copy()
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
    frame["stock_id"] = frame["stock_id"].map(normalize_stock_code)
    rows: list[dict[str, Any]] = []
    for sample_date, day in frame.groupby("sample_end_date", sort=True):
        row = _daily_score(day, top_k=top_k)
        if row is not None:
            rows.append({"sample_end_date": sample_date, **row})
    daily = pd.DataFrame(rows)
    return daily, summarize_daily(daily)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate D-series predictions with the official T+1 to T+5 proxy.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions, dtype={"stock_id": str})
    daily, summary = evaluate_official_proxy(predictions, top_k=args.top_k)
    daily.to_csv(output_dir / "official_proxy_daily.csv", index=False, encoding="utf-8")
    write_json(output_dir / "official_proxy_result.json", summary)


if __name__ == "__main__":
    main()

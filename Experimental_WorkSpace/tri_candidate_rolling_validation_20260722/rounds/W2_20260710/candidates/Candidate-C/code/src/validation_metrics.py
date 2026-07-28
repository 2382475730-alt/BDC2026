from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def robust_score(window_scores: Sequence[float]) -> float:
    if not window_scores:
        return 0.0
    scores = np.asarray(window_scores, dtype=np.float64)
    return float(0.7 * scores.mean() + 0.3 * scores.min())


def cv_robust_score(fold_scores: Sequence[float]) -> float:
    if not fold_scores:
        return 0.0
    scores = np.asarray(fold_scores, dtype=np.float64)
    return float(0.5 * scores.mean() + 0.3 * np.median(scores) + 0.2 * scores.min())


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2:
        return 0.0
    value = frame["left"].corr(frame["right"], method="spearman")
    if pd.isna(value):
        return 0.0
    return float(value)


def _split_dates_into_windows(dates: Sequence[pd.Timestamp], num_windows: int) -> list[list[pd.Timestamp]]:
    unique_dates = sorted(pd.Timestamp(date) for date in set(dates))
    if not unique_dates:
        return []
    window_count = max(1, min(int(num_windows), len(unique_dates)))
    return [
        [pd.Timestamp(date) for date in chunk]
        for chunk in np.array_split(np.asarray(unique_dates, dtype="datetime64[ns]"), window_count)
        if len(chunk) > 0
    ]


def daily_score_metrics(
    predictions: pd.DataFrame,
    score_col: str,
    date_col: str,
    label_col: str,
    relevance_col: str,
    top_ks: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for date, day in predictions.groupby(date_col, sort=True):
        day = day.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, label_col])
        if len(day) < 2:
            continue
        ordered = day.sort_values(score_col, ascending=False)
        true_ordered = day.sort_values(label_col, ascending=False)
        market_mean = float(day[label_col].mean())
        market_median = float(day[label_col].median())
        row: dict[str, float | str] = {
            "sample_end_date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "stock_count": float(len(day)),
            "rankic": safe_spearman(day[score_col], day[label_col]),
            "score_raw_spearman": safe_spearman(day[score_col], day[label_col]),
            "score_relevance_spearman": safe_spearman(day[score_col], day[relevance_col])
            if relevance_col in day.columns
            else 0.0,
        }
        for k in top_ks:
            k = min(int(k), len(day))
            if k <= 0:
                continue
            top = ordered.head(k)
            true_top = true_ordered.head(k)
            top_return = float(top[label_col].mean())
            row[f"top{k}_raw_return"] = top_return
            row[f"top{k}_excess_return"] = top_return - market_mean
            row[f"top{k}_positive_rate"] = float((top[label_col] > 0).mean())
            row[f"top{k}_above_median_rate"] = float((top[label_col] > market_median).mean())
            predicted_ids = set(top.index)
            true_ids = set(true_top.index)
            row[f"top{k}_true_top{k}_overlap"] = float(len(predicted_ids & true_ids) / k)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_one_score(
    predictions: pd.DataFrame,
    score_col: str,
    date_col: str = "sample_end_date",
    label_col: str = "raw_label",
    relevance_col: str = "relevance",
    top_ks: Sequence[int] = (5, 10, 20),
    num_windows: int = 3,
    primary_metric: str = "top5_excess_return",
) -> dict[str, object]:
    daily = daily_score_metrics(
        predictions=predictions,
        score_col=score_col,
        date_col=date_col,
        label_col=label_col,
        relevance_col=relevance_col,
        top_ks=top_ks,
    )
    if daily.empty:
        return {
            "daily_metrics": [],
            "window_metrics": [],
            "rankic_mean": 0.0,
            "rankic_std": 0.0,
            "rankic_ir": 0.0,
            "score_raw_spearman": 0.0,
            "score_relevance_spearman": 0.0,
            "window_scores": [],
            "mean_window_score": 0.0,
            "min_window_score": 0.0,
            "fold_robust_score": 0.0,
        }

    rankic_values = daily["rankic"].astype(float)
    rankic_std = float(rankic_values.std(ddof=0))
    date_values = pd.to_datetime(daily["sample_end_date"])
    window_rows: list[dict[str, object]] = []
    window_scores: list[float] = []
    for index, window_dates in enumerate(_split_dates_into_windows(date_values, num_windows), start=1):
        date_strings = {pd.Timestamp(date).strftime("%Y-%m-%d") for date in window_dates}
        window_daily = daily[daily["sample_end_date"].isin(date_strings)]
        metric_value = float(window_daily.get(primary_metric, pd.Series([0.0])).mean())
        window_scores.append(metric_value)
        window_rows.append(
            {
                "window": index,
                "start": min(date_strings),
                "end": max(date_strings),
                "daily_count": int(len(window_daily)),
                "score": metric_value,
            }
        )

    aggregate: dict[str, object] = {
        "daily_metrics": daily.to_dict(orient="records"),
        "window_metrics": window_rows,
        "rankic_mean": float(rankic_values.mean()),
        "rankic_std": rankic_std,
        "rankic_ir": float(rankic_values.mean() / (rankic_std + 1e-12)),
        "score_raw_spearman": safe_spearman(predictions[score_col], predictions[label_col]),
        "score_relevance_spearman": safe_spearman(predictions[score_col], predictions[relevance_col])
        if relevance_col in predictions.columns
        else 0.0,
        "window_scores": window_scores,
        "mean_window_score": float(np.mean(window_scores)) if window_scores else 0.0,
        "min_window_score": float(np.min(window_scores)) if window_scores else 0.0,
        "fold_robust_score": robust_score(window_scores),
    }
    for column in daily.columns:
        if column.startswith("top") and column != primary_metric:
            aggregate[column] = float(daily[column].astype(float).mean())
    if primary_metric in daily.columns:
        aggregate[primary_metric] = float(daily[primary_metric].astype(float).mean())
    return aggregate


def summarize_score_columns(
    predictions: pd.DataFrame,
    score_columns: Mapping[str, str],
    date_col: str = "sample_end_date",
    label_col: str = "raw_label",
    relevance_col: str = "relevance",
    top_ks: Sequence[int] = (5, 10, 20),
    num_windows: int = 3,
    primary_metric: str = "top5_excess_return",
) -> dict[str, object]:
    score_metrics = {
        name: summarize_one_score(
            predictions=predictions,
            score_col=column,
            date_col=date_col,
            label_col=label_col,
            relevance_col=relevance_col,
            top_ks=top_ks,
            num_windows=num_windows,
            primary_metric=primary_metric,
        )
        for name, column in score_columns.items()
    }

    rerank_gain: dict[str, float] = {}
    if "base" in score_metrics and "rerank" in score_metrics:
        base_score = float(score_metrics["base"]["fold_robust_score"])
        rerank_score = float(score_metrics["rerank"]["fold_robust_score"])
        rerank_gain = {
            "fold_robust_score": rerank_score - base_score,
            "cv_robust_score": rerank_score - base_score,
        }

    return {
        "score_metrics": score_metrics,
        "rerank_gain": rerank_gain,
    }


def summarize_cv(fold_metrics: Sequence[Mapping[str, object]], score_key: str = "fold_robust_score") -> dict[str, float]:
    scores = [float(metrics.get(score_key, 0.0)) for metrics in fold_metrics]
    if not scores:
        return {
            "mean_fold_score": 0.0,
            "median_fold_score": 0.0,
            "min_fold_score": 0.0,
            "std_fold_score": 0.0,
            "cv_robust_score": 0.0,
        }
    return {
        "mean_fold_score": float(np.mean(scores)),
        "median_fold_score": float(np.median(scores)),
        "min_fold_score": float(np.min(scores)),
        "std_fold_score": float(np.std(scores)),
        "cv_robust_score": cv_robust_score(scores),
    }

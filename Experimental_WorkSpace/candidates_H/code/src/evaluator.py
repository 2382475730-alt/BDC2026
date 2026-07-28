from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2:
        return 0.0
    value = frame["left"].corr(frame["right"], method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    order = np.argsort(scores)[::-1][:k]
    ideal = np.argsort(labels)[::-1][:k]
    gains = np.maximum(labels, 0.0)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(gains[ideal] * discounts))
    return dcg / (idcg + 1e-12)


def weights_from_scores(
    scores: np.ndarray,
    top_k: int = 5,
    mode: str = "topk_equal",
    temperature: float = 1.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    weights = np.zeros_like(scores, dtype=np.float64)
    if scores.size == 0:
        return weights
    k = min(top_k, scores.size)
    top_index = np.argsort(scores)[::-1][:k]
    if mode == "topk_equal":
        weights[top_index] = 1.0 / k
    elif mode in {"softmax", "capped_softmax"}:
        selected = scores[top_index] / max(float(temperature), 1e-6)
        selected = selected - selected.max()
        selected_weights = np.exp(selected)
        selected_weights = selected_weights / (selected_weights.sum() + 1e-12)
        if mode == "capped_softmax":
            selected_weights = np.clip(selected_weights, min_weight, max_weight)
            selected_weights = selected_weights / (selected_weights.sum() + 1e-12)
        weights[top_index] = selected_weights
    else:
        raise ValueError(f"unsupported portfolio mode: {mode}")
    return weights


def predictions_from_batch(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, object],
    score_key: str = "ranking_scores",
) -> pd.DataFrame:
    scores = outputs[score_key].detach().cpu().numpy()
    labels = batch["labels"].detach().cpu().numpy()  # type: ignore[union-attr]
    mask = batch["mask"].detach().cpu().numpy().astype(bool)  # type: ignore[union-attr]
    rows: list[dict[str, object]] = []
    sample_dates = batch["sample_dates"]  # type: ignore[assignment]
    stock_codes = batch["stock_codes"]  # type: ignore[assignment]
    for batch_index, sample_date in enumerate(sample_dates):
        count = int(mask[batch_index].sum())
        for stock_index in range(count):
            rows.append(
                {
                    "sample_end_date": sample_date,
                    "stock_id": stock_codes[batch_index][stock_index],
                    "score": float(scores[batch_index, stock_index]),
                    "label": float(labels[batch_index, stock_index]),
                }
            )
    return pd.DataFrame(rows)


def daily_metrics(
    predictions: pd.DataFrame,
    top_k: int = 5,
    portfolio_mode: str = "topk_equal",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_date, day in predictions.groupby("sample_end_date", sort=True):
        day = day.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "label"])
        if len(day) < 2:
            continue
        scores = day["score"].to_numpy(dtype=np.float64)
        labels = day["label"].to_numpy(dtype=np.float64)
        weights = weights_from_scores(scores, top_k=top_k, mode=portfolio_mode)
        portfolio_return = float(np.sum(weights * labels))
        k = min(top_k, len(day))
        ordered_returns = labels[np.argsort(scores)[::-1][:k]]
        true_top_returns = labels[np.argsort(labels)[::-1][:k]]
        pred_return_sum = float(np.sum(ordered_returns))
        max_return_sum = float(np.sum(true_top_returns))
        random_return_sum = float(k * np.mean(labels))
        denominator = max_return_sum - random_return_sum
        final_score = (
            (pred_return_sum - random_return_sum) / (denominator + 1e-12)
            if abs(denominator) > 1e-9
            else 0.0
        )
        rows.append(
            {
                "sample_end_date": str(sample_date),
                "stock_count": int(len(day)),
                "portfolio_return": portfolio_return,
                "top_return_mean": float(np.mean(ordered_returns)),
                "pred_return_sum": pred_return_sum,
                "max_return_sum": max_return_sum,
                "random_return_sum": random_return_sum,
                "final_score": float(final_score),
                "hit_rate": float(np.mean(ordered_returns > 0.0)),
                "rank_ic": safe_spearman(scores, labels),
                "ndcg_at_5": ndcg_at_k(scores, labels, k=5),
                "turnover_proxy": float(np.count_nonzero(weights) / max(1, len(weights))),
            }
        )
    return pd.DataFrame(rows)


def summarize_daily_metrics(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "mean_portfolio_return": 0.0,
            "median_portfolio_return": 0.0,
            "mean_final_score": 0.0,
            "median_final_score": 0.0,
            "min_final_score": 0.0,
            "robust_final_score": 0.0,
            "hit_rate": 0.0,
            "worst20_return": 0.0,
            "rank_ic": 0.0,
            "ic_ir": 0.0,
            "ndcg_at_5": 0.0,
            "fold_variance": 0.0,
            "stock_turnover": 0.0,
            "max_drawdown": 0.0,
            "downside_deviation": 0.0,
            "cvar_20": 0.0,
            "q10_return": 0.0,
            "stability_score": 0.0,
            "volatility": 0.0,
        }
    returns = daily["portfolio_return"].astype(float).to_numpy()
    final_scores = daily["final_score"].astype(float).to_numpy()
    cumulative = np.cumsum(returns)
    drawdown = cumulative - np.maximum.accumulate(cumulative)
    rank_ic = daily["rank_ic"].astype(float).to_numpy()
    downside = returns[returns < 0.0]
    cutoff = max(1, int(np.ceil(len(returns) * 0.2)))
    worst20 = np.sort(returns)[:cutoff]
    ic_std = float(np.std(rank_ic))
    mean_return = float(np.mean(returns))
    median_return = float(np.median(returns))
    cvar20 = float(np.mean(worst20))
    q10_return = float(np.quantile(returns, 0.10))
    volatility = float(np.std(returns))
    return {
        "mean_portfolio_return": mean_return,
        "median_portfolio_return": median_return,
        "mean_final_score": float(np.mean(final_scores)),
        "median_final_score": float(np.median(final_scores)),
        "min_final_score": float(np.min(final_scores)),
        "robust_final_score": float(0.7 * np.mean(final_scores) + 0.3 * np.min(final_scores)),
        "hit_rate": float(np.mean(returns > 0.0)),
        "worst20_return": float(np.mean(worst20)),
        "rank_ic": float(np.mean(rank_ic)),
        "ic_ir": float(np.mean(rank_ic) / (ic_std + 1e-12)),
        "ndcg_at_5": float(daily["ndcg_at_5"].astype(float).mean()),
        "fold_variance": float(np.var(returns)),
        "stock_turnover": float(daily["turnover_proxy"].astype(float).mean()),
        "max_drawdown": float(np.min(drawdown)),
        "downside_deviation": float(np.std(downside)) if len(downside) else 0.0,
        "cvar_20": cvar20,
        "q10_return": q10_return,
        "stability_score": float(
            0.35 * mean_return
            + 0.25 * median_return
            + 0.20 * cvar20
            + 0.10 * q10_return
            - 0.10 * volatility
        ),
        "volatility": volatility,
    }

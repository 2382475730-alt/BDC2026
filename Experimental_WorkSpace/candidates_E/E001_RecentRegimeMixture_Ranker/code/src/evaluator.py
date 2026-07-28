from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from score_contract import primary_scores


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
    portfolio_logits: np.ndarray | None = None,
    regime_risk: float | None = None,
    portfolio_config: dict[str, object] | None = None,
) -> np.ndarray:
    weights = np.zeros_like(scores, dtype=np.float64)
    if scores.size == 0:
        return weights
    config = portfolio_config or {}
    scoring_values = scores
    if mode in {"portfolio_logits", "regime_conditioned"} and portfolio_logits is not None:
        scoring_values = portfolio_logits
    if mode in {"stable_topk", "deterministic_regime", "stable_regime"}:
        mode = "topk_equal"
    effective_top_k = top_k
    cash_ratio = 0.0
    if mode == "regime_conditioned":
        risk = 0.0 if regime_risk is None or not np.isfinite(regime_risk) else float(regime_risk)
        threshold = float(config.get("regime_risk_threshold", 0.55))
        if risk >= threshold:
            effective_top_k = int(config.get("high_risk_top_k", max(top_k, 3)))
            cash_ratio = float(config.get("high_risk_cash_ratio", 0.0))
        else:
            effective_top_k = int(config.get("low_risk_top_k", top_k))
            cash_ratio = float(config.get("low_risk_cash_ratio", 0.0))
    k = min(effective_top_k, scores.size)
    top_index = np.argsort(scoring_values)[::-1][:k]
    invested_weight = max(0.0, min(1.0, 1.0 - cash_ratio))
    if mode == "topk_equal":
        weights[top_index] = invested_weight / k
    elif mode in {"softmax", "capped_softmax", "portfolio_logits", "regime_conditioned"}:
        selected = scoring_values[top_index] / max(float(temperature), 1e-6)
        selected = selected - selected.max()
        selected_weights = np.exp(selected)
        selected_weights = selected_weights / (selected_weights.sum() + 1e-12)
        if mode == "capped_softmax":
            selected_weights = _project_capped_simplex(selected_weights, max_weight)
        weights[top_index] = invested_weight * selected_weights
    else:
        raise ValueError(f"unsupported portfolio mode: {mode}")
    return weights


def _project_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    if values.size == 0:
        return values
    if cap * values.size < 1.0 - 1e-12:
        raise ValueError("max_weight is too small for a fully invested capped portfolio")
    low = float(np.min(values) - cap)
    high = float(np.max(values))
    for _ in range(80):
        midpoint = (low + high) / 2.0
        projected = np.clip(values - midpoint, 0.0, cap)
        if projected.sum() > 1.0:
            low = midpoint
        else:
            high = midpoint
    projected = np.clip(values - high, 0.0, cap)
    total = float(projected.sum())
    if total <= 0.0:
        projected[:] = 1.0 / len(projected)
    else:
        projected /= total
    return projected


def attach_portfolio_weights(predictions: pd.DataFrame, portfolio_config: dict[str, object]) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    frame = predictions.copy()
    weights = np.zeros(len(frame), dtype=np.float64)
    effective_top_k = np.zeros(len(frame), dtype=np.int64)
    cash_ratio = np.zeros(len(frame), dtype=np.float64)
    mode = str(portfolio_config.get("mode", "topk_equal"))
    top_k = int(portfolio_config.get("top_k", 5))
    for _, day in frame.groupby("sample_end_date", sort=True):
        scores = day["score"].to_numpy(dtype=np.float64)
        logits = day["portfolio_logit"].to_numpy(dtype=np.float64) if "portfolio_logit" in day else None
        risk = float(day["regime_risk"].astype(float).mean()) if "regime_risk" in day else None
        day_weights = weights_from_scores(
            scores,
            top_k=top_k,
            mode=mode,
            temperature=float(portfolio_config.get("softmax_temperature", 1.0)),
            min_weight=float(portfolio_config.get("min_weight", 0.0)),
            max_weight=float(portfolio_config.get("max_weight", 1.0)),
            portfolio_logits=logits,
            regime_risk=risk,
            portfolio_config=portfolio_config,
        )
        weights[day.index.to_numpy()] = day_weights
        effective_top_k[day.index.to_numpy()] = int(np.count_nonzero(day_weights > 1e-12))
        cash_ratio[day.index.to_numpy()] = max(0.0, 1.0 - float(day_weights.sum()))
    frame["portfolio_weight"] = weights
    frame["portfolio_selected"] = weights > 1e-12
    frame["effective_top_k"] = effective_top_k
    frame["cash_ratio"] = cash_ratio
    return frame


def predictions_from_batch(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, object],
    score_key: str = "ranking_scores",
) -> pd.DataFrame:
    score_tensor = primary_scores(outputs) if score_key == "ranking_scores" else outputs[score_key]
    scores = score_tensor.detach().cpu().numpy()
    portfolio_logits = outputs.get("portfolio_logits")
    portfolio_logits_np = portfolio_logits.detach().cpu().numpy() if portfolio_logits is not None else None
    regime_risk = outputs.get("regime_risk")
    regime_risk_np = regime_risk.detach().cpu().numpy() if regime_risk is not None else None
    stable = outputs.get("stable_score")
    recent = outputs.get("recent_residual")
    gate = outputs.get("gate")
    stable_np = stable.detach().cpu().numpy() if stable is not None else scores
    recent_np = recent.detach().cpu().numpy() if recent is not None else np.zeros_like(scores)
    gate_np = gate.detach().cpu().numpy() if gate is not None else np.zeros_like(scores)
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
                    "portfolio_logit": (
                        float(portfolio_logits_np[batch_index, stock_index])
                        if portfolio_logits_np is not None
                        else float(scores[batch_index, stock_index])
                    ),
                    "regime_risk": (
                        float(regime_risk_np[batch_index])
                        if regime_risk_np is not None
                        else 0.0
                    ),
                    "label": float(labels[batch_index, stock_index]),
                    "stable_score": float(stable_np[batch_index, stock_index]),
                    "recent_residual": float(recent_np[batch_index, stock_index]),
                    "gate": float(gate_np[batch_index, stock_index]),
                }
            )
    return pd.DataFrame(rows)


def daily_metrics(
    predictions: pd.DataFrame,
    top_k: int = 5,
    portfolio_mode: str = "topk_equal",
    portfolio_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_date, day in predictions.groupby("sample_end_date", sort=True):
        day = day.replace([np.inf, -np.inf], np.nan).dropna(subset=["score", "label"])
        if len(day) < 2:
            continue
        scores = day["score"].to_numpy(dtype=np.float64)
        labels = day["label"].to_numpy(dtype=np.float64)
        if "portfolio_weight" in day:
            weights = day["portfolio_weight"].to_numpy(dtype=np.float64)
        else:
            weights = weights_from_scores(scores, top_k=top_k, mode=portfolio_mode, portfolio_config=portfolio_config)
        portfolio_return = float(np.sum(weights * labels))
        k = min(top_k, len(day))
        selected_index = np.flatnonzero(weights > 1e-12)
        if selected_index.size == 0:
            selected_index = np.argsort(scores)[::-1][:k]
        ordered_returns = labels[selected_index]
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
                "weight_sum": float(np.sum(weights)),
                "cash_ratio": float(max(0.0, 1.0 - np.sum(weights))),
                "selected_count": int(np.count_nonzero(weights > 1e-12)),
                "regime_risk": float(day["regime_risk"].astype(float).mean()) if "regime_risk" in day else 0.0,
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
            "cvar_95_loss_abs": 0.0,
            "worst_4_period_cumulative_return": 0.0,
            "monthly_positive_rate": 0.0,
            "return_without_best_2_periods": 0.0,
            "robust_gate_score": 0.0,
            "volatility": 0.0,
        }
    returns = daily["portfolio_return"].astype(float).to_numpy()
    final_scores = daily["final_score"].astype(float).to_numpy()
    cumulative = np.cumsum(returns)
    drawdown = cumulative - np.maximum.accumulate(cumulative)
    max_drawdown_abs = abs(float(np.min(drawdown)))
    rank_ic = daily["rank_ic"].astype(float).to_numpy()
    downside = returns[returns < 0.0]
    cutoff = max(1, int(np.ceil(len(returns) * 0.2)))
    cvar95_cutoff = max(1, int(np.ceil(len(returns) * 0.05)))
    worst20 = np.sort(returns)[:cutoff]
    cvar95_loss_abs = max(0.0, float(np.mean(np.sort(-returns)[-cvar95_cutoff:])))
    worst_4 = float(min((np.prod(1.0 + returns[i : i + 4]) - 1.0 for i in range(max(1, len(returns) - 3))), default=np.min(returns)))
    best2 = np.sort(returns)[-min(2, len(returns)) :].sum()
    return_without_best2 = float(np.sum(returns) - best2)
    dated = daily.copy()
    dated["sample_end_date"] = pd.to_datetime(dated["sample_end_date"])
    monthly_positive = dated.groupby(dated["sample_end_date"].dt.to_period("M"))["portfolio_return"].mean() > 0.0
    monthly_positive_rate = float(monthly_positive.mean()) if len(monthly_positive) else 0.0
    ic_std = float(np.std(rank_ic))
    robust_gate_score = float(
        np.mean(returns)
        + 0.25 * np.median(returns)
        + 0.005 * monthly_positive_rate
        + 0.002 * np.mean(returns > 0.0)
        + min(0.0, return_without_best2 / max(1, len(returns)))
        + 0.25 * min(0.0, worst_4)
        - 0.25 * max_drawdown_abs
        - 0.5 * cvar95_loss_abs
    )
    return {
        "mean_portfolio_return": float(np.mean(returns)),
        "median_portfolio_return": float(np.median(returns)),
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
        "max_drawdown_abs": max_drawdown_abs,
        "downside_deviation": float(np.std(downside)) if len(downside) else 0.0,
        "cvar_20": float(np.mean(worst20)),
        "cvar_95_loss_abs": cvar95_loss_abs,
        "worst_4_period_cumulative_return": worst_4,
        "monthly_positive_rate": monthly_positive_rate,
        "return_without_best_2_periods": return_without_best2,
        "robust_gate_score": robust_gate_score,
        "volatility": float(np.std(returns)),
    }

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .common import normalize_stock_code, read_market_csv, safe_div
from .official_proxy_eval import evaluate_official_proxy
from .r4_protocol import summarize_weekly_rows
from .weekly_backtest import run_weekly_backtest


POLICY_IDS = ("P0", "P1", "P2", "P3", "P4")
CONTROLLER_FEATURES = (
    "top1_top2_margin",
    "top2_top3_margin",
    "top5_score_dispersion",
    "score_entropy",
    "selected_vol20",
    "selected_vol60",
    "selected_pair_correlation60",
    "market_vol20",
)
TIE_EPSILON = 1e-10


@dataclass
class CausalMarketFeatures:
    stock_features: pd.DataFrame
    market_features: pd.DataFrame
    returns_by_stock: dict[str, pd.Series]

    @classmethod
    def from_market_path(cls, market_path: str | Path) -> "CausalMarketFeatures":
        market, columns = read_market_csv(market_path)
        close_candidates = [name for name in ("收盘", "close", "收盘价") if name in market.columns]
        if not close_candidates:
            raise KeyError("market data has no recognized close-price column")
        return cls.from_market_frame(market, columns.stock, columns.date, close_candidates[0])

    @classmethod
    def from_market_frame(
        cls,
        market: pd.DataFrame,
        stock_column: str,
        date_column: str,
        close_column: str,
    ) -> "CausalMarketFeatures":
        frame = market[[stock_column, date_column, close_column]].copy()
        frame.columns = ["stock_id", "date", "close"]
        frame["stock_id"] = frame["stock_id"].map(normalize_stock_code)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.sort_values(["stock_id", "date"]).reset_index(drop=True)
        grouped = frame.groupby("stock_id", sort=False)
        frame["return_1"] = grouped["close"].pct_change(fill_method=None)
        frame["vol20"] = grouped["return_1"].transform(
            lambda values: values.rolling(20, min_periods=6).std(ddof=0)
        )
        frame["vol60"] = grouped["return_1"].transform(
            lambda values: values.rolling(60, min_periods=20).std(ddof=0)
        )
        stock_features = frame[["date", "stock_id", "vol20", "vol60"]].copy()
        daily = frame.groupby("date", sort=True)["return_1"].mean().rename("market_return_1").to_frame()
        daily["market_vol20"] = daily["market_return_1"].rolling(20, min_periods=5).std(ddof=0)
        market_features = daily.reset_index()[["date", "market_vol20"]]
        returns_by_stock = {
            stock: group.set_index("date")["return_1"].sort_index()
            for stock, group in frame.groupby("stock_id", sort=False)
        }
        return cls(stock_features=stock_features, market_features=market_features, returns_by_stock=returns_by_stock)

    def stock_volatility(self, date: pd.Timestamp | str, stock_id: str) -> tuple[float, float]:
        target = pd.Timestamp(date).normalize()
        stock = normalize_stock_code(stock_id)
        rows = self.stock_features[
            (self.stock_features["date"] == target) & (self.stock_features["stock_id"] == stock)
        ]
        if rows.empty:
            return math.nan, math.nan
        row = rows.iloc[-1]
        return float(row["vol20"]), float(row["vol60"])

    def combined_volatility(self, date: pd.Timestamp | str, stocks: Sequence[str]) -> np.ndarray:
        values = []
        for stock in stocks:
            vol20, vol60 = self.stock_volatility(date, stock)
            value = 0.5 * vol20 + 0.5 * vol60
            values.append(value)
        array = np.asarray(values, dtype=np.float64)
        valid = array[np.isfinite(array) & (array > 1e-12)]
        fallback = float(np.median(valid)) if valid.size else 1.0
        return np.where(np.isfinite(array) & (array > 1e-12), array, fallback)

    def pair_correlation60(self, date: pd.Timestamp | str, stock_a: str, stock_b: str) -> float:
        target = pd.Timestamp(date).normalize()
        left = self.returns_by_stock.get(normalize_stock_code(stock_a))
        right = self.returns_by_stock.get(normalize_stock_code(stock_b))
        if left is None or right is None:
            return 0.0
        joined = pd.concat(
            [left.loc[left.index <= target].rename("left"), right.loc[right.index <= target].rename("right")],
            axis=1,
            join="inner",
        ).dropna().tail(60)
        if len(joined) < 20:
            return 0.0
        value = float(joined["left"].corr(joined["right"]))
        return value if np.isfinite(value) else 0.0

    def market_vol20(self, date: pd.Timestamp | str) -> float:
        target = pd.Timestamp(date).normalize()
        rows = self.market_features[self.market_features["date"] == target]
        if rows.empty:
            return 0.0
        value = float(rows.iloc[-1]["market_vol20"])
        return value if np.isfinite(value) else 0.0


def robust_zscore(values: Sequence[float]) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    z = (scores - median) / (1.4826 * mad + 1e-12)
    return np.clip(z, -5.0, 5.0)


def bounded_simplex_projection(values: Sequence[float], lower: float, upper: float) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("bounded simplex input must be a non-empty vector")
    if lower * raw.size > 1.0 + 1e-12 or upper * raw.size < 1.0 - 1e-12:
        raise ValueError("weight bounds are infeasible for a fully invested simplex")
    raw = np.where(np.isfinite(raw), raw, 0.0)
    if np.all(raw <= 0.0):
        raw = np.ones_like(raw)
    raw = raw / max(float(raw.sum()), 1e-12)
    lo = float(np.min(raw - upper)) - 1.0
    hi = float(np.max(raw - lower)) + 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        projected = np.clip(raw - mid, lower, upper)
        if projected.sum() > 1.0:
            lo = mid
        else:
            hi = mid
    result = np.clip(raw - 0.5 * (lo + hi), lower, upper)
    residual = 1.0 - float(result.sum())
    if abs(residual) > 1e-12:
        room = (upper - result) if residual > 0 else (result - lower)
        for index in np.argsort(room)[::-1]:
            delta = math.copysign(min(abs(residual), float(room[index])), residual)
            result[index] += delta
            residual -= delta
            if abs(residual) <= 1e-12:
                break
    if not np.isclose(result.sum(), 1.0, atol=1e-10):
        raise AssertionError("bounded simplex projection did not sum to one")
    return result


def _policy_weights(
    day: pd.DataFrame,
    policy_id: str,
    market_features: CausalMarketFeatures,
) -> tuple[np.ndarray, np.ndarray, str]:
    valid = day.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"]).copy()
    valid = valid.sort_values(["score", "stock_id"], ascending=[False, True])
    if valid.empty:
        return np.asarray([], dtype=int), np.asarray([], dtype=np.float64), "no_valid_scores"
    scores = valid["score"].to_numpy(dtype=np.float64)
    zscores = robust_zscore(scores)
    decision_date = pd.Timestamp(valid["sample_end_date"].iloc[0]).normalize()
    if policy_id == "P0":
        selected = np.arange(min(2, len(valid)))
        weights = np.full(len(selected), 1.0 / len(selected), dtype=np.float64)
        reason = "top2_equal"
    elif policy_id == "P1":
        selected = np.arange(min(2, len(valid)))
        logits = zscores[selected]
        logits = logits - float(np.max(logits))
        weights = bounded_simplex_projection(np.exp(logits), 0.40, 0.60)
        reason = "top2_robust_z_softmax_t1_bounds_040_060"
    elif policy_id == "P2":
        selected = np.arange(min(2, len(valid)))
        stocks = valid.iloc[selected]["stock_id"].astype(str).tolist()
        vol = market_features.combined_volatility(decision_date, stocks)
        weights = bounded_simplex_projection(1.0 / np.maximum(vol, 1e-12), 0.40, 0.60)
        reason = "top2_inverse_vol20_vol60_bounds_040_060"
    elif policy_id == "P3":
        selected = np.arange(min(5, len(valid)))
        weights = np.full(len(selected), 1.0 / len(selected), dtype=np.float64)
        reason = "top5_equal"
    elif policy_id == "P4":
        selected = np.arange(min(5, len(valid)))
        stocks = valid.iloc[selected]["stock_id"].astype(str).tolist()
        vol = market_features.combined_volatility(decision_date, stocks)
        weights = bounded_simplex_projection(1.0 / np.maximum(vol, 1e-12), 0.10, 0.30)
        reason = "top5_inverse_vol20_vol60_bounds_010_030"
    else:
        raise ValueError(f"unsupported position policy: {policy_id}")
    return valid.iloc[selected].index.to_numpy(), weights, reason


def apply_static_policy(
    predictions: pd.DataFrame,
    policy_id: str,
    market_features: CausalMarketFeatures,
) -> pd.DataFrame:
    if policy_id not in POLICY_IDS:
        raise ValueError(f"unknown policy {policy_id}")
    frame = predictions.copy().reset_index(drop=True)
    required = {"sample_end_date", "stock_id", "score"}
    if not required.issubset(frame.columns):
        raise KeyError(f"predictions missing columns: {sorted(required - set(frame.columns))}")
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
    frame["stock_id"] = frame["stock_id"].map(normalize_stock_code)
    frame["portfolio_weight"] = 0.0
    frame["portfolio_selected"] = False
    frame["position_policy"] = policy_id
    frame["position_reason"] = "not_selected"
    for _, day in frame.groupby("sample_end_date", sort=True):
        selected_index, weights, reason = _policy_weights(day, policy_id, market_features)
        if len(selected_index) == 0:
            continue
        frame.loc[selected_index, "portfolio_weight"] = weights
        frame.loc[selected_index, "portfolio_selected"] = True
        frame.loc[selected_index, "position_reason"] = reason
        if not np.isclose(float(weights.sum()), 1.0, atol=1e-10):
            raise AssertionError(f"{policy_id} is not fully invested")
    return frame


def controller_feature_row(
    prediction_day: pd.DataFrame,
    market_features: CausalMarketFeatures,
) -> dict[str, float]:
    day = prediction_day.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"]).copy()
    day = day.sort_values(["score", "stock_id"], ascending=[False, True])
    if len(day) < 5:
        raise ValueError("controller feature row requires at least five ranked stocks")
    scores = day["score"].to_numpy(dtype=np.float64)
    top = scores[:5]
    z = robust_zscore(scores)
    logits = z - float(np.max(z))
    probs = np.exp(logits)
    probs /= max(float(probs.sum()), 1e-12)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))) / math.log(len(probs)))
    date = pd.Timestamp(day["sample_end_date"].iloc[0]).normalize()
    stocks = day.iloc[:2]["stock_id"].astype(str).tolist()
    vol_pairs = [market_features.stock_volatility(date, stock) for stock in stocks]
    vol20 = np.asarray([item[0] for item in vol_pairs], dtype=np.float64)
    vol60 = np.asarray([item[1] for item in vol_pairs], dtype=np.float64)
    selected_vol20 = float(np.nanmean(vol20)) if np.isfinite(vol20).any() else 0.0
    selected_vol60 = float(np.nanmean(vol60)) if np.isfinite(vol60).any() else 0.0
    result = {
        "top1_top2_margin": float(top[0] - top[1]),
        "top2_top3_margin": float(top[1] - top[2]),
        "top5_score_dispersion": float(np.std(top, ddof=0)),
        "score_entropy": entropy,
        "selected_vol20": selected_vol20,
        "selected_vol60": selected_vol60,
        "selected_pair_correlation60": market_features.pair_correlation60(date, stocks[0], stocks[1]),
        "market_vol20": market_features.market_vol20(date),
    }
    return {key: float(np.nan_to_num(result[key], nan=0.0, posinf=0.0, neginf=0.0)) for key in CONTROLLER_FEATURES}


def evaluate_weighted_predictions(
    weighted_predictions: pd.DataFrame,
    market_path: str | Path,
    *,
    validation_start: str,
    validation_end: str,
    top_k: int,
) -> dict[str, Any]:
    daily, official = evaluate_official_proxy(weighted_predictions, top_k=top_k)
    weekly_all, _ = run_weekly_backtest(
        weighted_predictions,
        market_path,
        validation_start=validation_start,
        validation_end=validation_end,
        top_k=top_k,
    )
    weekly = weekly_all[weekly_all["skipped_missing_decision_prediction"].astype(int) == 0].reset_index(drop=True)
    summary = summarize_weekly_rows(weekly)
    net = weekly["weekly_return_net"].astype(float).to_numpy(dtype=np.float64)
    downside = net[net < 0.0]
    downside_deviation = float(np.std(downside)) if downside.size else 0.0
    summary["downside_deviation"] = downside_deviation
    summary["Sortino"] = safe_div(float(np.mean(net)), downside_deviation)
    summary["Calmar"] = safe_div(summary["cumulative_return_net"], summary["max_drawdown_abs"])
    concentration = weight_path_metrics(weighted_predictions, weekly)
    summary.update(concentration)
    summary["planned_week_count"] = int(len(weekly_all))
    summary["skipped_missing_decision_prediction"] = int(
        weekly_all["skipped_missing_decision_prediction"].astype(int).sum()
    )
    summary["boundary_week_count"] = int(weekly_all["boundary_week"].astype(bool).sum())
    return {
        "official_daily": daily,
        "official": official,
        "weekly_all": weekly_all,
        "weekly": weekly,
        "weekly_summary": summary,
    }


def _weights_for_date(weighted_predictions: pd.DataFrame, date: str) -> dict[str, float]:
    day = weighted_predictions[
        pd.to_datetime(weighted_predictions["sample_end_date"]).dt.strftime("%Y-%m-%d") == date
    ]
    day = day[pd.to_numeric(day["portfolio_weight"], errors="coerce").fillna(0.0) > 1e-12]
    return {
        normalize_stock_code(stock): float(weight)
        for stock, weight in zip(day["stock_id"], day["portfolio_weight"])
    }


def weight_path_metrics(weighted_predictions: pd.DataFrame, weekly: pd.DataFrame) -> dict[str, Any]:
    paths: list[dict[str, float]] = []
    for date in weekly["decision_date"].astype(str):
        paths.append(_weights_for_date(weighted_predictions, date))
    turnovers = []
    for previous, current in zip(paths, paths[1:]):
        universe = set(previous) | set(current)
        turnovers.append(0.5 * sum(abs(current.get(stock, 0.0) - previous.get(stock, 0.0)) for stock in universe))
    frequencies: dict[str, int] = {}
    max_weight = 0.0
    for weights in paths:
        for stock, weight in weights.items():
            frequencies[stock] = frequencies.get(stock, 0) + 1
            max_weight = max(max_weight, float(weight))
    week_count = max(1, len(paths))
    max_frequency = max(frequencies.values(), default=0) / week_count
    return {
        "mean_one_way_weight_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "max_single_weight": max_weight,
        "max_holding_frequency": float(max_frequency),
        "top_holding_frequencies": dict(
            sorted(((stock, count / week_count) for stock, count in frequencies.items()), key=lambda item: item[1], reverse=True)[:20]
        ),
    }


def paired_position_metrics(
    challenger_weekly: pd.DataFrame,
    p0_weekly: pd.DataFrame,
    *,
    tie_epsilon: float = TIE_EPSILON,
) -> dict[str, Any]:
    merged = challenger_weekly.merge(
        p0_weekly[["week_start", "weekly_return_gross", "weekly_return_net", "cost"]],
        on="week_start",
        how="inner",
        suffixes=("_challenger", "_p0"),
    )
    if merged.empty:
        raise ValueError("paired weekly comparison is empty")
    diff = (
        merged["weekly_return_net_challenger"].astype(float)
        - merged["weekly_return_net_p0"].astype(float)
    ).to_numpy(dtype=np.float64)
    gross_diff = (
        merged["weekly_return_gross_challenger"].astype(float)
        - merged["weekly_return_gross_p0"].astype(float)
    ).to_numpy(dtype=np.float64)
    cost_diff = (
        merged["cost_challenger"].astype(float) - merged["cost_p0"].astype(float)
    ).to_numpy(dtype=np.float64)
    wins = int(np.sum(diff > tie_epsilon))
    losses = int(np.sum(diff < -tie_epsilon))
    ties = int(len(diff) - wins - losses)
    best2 = float(np.sum(np.sort(diff)[::-1][: min(2, len(diff))]))
    positive_sum = float(np.sum(diff[diff > tie_epsilon]))
    gross_improvement = float(np.sum(gross_diff))
    incremental_cost_drag = float(np.sum(cost_diff))
    return {
        "paired_week_count": int(len(diff)),
        "mean_weekly_excess_vs_P0": float(np.mean(diff)),
        "median_weekly_excess_vs_P0": float(np.median(diff)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "tie_epsilon": float(tie_epsilon),
        "tie_adjusted_PDR": float((wins + 0.5 * ties) / len(diff)),
        "non_tie_win_rate": float(wins / max(1, wins + losses)),
        "positive_diff_rate_strict": float(np.mean(diff > 0.0)),
        "diff_without_best_2_weeks": float(np.sum(diff) - best2),
        "best_2_weeks_contribution_ratio": safe_div(best2, positive_sum) if positive_sum > 0.0 else math.inf,
        "excess_sum": float(np.sum(diff)),
        "gross_improvement": gross_improvement,
        "incremental_cost_drag": incremental_cost_drag,
        "incremental_cost_drag_to_gross_improvement": (
            safe_div(incremental_cost_drag, gross_improvement) if gross_improvement > 0.0 else math.inf
        ),
        "weekly_diffs": diff.tolist(),
    }


def static_policy_gate(policy_metrics: dict[str, Any], p0_metrics: dict[str, Any]) -> dict[str, Any]:
    paired = policy_metrics["paired_vs_P0"]
    weekly = policy_metrics["weekly"]
    p0_weekly = p0_metrics["weekly"]
    checks = {
        "mean_weekly_excess_positive": paired["mean_weekly_excess_vs_P0"] > 0.0,
        "diff_without_best_2_positive": paired["diff_without_best_2_weeks"] > 0.0,
        "max_drawdown_within_1_05": weekly["max_drawdown_abs"] <= p0_weekly["max_drawdown_abs"] * 1.05 + 1e-12,
        "cost_drag_ratio_within_0_50": paired["incremental_cost_drag_to_gross_improvement"] <= 0.50 + 1e-12,
    }
    return {"passes": bool(all(checks.values())), "checks": checks}


def build_controller_training_data(
    predictions: pd.DataFrame,
    p0_weekly: pd.DataFrame,
    pstar_weekly: pd.DataFrame,
    market_features: CausalMarketFeatures,
) -> pd.DataFrame:
    merged = pstar_weekly.merge(
        p0_weekly[["week_start", "weekly_return_gross", "weekly_return_net", "cost"]],
        on="week_start",
        how="inner",
        suffixes=("_pstar", "_p0"),
    )
    pred = predictions.copy()
    pred["sample_end_date"] = pd.to_datetime(pred["sample_end_date"]).dt.strftime("%Y-%m-%d")
    pred_by_date = {date: day for date, day in pred.groupby("sample_end_date", sort=True)}
    rows = []
    for row in merged.itertuples(index=False):
        decision_date = str(getattr(row, "decision_date"))
        if decision_date not in pred_by_date:
            continue
        features = controller_feature_row(pred_by_date[decision_date], market_features)
        fold_values = pred_by_date[decision_date].get("fold_id")
        fold_id = str(fold_values.iloc[0]) if fold_values is not None else "unknown"
        pstar_net = float(getattr(row, "weekly_return_net_pstar"))
        p0_net = float(getattr(row, "weekly_return_net_p0"))
        rows.append(
            {
                "fold": fold_id,
                "week_start": str(getattr(row, "week_start")),
                "decision_date": decision_date,
                "action": "P_STAR" if pstar_net > p0_net else "P0",
                "label": int(pstar_net > p0_net),
                "p0_weekly_return_gross": float(getattr(row, "weekly_return_gross_p0")),
                "p0_weekly_return_net": p0_net,
                "p0_cost": float(getattr(row, "cost_p0")),
                "pstar_weekly_return_gross": float(getattr(row, "weekly_return_gross_pstar")),
                "pstar_weekly_return_net": pstar_net,
                "pstar_cost": float(getattr(row, "cost_pstar")),
                "weekly_excess_pstar_vs_p0": pstar_net - p0_net,
                **features,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("controller training data is empty")
    if frame["decision_date"].duplicated().any():
        raise ValueError("controller training data has duplicate decision dates")
    return frame.sort_values("decision_date").reset_index(drop=True)


def apply_controller_policy(
    predictions: pd.DataFrame,
    p0_weighted: pd.DataFrame,
    pstar_weighted: pd.DataFrame,
    model: Any,
    market_features: CausalMarketFeatures,
    *,
    threshold: float = 0.55,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = predictions.copy().reset_index(drop=True)
    for frame in (base, p0_weighted, pstar_weighted):
        frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
        frame["stock_id"] = frame["stock_id"].map(normalize_stock_code)
    key = ["sample_end_date", "stock_id"]
    left = p0_weighted[key + ["portfolio_weight"]].rename(columns={"portfolio_weight": "p0_weight"})
    right = pstar_weighted[key + ["portfolio_weight"]].rename(columns={"portfolio_weight": "pstar_weight"})
    base = base.merge(left, on=key, how="left").merge(right, on=key, how="left")
    base[["p0_weight", "pstar_weight"]] = base[["p0_weight", "pstar_weight"]].fillna(0.0)
    action_rows = []
    base["portfolio_weight"] = 0.0
    base["controller_probability_pstar"] = 0.0
    base["controller_action"] = "P0"
    for date, day in base.groupby("sample_end_date", sort=True):
        features = controller_feature_row(day, market_features)
        vector = pd.DataFrame([[features[name] for name in CONTROLLER_FEATURES]], columns=CONTROLLER_FEATURES)
        probability = float(model.predict_proba(vector)[0, 1])
        action = "P_STAR" if probability >= threshold else "P0"
        weights = day["pstar_weight"] if action == "P_STAR" else day["p0_weight"]
        base.loc[day.index, "portfolio_weight"] = weights.to_numpy(dtype=np.float64)
        base.loc[day.index, "controller_probability_pstar"] = probability
        base.loc[day.index, "controller_action"] = action
        action_rows.append(
            {
                "decision_date": date,
                "probability_pstar": probability,
                "action": action,
                "selection_reason": f"probability_{probability:.12f}_{'>=' if action == 'P_STAR' else '<'}_{threshold:.2f}",
                **features,
            }
        )
    return base, pd.DataFrame(action_rows)


def block_bootstrap_probability(
    weekly_diffs: Sequence[float],
    *,
    block_size: int = 4,
    samples: int = 10000,
    seed: int = 20260720,
) -> dict[str, Any]:
    values = np.asarray(weekly_diffs, dtype=np.float64)
    if values.size == 0:
        raise ValueError("bootstrap requires weekly differences")
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, len(values) - block_size + 1))
    means = np.empty(samples, dtype=np.float64)
    blocks_needed = int(math.ceil(len(values) / block_size))
    for index in range(samples):
        pieces = []
        for start in rng.choice(starts, size=blocks_needed, replace=True):
            pieces.append(values[int(start) : int(start) + block_size])
        means[index] = float(np.mean(np.concatenate(pieces)[: len(values)]))
    return {
        "block_size": block_size,
        "samples": samples,
        "seed": seed,
        "probability_mean_excess_positive": float(np.mean(means > 0.0)),
        "bootstrap_mean": float(np.mean(means)),
        "bootstrap_p05": float(np.quantile(means, 0.05)),
        "bootstrap_p50": float(np.quantile(means, 0.50)),
        "bootstrap_p95": float(np.quantile(means, 0.95)),
    }


def quarterly_and_regime_breakdown(
    challenger_weekly: pd.DataFrame,
    p0_weekly: pd.DataFrame,
) -> dict[str, Any]:
    merged = challenger_weekly.merge(
        p0_weekly[["week_start", "weekly_return_net"]],
        on="week_start",
        suffixes=("_challenger", "_p0"),
        how="inner",
    )
    merged["buy_date"] = pd.to_datetime(merged["buy_date"])
    merged["quarter"] = merged["buy_date"].dt.to_period("Q").astype(str)
    merged["excess"] = merged["weekly_return_net_challenger"] - merged["weekly_return_net_p0"]
    quarters = {
        quarter: {
            "week_count": int(len(group)),
            "mean_excess": float(group["excess"].mean()),
            "sum_excess": float(group["excess"].sum()),
            "challenger_mean_net": float(group["weekly_return_net_challenger"].mean()),
            "p0_mean_net": float(group["weekly_return_net_p0"].mean()),
        }
        for quarter, group in merged.groupby("quarter", sort=True)
    }
    regimes = {}
    for token in ("up", "down", "sideways", "high_vol", "low_vol", "high_breadth", "low_breadth"):
        group = merged[merged["market_state"].astype(str).str.contains(token, regex=False)]
        regimes[token] = {
            "week_count": int(len(group)),
            "mean_excess": float(group["excess"].mean()) if len(group) else 0.0,
            "challenger_mean_net": float(group["weekly_return_net_challenger"].mean()) if len(group) else 0.0,
            "p0_mean_net": float(group["weekly_return_net_p0"].mean()) if len(group) else 0.0,
        }
    return {"quarters": quarters, "regimes": regimes}


def controller_gate(
    *,
    official_pc1: dict[str, Any],
    official_p0: dict[str, Any],
    weekly_pc1: dict[str, Any],
    weekly_p0: dict[str, Any],
    paired: dict[str, Any],
    breakdown: dict[str, Any],
    bootstrap: dict[str, Any],
    changed_week_coverage: float,
    seed_excesses: Sequence[float] | None,
    reproducibility_pass: bool,
    leakage_pass: bool,
    official_baseline_score: float = 0.012528499297343995,
) -> dict[str, Any]:
    primary_quarters = [breakdown["quarters"].get(key, {"mean_excess": -math.inf}) for key in ("2025Q1", "2025Q2", "2025Q3", "2025Q4")]
    positive_quarters = sum(float(item["mean_excess"]) > 0.0 for item in primary_quarters)
    recent_quarter = float(breakdown["quarters"].get("2025Q4", {"mean_excess": -math.inf})["mean_excess"])
    gross_improvement = float(paired["gross_improvement"])
    incremental_cost = float(paired["incremental_cost_drag"])
    cost_ratio_ok = incremental_cost <= 0.5 * gross_improvement + 1e-12 if gross_improvement > 0 else False
    seed_values = list(seed_excesses or [])
    checks = {
        "leakage_audit": bool(leakage_pass),
        "reproducibility": bool(reproducibility_pass),
        "official_proxy_not_below_p0": official_pc1["official_proxy_final_score"] >= official_p0["official_proxy_final_score"] - 1e-12,
        "official_baseline_exceeded": official_pc1["official_proxy_final_score"] > official_baseline_score,
        "net_cumulative_above_p0": weekly_pc1["cumulative_return_net"] > weekly_p0["cumulative_return_net"],
        "mean_weekly_excess_min_0_001": paired["mean_weekly_excess_vs_P0"] >= 0.0010,
        "median_weekly_excess_nonnegative": paired["median_weekly_excess_vs_P0"] >= -1e-12,
        "tie_adjusted_pdr_min_0_52": paired["tie_adjusted_PDR"] >= 0.52,
        "non_tie_win_rate_min_0_55": paired["non_tie_win_rate"] >= 0.55,
        "changed_week_coverage_min_0_25": changed_week_coverage >= 0.25,
        "positive_quarters_at_least_3_of_4": positive_quarters >= 3,
        "recent_quarter_nonnegative": recent_quarter >= 0.0,
        "diff_without_best_2_nonnegative": paired["diff_without_best_2_weeks"] >= -1e-12,
        "best_2_contribution_max_0_50": paired["best_2_weeks_contribution_ratio"] <= 0.50 + 1e-12,
        "max_drawdown_not_worse": weekly_pc1["max_drawdown_abs"] <= weekly_p0["max_drawdown_abs"] + 1e-12,
        "cvar_not_worse": weekly_pc1["CVaR_95_loss_abs"] <= weekly_p0["CVaR_95_loss_abs"] + 1e-12,
        "worst_week_delta_min_minus_0_005": weekly_pc1["worst_week"] >= weekly_p0["worst_week"] - 0.005 - 1e-12,
        "incremental_cost_within_half_gross": cost_ratio_ok,
        "turnover_ratio_max_1_25": weekly_pc1["mean_one_way_weight_turnover"] <= 1.25 * weekly_p0["mean_one_way_weight_turnover"] + 1e-12,
        "single_weight_max_0_60": weekly_pc1["max_single_weight"] <= 0.60 + 1e-12,
        "holding_frequency_not_above_p0": weekly_pc1["max_holding_frequency"] <= weekly_p0["max_holding_frequency"] + 1e-12,
        "bootstrap_probability_min_0_95": bootstrap["probability_mean_excess_positive"] >= 0.95,
        "serious_three_seeds_present": len(seed_values) == 3,
        "seed_mean_excess_positive": len(seed_values) == 3 and float(np.mean(seed_values)) > 0.0,
        "seed_min_excess_nonnegative": len(seed_values) == 3 and float(np.min(seed_values)) >= 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "decision": "ACCEPT_AS_CHAMPION" if not failed else "REJECT",
        "checks": checks,
        "failed_checks": failed,
        "positive_2025_quarters": positive_quarters,
        "recent_quarter_mean_excess": recent_quarter,
        "seed_excesses": seed_values,
        "seed_mean_excess": float(np.mean(seed_values)) if seed_values else None,
        "seed_min_excess": float(np.min(seed_values)) if seed_values else None,
    }


def iter_feature_rows(
    predictions: pd.DataFrame,
    market_features: CausalMarketFeatures,
) -> Iterable[tuple[str, dict[str, float]]]:
    frame = predictions.copy()
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
    for date, day in frame.groupby("sample_end_date", sort=True):
        yield str(date), controller_feature_row(day, market_features)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from featurework import CLOSE_COL, DATE_COL, STOCK_COL


@dataclass(frozen=True)
class DefensiveContext:
    decision_date: pd.Timestamp
    candidates: pd.DataFrame
    correlation: pd.DataFrame
    pool_size_used: int
    universe_size: int
    feasible_capacity: int
    cluster_count: int
    largest_cluster: int
    regime_multiplier: float


def percentile_risk(values: pd.Series, *, missing_worst: bool = True) -> pd.Series:
    working = pd.to_numeric(values, errors="coerce").astype(float)
    finite = working[np.isfinite(working)]
    if finite.empty:
        return pd.Series(1.0 if missing_worst else 0.5, index=values.index)
    if missing_worst:
        fill_value = float(finite.max()) + max(float(finite.std()), 1e-12)
    else:
        fill_value = float(finite.median())
    working = working.fillna(fill_value)
    if len(working) == 1:
        return pd.Series(0.5, index=values.index)
    ranks = working.rank(method="average", ascending=True)
    return (ranks - 1.0) / (len(working) - 1.0)


def complete_clusters(
    correlation: pd.DataFrame,
    *,
    correlation_cut: float,
) -> dict[str, str]:
    stocks = list(correlation.index.astype(str))
    if len(stocks) == 1:
        return {stocks[0]: stocks[0]}
    values = correlation.to_numpy(dtype=float)
    distance = np.sqrt(np.clip((1.0 - values) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method="complete", optimal_ordering=True)
    threshold = math.sqrt((1.0 - correlation_cut) / 2.0)
    labels = fcluster(tree, t=threshold, criterion="distance")
    members: dict[int, list[str]] = {}
    for stock, label in zip(stocks, labels, strict=True):
        members.setdefault(int(label), []).append(stock)
    canonical = {label: min(group) for label, group in members.items()}
    return {
        stock: canonical[int(label)]
        for stock, label in zip(stocks, labels, strict=True)
    }


def candidate_pool_sizes(universe_size: int, pool_steps: Sequence[int | None]) -> list[int]:
    sizes: list[int] = []
    for raw in pool_steps:
        value = universe_size if raw is None else min(int(raw), universe_size)
        if value >= 5 and value not in sizes:
            sizes.append(value)
    return sizes


def _market_arrays(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = raw[[STOCK_COL, DATE_COL, CLOSE_COL]].copy()
    source[STOCK_COL] = (
        source[STOCK_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    source[DATE_COL] = pd.to_datetime(source[DATE_COL]).dt.normalize()
    source[CLOSE_COL] = pd.to_numeric(source[CLOSE_COL], errors="coerce")
    source = source.drop_duplicates([DATE_COL, STOCK_COL], keep="last")
    close = (
        source.pivot(index=DATE_COL, columns=STOCK_COL, values=CLOSE_COL)
        .sort_index()
        .sort_index(axis=1)
    )
    log_returns = np.log(close.where(close > 0.0)).diff()
    lower = log_returns.quantile(0.01, axis=1)
    upper = log_returns.quantile(0.99, axis=1)
    clipped = log_returns.clip(lower=lower, upper=upper, axis=0)
    sum5 = clipped.rolling(5, min_periods=4).sum()
    sum20 = clipped.rolling(20, min_periods=15).sum()
    return clipped, sum5, sum20


def _snapshot(
    *,
    returns: pd.DataFrame,
    sum5: pd.DataFrame,
    sum20: pd.DataFrame,
    decision_date: pd.Timestamp,
    stocks: Sequence[str],
    lookback: int,
    min_observations: int,
    shrinkage: float,
    missing_correlation: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    position = int(returns.index.searchsorted(decision_date, side="left")) - 1
    if position < lookback:
        raise ValueError(
            f"insufficient lag-1 market history before {decision_date.date()}"
        )
    window = returns.iloc[position - lookback + 1 : position + 1].reindex(
        columns=stocks
    )
    raw_corr = window.corr(min_periods=min_observations)
    values = raw_corr.to_numpy(dtype=float)
    off_diagonal = values[
        ~np.eye(len(stocks), dtype=bool) & np.isfinite(values)
    ]
    median_corr = float(np.median(off_diagonal)) if off_diagonal.size else 0.0
    correlation = (1.0 - shrinkage) * raw_corr + shrinkage * median_corr
    correlation = correlation.fillna(missing_correlation).clip(-1.0, 1.0)
    np.fill_diagonal(correlation.values, 1.0)

    volatility = (
        returns.rolling(20, min_periods=15).std(ddof=0).iloc[position].reindex(stocks)
    )
    recent5 = sum5.iloc[position].reindex(stocks)
    total20 = sum20.iloc[position].reindex(stocks)
    lead15 = total20 - recent5
    lead_rank = percentile_risk(lead15, missing_worst=False)
    reversal_down = (-recent5).clip(lower=0.0)
    reversal_down_rank = percentile_risk(reversal_down, missing_worst=True)
    reversal_rank = percentile_risk(
        lead_rank * reversal_down_rank,
        missing_worst=True,
    )
    return volatility, reversal_rank, correlation


def build_context(
    *,
    raw: pd.DataFrame,
    decision_date: pd.Timestamp,
    stock_codes: Sequence[str],
    scores: np.ndarray,
    settings: Mapping[str, Any],
) -> DefensiveContext:
    ordered = pd.DataFrame(
        {
            "stock_id": [str(stock).zfill(6) for stock in stock_codes],
            "score": np.asarray(scores, dtype=np.float64),
        }
    )
    ordered = (
        ordered.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates("stock_id", keep="first")
        .sort_values(["score", "stock_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if len(ordered) < 5:
        raise ValueError("fewer than five finite Candidate-D ranking scores")

    returns, sum5, sum20 = _market_arrays(raw)
    pool_steps = tuple(settings.get("pool_steps", [20, 50, 100, None]))
    max_cluster_holdings = int(settings["max_cluster_holdings"])
    chosen: tuple[
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.DataFrame,
        dict[str, str],
        int,
        int,
        int,
    ] | None = None
    for pool_size in candidate_pool_sizes(len(ordered), pool_steps):
        day = ordered.head(pool_size).copy().reset_index(drop=True)
        stocks = day["stock_id"].tolist()
        volatility, reversal, correlation = _snapshot(
            returns=returns,
            sum5=sum5,
            sum20=sum20,
            decision_date=decision_date,
            stocks=stocks,
            lookback=int(settings["correlation_lookback"]),
            min_observations=int(settings["correlation_min_observations"]),
            shrinkage=float(settings["correlation_shrinkage"]),
            missing_correlation=float(settings["correlation_missing_value"]),
        )
        clusters = complete_clusters(
            correlation,
            correlation_cut=float(settings["correlation_cluster_cut"]),
        )
        sizes = pd.Series([clusters[stock] for stock in stocks]).value_counts()
        capacity = int(sizes.clip(upper=max_cluster_holdings).sum())
        if capacity >= 5:
            chosen = (
                day,
                volatility,
                reversal,
                correlation,
                clusters,
                capacity,
                int(len(sizes)),
                int(sizes.max()),
            )
            break
    if chosen is None:
        raise ValueError("REJECTED_AT_COMBINATION_GATE: full ranking is infeasible")

    (
        day,
        volatility,
        reversal,
        correlation,
        clusters,
        capacity,
        cluster_count,
        largest_cluster,
    ) = chosen
    day["d_rank_score"] = np.linspace(1.0, 0.0, len(day))
    day["volatility_rank"] = percentile_risk(
        volatility.reindex(day["stock_id"]).reset_index(drop=True),
        missing_worst=True,
    ).to_numpy(dtype=float)
    day["reversal_risk_rank"] = (
        reversal.reindex(day["stock_id"]).reset_index(drop=True).to_numpy(dtype=float)
    )
    day["cluster_id"] = day["stock_id"].map(clusters)
    return DefensiveContext(
        decision_date=decision_date,
        candidates=day,
        correlation=correlation,
        pool_size_used=int(len(day)),
        universe_size=int(len(ordered)),
        feasible_capacity=capacity,
        cluster_count=cluster_count,
        largest_cluster=largest_cluster,
        regime_multiplier=float(settings.get("regime_multiplier", 1.0)),
    )


def select_defensive(
    context: DefensiveContext,
    *,
    settings: Mapping[str, Any],
    previous_weights: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = context.candidates
    previous = dict(previous_weights or {})
    selected: list[dict[str, Any]] = []
    selected_stocks: list[str] = []
    cluster_counts: dict[str, int] = {}
    max_cluster_holdings = int(settings["max_cluster_holdings"])
    for order in range(5):
        choices: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for row in candidates.itertuples(index=False):
            stock = str(row.stock_id)
            if stock in selected_stocks:
                continue
            cluster = str(row.cluster_id)
            cluster_count = cluster_counts.get(cluster, 0)
            if cluster_count >= max_cluster_holdings:
                continue
            cluster_penalty = 1.0 if cluster_count == 1 else 0.0
            turnover_penalty = (
                0.0 if not previous or stock in previous else 1.0
            )
            max_corr = (
                max(
                    float(context.correlation.loc[stock, prior])
                    for prior in selected_stocks
                )
                if selected_stocks
                else 0.0
            )
            defensive_score = (
                float(row.d_rank_score)
                - context.regime_multiplier
                * float(settings["lambda_volatility"])
                * float(row.volatility_rank)
                - context.regime_multiplier
                * float(settings["lambda_correlation"])
                * cluster_penalty
                - float(settings["lambda_turnover"]) * turnover_penalty
                - context.regime_multiplier
                * float(settings["lambda_reversal"])
                * float(row.reversal_risk_rank)
            )
            payload = {
                "stock_id": stock,
                "selection_order": order + 1,
                "cluster_id": cluster,
                "defensive_score": defensive_score,
                "d_rank_score": float(row.d_rank_score),
                "volatility_rank": float(row.volatility_rank),
                "reversal_risk_rank": float(row.reversal_risk_rank),
                "max_selected_correlation": max_corr,
                "turnover_penalty": turnover_penalty,
                "cluster_penalty": cluster_penalty,
            }
            key = (
                -defensive_score,
                -float(row.d_rank_score),
                float(row.volatility_rank),
                max_corr,
                0 if stock in previous else 1,
                stock,
            )
            choices.append((key, payload))
        if not choices:
            raise ValueError("unable to select five stocks under cluster cap")
        _, chosen = min(choices, key=lambda item: item[0])
        selected.append(chosen)
        selected_stocks.append(chosen["stock_id"])
        cluster_counts[chosen["cluster_id"]] = (
            cluster_counts.get(chosen["cluster_id"], 0) + 1
        )

    weights = [float(value) for value in settings["weights"]]
    if len(weights) != 5 or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("defensive weights must contain five values summing to one")
    for item, weight in zip(selected, weights, strict=True):
        item["weight"] = weight
    detail = pd.DataFrame(selected)
    result = detail[["stock_id", "weight"]].copy()

    pair_values: list[float] = []
    for left in range(len(selected_stocks)):
        for right in range(left + 1, len(selected_stocks)):
            pair_values.append(
                float(
                    context.correlation.loc[
                        selected_stocks[left], selected_stocks[right]
                    ]
                )
            )
    diagnostics = {
        "classification": "FORCED_TEST_PACKAGE_NOT_RESEARCH_GATE_APPROVED",
        "policy_id": str(settings["policy_id"]),
        "decision_date": context.decision_date.strftime("%Y-%m-%d"),
        "pool_size_used": context.pool_size_used,
        "universe_size": context.universe_size,
        "feasible_capacity": context.feasible_capacity,
        "cluster_count": context.cluster_count,
        "largest_cluster": context.largest_cluster,
        "regime_multiplier": context.regime_multiplier,
        "hhi": float(np.square(weights).sum()),
        "max_weight": float(max(weights)),
        "weighted_pair_correlation": (
            float(np.mean(pair_values)) if pair_values else 0.0
        ),
        "selected": detail.to_dict(orient="records"),
    }
    return result, diagnostics


def build_defensive_result(
    *,
    raw: pd.DataFrame,
    decision_date: pd.Timestamp,
    stock_codes: Sequence[str],
    scores: np.ndarray,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    context = build_context(
        raw=raw,
        decision_date=pd.Timestamp(decision_date).normalize(),
        stock_codes=stock_codes,
        scores=scores,
        settings=settings,
    )
    return select_defensive(context, settings=settings)

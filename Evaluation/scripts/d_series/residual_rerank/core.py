from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


GRID_LAMBDAS: tuple[tuple[float, float, float], ...] = tuple(
    product(
        (0.00, 0.05, 0.10, 0.15),
        (0.00, 0.05, 0.10),
        (0.00, 0.05, 0.10),
    )
)

DEFAULT_COLUMNS = {
    "stock": "stock_id",
    "date": "date",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "amount",
}

POLICY_IDS = (
    "TOP2_EQUAL",
    "TOP5_EQUAL",
    "TOP5_FIXED_30_25_20_15_10",
)


def _finite_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def cross_section_rank(values: pd.Series) -> pd.Series:
    """Winsorize, percentile-rank, map to [-1, 1], and clip to [-.95, .95]."""

    numeric = _finite_numeric(values)
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series(0.0, index=values.index, dtype=float)
    lower = float(finite.quantile(0.025))
    upper = float(finite.quantile(0.975))
    clipped = numeric.clip(lower=lower, upper=upper)
    ranked = clipped.rank(method="average", pct=True, ascending=True)
    normalized = (2.0 * (ranked - 0.5)).clip(-0.95, 0.95)
    return normalized.fillna(0.0).astype(float)


def _positive_percentile(values: pd.Series) -> pd.Series:
    numeric = _finite_numeric(values)
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series(0.0, index=values.index, dtype=float)
    lower = float(finite.quantile(0.025))
    upper = float(finite.quantile(0.975))
    return (
        numeric.clip(lower=lower, upper=upper)
        .rank(method="average", pct=True, ascending=True)
        .fillna(0.0)
        .astype(float)
    )


def _ols_residuals(stock_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
    mask = np.isfinite(stock_returns) & np.isfinite(market_returns)
    if int(mask.sum()) < 40:
        return np.full(stock_returns.shape, np.nan, dtype=np.float64)
    y = stock_returns[mask].astype(np.float64)
    x = market_returns[mask].astype(np.float64)
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    result = np.full(stock_returns.shape, np.nan, dtype=np.float64)
    result[mask] = y - design @ coefficients
    return result


def _safe_log_ratio(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or value <= 0 or reference <= 0:
        return np.nan
    return float(np.log(value / reference))


def build_causal_factor_table(
    raw: pd.DataFrame,
    decision_dates: Iterable[pd.Timestamp | str],
    columns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build lag-1 residual factors.

    Every factor for decision date T is computed from rows strictly earlier than
    T. Rows at or after T are ignored, making the function prefix invariant.
    """

    names = {**DEFAULT_COLUMNS, **(dict(columns) if columns else {})}
    missing = [value for value in names.values() if value not in raw.columns]
    if missing:
        raise KeyError(f"missing factor input columns: {sorted(missing)}")

    frame = raw[list(names.values())].copy()
    frame = frame.rename(columns={value: key for key, value in names.items()})
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["stock"] = (
        frame["stock"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    for column in ("open", "close", "high", "low", "volume", "amount"):
        frame[column] = _finite_numeric(frame[column])
    frame = (
        frame.drop_duplicates(["stock", "date"], keep="last")
        .sort_values(["stock", "date"])
        .reset_index(drop=True)
    )
    if int(frame["stock"].nunique()) < 10:
        raise ValueError("factor construction requires at least 10 stocks")

    frame["log_return"] = frame.groupby("stock", sort=False)["close"].transform(
        lambda values: np.log(values / values.shift(1))
    )
    market = frame.groupby("date", sort=True)["log_return"].mean()
    frame["market_log_return"] = frame["date"].map(market)

    output: list[pd.DataFrame] = []
    available_dates = np.asarray(sorted(frame["date"].dropna().unique()))
    grouped = {stock: group.reset_index(drop=True) for stock, group in frame.groupby("stock", sort=True)}

    for decision_value in sorted({pd.Timestamp(value).normalize() for value in decision_dates}):
        prior_dates = available_dates[available_dates < np.datetime64(decision_value)]
        if prior_dates.size == 0:
            raise ValueError(f"no lag-1 data before decision date {decision_value.date()}")
        asof = pd.Timestamp(prior_dates[-1]).normalize()
        rows: list[dict[str, object]] = []
        for stock, full_group in grouped.items():
            group = full_group.loc[full_group["date"] <= asof].tail(61).copy()
            if len(group) < 41:
                rows.append(
                    {
                        "decision_date": decision_value,
                        "factor_asof_date": asof,
                        "stock_id": stock,
                    }
                )
                continue

            regression = group.tail(60)
            residuals = _ols_residuals(
                regression["log_return"].to_numpy(dtype=np.float64),
                regression["market_log_return"].to_numpy(dtype=np.float64),
            )
            valid_residuals = residuals[np.isfinite(residuals)]
            current = group.iloc[-1]
            previous = group.iloc[-2]
            prior20 = group.iloc[-21:-1]

            high_reference = float(prior20["high"].max())
            breakout = (
                float(current["close"]) / high_reference - 1.0
                if np.isfinite(high_reference) and high_reference > 0
                else np.nan
            )
            volume_confirmation = _safe_log_ratio(
                float(current["volume"]),
                float(prior20["volume"].median()),
            )
            amount_confirmation = _safe_log_ratio(
                float(current["amount"]),
                float(prior20["amount"].median()),
            )
            gap_up = max(
                float(current["open"]) / float(previous["close"]) - 1.0,
                0.0,
            )
            intraday_fade = max(
                1.0 - float(current["close"]) / float(current["open"]),
                0.0,
            )
            price_range = max(float(current["high"]) - float(current["low"]), 1e-12)
            upper_shadow = max(
                float(current["high"]) - max(float(current["open"]), float(current["close"])),
                0.0,
            ) / price_range

            rows.append(
                {
                    "decision_date": decision_value,
                    "factor_asof_date": asof,
                    "stock_id": stock,
                    "residual_5": (
                        float(valid_residuals[-5:].sum())
                        if len(valid_residuals) >= 20
                        else np.nan
                    ),
                    "residual_20": (
                        float(valid_residuals[-20:].sum())
                        if len(valid_residuals) >= 20
                        else np.nan
                    ),
                    "residual_20_ex5": (
                        float(valid_residuals[-20:-5].sum())
                        if len(valid_residuals) >= 20
                        else np.nan
                    ),
                    "breakout_20": breakout,
                    "signed_volume_confirmation": (
                        float(np.sign(breakout) * max(volume_confirmation, 0.0))
                        if np.isfinite(breakout) and np.isfinite(volume_confirmation)
                        else np.nan
                    ),
                    "signed_amount_confirmation": (
                        float(np.sign(breakout) * max(amount_confirmation, 0.0))
                        if np.isfinite(breakout) and np.isfinite(amount_confirmation)
                        else np.nan
                    ),
                    "gap_fade": gap_up * intraday_fade,
                    "upper_shadow": upper_shadow,
                    "short_reversal": (
                        max(-float(valid_residuals[-3:].sum()), 0.0)
                        if len(valid_residuals) >= 3
                        else np.nan
                    ),
                }
            )

        day = pd.DataFrame(rows)
        momentum_components = pd.concat(
            [
                cross_section_rank(day["residual_5"]),
                cross_section_rank(day["residual_20"]),
                cross_section_rank(day["residual_20_ex5"]),
            ],
            axis=1,
        ).mean(axis=1)
        day["momentum_rank"] = cross_section_rank(momentum_components)

        volume_components = (
            0.50 * cross_section_rank(day["breakout_20"])
            + 0.25 * cross_section_rank(day["signed_volume_confirmation"])
            + 0.25 * cross_section_rank(day["signed_amount_confirmation"])
        )
        day["volume_rank"] = cross_section_rank(volume_components)

        leader = day["momentum_rank"].clip(lower=0.0)
        risk_raw = leader * (
            _positive_percentile(day["gap_fade"])
            + _positive_percentile(day["upper_shadow"])
            + _positive_percentile(day["short_reversal"])
        ) / 3.0
        day["reversal_rank"] = cross_section_rank(risk_raw)
        required = [
            "residual_5",
            "residual_20",
            "residual_20_ex5",
            "breakout_20",
            "signed_volume_confirmation",
            "signed_amount_confirmation",
            "gap_fade",
            "upper_shadow",
            "short_reversal",
        ]
        day["factor_complete"] = day[required].notna().all(axis=1)
        output.append(day)

    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["decision_date", "stock_id"])
        .reset_index(drop=True)
    )


def combine_alpha_scores(
    scores: pd.DataFrame,
    factors: pd.DataFrame,
    lambdas: Sequence[float],
) -> pd.DataFrame:
    if tuple(float(value) for value in lambdas) not in GRID_LAMBDAS:
        raise ValueError(f"lambdas must be one of the 36 preregistered points: {lambdas}")
    required_scores = {"decision_date", "stock_id", "score"}
    required_factors = {
        "decision_date",
        "stock_id",
        "momentum_rank",
        "volume_rank",
        "reversal_rank",
    }
    if missing := required_scores.difference(scores.columns):
        raise KeyError(f"missing score columns: {sorted(missing)}")
    if missing := required_factors.difference(factors.columns):
        raise KeyError(f"missing factor columns: {sorted(missing)}")

    left = scores.copy()
    right = factors.copy()
    left["decision_date"] = pd.to_datetime(left["decision_date"]).dt.normalize()
    right["decision_date"] = pd.to_datetime(right["decision_date"]).dt.normalize()
    left["stock_id"] = left["stock_id"].astype(str).str.zfill(6)
    right["stock_id"] = right["stock_id"].astype(str).str.zfill(6)
    merged = left.merge(
        right,
        on=["decision_date", "stock_id"],
        how="left",
        validate="one_to_one",
    )
    if merged[["momentum_rank", "volume_rank", "reversal_rank"]].isna().any().any():
        raise ValueError("score/factor merge produced missing factor ranks")
    merged["d_rank_score"] = (
        merged.groupby("decision_date", group_keys=False)["score"]
        .apply(cross_section_rank)
        .sort_index()
    )
    lambda_momentum, lambda_volume, lambda_reversal = [
        float(value) for value in lambdas
    ]
    merged["alpha_score"] = (
        merged["d_rank_score"]
        + lambda_momentum * merged["momentum_rank"]
        + lambda_volume * merged["volume_rank"]
        - lambda_reversal * merged["reversal_rank"]
    )
    merged["lambda_momentum"] = lambda_momentum
    merged["lambda_volume"] = lambda_volume
    merged["lambda_reversal"] = lambda_reversal
    return merged.sort_values(["decision_date", "stock_id"]).reset_index(drop=True)


def policy_weights(policy_id: str) -> np.ndarray:
    if policy_id == "TOP2_EQUAL":
        return np.asarray([0.50, 0.50], dtype=np.float64)
    if policy_id == "TOP5_EQUAL":
        return np.full(5, 0.20, dtype=np.float64)
    if policy_id == "TOP5_FIXED_30_25_20_15_10":
        return np.asarray([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64)
    raise ValueError(f"unsupported frozen policy: {policy_id}")


def select_portfolios(alpha_scores: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    score_column = "score" if policy_id == "TOP2_EQUAL" else "alpha_score"
    if score_column not in alpha_scores:
        raise KeyError(f"missing selection score: {score_column}")
    weights = policy_weights(policy_id)
    rows: list[pd.DataFrame] = []
    for decision_date, day in alpha_scores.groupby("decision_date", sort=True):
        selected = (
            day.sort_values(
                [score_column, "score", "stock_id"],
                ascending=[False, False, True],
            )
            .head(len(weights))
            .copy()
        )
        if len(selected) != len(weights):
            raise ValueError(f"{decision_date} has fewer stocks than {policy_id} requires")
        selected["weight"] = weights
        selected["portfolio_rank"] = np.arange(1, len(weights) + 1)
        selected["policy_id"] = policy_id
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def calibration_choice(metrics: pd.DataFrame) -> dict[str, object]:
    required = {
        "lambda_momentum",
        "lambda_volume",
        "lambda_reversal",
        "stability_score",
        "mean_excess",
        "median_excess",
    }
    if missing := required.difference(metrics.columns):
        raise KeyError(f"missing calibration metrics: {sorted(missing)}")
    working = metrics.copy()
    working["lambda_sum"] = (
        working["lambda_momentum"]
        + working["lambda_volume"]
        + working["lambda_reversal"]
    )
    zero = working.loc[working["lambda_sum"].abs() <= 1e-12]
    nonzero = working.loc[working["lambda_sum"] > 1e-12].copy()
    if len(zero) != 1 or nonzero.empty:
        raise ValueError("calibration requires exactly one zero point and nonzero points")
    nonzero = nonzero.sort_values(
        [
            "stability_score",
            "lambda_sum",
            "lambda_momentum",
            "lambda_volume",
            "lambda_reversal",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    winner = nonzero.iloc[0]
    zero_score = float(zero.iloc[0]["stability_score"])
    passed = (
        float(winner["stability_score"]) > zero_score + 1e-12
        and float(winner["mean_excess"]) > 0.0
        and float(winner["median_excess"]) > 0.0
    )
    return {
        "status": "PASS" if passed else "REJECTED_AT_CALIBRATION_GATE",
        "selected_lambdas": {
            "lambda_momentum": float(winner["lambda_momentum"]),
            "lambda_volume": float(winner["lambda_volume"]),
            "lambda_reversal": float(winner["lambda_reversal"]),
        },
        "selected_stability_score": float(winner["stability_score"]),
        "zero_stability_score": zero_score,
        "selected_mean_excess": float(winner["mean_excess"]),
        "selected_median_excess": float(winner["median_excess"]),
    }

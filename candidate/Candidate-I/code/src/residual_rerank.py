from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from featurework import (
    AMOUNT_COL,
    CLOSE_COL,
    DATE_COL,
    HIGH_COL,
    LOW_COL,
    OPEN_COL,
    STOCK_COL,
    VOLUME_COL,
)


ALLOWED_LAMBDAS = {
    "lambda_momentum": (0.00, 0.05, 0.10, 0.15),
    "lambda_volume": (0.00, 0.05, 0.10),
    "lambda_reversal": (0.00, 0.05, 0.10),
}


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def cross_section_rank(values: pd.Series) -> pd.Series:
    numeric = _numeric(values)
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series(0.0, index=values.index, dtype=float)
    lower = float(finite.quantile(0.025))
    upper = float(finite.quantile(0.975))
    ranked = (
        numeric.clip(lower=lower, upper=upper)
        .rank(method="average", pct=True, ascending=True)
    )
    return (2.0 * (ranked - 0.5)).clip(-0.95, 0.95).fillna(0.0)


def _positive_percentile(values: pd.Series) -> pd.Series:
    numeric = _numeric(values)
    finite = numeric.dropna()
    if finite.empty:
        return pd.Series(0.0, index=values.index, dtype=float)
    lower = float(finite.quantile(0.025))
    upper = float(finite.quantile(0.975))
    return (
        numeric.clip(lower=lower, upper=upper)
        .rank(method="average", pct=True, ascending=True)
        .fillna(0.0)
    )


def _ols_residuals(
    stock_returns: np.ndarray,
    market_returns: np.ndarray,
) -> np.ndarray:
    mask = np.isfinite(stock_returns) & np.isfinite(market_returns)
    if int(mask.sum()) < 40:
        return np.full(stock_returns.shape, np.nan, dtype=np.float64)
    y = stock_returns[mask].astype(np.float64)
    x = market_returns[mask].astype(np.float64)
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = np.full(stock_returns.shape, np.nan, dtype=np.float64)
    residuals[mask] = y - design @ coefficients
    return residuals


def _safe_log_ratio(value: float, reference: float) -> float:
    if (
        not np.isfinite(value)
        or not np.isfinite(reference)
        or value <= 0.0
        or reference <= 0.0
    ):
        return np.nan
    return float(np.log(value / reference))


def build_lag1_residual_factors(
    raw: pd.DataFrame,
    decision_date: pd.Timestamp | str,
) -> pd.DataFrame:
    required = {
        STOCK_COL,
        DATE_COL,
        OPEN_COL,
        CLOSE_COL,
        HIGH_COL,
        LOW_COL,
        VOLUME_COL,
        AMOUNT_COL,
    }
    if missing := required.difference(raw.columns):
        raise KeyError(f"missing residual-rerank columns: {sorted(missing)}")

    frame = raw[list(required)].copy().rename(
        columns={
            STOCK_COL: "stock_id",
            DATE_COL: "date",
            OPEN_COL: "open",
            CLOSE_COL: "close",
            HIGH_COL: "high",
            LOW_COL: "low",
            VOLUME_COL: "volume",
            AMOUNT_COL: "amount",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["stock_id"] = (
        frame["stock_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    for column in ("open", "close", "high", "low", "volume", "amount"):
        frame[column] = _numeric(frame[column])
    frame = (
        frame.drop_duplicates(["stock_id", "date"], keep="last")
        .sort_values(["stock_id", "date"])
        .reset_index(drop=True)
    )

    decision = pd.Timestamp(decision_date).normalize()
    prior_dates = sorted(frame.loc[frame["date"] < decision, "date"].unique())
    if not prior_dates:
        raise ValueError("no lag-1 market data before the decision date")
    asof = pd.Timestamp(prior_dates[-1]).normalize()
    frame = frame.loc[frame["date"] <= asof].copy()
    frame["log_return"] = frame.groupby("stock_id", sort=False)[
        "close"
    ].transform(lambda values: np.log(values / values.shift(1)))
    market = frame.groupby("date", sort=True)["log_return"].mean()
    frame["market_log_return"] = frame["date"].map(market)

    rows: list[dict[str, object]] = []
    for stock_id, full_group in frame.groupby("stock_id", sort=True):
        group = full_group.tail(61).copy()
        row: dict[str, object] = {
            "decision_date": decision,
            "factor_asof_date": asof,
            "stock_id": stock_id,
        }
        if len(group) < 41:
            rows.append(row)
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
            if np.isfinite(high_reference) and high_reference > 0.0
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
        price_range = max(
            float(current["high"]) - float(current["low"]),
            1e-12,
        )
        upper_shadow = max(
            float(current["high"])
            - max(float(current["open"]), float(current["close"])),
            0.0,
        ) / price_range
        row.update(
            {
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
                    if np.isfinite(breakout)
                    and np.isfinite(volume_confirmation)
                    else np.nan
                ),
                "signed_amount_confirmation": (
                    float(np.sign(breakout) * max(amount_confirmation, 0.0))
                    if np.isfinite(breakout)
                    and np.isfinite(amount_confirmation)
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
        rows.append(row)

    factors = pd.DataFrame(rows)
    momentum_components = pd.concat(
        [
            cross_section_rank(factors["residual_5"]),
            cross_section_rank(factors["residual_20"]),
            cross_section_rank(factors["residual_20_ex5"]),
        ],
        axis=1,
    ).mean(axis=1)
    factors["momentum_rank"] = cross_section_rank(momentum_components)
    volume_components = (
        0.50 * cross_section_rank(factors["breakout_20"])
        + 0.25 * cross_section_rank(factors["signed_volume_confirmation"])
        + 0.25 * cross_section_rank(factors["signed_amount_confirmation"])
    )
    factors["volume_rank"] = cross_section_rank(volume_components)
    leader = factors["momentum_rank"].clip(lower=0.0)
    risk_raw = leader * (
        _positive_percentile(factors["gap_fade"])
        + _positive_percentile(factors["upper_shadow"])
        + _positive_percentile(factors["short_reversal"])
    ) / 3.0
    factors["reversal_rank"] = cross_section_rank(risk_raw)
    component_columns = [
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
    factors["factor_complete"] = factors[component_columns].notna().all(axis=1)
    return factors.sort_values("stock_id").reset_index(drop=True)


def _validate_lambdas(lambdas: Mapping[str, float]) -> None:
    for name, allowed in ALLOWED_LAMBDAS.items():
        value = float(lambdas[name])
        if value not in allowed:
            raise ValueError(f"{name}={value} is outside the preregistered grid")


def build_residual_rerank_portfolio(
    *,
    raw: pd.DataFrame,
    decision_date: pd.Timestamp | str,
    stock_codes: Sequence[object],
    model_scores: Sequence[float],
    lambdas: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    _validate_lambdas(lambdas)
    scores = pd.DataFrame(
        {
            "stock_id": [
                str(value).replace(".0", "").zfill(6) for value in stock_codes
            ],
            "model_score": np.asarray(model_scores, dtype=np.float64),
        }
    )
    if scores["stock_id"].duplicated().any():
        raise ValueError("Candidate-D score export contains duplicate stocks")
    factors = build_lag1_residual_factors(raw, decision_date)
    merged = scores.merge(
        factors,
        on="stock_id",
        how="left",
        validate="one_to_one",
    )
    required_ranks = ["momentum_rank", "volume_rank", "reversal_rank"]
    if merged[required_ranks].isna().any().any():
        raise ValueError("one or more model stocks lack residual-factor ranks")
    merged["d_rank_score"] = cross_section_rank(merged["model_score"])
    merged["alpha_score"] = (
        merged["d_rank_score"]
        + float(lambdas["lambda_momentum"]) * merged["momentum_rank"]
        + float(lambdas["lambda_volume"]) * merged["volume_rank"]
        - float(lambdas["lambda_reversal"]) * merged["reversal_rank"]
    )
    selected = (
        merged.sort_values(
            ["alpha_score", "model_score", "stock_id"],
            ascending=[False, False, True],
        )
        .head(5)
        .copy()
    )
    if len(selected) != 5:
        raise ValueError("residual rerank requires five selected stocks")
    result = selected[["stock_id"]].copy()
    result["weight"] = 0.20
    diagnostics = {
        "full_score_count": int(len(scores)),
        "factor_asof_date": pd.Timestamp(
            factors["factor_asof_date"].iloc[0]
        ).strftime("%Y-%m-%d"),
        "factor_complete_count": int(factors["factor_complete"].sum()),
        "factor_universe_count": int(len(factors)),
        "selected_incomplete_factor_count": int(
            (~selected["factor_complete"].fillna(False)).sum()
        ),
        "hhi": float(np.square(result["weight"]).sum()),
    }
    return result, diagnostics


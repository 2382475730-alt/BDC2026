from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from featurework import DATE_COL, STOCK_COL, OPEN_COL, CLOSE_COL


BUY_COMMISSION = 0.0005
SELL_COMMISSION = 0.0005
STAMP_TAX = 0.0005


def _project_capped_simplex(values: np.ndarray, cap: float, total: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) * cap < total - 1e-12:
        raise ValueError("cap cannot support requested invested weight")
    low, high = float(values.min() - cap), float(values.max())
    for _ in range(100):
        mid = (low + high) / 2.0
        projected = np.clip(values - mid, 0.0, cap)
        if projected.sum() > total:
            low = mid
        else:
            high = mid
    result = np.clip(values - high, 0.0, cap)
    if result.sum() <= 0:
        result[:] = total / len(result)
    else:
        result *= total / result.sum()
    return result


def _bounded_softmax(scores: np.ndarray, q: float, cap_fraction: float, temperature: float) -> np.ndarray:
    centered = scores / max(float(temperature), 1e-6)
    centered -= centered.max()
    raw = np.exp(centered)
    raw /= raw.sum()
    invested_fraction = _project_capped_simplex(raw, cap=cap_fraction, total=1.0)
    return q * invested_fraction


def portfolio_weights(
    scores: np.ndarray,
    policy: str,
    covariance: np.ndarray | None = None,
    previous: np.ndarray | None = None,
) -> np.ndarray:
    """Frozen-capacity E_R1 P0-P3 policies; all are long-only and unlevered."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 2:
        raise ValueError("portfolio requires at least two scores")
    result = np.zeros_like(scores)
    if policy == "P0_TOP2_EQUAL":
        chosen = np.argsort(scores)[-2:]
        result[chosen] = 0.5
        return result
    k = 4 if policy in {"P1_TOP4_BOUNDED_SOFTMAX", "P3_CONFIDENCE_EXPOSURE_TOP4"} else 5
    chosen = np.argsort(scores)[-min(k, len(scores)):]
    selected_scores = scores[chosen]
    if policy == "P1_TOP4_BOUNDED_SOFTMAX":
        result[chosen] = _bounded_softmax(selected_scores, q=1.0, cap_fraction=0.35, temperature=0.50)
    elif policy == "P2_SHRUNK_COV_OPT_TOP5":
        cov = np.eye(len(scores), dtype=np.float64) if covariance is None else np.asarray(covariance, dtype=np.float64)
        sub_cov = cov[np.ix_(chosen, chosen)]
        diagonal = np.diag(np.diag(sub_cov))
        shrunk = 0.5 * sub_cov + 0.5 * diagonal
        alpha = (selected_scores - selected_scores.mean()) / (selected_scores.std() + 1e-8)
        weights = np.full(len(chosen), 1.0 / len(chosen))
        prior = np.zeros(len(chosen)) if previous is None else np.asarray(previous, dtype=np.float64)[chosen]
        for _ in range(80):
            gradient = 0.20 * (shrunk @ weights) - 0.03 * alpha + 0.01 * np.sign(weights - prior)
            weights = _project_capped_simplex(weights - 0.10 * gradient, cap=0.35, total=1.0)
        result[chosen] = weights
    elif policy == "P3_CONFIDENCE_EXPOSURE_TOP4":
        ordered = np.sort(selected_scores)
        gap = float(ordered[-1] - ordered[-2])
        dispersion = float(np.std(scores)) + 1e-8
        confidence = gap / dispersion
        q = float(np.clip(0.50 + 0.25 * confidence, 0.50, 1.00))
        result[chosen] = _bounded_softmax(selected_scores, q=q, cap_fraction=0.35, temperature=0.75)
    else:
        raise ValueError(f"unknown portfolio policy: {policy}")
    assert_portfolio_constraints(result, main_strategy=True)
    return result


def assert_portfolio_constraints(weights: np.ndarray, main_strategy: bool) -> bool:
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < -1e-12).any() or weights.sum() > 1.0 + 1e-10:
        raise AssertionError("long-only/no-leverage constraint failed")
    selected = weights[weights > 1e-12]
    if main_strategy:
        if not 3 <= len(selected) <= 5 or not 0.50 - 1e-10 <= weights.sum() <= 1.0 + 1e-10:
            raise AssertionError("main-strategy stock-count or q constraint failed")
        invested = selected / selected.sum()
        if invested.max() > 0.35 + 1e-8 or float(np.square(invested).sum()) > 0.34 + 1e-8:
            raise AssertionError("single-name cap or invested HHI failed")
    return True


def attach_policy_weights(predictions: pd.DataFrame, policy: str, market: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = predictions.copy()
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.normalize()
    frame["portfolio_weight"] = 0.0
    previous_by_stock: dict[str, float] = {}
    return_panel: pd.DataFrame | None = None
    if policy == "P2_SHRUNK_COV_OPT_TOP5" and market is not None:
        history = market.copy()
        history[DATE_COL] = pd.to_datetime(history[DATE_COL]).dt.normalize()
        history[STOCK_COL] = history[STOCK_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        history = history.sort_values([STOCK_COL, DATE_COL])
        history["ret"] = history.groupby(STOCK_COL, sort=False)[CLOSE_COL].pct_change(fill_method=None)
        return_panel = history.pivot(index=DATE_COL, columns=STOCK_COL, values="ret").sort_index()
    for date, day in frame.groupby("sample_end_date", sort=True):
        scores = day["score"].to_numpy(float)
        covariance = None
        if return_panel is not None:
            panel = return_panel.loc[return_panel.index <= date].tail(60)
            codes = day["stock_id"].astype(str).tolist()
            covariance = panel.reindex(columns=codes).fillna(0.0).cov().to_numpy(float)
        previous = np.asarray([previous_by_stock.get(str(code), 0.0) for code in day["stock_id"]], dtype=np.float64)
        weights = portfolio_weights(scores, policy=policy, covariance=covariance, previous=previous)
        frame.loc[day.index, "portfolio_weight"] = weights
        previous_by_stock = {str(code): float(weight) for code, weight in zip(day["stock_id"], weights) if weight > 1e-12}
    return frame


def _week_groups(dates: Iterable[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for date in dates:
        iso = pd.Timestamp(date).isocalendar()
        groups.setdefault((int(iso.year), int(iso.week)), []).append(pd.Timestamp(date).normalize())
    return [sorted(groups[key]) for key in sorted(groups)]


def run_weekly_backtest_exact_cost(
    predictions: pd.DataFrame,
    market: pd.DataFrame | str | Path,
    validation_start: str,
    validation_end: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    if not isinstance(market, pd.DataFrame):
        market = pd.read_csv(market, dtype={STOCK_COL: str})
    market = market.copy()
    market[DATE_COL] = pd.to_datetime(market[DATE_COL]).dt.normalize()
    market[STOCK_COL] = market[STOCK_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    dates = sorted(market[DATE_COL].dropna().unique())
    date_index = {pd.Timestamp(value).normalize(): index for index, value in enumerate(dates)}
    start, end = pd.Timestamp(validation_start).normalize(), pd.Timestamp(validation_end).normalize()
    scored_dates = [pd.Timestamp(value).normalize() for value in dates if start <= value <= end]
    opens = market.set_index([STOCK_COL, DATE_COL])[OPEN_COL].astype(float)
    pred = predictions.copy()
    pred["sample_end_date"] = pd.to_datetime(pred["sample_end_date"]).dt.normalize()
    pred["stock_id"] = pred["stock_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    pred_by_date = {date: day for date, day in pred.groupby("sample_end_date", sort=True)}
    rows: list[dict[str, object]] = []
    previous: dict[str, float] = {}
    for week in _week_groups(scored_dates):
        buy_date, sell_date = week[0], week[-1]
        decision = pd.Timestamp(dates[date_index[buy_date] - 1]).normalize()
        if decision not in pred_by_date:
            raise AssertionError(f"missing decision prediction: {decision:%Y-%m-%d}")
        day = pred_by_date[decision].copy()
        if "portfolio_weight" not in day:
            raise AssertionError("portfolio_weight is required")
        day = day[pd.to_numeric(day["portfolio_weight"], errors="coerce").fillna(0.0) > 1e-12]
        weights = day["portfolio_weight"].to_numpy(float)
        stocks = day["stock_id"].astype(str).tolist()
        stock_returns: list[float] = []
        for stock in stocks:
            buy_open = float(opens.get((stock, buy_date), np.nan))
            sell_open = float(opens.get((stock, sell_date), np.nan))
            if not np.isfinite(buy_open) or not np.isfinite(sell_open) or buy_open <= 0:
                raise AssertionError(f"missing official open for {stock} in week {buy_date:%Y-%m-%d}")
            stock_returns.append(sell_open / buy_open - 1.0)
        stock_returns_array = np.asarray(stock_returns, dtype=np.float64)
        gross = float(np.dot(weights, stock_returns_array))
        buy_notional = float(weights.sum())
        sell_notional = float(np.dot(weights, 1.0 + stock_returns_array))
        cost = BUY_COMMISSION * buy_notional + (SELL_COMMISSION + STAMP_TAX) * sell_notional
        net = gross - cost
        current = dict(zip(stocks, weights))
        all_codes = set(previous) | set(current)
        composition_turnover = 0.5 * sum(abs(current.get(code, 0.0) - previous.get(code, 0.0)) for code in all_codes)
        invested = weights / weights.sum()
        rows.append({
            "week_start": buy_date.strftime("%Y-%m-%d"), "buy_date": buy_date.strftime("%Y-%m-%d"),
            "sell_date": sell_date.strftime("%Y-%m-%d"), "decision_date": decision.strftime("%Y-%m-%d"),
            "weekly_return_gross": gross, "weekly_return_net": net, "cost": cost,
            "buy_notional": buy_notional, "sell_notional": sell_notional,
            "weight_sum": buy_notional, "cash_ratio": 1.0 - buy_notional,
            "selected_count": len(stocks), "selected_stocks": ",".join(stocks),
            "invested_hhi": float(np.square(invested).sum()), "max_invested_name_weight": float(invested.max()),
            "composition_turnover": float(composition_turnover),
        })
        previous = current
    weekly = pd.DataFrame(rows)
    net = weekly["weekly_return_net"].to_numpy(float)
    gross = weekly["weekly_return_gross"].to_numpy(float)
    nav = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    peak = np.maximum.accumulate(nav)
    drawdown = nav / peak - 1.0
    tail_count = max(1, int(math.ceil(0.05 * len(net))))
    metrics: dict[str, float | int | str] = {
        "week_count": int(len(weekly)), "mean_weekly_return_gross": float(gross.mean()),
        "mean_weekly_return_net": float(net.mean()), "median_weekly_return_net": float(np.median(net)),
        "cumulative_return_net": float(np.prod(1.0 + net) - 1.0), "positive_week_rate": float(np.mean(net > 0)),
        "max_drawdown_abs": float(abs(drawdown.min())),
        "CVaR_95_loss_abs": float(max(0.0, np.sort(-net)[-tail_count:].mean())),
        "cost_drag": float((gross - net).sum()), "average_cash_ratio": float(weekly["cash_ratio"].mean()),
        "average_invested_hhi": float(weekly["invested_hhi"].mean()),
        "average_composition_turnover": float(weekly["composition_turnover"].mean()),
        "date_start": str(weekly["buy_date"].min()), "date_end": str(weekly["sell_date"].max()),
    }
    return weekly, metrics

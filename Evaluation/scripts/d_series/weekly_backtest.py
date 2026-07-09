from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import normalize_stock_code, read_market_csv, safe_div, sorted_trading_dates, write_json


def _open_lookup(market: pd.DataFrame, stock_col: str, date_col: str, open_col: str) -> dict[tuple[str, pd.Timestamp], float]:
    lookup: dict[tuple[str, pd.Timestamp], float] = {}
    subset = market[[stock_col, date_col, open_col]].copy()
    subset[stock_col] = subset[stock_col].map(normalize_stock_code)
    subset[open_col] = pd.to_numeric(subset[open_col], errors="coerce")
    for row in subset.itertuples(index=False):
        lookup[(str(row[0]), pd.Timestamp(row[1]).normalize())] = float(row[2]) if not pd.isna(row[2]) else math.nan
    return lookup


def _week_groups(dates: list[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for date in dates:
        iso = date.isocalendar()
        groups.setdefault((int(iso.year), int(iso.week)), []).append(date)
    return [groups[key] for key in sorted(groups)]


def _portfolio_return(
    open_by_key: dict[tuple[str, pd.Timestamp], float],
    stocks: list[str],
    weights: np.ndarray,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
) -> tuple[float, list[dict[str, Any]]]:
    gross = 0.0
    details: list[dict[str, Any]] = []
    for stock, weight in zip(stocks, weights):
        buy_open = open_by_key.get((stock, buy_date), math.nan)
        sell_open = open_by_key.get((stock, sell_date), math.nan)
        stock_return = (sell_open / buy_open - 1.0) if buy_open and np.isfinite(buy_open) and np.isfinite(sell_open) else 0.0
        weighted = float(weight) * stock_return
        gross += weighted
        details.append(
            {
                "stock_id": stock,
                "weight": float(weight),
                "buy_open": buy_open,
                "sell_open": sell_open,
                "stock_return": float(stock_return),
                "weighted_return": float(weighted),
            }
        )
    return float(gross), details


def _summarize(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "week_count": 0,
            "mean_weekly_return_net": 0.0,
            "median_weekly_return_net": 0.0,
            "cumulative_return_net": 0.0,
            "max_drawdown_abs": 0.0,
            "worst_week": 0.0,
            "CVaR_95_loss_abs": 0.0,
            "cost_drag": 0.0,
        }
    net = rows["weekly_return_net"].astype(float).to_numpy()
    gross = rows["weekly_return_gross"].astype(float).to_numpy()
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    losses = -net
    worst_count = max(1, int(math.ceil(0.05 * len(net))))
    cvar_loss = float(np.mean(np.sort(losses)[-worst_count:]))
    worst4 = float(min((np.prod(1.0 + net[i : i + 4]) - 1.0 for i in range(max(1, len(net) - 3))), default=net.min()))
    downside = net[net < 0.0]
    cost_drag = float(np.sum(gross - net))
    mean_net = float(np.mean(net))
    overlap = rows["jaccard_prev_week"].dropna().astype(float)
    mean_overlap = float(overlap.mean()) if len(overlap) else 0.0
    return {
        "week_count": int(len(rows)),
        "date_start": str(rows["buy_date"].min()),
        "date_end": str(rows["sell_date"].max()),
        "mean_weekly_return_net": mean_net,
        "median_weekly_return_net": float(np.median(net)),
        "cumulative_return_net": float(np.prod(1.0 + net) - 1.0),
        "geometric_mean_weekly_return_net": float(np.prod(1.0 + net) ** (1.0 / len(net)) - 1.0),
        "positive_week_rate": float(np.mean(net > 0.0)),
        "worst_week": float(np.min(net)),
        "worst_4_week_cumulative_return": worst4,
        "max_drawdown_abs": float(abs(np.min(drawdown))),
        "max_drawdown_duration": int(_max_drawdown_duration(drawdown)),
        "CVaR_95_loss_abs": max(0.0, cvar_loss),
        "downside_deviation": float(np.std(downside)) if len(downside) else 0.0,
        "Sortino": safe_div(mean_net, float(np.std(downside)) if len(downside) else 0.0),
        "Calmar": safe_div(float(np.prod(1.0 + net) - 1.0), float(abs(np.min(drawdown)))),
        "average_weight_sum": float(rows["weight_sum"].astype(float).mean()),
        "cash_ratio": float((1.0 - rows["weight_sum"].astype(float)).mean()),
        "buy_turnover": float(rows["buy_turnover"].astype(float).mean()),
        "sell_turnover": float(rows["sell_turnover"].astype(float).mean()),
        "cost_drag": cost_drag,
        "number_of_selected_stocks": float(rows["selected_count"].astype(float).mean()),
        "top5_jaccard_between_weeks": mean_overlap,
        "average_holding_overlap_rate": mean_overlap,
        "topk_turnover": float(1.0 - mean_overlap),
        "up_market_return": _state_mean(rows, "up"),
        "down_market_return": _state_mean(rows, "down"),
        "high_vol_return": _state_mean(rows, "high_vol"),
        "low_vol_return": _state_mean(rows, "low_vol"),
        "high_breadth_return": _state_mean(rows, "high_breadth"),
        "low_breadth_return": _state_mean(rows, "low_breadth"),
        "decision_down_regime_return": _decision_state_mean(rows, "down"),
        "decision_up_regime_return": _decision_state_mean(rows, "up"),
        "decision_high_vol_return": _decision_state_mean(rows, "high_vol"),
        "decision_low_vol_return": _decision_state_mean(rows, "low_vol"),
        "decision_high_breadth_return": _decision_state_mean(rows, "high_breadth"),
        "decision_low_breadth_return": _decision_state_mean(rows, "low_breadth"),
        "mean_regime_risk": float(rows["regime_risk"].astype(float).mean()) if "regime_risk" in rows else 0.0,
        "skipped_missing_decision_prediction": int(rows["skipped_missing_decision_prediction"].sum()),
        "boundary_week_count": int(rows["boundary_week"].sum()),
    }


def _max_drawdown_duration(drawdown: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in drawdown:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _state_mean(rows: pd.DataFrame, token: str) -> float:
    subset = rows[rows["market_state"].astype(str).str.contains(token, regex=False)]
    return float(subset["weekly_return_net"].astype(float).mean()) if not subset.empty else 0.0


def _decision_state_mean(rows: pd.DataFrame, token: str) -> float:
    if "decision_regime" not in rows.columns:
        return 0.0
    subset = rows[rows["decision_regime"].astype(str).str.contains(token, regex=False)]
    return float(subset["weekly_return_net"].astype(float).mean()) if not subset.empty else 0.0


def run_weekly_backtest(
    predictions: pd.DataFrame,
    market_path: str | Path,
    *,
    validation_start: str,
    validation_end: str,
    top_k: int = 5,
    buy_fee: float = 0.0005,
    sell_fee: float = 0.0005,
    stamp_tax: float = 0.0005,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    market, columns = read_market_csv(market_path)
    dates = sorted_trading_dates(market, columns.date)
    date_to_index = {date: idx for idx, date in enumerate(dates)}
    start = pd.Timestamp(validation_start).normalize()
    end = pd.Timestamp(validation_end).normalize()
    validation_dates = [date for date in dates if start <= date <= end]
    open_by_key = _open_lookup(market, columns.stock, columns.date, columns.open)

    pred = predictions.copy()
    pred["sample_end_date"] = pd.to_datetime(pred["sample_end_date"]).dt.normalize()
    pred["stock_id"] = pred["stock_id"].map(normalize_stock_code)
    pred_by_date = {date: day for date, day in pred.groupby("sample_end_date", sort=True)}

    rows: list[dict[str, Any]] = []
    previous_stocks: set[str] | None = None
    for week_dates in _week_groups(validation_dates):
        buy_date = week_dates[0]
        sell_date = week_dates[-1]
        decision_index = date_to_index.get(buy_date, 0) - 1
        decision_date = dates[decision_index] if decision_index >= 0 else None
        missing_decision = decision_date not in pred_by_date if decision_date is not None else True
        if missing_decision:
            rows.append(
                {
                    "week_start": buy_date.strftime("%Y-%m-%d"),
                    "buy_date": buy_date.strftime("%Y-%m-%d"),
                    "sell_date": sell_date.strftime("%Y-%m-%d"),
                    "decision_date": decision_date.strftime("%Y-%m-%d") if decision_date is not None else "",
                    "weekly_return_gross": 0.0,
                    "weekly_return_net": 0.0,
                    "cost": 0.0,
                    "weight_sum": 0.0,
                    "selected_count": 0,
                    "selected_stocks": "",
                    "boundary_week": buy_date == validation_dates[0] or sell_date == validation_dates[-1],
                    "skipped_missing_decision_prediction": 1,
                    "buy_turnover": 0.0,
                    "sell_turnover": 0.0,
                    "jaccard_prev_week": np.nan,
                    "market_state": "unknown",
                    "decision_regime": "unknown",
                    "regime_risk": 0.0,
                    "cash_ratio": 1.0,
                }
            )
            continue
        day = pred_by_date[decision_date].replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
        if "portfolio_weight" in day.columns and day["portfolio_weight"].notna().any():
            day = day.copy()
            day["portfolio_weight"] = pd.to_numeric(day["portfolio_weight"], errors="coerce").fillna(0.0)
            day = day[day["portfolio_weight"] > 1e-12].sort_values(["portfolio_weight", "score"], ascending=False)
            stocks = day["stock_id"].astype(str).tolist()
            weights = day["portfolio_weight"].to_numpy(dtype=np.float64)
        else:
            day = day.sort_values("score", ascending=False).head(top_k)
            stocks = day["stock_id"].astype(str).tolist()
            weights = np.full(len(stocks), 1.0 / len(stocks), dtype=np.float64) if len(stocks) else np.asarray([], dtype=np.float64)
        selected_count = len(stocks)
        gross, _ = _portfolio_return(open_by_key, stocks, weights, buy_date, sell_date)
        invested = float(np.sum(weights))
        cost = (buy_fee + sell_fee + stamp_tax) * invested
        current_stocks = set(stocks)
        jaccard = (
            len(current_stocks & previous_stocks) / max(1, len(current_stocks | previous_stocks))
            if previous_stocks is not None
            else np.nan
        )
        previous_stocks = current_stocks
        market_state = _classify_market_state(market, columns, open_by_key, buy_date, sell_date)
        regime_risk = float(day["regime_risk"].astype(float).mean()) if "regime_risk" in day.columns and not day.empty else 0.0
        decision_regime = str(day["deterministic_regime"].iloc[0]) if "deterministic_regime" in day.columns and not day.empty else "unknown"
        turnover = 0.0 if previous_stocks is None or np.isnan(jaccard) else 1.0 - float(jaccard)
        rows.append(
            {
                "week_start": buy_date.strftime("%Y-%m-%d"),
                "buy_date": buy_date.strftime("%Y-%m-%d"),
                "sell_date": sell_date.strftime("%Y-%m-%d"),
                "decision_date": decision_date.strftime("%Y-%m-%d"),
                "weekly_return_gross": gross,
                "weekly_return_net": gross - cost,
                "cost": cost,
                "weight_sum": invested,
                "cash_ratio": float(max(0.0, 1.0 - invested)),
                "selected_count": selected_count,
                "selected_stocks": ",".join(stocks),
                "boundary_week": buy_date == validation_dates[0] or sell_date == validation_dates[-1],
                "skipped_missing_decision_prediction": 0,
                "buy_turnover": turnover,
                "sell_turnover": turnover,
                "jaccard_prev_week": jaccard,
                "market_state": market_state,
                "decision_regime": decision_regime,
                "regime_risk": regime_risk,
            }
        )
    weekly = pd.DataFrame(rows)
    scored = weekly[weekly["skipped_missing_decision_prediction"] == 0].copy()
    summary = _summarize(scored)
    summary["planned_week_count"] = int(len(weekly))
    summary["skipped_missing_decision_prediction"] = int(weekly["skipped_missing_decision_prediction"].sum()) if not weekly.empty else 0
    summary["boundary_week_count"] = int(weekly["boundary_week"].sum()) if not weekly.empty else 0
    return weekly, summary


def _classify_market_state(
    market: pd.DataFrame,
    columns: Any,
    open_by_key: dict[tuple[str, pd.Timestamp], float],
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
) -> str:
    stocks = market[market[columns.date] == buy_date][columns.stock].map(normalize_stock_code).unique().tolist()
    returns = []
    for stock in stocks:
        buy_open = open_by_key.get((stock, buy_date), math.nan)
        sell_open = open_by_key.get((stock, sell_date), math.nan)
        if buy_open and np.isfinite(buy_open) and np.isfinite(sell_open):
            returns.append(sell_open / buy_open - 1.0)
    if not returns:
        return "unknown"
    values = np.asarray(returns, dtype=np.float64)
    direction = "up" if values.mean() > 0.005 else "down" if values.mean() < -0.005 else "sideways"
    vol = "high_vol" if values.std() > 0.04 else "low_vol"
    breadth = "high_breadth" if float(np.mean(values > 0.0)) >= 0.5 else "low_breadth"
    return f"{direction}_{vol}_{breadth}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-series non-overlapping weekly validation/backtest.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--market-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions, dtype={"stock_id": str})
    weekly, summary = run_weekly_backtest(
        predictions,
        args.market_data,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        top_k=args.top_k,
    )
    weekly.to_csv(output_dir / "weekly_backtest.csv", index=False, encoding="utf-8")
    write_json(output_dir / "weekly_backtest_result.json", summary)


if __name__ == "__main__":
    main()

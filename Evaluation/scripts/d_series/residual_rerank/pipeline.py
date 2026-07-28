from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from core import (
    GRID_LAMBDAS,
    combine_alpha_scores,
    select_portfolios,
)
from workflow import _ensure_not_exposed, verify_prediction_lock


GRID_KEYS = [
    "lambda_momentum",
    "lambda_volume",
    "lambda_reversal",
]
PORTFOLIO_KEYS = [
    "seed",
    "fold_id",
    "phase",
    "decision_date",
    *GRID_KEYS,
    "policy_id",
]


def five_day_decision_dates(
    values: Iterable[pd.Timestamp | str],
) -> list[pd.Timestamp]:
    dates = sorted({pd.Timestamp(value).normalize() for value in values})
    return dates[::5]


def build_grid_holdings(
    scores: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "decision_date",
        "stock_id",
        "score",
        "seed",
        "fold_id",
        "phase",
    }
    if missing := required.difference(scores.columns):
        raise KeyError(f"missing grid score columns: {sorted(missing)}")
    score_frame = scores.copy()
    score_frame["decision_date"] = pd.to_datetime(
        score_frame["decision_date"]
    ).dt.normalize()
    factor_frame = factors.copy()
    factor_frame["decision_date"] = pd.to_datetime(
        factor_frame["decision_date"]
    ).dt.normalize()
    rows: list[pd.DataFrame] = []
    for (seed, fold_id, phase), group in score_frame.groupby(
        ["seed", "fold_id", "phase"],
        sort=True,
    ):
        decisions = five_day_decision_dates(group["decision_date"])
        base = group.loc[group["decision_date"].isin(decisions)].copy()
        relevant_factors = factor_frame.loc[
            factor_frame["decision_date"].isin(decisions)
        ].copy()
        for lambdas in GRID_LAMBDAS:
            combined = combine_alpha_scores(base, relevant_factors, lambdas)
            for policy_id in (
                "TOP2_EQUAL",
                "TOP5_EQUAL",
                "TOP5_FIXED_30_25_20_15_10",
            ):
                selected = select_portfolios(combined, policy_id)
                selected["seed"] = int(seed)
                selected["fold_id"] = str(fold_id)
                selected["phase"] = str(phase)
                rows.append(selected)
    output = pd.concat(rows, ignore_index=True)
    columns = [
        "seed",
        "fold_id",
        "phase",
        "decision_date",
        *GRID_KEYS,
        "policy_id",
        "portfolio_rank",
        "stock_id",
        "weight",
        "score",
        "d_rank_score",
        "alpha_score",
    ]
    return output[columns].sort_values(
        PORTFOLIO_KEYS + ["portfolio_rank", "stock_id"]
    ).reset_index(drop=True)


def build_forward_outcomes(
    raw: pd.DataFrame,
    decision_dates: Iterable[pd.Timestamp | str],
    columns: Mapping[str, str],
) -> pd.DataFrame:
    required_keys = {"stock", "date", "open"}
    if missing := required_keys.difference(columns):
        raise KeyError(f"missing outcome column mapping: {sorted(missing)}")
    required_columns = {columns[key] for key in required_keys}
    if missing := required_columns.difference(raw.columns):
        raise KeyError(f"missing outcome source columns: {sorted(missing)}")
    frame = raw[list(required_columns)].copy().rename(
        columns={value: key for key, value in columns.items() if key in required_keys}
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["stock"] = (
        frame["stock"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    wanted = {pd.Timestamp(value).normalize() for value in decision_dates}
    rows = []
    for stock, group in frame.groupby("stock", sort=True):
        group = (
            group.drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        position_by_date = {
            pd.Timestamp(value).normalize(): index
            for index, value in enumerate(group["date"])
        }
        for decision_date in sorted(wanted):
            position = position_by_date.get(decision_date)
            if position is None or position + 5 >= len(group):
                continue
            buy_open = float(group.iloc[position + 1]["open"])
            sell_open = float(group.iloc[position + 5]["open"])
            if (
                not np.isfinite(buy_open)
                or not np.isfinite(sell_open)
                or buy_open <= 0.0
            ):
                continue
            rows.append(
                {
                    "decision_date": decision_date,
                    "stock_id": stock,
                    "buy_date": group.iloc[position + 1]["date"],
                    "sell_date": group.iloc[position + 5]["date"],
                    "buy_open": buy_open,
                    "sell_open": sell_open,
                    "forward_return": sell_open / buy_open - 1.0,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["decision_date", "stock_id"]
    ).reset_index(drop=True)


def score_grid_locked(
    holdings_path: str | Path,
    outcomes: pd.DataFrame,
    lock_path: str | Path,
    cost_bps_per_side: float,
) -> pd.DataFrame:
    lock = verify_prediction_lock(lock_path)
    holdings_file = Path(holdings_path).resolve()
    locked_paths = {str(Path(item["path"]).resolve()) for item in lock["files"]}
    if str(holdings_file) not in locked_paths:
        raise ValueError("grid holdings are not covered by the prediction lock")
    holdings = pd.read_csv(holdings_file, dtype={"stock_id": str})
    required = {*PORTFOLIO_KEYS, "stock_id", "weight"}
    if missing := required.difference(holdings.columns):
        raise KeyError(f"missing grid holding columns: {sorted(missing)}")
    holdings["decision_date"] = pd.to_datetime(
        holdings["decision_date"]
    ).dt.normalize()
    _ensure_not_exposed(holdings["decision_date"].unique())
    holdings["stock_id"] = holdings["stock_id"].astype(str).str.zfill(6)
    labels = outcomes.copy()
    labels["decision_date"] = pd.to_datetime(labels["decision_date"]).dt.normalize()
    labels["stock_id"] = labels["stock_id"].astype(str).str.zfill(6)
    labels = labels[["decision_date", "stock_id", "forward_return"]]
    if labels.duplicated(["decision_date", "stock_id"]).any():
        raise ValueError("outcomes contain duplicate decision/stock rows")
    merged = holdings.merge(
        labels,
        on=["decision_date", "stock_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["forward_return"].isna().any():
        raise ValueError("locked grid holdings contain missing outcomes")

    cost = 2.0 * float(cost_bps_per_side) / 10_000.0
    rows: list[dict[str, object]] = []
    for keys, group in merged.groupby(PORTFOLIO_KEYS, sort=True):
        weights = pd.to_numeric(group["weight"], errors="raise").to_numpy(
            dtype=np.float64
        )
        if len(weights) > 5 or len(weights) < 1:
            raise ValueError("grid portfolio must contain between one and five stocks")
        if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("grid portfolio must be fully invested")
        gross = float(
            np.sum(
                weights
                * pd.to_numeric(
                    group["forward_return"],
                    errors="raise",
                ).to_numpy(dtype=np.float64)
            )
        )
        row = dict(zip(PORTFOLIO_KEYS, keys))
        row.update(
            {
                "gross_return": gross,
                "cost_bps_per_side": float(cost_bps_per_side),
                "cost": cost,
                "net_return": gross - cost,
                "hhi": float(np.square(weights).sum()),
                "stock_count": int(len(weights)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(PORTFOLIO_KEYS).reset_index(drop=True)


def _metrics(values: pd.Series) -> dict[str, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(
        dtype=np.float64
    )
    if not len(array):
        raise ValueError("cannot summarize an empty calibration series")
    tail_count = max(1, int(math.ceil(len(array) * 0.20)))
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "q10": float(np.quantile(array, 0.10)),
        "cvar20": float(np.sort(array)[:tail_count].mean()),
    }


def calibration_metrics(scored_grid: pd.DataFrame) -> pd.DataFrame:
    primary = scored_grid.loc[
        scored_grid["policy_id"] == "TOP5_EQUAL"
    ].copy()
    zero = primary.loc[
        (primary["lambda_momentum"].abs() <= 1e-12)
        & (primary["lambda_volume"].abs() <= 1e-12)
        & (primary["lambda_reversal"].abs() <= 1e-12)
    ][["seed", "fold_id", "phase", "decision_date", "net_return"]].rename(
        columns={"net_return": "zero_net_return"}
    )
    rows = []
    for lambdas, group in primary.groupby(GRID_KEYS, sort=True):
        paired = group.merge(
            zero,
            on=["seed", "fold_id", "phase", "decision_date"],
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(group):
            raise ValueError("calibration grid is not exactly paired to zero control")
        excess = paired["net_return"] - paired["zero_net_return"]
        summary = _metrics(excess)
        rows.append(
            {
                **dict(zip(GRID_KEYS, lambdas)),
                "decision_count": int(len(excess)),
                "mean_excess": summary["mean"],
                "median_excess": summary["median"],
                "std_excess": summary["std"],
                "q10_excess": summary["q10"],
                "cvar20_excess": summary["cvar20"],
                "stability_score": (
                    0.35 * summary["mean"]
                    + 0.25 * summary["median"]
                    + 0.20 * summary["cvar20"]
                    + 0.10 * summary["q10"]
                    - 0.10 * summary["std"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(GRID_KEYS).reset_index(drop=True)

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SEALED_RANGES = (
    ("2025-10-01", "2025-12-31"),
    ("2026-04-03", "2026-07-03"),
)
ALLOWED_RESIDUAL_LAMBDAS = (0.0, 0.05, 0.10)


@dataclass(frozen=True)
class DevelopmentFold:
    fold_id: str
    train_end: str
    inner_start: str
    inner_end: str
    outer_train_end: str
    outer_start: str
    outer_end: str


DEVELOPMENT_FOLDS = (
    DevelopmentFold(
        "2025Q1",
        "2024-09-20",
        "2024-10-01",
        "2024-12-24",
        "2024-12-24",
        "2025-01-03",
        "2025-03-31",
    ),
    DevelopmentFold(
        "2025Q2",
        "2024-12-24",
        "2025-01-03",
        "2025-03-24",
        "2025-03-24",
        "2025-04-01",
        "2025-06-30",
    ),
    DevelopmentFold(
        "2025Q3",
        "2025-03-24",
        "2025-04-01",
        "2025-06-23",
        "2025-06-23",
        "2025-07-01",
        "2025-09-30",
    ),
    DevelopmentFold(
        "2026Q1",
        "2025-06-23",
        "2025-07-01",
        "2025-09-23",
        "2025-09-23",
        "2026-01-05",
        "2026-03-27",
    ),
)


def _overlaps(
    start: pd.Timestamp,
    end: pd.Timestamp,
    blocked_start: pd.Timestamp,
    blocked_end: pd.Timestamp,
) -> bool:
    return start <= blocked_end and end >= blocked_start


def validate_fold_plan(folds: Sequence[DevelopmentFold]) -> None:
    if len(folds) != 4:
        raise ValueError("the preregistered development plan must contain four folds")
    for fold in folds:
        train_end = pd.Timestamp(fold.train_end)
        inner_start = pd.Timestamp(fold.inner_start)
        inner_end = pd.Timestamp(fold.inner_end)
        outer_train_end = pd.Timestamp(fold.outer_train_end)
        outer_start = pd.Timestamp(fold.outer_start)
        outer_end = pd.Timestamp(fold.outer_end)
        if not train_end < inner_start <= inner_end < outer_start <= outer_end:
            raise ValueError(f"invalid chronological order in {fold.fold_id}")
        if not outer_train_end < outer_start:
            raise ValueError(f"outer training leaks into {fold.fold_id}")
        for blocked_start, blocked_end in SEALED_RANGES:
            blocked_left = pd.Timestamp(blocked_start)
            blocked_right = pd.Timestamp(blocked_end)
            if _overlaps(inner_start, inner_end, blocked_left, blocked_right):
                raise ValueError(f"inner window overlaps sealed labels in {fold.fold_id}")
            if _overlaps(outer_start, outer_end, blocked_left, blocked_right):
                raise ValueError(f"outer window overlaps sealed labels in {fold.fold_id}")
            if blocked_left <= outer_train_end <= blocked_right:
                raise ValueError(f"outer training ends inside sealed labels in {fold.fold_id}")


def filter_development_dates(
    dates: Iterable[pd.Timestamp | str],
    end: pd.Timestamp | str,
    start: pd.Timestamp | str | None = None,
) -> list[pd.Timestamp]:
    """Return only label dates authorized for development materialization."""

    lower = pd.Timestamp(start).normalize() if start is not None else None
    upper = pd.Timestamp(end).normalize()
    allowed: list[pd.Timestamp] = []
    for value in dates:
        current = pd.Timestamp(value).normalize()
        if current > upper or (lower is not None and current < lower):
            continue
        if any(
            pd.Timestamp(blocked_start) <= current <= pd.Timestamp(blocked_end)
            for blocked_start, blocked_end in SEALED_RANGES
        ):
            continue
        allowed.append(current)
    return sorted(set(allowed))


def should_evaluate_validation(
    epoch: int,
    selected_epochs: int,
    outer_once: bool,
) -> bool:
    return not outer_once or int(epoch) == int(selected_epochs)


def apply_portfolio_policy(
    frame: pd.DataFrame,
    k: int,
    weighting: str = "equal",
) -> pd.DataFrame:
    required = {"stock_id", "score", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing policy columns: {sorted(missing)}")
    selected = (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score", "label"])
        .sort_values(["score", "stock_id"], ascending=[False, True])
        .head(max(0, int(k)))
        .copy()
    )
    if selected.empty:
        selected["weight"] = pd.Series(dtype=float)
        return selected
    if weighting == "equal":
        raw = np.ones(len(selected), dtype=np.float64)
    elif weighting == "rank":
        raw = np.arange(len(selected), 0, -1, dtype=np.float64)
    elif weighting == "fixed_30_25_20_15_10":
        if len(selected) != 5 or int(k) != 5:
            raise ValueError("fixed_30_25_20_15_10 requires five selected stocks")
        raw = np.asarray([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float64)
    else:
        raise ValueError(f"unsupported weighting: {weighting}")
    selected["weight"] = raw / raw.sum()
    return selected


def five_trading_day_portfolios(
    predictions: pd.DataFrame,
    k: int = 5,
    weighting: str = "equal",
) -> pd.DataFrame:
    required = {"sample_end_date", "stock_id", "score", "label"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    working = predictions.copy()
    working["sample_end_date"] = pd.to_datetime(
        working["sample_end_date"]
    ).dt.normalize()
    decision_dates = sorted(working["sample_end_date"].dropna().unique())[::5]
    rows: list[dict[str, object]] = []
    for decision_date in decision_dates:
        day = working.loc[working["sample_end_date"] == decision_date]
        selected = apply_portfolio_policy(day, k=k, weighting=weighting)
        if selected.empty:
            continue
        rows.append(
            {
                "decision_date": pd.Timestamp(decision_date).strftime("%Y-%m-%d"),
                "portfolio_return": float(
                    (selected["weight"] * selected["label"]).sum()
                ),
                "hhi": float((selected["weight"] ** 2).sum()),
                "stock_count": int(len(selected)),
                "holdings": "|".join(selected["stock_id"].astype(str)),
                "weights": "|".join(
                    f"{value:.8f}" for value in selected["weight"].astype(float)
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_weekly_metrics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            key: 0.0
            for key in (
                "mean",
                "median",
                "hit_rate",
                "std",
                "q10",
                "cvar20",
                "worst",
                "sum_remove_best1",
                "sum_remove_best2",
                "best_positive_contribution",
            )
        }
    tail_count = max(1, int(np.ceil(array.size * 0.20)))
    ordered_desc = np.sort(array)[::-1]
    positives = array[array > 0.0]
    positive_sum = float(positives.sum())
    best_contribution = (
        float(positives.max() / positive_sum) if positive_sum > 0.0 else 1.0
    )
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "hit_rate": float(np.mean(array > 0.0)),
        "std": float(array.std()),
        "q10": float(np.quantile(array, 0.10)),
        "cvar20": float(np.sort(array)[:tail_count].mean()),
        "worst": float(array.min()),
        "sum_remove_best1": float(ordered_desc[1:].sum()),
        "sum_remove_best2": float(ordered_desc[2:].sum()),
        "best_positive_contribution": best_contribution,
    }


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: dict[str, bool]
    failed_checks: tuple[str, ...]


def quick_gate(metrics: Mapping[str, object]) -> GateResult:
    seed_means = [float(value) for value in metrics["seed_mean_excess"]]  # type: ignore[index]
    checks = {
        "positive_folds_at_least_3_of_4": int(metrics["positive_folds"]) >= 3,
        "recent_fold_positive": float(metrics["recent_fold_excess"]) > 0.0,
        "both_seed_means_positive": len(seed_means) == 2
        and all(value > 0.0 for value in seed_means),
        "strict_weekly_win_rate_at_least_55pct": float(
            metrics["strict_weekly_win_rate"]
        )
        >= 0.55,
        "excess_positive_after_best_two_removed": float(
            metrics["excess_sum_remove_best2"]
        )
        > 0.0,
        "cvar20_not_worse_than_control": float(metrics["candidate_cvar20"])
        >= float(metrics["control_cvar20"]),
        "hhi_at_most_0_30": float(metrics["hhi"]) <= 0.30,
        "best_week_at_most_20pct_positive_sum": float(
            metrics["best_positive_contribution"]
        )
        <= 0.20,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return GateResult(not failed, checks, failed)


def paired_quick_metrics(
    control: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    keys = ["seed", "fold_id", "decision_date"]
    required = {*keys, "portfolio_return", "hhi"}
    for name, frame in (("control", control), ("candidate", candidate)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing paired columns: {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} contains duplicate paired decisions")
    paired = control.merge(
        candidate,
        on=keys,
        how="inner",
        suffixes=("_control", "_candidate"),
        validate="one_to_one",
    )
    if len(paired) != len(control) or len(paired) != len(candidate):
        raise ValueError("control and candidate decisions are not exactly paired")
    paired["excess"] = (
        paired["portfolio_return_candidate"]
        - paired["portfolio_return_control"]
    )
    seed_fold = (
        paired.groupby(["seed", "fold_id"], sort=True)["excess"]
        .mean()
        .reset_index()
    )
    positive_by_seed = (
        seed_fold.assign(positive=seed_fold["excess"] > 0.0)
        .groupby("seed")["positive"]
        .sum()
    )
    latest_fold = (
        paired.assign(_date=pd.to_datetime(paired["decision_date"]))
        .groupby("fold_id")["_date"]
        .max()
        .idxmax()
    )
    seed_means = (
        paired.groupby("seed", sort=True)["excess"].mean().astype(float).tolist()
    )
    excess_metrics = compute_weekly_metrics(paired["excess"])
    candidate_metrics = compute_weekly_metrics(
        paired["portfolio_return_candidate"]
    )
    control_metrics = compute_weekly_metrics(paired["portfolio_return_control"])
    metrics: dict[str, object] = {
        "positive_folds": int(positive_by_seed.min()),
        "positive_folds_by_seed": {
            str(int(seed)): int(count)
            for seed, count in positive_by_seed.items()
        },
        "recent_fold_id": str(latest_fold),
        "recent_fold_excess": float(
            paired.loc[paired["fold_id"] == latest_fold, "excess"].mean()
        ),
        "seed_mean_excess": seed_means,
        "strict_weekly_win_rate": float(np.mean(paired["excess"] > 0.0)),
        "excess_sum_remove_best2": excess_metrics["sum_remove_best2"],
        "candidate_cvar20": candidate_metrics["cvar20"],
        "control_cvar20": control_metrics["cvar20"],
        "hhi": float(paired["hhi_candidate"].max()),
        "best_positive_contribution": candidate_metrics[
            "best_positive_contribution"
        ],
        "paired_decision_count": int(len(paired)),
    }
    return metrics, paired


def combine_residual_rank(
    base_rank: np.ndarray,
    auxiliary_rank: np.ndarray,
    residual_lambda: float,
) -> np.ndarray:
    value = float(residual_lambda)
    if value not in ALLOWED_RESIDUAL_LAMBDAS:
        raise ValueError(
            f"lambda must be one of {ALLOWED_RESIDUAL_LAMBDAS}, got {value}"
        )
    base = np.asarray(base_rank, dtype=np.float64)
    auxiliary = np.asarray(auxiliary_rank, dtype=np.float64)
    if base.shape != auxiliary.shape:
        raise ValueError("base and auxiliary ranks must have identical shapes")
    return base + value * np.clip(auxiliary, -1.0, 1.0)


validate_fold_plan(DEVELOPMENT_FOLDS)

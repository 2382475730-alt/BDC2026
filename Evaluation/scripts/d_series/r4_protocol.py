from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .common import md5_file, normalize_stock_code, read_json, safe_div, write_json
from .official_proxy_eval import evaluate_official_proxy
from .weekly_backtest import run_weekly_backtest


R4_SEEDS = [20260710, 20260711, 20260712, 20260713, 20260714]
FULL_INVESTMENT_TOLERANCE = 1e-8
WEEKLY_ROUND_TRIP_COST = 0.0005 + 0.0005 + 0.0005


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _week_groups(dates: Iterable[pd.Timestamp]) -> dict[tuple[int, int], list[pd.Timestamp]]:
    groups: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for date in dates:
        normalized = pd.Timestamp(date).normalize()
        iso = normalized.isocalendar()
        groups.setdefault((int(iso.year), int(iso.week)), []).append(normalized)
    return {key: sorted(values) for key, values in sorted(groups.items())}


def _week_payload(key: tuple[int, int], dates: Sequence[pd.Timestamp]) -> dict[str, Any]:
    return {
        "iso_year": key[0],
        "iso_week": key[1],
        "start": dates[0].strftime("%Y-%m-%d"),
        "end": dates[-1].strftime("%Y-%m-%d"),
        "trading_day_count": len(dates),
        "trading_dates": [date.strftime("%Y-%m-%d") for date in dates],
    }


def build_r4_split_manifest(
    *,
    data_path: str | Path,
    d02_split_path: str | Path,
    d02_metrics_path: str | Path,
    d02_weekly_path: str | Path,
    d02_predictions_path: str | Path,
    sealed_week_count: int = 13,
    sell_offset: int = 5,
) -> dict[str, Any]:
    data_path = Path(data_path).resolve()
    d02_split_path = Path(d02_split_path).resolve()
    d02_metrics_path = Path(d02_metrics_path).resolve()
    d02_weekly_path = Path(d02_weekly_path).resolve()
    d02_predictions_path = Path(d02_predictions_path).resolve()
    d02_split = read_json(d02_split_path, default={}) or {}
    validation = d02_split.get("validation", {})
    validation_start = pd.Timestamp(validation["sample_start"]).normalize()
    validation_end = pd.Timestamp(validation["sample_end"]).normalize()

    header = pd.read_csv(data_path, nrows=0)
    date_col = str(header.columns[1])
    date_frame = pd.read_csv(data_path, usecols=[date_col])
    trading_dates = [
        pd.Timestamp(value).normalize()
        for value in sorted(pd.to_datetime(date_frame[date_col]).dropna().unique())
    ]
    full_groups = _week_groups(trading_dates)
    validation_dates = [date for date in trading_dates if validation_start <= date <= validation_end]
    validation_groups = _week_groups(validation_dates)
    complete_groups = [
        (key, dates)
        for key, dates in validation_groups.items()
        if dates == full_groups.get(key, [])
    ]
    if len(complete_groups) <= sealed_week_count:
        raise ValueError(
            f"validation has {len(complete_groups)} complete weeks; need more than sealed count {sealed_week_count}"
        )

    holdout_groups = complete_groups[-sealed_week_count:]
    holdout_start = holdout_groups[0][1][0]
    holdout_end = holdout_groups[-1][1][-1]
    holdout_start_index = trading_dates.index(holdout_start)
    if holdout_start_index <= sell_offset:
        raise ValueError("not enough pre-holdout trading dates to apply T+5 purge")
    holdout_decision_date = trading_dates[holdout_start_index - 1]
    development_sample_end = trading_dates[holdout_start_index - sell_offset - 1]
    pre_holdout_groups = complete_groups[:-sealed_week_count]
    development_groups: list[tuple[tuple[int, int], list[pd.Timestamp]]] = []
    embargo_groups: list[tuple[tuple[int, int], list[pd.Timestamp]]] = []
    for key, dates in pre_holdout_groups:
        buy_index = trading_dates.index(dates[0])
        decision_date = trading_dates[buy_index - 1] if buy_index > 0 else dates[0]
        if decision_date <= development_sample_end:
            development_groups.append((key, dates))
        else:
            embargo_groups.append((key, dates))
    if not development_groups:
        raise ValueError("T+5 purge removed every development week")
    development_week_start = development_groups[0][1][0]
    development_week_end = development_groups[-1][1][-1]

    manifest = {
        "schema_version": "r4_nested_validation_v1",
        "policy": {
            "development_use": "checkpoint and parameter selection only",
            "sealed_holdout_use": "first opened only after finalist lock",
            "sealed_week_count": sealed_week_count,
            "complete_week_definition": "all market trading dates in the ISO week are inside original Research Validation",
            "label_purge_rule": "development sample T requires label_end(T) < sealed holdout first buy date",
            "full_validation_final_gate": "unchanged D02 Research Validation and Weekly Backtest protocol",
        },
        "source": {
            "data_path": str(data_path),
            "data_md5": md5_file(data_path),
            "data_sha256": sha256_file(data_path),
            "d02_split_path": str(d02_split_path),
            "d02_split_sha256": sha256_file(d02_split_path),
            "d02_metrics_path": str(d02_metrics_path),
            "d02_metrics_sha256": sha256_file(d02_metrics_path),
            "d02_weekly_path": str(d02_weekly_path),
            "d02_weekly_sha256": sha256_file(d02_weekly_path),
            "d02_predictions_path": str(d02_predictions_path),
            "d02_predictions_sha256": sha256_file(d02_predictions_path),
        },
        "original_research_validation": {
            "sample_start": validation_start.strftime("%Y-%m-%d"),
            "sample_end": validation_end.strftime("%Y-%m-%d"),
            "sample_count": int(validation.get("sample_count", 0)),
            "validation_trading_day_count": len(validation_dates),
            "iso_week_group_count": len(validation_groups),
            "complete_week_count": len(complete_groups),
        },
        "development": {
            "validation_sample_start": validation_start.strftime("%Y-%m-%d"),
            "validation_sample_end": development_sample_end.strftime("%Y-%m-%d"),
            "weekly_buy_start": development_week_start.strftime("%Y-%m-%d"),
            "weekly_sell_end": development_week_end.strftime("%Y-%m-%d"),
            "complete_week_count": len(development_groups),
            "weeks": [_week_payload(key, dates) for key, dates in development_groups],
        },
        "embargo": {
            "reason": "weekly decision date requires a T+5 validation label that would cross the sealed holdout boundary",
            "complete_week_count": len(embargo_groups),
            "weeks": [_week_payload(key, dates) for key, dates in embargo_groups],
        },
        "sealed_holdout": {
            "status": "sealed",
            "decision_date_for_first_week": holdout_decision_date.strftime("%Y-%m-%d"),
            "weekly_buy_start": holdout_start.strftime("%Y-%m-%d"),
            "weekly_sell_end": holdout_end.strftime("%Y-%m-%d"),
            "complete_week_count": len(holdout_groups),
            "weeks": [_week_payload(key, dates) for key, dates in holdout_groups],
        },
        "full_validation": {
            "sample_start": validation_start.strftime("%Y-%m-%d"),
            "sample_end": validation_end.strftime("%Y-%m-%d"),
            "weekly_validation_start": validation_start.strftime("%Y-%m-%d"),
            "weekly_validation_end": validation_end.strftime("%Y-%m-%d"),
        },
        "purge_checks": {
            "development_last_label_end": holdout_decision_date.strftime("%Y-%m-%d"),
            "sealed_holdout_first_buy": holdout_start.strftime("%Y-%m-%d"),
            "development_label_end_before_holdout_buy": holdout_decision_date < holdout_start,
            "date_level_split": True,
            "sell_offset": sell_offset,
        },
    }
    return manifest


def filter_weekly_window(weekly: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frame = weekly.copy()
    frame["buy_date"] = pd.to_datetime(frame["buy_date"]).dt.normalize()
    frame["sell_date"] = pd.to_datetime(frame["sell_date"]).dt.normalize()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    frame = frame[(frame["buy_date"] >= start_ts) & (frame["sell_date"] <= end_ts)]
    if "skipped_missing_decision_prediction" in frame:
        frame = frame[frame["skipped_missing_decision_prediction"].astype(int) == 0]
    return frame.reset_index(drop=True)


def summarize_weekly_rows(weekly: pd.DataFrame) -> dict[str, Any]:
    if weekly.empty:
        return {
            "week_count": 0,
            "mean_weekly_return_net": 0.0,
            "median_weekly_return_net": 0.0,
            "cumulative_return_net": 0.0,
            "positive_week_rate": 0.0,
            "worst_week": 0.0,
            "max_drawdown_abs": 0.0,
            "CVaR_95_loss_abs": 0.0,
            "cost_drag": 0.0,
        }
    net = weekly["weekly_return_net"].astype(float).to_numpy()
    gross = weekly["weekly_return_gross"].astype(float).to_numpy()
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    cvar_count = max(1, int(math.ceil(0.05 * len(net))))
    cvar_loss = float(np.mean(np.sort(-net)[-cvar_count:]))
    worst4 = float(
        min(
            (np.prod(1.0 + net[index : index + 4]) - 1.0 for index in range(max(1, len(net) - 3))),
            default=float(np.min(net)),
        )
    )
    return {
        "week_count": int(len(weekly)),
        "date_start": str(weekly["buy_date"].min()),
        "date_end": str(weekly["sell_date"].max()),
        "mean_weekly_return_net": float(np.mean(net)),
        "median_weekly_return_net": float(np.median(net)),
        "cumulative_return_net": float(np.prod(1.0 + net) - 1.0),
        "positive_week_rate": float(np.mean(net > 0.0)),
        "worst_week": float(np.min(net)),
        "worst_4_week_cumulative_return": worst4,
        "max_drawdown_abs": float(abs(np.min(drawdown))),
        "CVaR_95_loss_abs": max(0.0, cvar_loss),
        "average_weight_sum": float(weekly["weight_sum"].astype(float).mean()),
        "cash_ratio": float((1.0 - weekly["weight_sum"].astype(float)).mean()),
        "cost_drag": float(np.sum(gross - net)),
        "number_of_selected_stocks": float(weekly["selected_count"].astype(float).mean()),
    }


def paired_weekly_metrics(challenger: pd.DataFrame, champion: pd.DataFrame) -> dict[str, Any]:
    left = challenger.copy()
    right = champion.copy()
    if "skipped_missing_decision_prediction" in left:
        left = left[left["skipped_missing_decision_prediction"].astype(int) == 0]
    if "skipped_missing_decision_prediction" in right:
        right = right[right["skipped_missing_decision_prediction"].astype(int) == 0]
    merged = left.merge(
        right[["week_start", "weekly_return_net"]],
        on="week_start",
        how="inner",
        suffixes=("_challenger", "_champion"),
    )
    if merged.empty:
        return {}
    diff = (
        merged["weekly_return_net_challenger"].astype(float)
        - merged["weekly_return_net_champion"].astype(float)
    ).to_numpy(dtype=np.float64)
    sorted_diff = np.sort(diff)[::-1]
    best2 = float(np.sum(sorted_diff[: min(2, len(sorted_diff))]))
    total_positive = float(np.sum(diff[diff > 0.0]))
    return {
        "paired_week_count": int(len(diff)),
        "mean_weekly_excess": float(np.mean(diff)),
        "median_weekly_excess": float(np.median(diff)),
        "positive_diff_rate": float(np.mean(diff > 0.0)),
        "diff_without_best_2_weeks": float(np.sum(diff) - best2),
        "best_2_weeks_contribution_ratio": safe_div(best2, total_positive) if total_positive > 0 else 0.0,
        "excess_sum": float(np.sum(diff)),
        "weekly_diffs": diff.tolist(),
    }


def _capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    if values.size == 0:
        return values
    cap = float(cap)
    if cap * len(values) < 1.0 - 1e-12:
        raise ValueError(f"max weight {cap} cannot fully invest across {len(values)} holdings")
    weights = values.astype(np.float64)
    weights = weights / max(weights.sum(), 1e-12)
    fixed = np.zeros(len(weights), dtype=bool)
    for _ in range(len(weights) + 1):
        over = (~fixed) & (weights > cap + 1e-12)
        if not over.any():
            break
        weights[over] = cap
        fixed |= over
        remaining = 1.0 - float(weights[fixed].sum())
        free = ~fixed
        if not free.any():
            break
        free_values = values[free].astype(np.float64)
        weights[free] = remaining * free_values / max(free_values.sum(), 1e-12)
    weights = np.minimum(weights, cap)
    residual = 1.0 - float(weights.sum())
    if residual > 1e-10:
        for index in np.argsort(weights):
            room = cap - weights[index]
            add = min(room, residual)
            weights[index] += add
            residual -= add
            if residual <= 1e-10:
                break
    return weights


def apply_portfolio_policy(
    predictions: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    frame = predictions.copy().reset_index(drop=True)
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
    frame["stock_id"] = frame["stock_id"].map(normalize_stock_code)
    frame["portfolio_weight"] = 0.0
    frame["effective_top_k"] = 0
    frame["cash_ratio"] = 0.0
    frame["policy_reason"] = "not_selected"
    mode = str(policy.get("mode", "top2_equal"))
    previous_selected: set[str] = set()

    for _, day in frame.groupby("sample_end_date", sort=True):
        valid = day.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
        if valid.empty:
            continue
        scores = valid["score"].astype(float).to_numpy()
        std = float(np.std(scores))
        normalized = (scores - float(np.mean(scores))) / (std + 1e-12)
        order = list(np.argsort(normalized)[::-1])
        selected: list[int]
        raw_weights: np.ndarray
        reason = mode

        if mode == "top2_equal":
            k = min(2, len(order))
            selected = order[:k]
            raw_weights = np.full(k, 1.0 / k, dtype=np.float64)
        elif mode == "dynamic_gap":
            top2_gap = normalized[order[1]] - normalized[order[2]] if len(order) >= 3 else float("inf")
            top3_gap = normalized[order[2]] - normalized[order[4]] if len(order) >= 5 else float("inf")
            if top2_gap >= float(policy.get("top2_gap_threshold", 0.30)):
                k = 2
                reason = "dynamic_gap_top2"
            elif top3_gap >= float(policy.get("top3_gap_threshold", 0.20)):
                k = 3
                reason = "dynamic_gap_top3"
            else:
                k = 5
                reason = "dynamic_gap_top5"
            k = min(k, len(order))
            selected = order[:k]
            raw_weights = np.full(k, 1.0 / k, dtype=np.float64)
        elif mode == "temperature_softmax":
            k = min(int(policy.get("top_k", 5)), len(order))
            selected = order[:k]
            temperature = max(float(policy.get("temperature", 1.0)), 1e-6)
            logits = normalized[selected] / temperature
            logits = logits - float(np.max(logits))
            raw_weights = np.exp(logits)
            raw_weights = _capped_simplex(raw_weights, float(policy.get("max_weight", 0.5)))
        elif mode == "rank_hysteresis":
            k = min(int(policy.get("top_k", 2)), len(order))
            pool_size = min(int(policy.get("candidate_pool", 5)), len(order))
            cutoff = normalized[order[k - 1]]
            margin = float(policy.get("margin", 0.10))
            retained: list[int] = []
            for position in order[:pool_size]:
                stock = str(valid.iloc[position]["stock_id"])
                if stock in previous_selected and normalized[position] >= cutoff - margin:
                    retained.append(position)
                    if len(retained) >= k:
                        break
            selected = list(retained)
            for position in order:
                if len(selected) >= k:
                    break
                if position not in selected:
                    selected.append(position)
            raw_weights = np.full(k, 1.0 / k, dtype=np.float64)
            reason = f"rank_hysteresis_retained_{len(retained)}"
        else:
            raise ValueError(f"unsupported R4 portfolio policy: {mode}")

        selected_indices = valid.iloc[selected].index.to_numpy()
        frame.loc[selected_indices, "portfolio_weight"] = raw_weights
        frame.loc[valid.index, "effective_top_k"] = len(selected)
        frame.loc[selected_indices, "policy_reason"] = reason
        previous_selected = set(valid.iloc[selected]["stock_id"].astype(str).tolist())

    return frame


def aggregate_predictions(
    prediction_frames: Sequence[pd.DataFrame],
    *,
    method: str,
    disagreement_penalty: float = 0.25,
) -> pd.DataFrame:
    if len(prediction_frames) < 2:
        raise ValueError("consensus requires at least two prediction frames")
    pieces: list[pd.DataFrame] = []
    for model_index, source in enumerate(prediction_frames):
        frame = source[["sample_end_date", "stock_id", "score", "label"]].copy()
        frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
        frame["stock_id"] = frame["stock_id"].map(normalize_stock_code)
        frame["model_index"] = model_index
        frame["universe_size"] = frame.groupby("sample_end_date")["stock_id"].transform("count")
        frame["percentile"] = frame.groupby("sample_end_date")["score"].rank(method="average", pct=True)
        frame["rank_desc"] = frame.groupby("sample_end_date")["score"].rank(method="average", ascending=False)
        grouped = frame.groupby("sample_end_date")["score"]
        frame["standardized"] = (frame["score"] - grouped.transform("mean")) / (
            grouped.transform("std").fillna(0.0) + 1e-12
        )
        pieces.append(frame)
    stacked = pd.concat(pieces, ignore_index=True)
    key = ["sample_end_date", "stock_id"]
    model_counts = stacked.groupby(key)["model_index"].nunique()
    complete_keys = model_counts[model_counts == len(prediction_frames)].index
    stacked = stacked.set_index(key).loc[complete_keys].reset_index()

    label_span = stacked.groupby(key)["label"].agg(lambda values: float(np.max(values) - np.min(values)))
    if float(label_span.max()) > 1e-8:
        raise ValueError("consensus inputs have inconsistent labels")

    rows: list[dict[str, Any]] = []
    for (sample_date, stock_id), group in stacked.groupby(key, sort=True):
        percentiles = group["percentile"].astype(float).to_numpy()
        standardized = group["standardized"].astype(float).to_numpy()
        if method == "mean_percentile_rank":
            score = float(np.mean(percentiles))
        elif method == "trimmed_mean_standardized":
            ordered = np.sort(standardized)
            trimmed = ordered[1:-1] if len(ordered) >= 5 else ordered
            score = float(np.mean(trimmed))
        elif method == "borda_rank_average":
            sizes = group["universe_size"].astype(float).to_numpy()
            borda = (sizes - group["rank_desc"].astype(float).to_numpy()) / np.maximum(sizes - 1.0, 1.0)
            score = float(np.mean(borda))
        elif method == "disagreement_penalized_consensus":
            score = float(np.mean(percentiles) - disagreement_penalty * np.std(percentiles))
        else:
            raise ValueError(f"unsupported consensus method: {method}")
        rows.append(
            {
                "sample_end_date": sample_date,
                "stock_id": stock_id,
                "score": score,
                "label": float(group["label"].iloc[0]),
                "ensemble_size": len(prediction_frames),
                "score_disagreement": float(np.std(percentiles)),
                "consensus_method": method,
            }
        )
    return pd.DataFrame(rows)


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    market_path: str | Path,
    weekly_start: str,
    weekly_end: str,
    top_k: int = 5,
) -> dict[str, Any]:
    official_daily, official_summary = evaluate_official_proxy(predictions, top_k=top_k)
    weekly, weekly_summary = run_weekly_backtest(
        predictions,
        market_path,
        validation_start=weekly_start,
        validation_end=weekly_end,
        top_k=top_k,
    )
    return {
        "official_daily": official_daily,
        "official_summary": official_summary,
        "weekly": weekly,
        "weekly_summary": weekly_summary,
    }


def cost_semantics_audit(
    challenger_weekly: pd.DataFrame,
    champion_weekly: pd.DataFrame,
    *,
    tolerance: float = FULL_INVESTMENT_TOLERANCE,
) -> dict[str, Any]:
    challenger = challenger_weekly.copy()
    champion = champion_weekly.copy()
    if "skipped_missing_decision_prediction" in challenger:
        challenger = challenger[challenger["skipped_missing_decision_prediction"].astype(int) == 0]
    if "skipped_missing_decision_prediction" in champion:
        champion = champion[champion["skipped_missing_decision_prediction"].astype(int) == 0]

    challenger_expected = WEEKLY_ROUND_TRIP_COST * challenger["weight_sum"].astype(float)
    champion_expected = WEEKLY_ROUND_TRIP_COST * champion["weight_sum"].astype(float)
    checks = {
        "challenger_weight_sum_one": bool(
            np.all(np.abs(challenger["weight_sum"].astype(float).to_numpy() - 1.0) <= tolerance)
        ),
        "champion_weight_sum_one": bool(
            np.all(np.abs(champion["weight_sum"].astype(float).to_numpy() - 1.0) <= tolerance)
        ),
        "challenger_max_five_stocks": bool((challenger["selected_count"].astype(int) <= 5).all()),
        "champion_max_five_stocks": bool((champion["selected_count"].astype(int) <= 5).all()),
        "challenger_cost_formula": bool(
            np.all(np.abs(challenger["cost"].astype(float).to_numpy() - challenger_expected.to_numpy()) <= tolerance)
        ),
        "champion_cost_formula": bool(
            np.all(np.abs(champion["cost"].astype(float).to_numpy() - champion_expected.to_numpy()) <= tolerance)
        ),
    }
    challenger_cost_drag = float(challenger["cost"].astype(float).sum())
    champion_cost_drag = float(champion["cost"].astype(float).sum())
    checks["equal_full_investment_cost"] = (
        len(challenger) == len(champion)
        and abs(challenger_cost_drag - champion_cost_drag) <= tolerance * max(1, len(challenger))
    )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "semantics": "weekly forced liquidation; full-investment cost must be formula-correct and equal, not lower through cash",
        "weekly_round_trip_cost": WEEKLY_ROUND_TRIP_COST,
        "challenger_week_count": len(challenger),
        "champion_week_count": len(champion),
        "challenger_cost_drag": challenger_cost_drag,
        "champion_cost_drag": champion_cost_drag,
        "checks": checks,
    }


def pdr_checkpoint_score(
    *,
    candidate_summary: dict[str, Any],
    champion_summary: dict[str, Any],
    paired: dict[str, Any],
) -> tuple[bool, int, float, dict[str, bool]]:
    constraints = {
        "positive_diff_rate": float(paired.get("positive_diff_rate", -1.0)) >= 0.50,
        "mean_weekly_excess": float(paired.get("mean_weekly_excess", -1.0)) > 0.0,
        "diff_without_best_2": float(paired.get("diff_without_best_2_weeks", -1.0)) > 0.0,
        "cvar_not_materially_worse": float(candidate_summary.get("CVaR_95_loss_abs", math.inf))
        <= float(champion_summary.get("CVaR_95_loss_abs", 0.0)) * 1.05 + 1e-12,
        "drawdown_not_materially_worse": float(candidate_summary.get("max_drawdown_abs", math.inf))
        <= float(champion_summary.get("max_drawdown_abs", 0.0)) * 1.05 + 1e-12,
    }
    paired_count = max(1, int(paired.get("paired_week_count", 0)))
    score = float(
        float(paired.get("mean_weekly_excess", 0.0))
        + 0.20 * float(paired.get("median_weekly_excess", 0.0))
        + 0.01 * float(paired.get("positive_diff_rate", 0.0))
        + 0.25 * min(0.0, float(paired.get("diff_without_best_2_weeks", 0.0)) / paired_count)
        - 0.25
        * max(
            0.0,
            float(candidate_summary.get("CVaR_95_loss_abs", 0.0))
            - float(champion_summary.get("CVaR_95_loss_abs", 0.0)),
        )
        - 0.10
        * max(
            0.0,
            float(candidate_summary.get("max_drawdown_abs", 0.0))
            - float(champion_summary.get("max_drawdown_abs", 0.0)),
        )
    )
    return all(constraints.values()), sum(constraints.values()), score, constraints


def select_pdr_checkpoint(
    *,
    run_dir: str | Path,
    market_path: str | Path,
    split_manifest: dict[str, Any],
    champion_weekly_path: str | Path,
    portfolio_policy: dict[str, Any],
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    champion_full = pd.read_csv(champion_weekly_path)
    development = split_manifest["development"]
    champion_weekly = filter_weekly_window(
        champion_full,
        development["weekly_buy_start"],
        development["weekly_sell_end"],
    )
    champion_summary = summarize_weekly_rows(champion_weekly)
    candidates: list[dict[str, Any]] = []
    for prediction_path in sorted((run_dir / "epoch_predictions").glob("epoch_[0-9][0-9][0-9].csv")):
        epoch = int(prediction_path.stem.split("_")[-1])
        predictions = pd.read_csv(prediction_path, dtype={"stock_id": str})
        predictions = apply_portfolio_policy(predictions, portfolio_policy)
        evaluated = evaluate_prediction_frame(
            predictions,
            market_path=market_path,
            weekly_start=development["weekly_buy_start"],
            weekly_end=development["weekly_sell_end"],
            top_k=5,
        )
        candidate_weekly = filter_weekly_window(
            evaluated["weekly"],
            development["weekly_buy_start"],
            development["weekly_sell_end"],
        )
        candidate_summary = summarize_weekly_rows(candidate_weekly)
        paired = paired_weekly_metrics(candidate_weekly, champion_weekly)
        feasible, pass_count, score, constraints = pdr_checkpoint_score(
            candidate_summary=candidate_summary,
            champion_summary=champion_summary,
            paired=paired,
        )
        candidates.append(
            {
                "epoch": epoch,
                "feasible": feasible,
                "constraint_pass_count": pass_count,
                "robust_score": score,
                "constraints": constraints,
                "weekly_summary": candidate_summary,
                "paired": paired,
                "official_proxy": evaluated["official_summary"],
                "predictions": predictions,
                "weekly": evaluated["weekly"],
            }
        )
    if not candidates:
        raise FileNotFoundError(f"no epoch predictions under {run_dir / 'epoch_predictions'}")
    feasible_candidates = [item for item in candidates if item["feasible"]]
    selection_pool = feasible_candidates or candidates
    selected = max(
        selection_pool,
        key=lambda item: (item["constraint_pass_count"], item["robust_score"], item["epoch"]),
    )
    selected_epoch = int(selected["epoch"])
    selected_checkpoint = run_dir / "epoch_checkpoints" / f"epoch_{selected_epoch:03d}.pth"
    if not selected_checkpoint.exists():
        raise FileNotFoundError(selected_checkpoint)
    original_best = run_dir / "best_model.pth"
    original_backup = run_dir / "best_model_before_r4_selection.pth"
    if original_best.exists() and not original_backup.exists():
        shutil.copy2(original_best, original_backup)
    shutil.copy2(selected_checkpoint, original_best)
    selected["predictions"].to_csv(run_dir / "best_val_predictions.csv", index=False, encoding="utf-8")
    selected["weekly"].to_csv(run_dir / "development_weekly_backtest.csv", index=False, encoding="utf-8")
    write_json(run_dir / "development_weekly_backtest_result.json", selected["weekly_summary"])
    serializable_candidates = [
        {key: value for key, value in item.items() if key not in {"predictions", "weekly"}}
        for item in candidates
    ]
    payload = {
        "selection_rule": "development-only PDR constraints, then robust score",
        "sealed_holdout_used": False,
        "portfolio_policy": portfolio_policy,
        "selected_epoch": selected_epoch,
        "selected_feasible": bool(selected["feasible"]),
        "feasible_epoch_count": len(feasible_candidates),
        "champion_development_summary": champion_summary,
        "selected": {key: value for key, value in selected.items() if key not in {"predictions", "weekly"}},
        "epochs": serializable_candidates,
    }
    write_json(run_dir / "r4_checkpoint_selection.json", payload)
    return payload


def evaluate_r4_gate(
    *,
    full_metrics: dict[str, Any],
    sealed_holdout_paired: dict[str, Any],
    seed_results: Sequence[dict[str, Any]],
    leakage_pass: bool,
    reproducibility_pass: bool,
    cost_audit: dict[str, Any],
    cold_start_prediction_seconds: float,
    champion: dict[str, float] | None = None,
) -> dict[str, Any]:
    champion = champion or {
        "official_proxy": 0.0371198358,
        "weekly_cumulative": 2.4696850616,
        "max_drawdown_abs": 0.3005223607,
        "worst_week": -0.138453547,
    }
    official = full_metrics.get("official_proxy", {})
    weekly = full_metrics.get("weekly_backtest", {})
    paired = full_metrics.get("paired_vs_champion", {})
    seed_count = len(seed_results)
    seed_beats = sum(
        1
        for item in seed_results
        if float(item.get("official_proxy", -math.inf)) > champion["official_proxy"]
        and float(item.get("weekly_cumulative", -math.inf)) > champion["weekly_cumulative"]
    )
    seed_excesses = [float(item.get("mean_weekly_excess", 0.0)) for item in seed_results]
    seed_min_excess = min(seed_excesses) if seed_excesses else -math.inf
    cumulative = float(weekly.get("cumulative_return_net", -math.inf))
    drawdown = float(weekly.get("max_drawdown_abs", math.inf))
    cumulative_improvement = cumulative - champion["weekly_cumulative"]
    drawdown_compensation = (
        drawdown <= champion["max_drawdown_abs"]
        or (
            drawdown <= champion["max_drawdown_abs"] * 1.05
            and cumulative_improvement >= max(0.25, champion["weekly_cumulative"] * 0.10)
        )
    )
    required_seed_beats = 3 if seed_count >= 5 else 2 if seed_count >= 3 else math.inf
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, value: Any, threshold: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "value": value, "threshold": threshold})

    add("full_validation_official_proxy", float(official.get("official_proxy_final_score", -math.inf)) > champion["official_proxy"], official.get("official_proxy_final_score"), champion["official_proxy"])
    add("weekly_cumulative_net", cumulative > champion["weekly_cumulative"], cumulative, champion["weekly_cumulative"])
    add("paired_positive_diff_rate", float(paired.get("positive_diff_rate", -math.inf)) >= 0.52, paired.get("positive_diff_rate"), 0.52)
    add("paired_mean_weekly_excess", float(paired.get("mean_weekly_excess", -math.inf)) > 0.0, paired.get("mean_weekly_excess"), 0.0)
    add("diff_without_best_2_weeks", float(paired.get("diff_without_best_2_weeks", -math.inf)) > 0.0, paired.get("diff_without_best_2_weeks"), 0.0)
    add("drawdown_or_compensated", drawdown_compensation, drawdown, {"base": champion["max_drawdown_abs"], "hard_cap": champion["max_drawdown_abs"] * 1.05})
    add("CVaR95_loss_abs", float(weekly.get("CVaR_95_loss_abs", math.inf)) <= 0.116554, weekly.get("CVaR_95_loss_abs"), 0.116554)
    add("worst_week_not_materially_worse", float(weekly.get("worst_week", -math.inf)) >= champion["worst_week"] * 1.10, weekly.get("worst_week"), champion["worst_week"] * 1.10)
    add("sealed_holdout_mean_excess", float(sealed_holdout_paired.get("mean_weekly_excess", -math.inf)) > 0.0, sealed_holdout_paired.get("mean_weekly_excess"), 0.0)
    add("sealed_holdout_positive_diff_rate", float(sealed_holdout_paired.get("positive_diff_rate", -math.inf)) >= 0.50, sealed_holdout_paired.get("positive_diff_rate"), 0.50)
    add("seed_majority_beats_d02", seed_count >= 3 and seed_beats >= required_seed_beats, seed_beats, required_seed_beats)
    add("seed_min_excess", seed_min_excess >= -0.005, seed_min_excess, -0.005)
    add("leakage_audit", leakage_pass, leakage_pass, True)
    add("reproducibility_check", reproducibility_pass, reproducibility_pass, True)
    add("full_investment_cost_semantics", cost_audit.get("status") == "pass", cost_audit.get("status"), "pass")
    add("competition_format_max_five", float(weekly.get("number_of_selected_stocks", math.inf)) <= 5.0, weekly.get("number_of_selected_stocks"), 5)
    add("competition_format_weight_sum_one", abs(float(weekly.get("average_weight_sum", 0.0)) - 1.0) <= FULL_INVESTMENT_TOLERANCE, weekly.get("average_weight_sum"), 1.0)
    add("cold_start_prediction_time", cold_start_prediction_seconds <= 300.0, cold_start_prediction_seconds, 300.0)
    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "gate_version": "R4_corrected_full_investment_v1",
        "decision": "ACCEPT_AS_CHAMPION" if not failed else "RESEARCH_CANDIDATE_ONLY",
        "failed_checks": failed,
        "checks": checks,
        "seed_count": seed_count,
        "seed_beats_d02": seed_beats,
        "seed_min_excess": seed_min_excess,
        "cost_semantics": cost_audit,
    }


def build_artifact_manifest(root: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output_path).resolve() if output_path else root / "artifact_manifest.json"
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.resolve() != output):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {"root": str(root), "file_count": len(files), "files": files}
    write_json(output, payload)
    return payload

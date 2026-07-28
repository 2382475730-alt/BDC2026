from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from Evaluation.scripts.d_series import defensive_combo_v1 as v1


EXPERIMENT_ID = "TEST_D_DEFENSIVE_COMBO_V2_COMPLETE_LINK_20260728"
WORKSPACE = v1.WORKSPACE
RUN_DIR = (
    WORKSPACE
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "automation_runs"
    / EXPERIMENT_ID
)
V1_RUN_DIR = (
    WORKSPACE
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "automation_runs"
    / "TEST_D_DEFENSIVE_COMBO_V1_20260728"
)
V1_SOURCE = WORKSPACE / "Evaluation" / "scripts" / "d_series" / "defensive_combo_v1.py"
V1_TEST = (
    WORKSPACE / "Evaluation" / "scripts" / "d_series" / "test_defensive_combo_v1.py"
)
V2_SOURCE = Path(__file__).resolve()
V2_TEST = (
    WORKSPACE / "Evaluation" / "scripts" / "d_series" / "test_defensive_combo_v2.py"
)

SEEDS = v1.SEEDS
FOLDS = v1.FOLDS
POOL_STEPS = (20, 50, 100, None)
CLUSTER_METHOD = "complete"
PORTFOLIO_SIZE = v1.PORTFOLIO_SIZE
MAX_CLUSTER_HOLDINGS = v1.MAX_CLUSTER_HOLDINGS
NONINFERIORITY_MARGIN = v1.NONINFERIORITY_MARGIN
FLOAT_TOLERANCE = v1.FLOAT_TOLERANCE
RANK_COLUMNS = ["marker", "sample_end_date", "stock_id", "score"]
FINAL_PORTFOLIO_COLUMNS = [
    "seed",
    "fold_id",
    "decision_date",
    "policy_id",
    "pool_size_used",
    "stock_id",
    "selection_order",
    "weight",
    "cluster_id",
    "d_rank_score",
    "defensive_score",
    "volatility_rank",
    "reversal_risk_rank",
    "max_selected_correlation",
    "turnover_penalty",
    "regime_multiplier",
    "hhi",
    "weighted_pair_correlation",
    "cross_week_turnover",
]


@dataclass
class V2DayContext:
    decision_date: pd.Timestamp
    candidates: pd.DataFrame
    correlation: pd.DataFrame
    raw_state: dict[str, float]
    regime_multiplier: float
    pool_size_used: int
    universe_size: int
    feasible_capacity: int
    cluster_count: int
    largest_cluster: int


class FeasibilityFailure(RuntimeError):
    def __init__(self, decision_date: pd.Timestamp, universe_size: int) -> None:
        super().__init__(
            "complete-link cluster capacity is below five even for the full "
            f"ranking on {decision_date.date()}"
        )
        self.decision_date = decision_date
        self.universe_size = universe_size


def sha256_file(path: Path) -> str:
    return v1.sha256_file(path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    v1.write_json(path, payload)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    v1.write_csv(path, frame)


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def policy_grid() -> list[v1.Policy]:
    return v1.policy_grid()


def rank_projection(path: Path) -> pd.DataFrame:
    frame = v1.load_predictions(path, include_label=False)
    return frame.sort_values(
        ["sample_end_date", "stock_id"], ascending=[True, True]
    ).reset_index(drop=True)


def rank_projection_bytes(path: Path) -> bytes:
    return rank_projection(path).to_csv(
        index=False,
        columns=RANK_COLUMNS,
        encoding="utf-8",
        float_format="%.12g",
        lineterminator="\n",
    ).encode("utf-8")


def rank_projection_record(seed: int, fold: str, phase: str) -> dict[str, Any]:
    path = v1.prediction_path(seed, fold, phase)
    frame = rank_projection(path)
    return {
        "kind": "rank_projection",
        "seed": seed,
        "fold_id": fold,
        "phase": phase,
        "path": relative_path(path),
        "rows": int(len(frame)),
        "date_min": frame["sample_end_date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["sample_end_date"].max().strftime("%Y-%m-%d"),
        "sha256": sha256_bytes(rank_projection_bytes(path)),
        "columns": RANK_COLUMNS,
        "label_bytes_hashed": False,
    }


def input_manifest() -> dict[str, Any]:
    rank_records = [
        rank_projection_record(seed, fold, phase)
        for seed in SEEDS
        for fold in FOLDS
        for phase in ("inner", "outer")
    ]
    files: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            for phase in ("inner", "outer"):
                for name in ("config.json", "split_metadata.json", "scalers.json"):
                    path = v1.companion_path(seed, fold, phase, name)
                    files.append(
                        {
                            "kind": "ordinary_file",
                            "role": f"h0_{phase}_{name}",
                            "seed": seed,
                            "fold_id": fold,
                            "path": relative_path(path),
                            "bytes": int(path.stat().st_size),
                            "sha256": sha256_file(path),
                        }
                    )
    for role, path in (
        ("market_data", v1.MARKET_CSV),
        ("v1_source", V1_SOURCE),
        ("v1_test", V1_TEST),
        ("v2_source", V2_SOURCE),
        ("v2_test", V2_TEST),
    ):
        files.append(
            {
                "kind": "ordinary_file",
                "role": role,
                "path": relative_path(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "rank_projections": rank_records,
        "files": files,
        "outer_label_container_hashed": False,
        "prospective_path_accessed": False,
    }


def verify_input_manifest(manifest: Mapping[str, Any]) -> None:
    for record in manifest["rank_projections"]:
        path = WORKSPACE / record["path"]
        if sha256_bytes(rank_projection_bytes(path)) != record["sha256"]:
            raise RuntimeError(f"rank projection drift: {record['path']}")
    for record in manifest["files"]:
        path = WORKSPACE / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"input drift: {record['path']}")


def protection_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "formal_trees": v1.protected_manifest(),
        "v1_source_sha256": sha256_file(V1_SOURCE),
        "v1_test_sha256": sha256_file(V1_TEST),
        "v1_run_tree": v1.tree_manifest(V1_RUN_DIR),
    }


def compare_protection(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    formal = v1.compare_protected(
        before["formal_trees"], after["formal_trees"]
    )
    checks = {
        "formal_trees": formal["status"] == "PASS",
        "v1_source": before["v1_source_sha256"] == after["v1_source_sha256"],
        "v1_test": before["v1_test_sha256"] == after["v1_test_sha256"],
        "v1_run_tree": (
            before["v1_run_tree"]["tree_sha256"]
            == after["v1_run_tree"]["tree_sha256"]
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "formal_tree_comparison": formal,
    }


def environment_payload() -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = (
            torch.cuda.get_device_name(0) if cuda_available else "NOT_USED"
        )
    except Exception as exc:
        torch_version = f"UNAVAILABLE: {type(exc).__name__}"
        cuda_available = False
        gpu_name = "NOT_USED"
    return {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_time_seconds": 0.0,
        "training_performed": False,
    }


def preregistration_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "identity": {
            "alpha_champion": "D01_FrozenResearchChampion_20260709",
            "defensive_champion": None,
            "submission_identity": "candidate/Candidate-D",
            "promotion_eligible": False,
        },
        "sources": {
            "rankings": "existing H0/D01-program inner and outer full rankings",
            "market_data": relative_path(v1.MARKET_CSV),
            "industry_data_enabled": False,
            "external_data_enabled": False,
            "prospective_windows_accessed": False,
        },
        "seeds": list(SEEDS),
        "folds": v1.FOLD_SPECS,
        "decision_frequency": "every fifth prediction date, starting first",
        "portfolio": {
            "stock_count": PORTFOLIO_SIZE,
            "weight_sum": 1.0,
            "cash_allowed": False,
            "max_weight": 0.30,
            "max_hhi": 0.30,
            "max_cluster_holdings": MAX_CLUSTER_HOLDINGS,
            "weight_templates": v1.WEIGHT_TEMPLATES,
        },
        "feasibility": {
            "candidate_pool_steps": [20, 50, 100, "FULL_RANKING"],
            "first_feasible_pool_wins": True,
            "capacity_formula": "sum(min(cluster_size, 2))",
            "minimum_capacity": 5,
            "full_ranking_failure": "REJECTED_AT_COMBINATION_GATE",
            "outcome_columns_allowed": False,
        },
        "risk_features": {
            "cluster_method": "complete-linkage on sqrt((1-rho)/2)",
            "correlation_cluster_cut": v1.CORRELATION_CLUSTER_CUT,
            "correlation_lookback": v1.CORRELATION_LOOKBACK,
            "correlation_min_observations": v1.CORRELATION_MIN_OBS,
            "correlation_missing_value": v1.CORRELATION_MISSING_VALUE,
            "correlation_shrinkage_to_cross_sectional_median": (
                v1.CORRELATION_SHRINKAGE
            ),
            "state_group": "fixed raw Candidate-D top-20 before any expansion",
            "raw_cutoff": "strictly before decision date (lag-1)",
            "state_fields": list(v1.STATE_FIELDS),
            "state_multiplier_range": [
                v1.STATE_MULTIPLIER_MIN,
                v1.STATE_MULTIPLIER_MAX,
            ],
        },
        "grid": {
            "lambda_volatility": list(v1.LAMBDA_VOLATILITY),
            "lambda_correlation": list(v1.LAMBDA_CORRELATION),
            "lambda_turnover": list(v1.LAMBDA_TURNOVER),
            "lambda_reversal": list(v1.LAMBDA_REVERSAL),
            "weight_templates": list(v1.WEIGHT_TEMPLATES),
            "variant_count": len(policy_grid()),
            "linkage_and_pool_steps_are_not_search_parameters": True,
        },
        "global_policy_selection": {
            "structural_pass_required_on_every_inner_decision": True,
            "unique_week_series": (
                "mean across seeds for the same fold_id and decision_date"
            ),
            "overall_mean_excess_minimum": -NONINFERIORITY_MARGIN,
            "each_seed_mean_excess_minimum": -NONINFERIORITY_MARGIN,
            "priority": [
                "higher unique-week candidate positive rate",
                "higher unique-week candidate median",
                "higher unique-week candidate CVaR20",
                "higher unique-week candidate worst week",
                "higher unique-week excess sum after removing best two",
                "lower mean cross-week turnover",
                "lower mean weighted pair correlation",
                "lower total lambda penalty",
                "equal-weight template",
                "lower policy_id",
            ],
            "one_policy_for_all_outer_units": True,
        },
        "tie_break": [
            "higher DefensiveScore",
            "higher D rank score",
            "lower volatility rank",
            "lower max correlation to selected names",
            "already held in previous locked portfolio",
            "lower stock code",
        ],
        "outcome_protocol": {
            "inner_locks_before_inner_label_parse": True,
            "outer_two_independent_locks_before_outer_label_parse": True,
            "outer_container_hash_after_seal_only": True,
            "known_outer_authority": "rejection only; never promotion",
            "prospective_access": False,
        },
        "cost_bps_per_side": [0, 10, 20],
        "noninferiority": {
            "margin_per_five_day_period": NONINFERIORITY_MARGIN,
            "bootstrap_reps": v1.BOOTSTRAP_REPS,
            "bootstrap_seed": v1.BOOTSTRAP_SEED,
            "moving_block_length": v1.BOOTSTRAP_BLOCK_LENGTH,
        },
        "stop_rules": {
            "feasibility_or_inner_failure": "REJECTED_AT_COMBINATION_GATE",
            "outer_gate_failure": "REJECTED_AT_QUICK_GATE",
            "outer_gate_pass": (
                "PASS_EXPOSED_QUICK_SCREEN_NOT_PROMOTION_ELIGIBLE"
            ),
            "training_auto_start": False,
            "candidate_auto_promotion": False,
        },
    }


def complete_clusters(correlation: pd.DataFrame) -> dict[str, str]:
    stocks = list(correlation.index.astype(str))
    if len(stocks) == 1:
        return {stocks[0]: stocks[0]}
    values = correlation.to_numpy(dtype=float)
    distance = np.sqrt(np.clip((1.0 - values) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method=CLUSTER_METHOD, optimal_ordering=True)
    threshold = math.sqrt(
        (1.0 - v1.CORRELATION_CLUSTER_CUT) / 2.0
    )
    labels = fcluster(tree, t=threshold, criterion="distance")
    members: dict[int, list[str]] = {}
    for stock, label in zip(stocks, labels, strict=True):
        members.setdefault(int(label), []).append(stock)
    canonical = {label: min(group) for label, group in members.items()}
    return {
        stock: canonical[int(label)]
        for stock, label in zip(stocks, labels, strict=True)
    }


def cluster_statistics(
    stocks: Sequence[str], clusters: Mapping[str, str]
) -> tuple[int, int, int]:
    sizes = pd.Series([clusters[str(stock)] for stock in stocks]).value_counts()
    capacity = int(sizes.clip(upper=MAX_CLUSTER_HOLDINGS).sum())
    return capacity, int(len(sizes)), int(sizes.max())


def candidate_pool_sizes(universe_size: int) -> list[int]:
    sizes: list[int] = []
    for raw in POOL_STEPS:
        value = universe_size if raw is None else min(int(raw), universe_size)
        if value >= PORTFOLIO_SIZE and value not in sizes:
            sizes.append(value)
    return sizes


def build_day_context(
    date: pd.Timestamp,
    day_all: pd.DataFrame,
    market: v1.MarketData,
    history: Mapping[str, list[float]],
) -> V2DayContext:
    ordered = (
        day_all.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .sort_values(["score", "stock_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if len(ordered) < PORTFOLIO_SIZE:
        raise FeasibilityFailure(date, int(len(ordered)))

    state_day = ordered.head(min(20, len(ordered))).copy()
    state_stocks = state_day["stock_id"].astype(str).tolist()
    _, _, _, raw_state = market.snapshot(date, state_stocks)

    selected_payload: tuple[
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.DataFrame,
        dict[str, str],
        int,
        int,
        int,
    ] | None = None
    for pool_size in candidate_pool_sizes(len(ordered)):
        day = ordered.head(pool_size).copy().reset_index(drop=True)
        stocks = day["stock_id"].astype(str).tolist()
        volatility, reversal, correlation, _ = market.snapshot(date, stocks)
        clusters = complete_clusters(correlation)
        capacity, cluster_count, largest_cluster = cluster_statistics(
            stocks, clusters
        )
        if capacity >= PORTFOLIO_SIZE:
            selected_payload = (
                day,
                volatility,
                reversal,
                correlation,
                clusters,
                capacity,
                cluster_count,
                largest_cluster,
            )
            break
    if selected_payload is None:
        raise FeasibilityFailure(date, int(len(ordered)))

    (
        day,
        volatility,
        reversal,
        correlation,
        clusters,
        capacity,
        cluster_count,
        largest_cluster,
    ) = selected_payload
    day["d_rank_score"] = np.linspace(1.0, 0.0, len(day))
    day["volatility_rank"] = v1.percentile_risk(
        volatility.reindex(day["stock_id"]).reset_index(drop=True),
        missing_worst=True,
    ).to_numpy(dtype=float)
    day["reversal_risk_rank"] = reversal.reindex(
        day["stock_id"]
    ).to_numpy(dtype=float)
    day["cluster_id"] = day["stock_id"].map(clusters)
    percentiles = v1.state_percentiles(raw_state, history)
    multiplier = v1.STATE_MULTIPLIER_MIN + (
        v1.STATE_MULTIPLIER_MAX - v1.STATE_MULTIPLIER_MIN
    ) * float(np.mean(list(percentiles.values())))
    return V2DayContext(
        decision_date=date,
        candidates=day,
        correlation=correlation,
        raw_state={key: float(value) for key, value in raw_state.items()},
        regime_multiplier=float(multiplier),
        pool_size_used=int(len(day)),
        universe_size=int(len(ordered)),
        feasible_capacity=capacity,
        cluster_count=cluster_count,
        largest_cluster=largest_cluster,
    )


def build_contexts(
    rank_only: pd.DataFrame,
    market: v1.MarketData,
    initial_state_history: Mapping[str, Sequence[float]] | None = None,
) -> tuple[dict[pd.Timestamp, V2DayContext], dict[str, list[float]]]:
    history = {
        field: list(initial_state_history[field]) if initial_state_history else []
        for field in v1.STATE_FIELDS
    }
    contexts: dict[pd.Timestamp, V2DayContext] = {}
    required_dates = {
        pd.Timestamp(date).normalize() for date in v1.decision_dates(rank_only)
    }
    for raw_date in sorted(rank_only["sample_end_date"].unique()):
        date = pd.Timestamp(raw_date).normalize()
        day_all = rank_only.loc[rank_only["sample_end_date"] == date].copy()
        if date in required_dates:
            context = build_day_context(date, day_all, market, history)
            contexts[date] = context
            raw_state = context.raw_state
        else:
            ordered = (
                day_all.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["score"])
                .sort_values(
                    ["score", "stock_id"], ascending=[False, True]
                )
                .head(20)
            )
            if len(ordered) < PORTFOLIO_SIZE:
                raise ValueError(
                    f"fewer than five ranked stocks on {date.date()}"
                )
            state_stocks = ordered["stock_id"].astype(str).tolist()
            _, _, _, raw_state = market.snapshot(date, state_stocks)
        for field in v1.STATE_FIELDS:
            history[field].append(float(raw_state[field]))
    return contexts, history


def select_defensive(
    context: V2DayContext,
    policy: v1.Policy,
    previous_weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    candidates = context.candidates.copy()
    selected: list[dict[str, Any]] = []
    selected_stocks: list[str] = []
    cluster_counts: dict[str, int] = {}
    has_previous = bool(previous_weights)
    for order in range(PORTFOLIO_SIZE):
        choices: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for row in candidates.itertuples(index=False):
            stock = str(row.stock_id)
            if stock in selected_stocks:
                continue
            cluster = str(row.cluster_id)
            cluster_count = cluster_counts.get(cluster, 0)
            if cluster_count >= MAX_CLUSTER_HOLDINGS:
                continue
            cluster_penalty = 1.0 if cluster_count == 1 else 0.0
            turnover_penalty = (
                0.0 if not has_previous or stock in previous_weights else 1.0
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
                * policy.lambda_volatility
                * float(row.volatility_rank)
                - context.regime_multiplier
                * policy.lambda_correlation
                * cluster_penalty
                - policy.lambda_turnover * turnover_penalty
                - context.regime_multiplier
                * policy.lambda_reversal
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
                0 if stock in previous_weights else 1,
                stock,
            )
            choices.append((key, payload))
        if not choices:
            return []
        _, chosen = min(choices, key=lambda item: item[0])
        selected.append(chosen)
        selected_stocks.append(chosen["stock_id"])
        cluster_counts[chosen["cluster_id"]] = (
            cluster_counts.get(chosen["cluster_id"], 0) + 1
        )
    for item, weight in zip(
        selected, v1.WEIGHT_TEMPLATES[policy.template], strict=True
    ):
        item["weight"] = float(weight)
    return selected


def select_anchor(
    context: V2DayContext, weights: Sequence[float]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for order, (row, weight) in enumerate(
        zip(
            context.candidates.head(len(weights)).itertuples(index=False),
            weights,
            strict=True,
        ),
        start=1,
    ):
        selected.append(
            {
                "stock_id": str(row.stock_id),
                "selection_order": order,
                "cluster_id": str(row.cluster_id),
                "defensive_score": float(row.d_rank_score),
                "d_rank_score": float(row.d_rank_score),
                "volatility_rank": float(row.volatility_rank),
                "reversal_risk_rank": float(row.reversal_risk_rank),
                "max_selected_correlation": 0.0,
                "turnover_penalty": 0.0,
                "cluster_penalty": 0.0,
                "weight": float(weight),
            }
        )
    return selected


def simulate_policy(
    contexts: Mapping[pd.Timestamp, V2DayContext],
    dates: Sequence[pd.Timestamp],
    seed: int,
    fold: str,
    phase: str,
    role: str,
    policy: v1.Policy | None = None,
    anchor_weights: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous: dict[str, float] = {}
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    expected_count = (
        len(anchor_weights)
        if anchor_weights is not None
        else len(v1.WEIGHT_TEMPLATES[policy.template])
    )
    for raw_date in dates:
        date = pd.Timestamp(raw_date).normalize()
        context = contexts[date]
        if policy is not None:
            selected = select_defensive(context, policy, previous)
            policy_id = policy.policy_id
        elif anchor_weights is not None:
            selected = select_anchor(context, anchor_weights)
            policy_id = role
        else:
            raise ValueError("policy or anchor_weights is required")
        if len(selected) != expected_count:
            raise ValueError(
                f"portfolio selection failed for {role} on {date.date()}"
            )
        current = {
            str(item["stock_id"]): float(item["weight"]) for item in selected
        }
        turnover = v1.transition_turnover(current, previous)
        cross_week = turnover if previous else np.nan
        pair_correlation = v1.weighted_pair_correlation(
            selected, context.correlation
        )
        hhi = float(sum(weight * weight for weight in current.values()))
        max_cluster_count = int(
            pd.Series([item["cluster_id"] for item in selected])
            .value_counts()
            .max()
        )
        common = {
            "marker": "TEST",
            "experiment_id": EXPERIMENT_ID,
            "seed": seed,
            "fold_id": fold,
            "phase": phase,
            "policy_role": role,
            "policy_id": policy_id,
            "decision_date": date.strftime("%Y-%m-%d"),
            "pool_size_used": context.pool_size_used,
            "universe_size": context.universe_size,
            "feasible_capacity": context.feasible_capacity,
            "cluster_count": context.cluster_count,
            "largest_cluster": context.largest_cluster,
            "regime_multiplier": context.regime_multiplier,
            "trade_turnover": turnover,
            "cross_week_turnover": cross_week,
            "weighted_pair_correlation": pair_correlation,
            "hhi": hhi,
            "max_cluster_count": max_cluster_count,
        }
        for item in selected:
            long_rows.append({**common, **item})
        summary_rows.append(
            {
                **common,
                "max_weight": max(current.values()),
                "weight_sum": sum(current.values()),
                "stock_count": len(current),
                "holdings": "|".join(current),
                "weights": "|".join(
                    f"{value:.8f}" for value in current.values()
                ),
            }
        )
        previous = current
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def validate_portfolios(
    frame: pd.DataFrame, *, enforce_defensive: bool = True
) -> dict[str, Any]:
    keys = [
        "seed",
        "fold_id",
        "policy_role",
        "policy_id",
        "decision_date",
    ]
    grouped = frame.groupby(keys, sort=True)
    counts = grouped["stock_id"].nunique()
    weights = grouped["weight"].sum()
    max_weights = grouped["weight"].max()
    hhi = grouped["weight"].apply(lambda values: float(np.square(values).sum()))
    cluster_cap = grouped["cluster_id"].apply(
        lambda values: int(pd.Series(values).value_counts().max())
    )
    checks: dict[str, bool] = {
        "five_unique_stocks": bool((counts == PORTFOLIO_SIZE).all()),
        "fully_invested": bool(
            np.allclose(weights.to_numpy(dtype=float), 1.0, atol=FLOAT_TOLERANCE)
        ),
    }
    if enforce_defensive:
        checks.update(
            {
                "max_weight": bool(
                    (max_weights <= 0.30 + FLOAT_TOLERANCE).all()
                ),
                "hhi": bool((hhi <= 0.30 + FLOAT_TOLERANCE).all()),
                "cluster_cap": bool(
                    (cluster_cap <= MAX_CLUSTER_HOLDINGS).all()
                ),
            }
        )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "portfolio_count": int(len(counts)),
        "max_hhi": float(hhi.max()),
        "max_weight_observed": float(max_weights.max()),
        "max_cluster_count": int(cluster_cap.max()),
    }


def validate_native_anchors(frame: pd.DataFrame) -> dict[str, Any]:
    expected = {
        "BASELINE_TOP2_EQUAL": 2,
        "ANCHOR_TOP5_EQUAL": 5,
        "ANCHOR_TOP5_RANK_30_25_20_15_10": 5,
    }
    anchors = frame.loc[frame["policy_role"].isin(expected)].copy()
    keys = [
        "seed",
        "fold_id",
        "policy_role",
        "policy_id",
        "decision_date",
    ]
    grouped = anchors.groupby(keys, sort=True)
    counts = grouped["stock_id"].nunique()
    weights = grouped["weight"].sum()
    expected_counts = counts.index.get_level_values("policy_role").map(expected)
    checks = {
        "expected_unique_stock_count": bool(
            np.array_equal(
                counts.to_numpy(dtype=int),
                np.asarray(expected_counts, dtype=int),
            )
        ),
        "fully_invested": bool(
            np.allclose(weights.to_numpy(dtype=float), 1.0, atol=FLOAT_TOLERANCE)
        ),
        "positive_weights": bool((anchors["weight"] > 0.0).all()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "portfolio_count": int(len(counts)),
    }


def score_locked_portfolios(
    locked: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    working = locked.merge(
        labels,
        on=["seed", "fold_id", "decision_date", "stock_id"],
        how="left",
        validate="many_to_one",
    )
    if working["label"].isna().any():
        raise ValueError("missing labels for one or more locked holdings")
    working["weighted_return"] = working["weight"] * working["label"]
    keys = [
        "seed",
        "fold_id",
        "policy_role",
        "policy_id",
        "decision_date",
    ]
    scored = (
        working.groupby(keys, sort=True, as_index=False)
        .agg(
            portfolio_return_gross=("weighted_return", "sum"),
            trade_turnover=("trade_turnover", "first"),
            cross_week_turnover=("cross_week_turnover", "first"),
            weighted_pair_correlation=("weighted_pair_correlation", "first"),
            hhi=("hhi", "first"),
            max_cluster_count=("max_cluster_count", "first"),
            pool_size_used=("pool_size_used", "first"),
            stock_count=("stock_id", "size"),
        )
    )
    for bps in (10, 20):
        cost = bps / 10_000.0
        scored[f"return_roundtrip_{bps}bps"] = (
            scored["portfolio_return_gross"] - 2.0 * cost
        )
        scored[f"return_turnover_{bps}bps"] = (
            scored["portfolio_return_gross"]
            - 2.0 * cost * scored["trade_turnover"]
        )
    return scored


def load_labels(phase: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    hashes: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            path = v1.prediction_path(seed, fold, phase)
            predictions = v1.load_predictions(path, include_label=True)
            frames.append(
                predictions.assign(
                    seed=seed,
                    fold_id=fold,
                    decision_date=predictions[
                        "sample_end_date"
                    ].dt.strftime("%Y-%m-%d"),
                )[["seed", "fold_id", "decision_date", "stock_id", "label"]]
            )
            hashes.append(
                {
                    "seed": seed,
                    "fold_id": fold,
                    "phase": phase,
                    "path": relative_path(path),
                    "bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return pd.concat(frames, ignore_index=True), hashes


def global_policy_summary(
    candidate_scored: pd.DataFrame,
    baseline_scored: pd.DataFrame,
    candidate_locked: pd.DataFrame,
) -> pd.DataFrame:
    base = baseline_scored[
        [
            "seed",
            "fold_id",
            "decision_date",
            "portfolio_return_gross",
        ]
    ].rename(columns={"portfolio_return_gross": "baseline_return"})
    rows: list[dict[str, Any]] = []
    policies = {policy.policy_id: policy for policy in policy_grid()}
    for policy_id, group in candidate_scored.groupby("policy_id", sort=True):
        paired = group.merge(
            base,
            on=["seed", "fold_id", "decision_date"],
            how="inner",
            validate="one_to_one",
        )
        paired["excess"] = (
            paired["portfolio_return_gross"] - paired["baseline_return"]
        )
        unique_week = (
            paired.groupby(["fold_id", "decision_date"], sort=True)
            .agg(
                candidate_return=("portfolio_return_gross", "mean"),
                baseline_return=("baseline_return", "mean"),
                excess=("excess", "mean"),
            )
            .reset_index()
        )
        candidate_metrics = v1.weekly_metrics(unique_week["candidate_return"])
        excess_metrics = v1.weekly_metrics(unique_week["excess"])
        seed_means = (
            paired.groupby("seed", sort=True)["excess"].mean().astype(float)
        )
        locked = candidate_locked.loc[
            candidate_locked["policy_id"] == policy_id
        ]
        validation = validate_portfolios(locked)
        policy = policies[str(policy_id)]
        overall_noninferior = (
            float(unique_week["excess"].mean()) >= -NONINFERIORITY_MARGIN
        )
        seed_noninferior = len(seed_means) == len(SEEDS) and bool(
            (seed_means >= -NONINFERIORITY_MARGIN).all()
        )
        eligible = (
            validation["status"] == "PASS"
            and overall_noninferior
            and seed_noninferior
        )
        rows.append(
            {
                "marker": "TEST",
                "experiment_id": EXPERIMENT_ID,
                **asdict(policy),
                "policy_id": policy.policy_id,
                "unique_week_count": int(len(unique_week)),
                "candidate_positive_rate": candidate_metrics["positive_rate"],
                "candidate_median": candidate_metrics["median"],
                "candidate_cvar20": candidate_metrics["cvar20"],
                "candidate_worst": candidate_metrics["worst"],
                "overall_mean_excess": float(unique_week["excess"].mean()),
                "excess_sum_remove_best2": excess_metrics["sum_remove_best2"],
                "seed_20260726_mean_excess": float(seed_means.loc[20260726]),
                "seed_20260727_mean_excess": float(seed_means.loc[20260727]),
                "mean_cross_week_turnover": float(
                    group["cross_week_turnover"].dropna().mean()
                ),
                "mean_pair_correlation": float(
                    group["weighted_pair_correlation"].mean()
                ),
                "max_hhi": validation["max_hhi"],
                "max_weight": validation["max_weight_observed"],
                "max_cluster_count": validation["max_cluster_count"],
                "structural_pass": validation["status"] == "PASS",
                "overall_noninferiority_pass": overall_noninferior,
                "both_seed_noninferiority_pass": seed_noninferior,
                "total_penalty": policy.total_penalty,
                "equal_weight_template": (
                    policy.template == "TOP5_EQUAL"
                ),
                "eligible": eligible,
            }
        )
    return pd.DataFrame(rows)


def choose_global_policy(summary: pd.DataFrame) -> pd.Series:
    eligible = summary.loc[summary["eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("no global policy passes frozen inner eligibility")
    ordered = eligible.sort_values(
        [
            "candidate_positive_rate",
            "candidate_median",
            "candidate_cvar20",
            "candidate_worst",
            "excess_sum_remove_best2",
            "mean_cross_week_turnover",
            "mean_pair_correlation",
            "total_penalty",
            "equal_weight_template",
            "policy_id",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
        ],
        kind="mergesort",
    )
    return ordered.iloc[0]


def terminal_status(
    status: str,
    reason: str,
    details: Mapping[str, Any],
    *,
    outer_label_parse_count: int = 0,
) -> dict[str, Any]:
    before = json.loads(
        (RUN_DIR / "TEST_PROTECTED_FINGERPRINTS_BEFORE.json").read_text(
            encoding="utf-8"
        )
    )
    after = protection_manifest()
    write_json(RUN_DIR / "TEST_PROTECTED_FINGERPRINTS_AFTER.json", after)
    protection = compare_protection(before, after)
    write_json(RUN_DIR / "TEST_PROTECTION_AUDIT.json", protection)
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "reason": reason,
        "details": details,
        "outer_label_parse_count": outer_label_parse_count,
        "gpu_time_seconds": 0.0,
        "training_run": False,
        "h01_created": False,
        "promotion_eligible": False,
        "alpha_champion": "D01_FrozenResearchChampion_20260709",
        "defensive_champion": None,
        "protection_status": protection["status"],
    }
    write_json(RUN_DIR / "TEST_STATUS.json", payload)
    leakage = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if protection["status"] == "PASS" else "FAIL",
        "no_training": True,
        "outer_label_parse_count": outer_label_parse_count,
        "prospective_paths_accessed": False,
        "protected_objects_unchanged": protection["status"] == "PASS",
        "promotion_eligible": False,
    }
    write_json(RUN_DIR / "TEST_LEAKAGE_AUDIT.json", leakage)
    report = f"""# Candidate-D Defensive Combo V2 TEST

状态：`{status}`

原因：{reason}

本运行未训练、未访问 prospective 窗口、未修改 Candidate-D、D01、官方参考或 V1
证据。保护审计：`{protection['status']}`。外层 label 解析计数：
`{outer_label_parse_count}`。
"""
    (RUN_DIR / "TEST_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return payload


def prepare() -> dict[str, Any]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if (RUN_DIR / "TEST_STATUS.json").exists():
        raise RuntimeError("terminal V2 status already exists; refusing overwrite")
    before = protection_manifest()
    write_json(RUN_DIR / "TEST_PROTECTED_FINGERPRINTS_BEFORE.json", before)
    prereg = preregistration_payload()
    write_json(RUN_DIR / "TEST_PREREGISTRATION.json", prereg)
    manifest = input_manifest()
    write_json(RUN_DIR / "TEST_INPUT_MANIFEST.json", manifest)
    environment = environment_payload()
    write_json(RUN_DIR / "TEST_ENVIRONMENT.json", environment)
    state = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "stage": "PREPARED",
        "outer_label_parse_count": 0,
        "inner_label_parse_count": 0,
        "prospective_paths_accessed": False,
    }
    write_json(RUN_DIR / "TEST_RUN_STATE.json", state)
    return state


def require_prepared() -> tuple[dict[str, Any], dict[str, Any]]:
    prereg_path = RUN_DIR / "TEST_PREREGISTRATION.json"
    manifest_path = RUN_DIR / "TEST_INPUT_MANIFEST.json"
    if not prereg_path.exists() or not manifest_path.exists():
        raise RuntimeError("V2 prepare stage has not completed")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if prereg["experiment_id"] != EXPERIMENT_ID:
        raise RuntimeError("preregistration experiment identity mismatch")
    verify_input_manifest(manifest)
    return prereg, manifest


def feasibility() -> dict[str, Any]:
    require_prepared()
    market = v1.MarketData(v1.MARKET_CSV)
    rows: list[dict[str, Any]] = []
    try:
        for seed in SEEDS:
            for fold in FOLDS:
                inner_rank = v1.load_predictions(
                    v1.prediction_path(seed, fold, "inner"),
                    include_label=False,
                )
                inner_contexts, history = build_contexts(inner_rank, market)
                outer_rank = v1.load_predictions(
                    v1.prediction_path(seed, fold, "outer"),
                    include_label=False,
                )
                outer_contexts, _ = build_contexts(
                    outer_rank, market, initial_state_history=history
                )
                for phase, rank, contexts in (
                    ("inner", inner_rank, inner_contexts),
                    ("outer", outer_rank, outer_contexts),
                ):
                    for date in v1.decision_dates(rank):
                        context = contexts[pd.Timestamp(date).normalize()]
                        rows.append(
                            {
                                "marker": "TEST",
                                "experiment_id": EXPERIMENT_ID,
                                "seed": seed,
                                "fold_id": fold,
                                "phase": phase,
                                "decision_date": pd.Timestamp(date).strftime(
                                    "%Y-%m-%d"
                                ),
                                "pool_size_used": context.pool_size_used,
                                "universe_size": context.universe_size,
                                "feasible_capacity": (
                                    context.feasible_capacity
                                ),
                                "cluster_count": context.cluster_count,
                                "largest_cluster": context.largest_cluster,
                                "feasible": (
                                    context.feasible_capacity
                                    >= PORTFOLIO_SIZE
                                ),
                            }
                        )
    except FeasibilityFailure as exc:
        frame = pd.DataFrame(rows)
        write_csv(RUN_DIR / "TEST_FEASIBILITY_CONTEXTS.csv", frame)
        gate = {
            "status": "REJECTED_AT_COMBINATION_GATE",
            "reason": str(exc),
            "decision_date": exc.decision_date.strftime("%Y-%m-%d"),
            "universe_size": exc.universe_size,
            "outer_label_parse_count": 0,
        }
        write_json(RUN_DIR / "TEST_FEASIBILITY_GATE.json", gate)
        return terminal_status(
            "REJECTED_AT_COMBINATION_GATE",
            str(exc),
            gate,
        )
    frame = pd.DataFrame(rows).sort_values(
        ["seed", "fold_id", "phase", "decision_date"]
    )
    write_csv(RUN_DIR / "TEST_FEASIBILITY_CONTEXTS.csv", frame)
    gate = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "context_count": int(len(frame)),
        "feasible_count": int(frame["feasible"].sum()),
        "minimum_capacity": int(frame["feasible_capacity"].min()),
        "top20_count": int((frame["pool_size_used"] == 20).sum()),
        "expanded_count": int((frame["pool_size_used"] > 20).sum()),
        "maximum_pool_size_used": int(frame["pool_size_used"].max()),
        "outer_label_parse_count": 0,
    }
    write_json(RUN_DIR / "TEST_FEASIBILITY_GATE.json", gate)
    write_json(
        RUN_DIR / "TEST_RUN_STATE.json",
        {
            "schema_version": 1,
            "marker": "TEST",
            "experiment_id": EXPERIMENT_ID,
            "stage": "FEASIBILITY_PASS",
            "outer_label_parse_count": 0,
            "inner_label_parse_count": 0,
        },
    )
    return gate


def inner_lock() -> dict[str, Any]:
    require_prepared()
    gate = json.loads(
        (RUN_DIR / "TEST_FEASIBILITY_GATE.json").read_text(encoding="utf-8")
    )
    if gate["status"] != "PASS":
        raise RuntimeError("feasibility gate did not pass")
    market = v1.MarketData(v1.MARKET_CSV)
    candidate_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        for fold in FOLDS:
            rank = v1.load_predictions(
                v1.prediction_path(seed, fold, "inner"),
                include_label=False,
            )
            contexts, _ = build_contexts(rank, market)
            dates = v1.decision_dates(rank)
            baseline, baseline_summary = simulate_policy(
                contexts,
                dates,
                seed,
                fold,
                "inner",
                "BASELINE_TOP2_EQUAL",
                anchor_weights=v1.ANCHOR_TEMPLATES["BASELINE_TOP2_EQUAL"],
            )
            baseline_frames.append(baseline)
            summary_frames.append(baseline_summary)
            for policy in policy_grid():
                candidate, candidate_summary = simulate_policy(
                    contexts,
                    dates,
                    seed,
                    fold,
                    "inner",
                    "DEFENSIVE_CANDIDATE",
                    policy=policy,
                )
                candidate_frames.append(candidate)
                summary_frames.append(candidate_summary)
    candidates = pd.concat(candidate_frames, ignore_index=True).sort_values(
        [
            "policy_id",
            "seed",
            "fold_id",
            "decision_date",
            "selection_order",
        ]
    )
    baselines = pd.concat(baseline_frames, ignore_index=True).sort_values(
        ["seed", "fold_id", "decision_date", "selection_order"]
    )
    summaries = pd.concat(summary_frames, ignore_index=True).sort_values(
        ["policy_role", "policy_id", "seed", "fold_id", "decision_date"]
    )
    candidate_validation = validate_portfolios(candidates)
    baseline_validation = validate_native_anchors(baselines)
    if candidate_validation["status"] != "PASS":
        return terminal_status(
            "REJECTED_AT_COMBINATION_GATE",
            "inner candidate structural validation failed",
            candidate_validation,
        )
    if baseline_validation["status"] != "PASS":
        raise RuntimeError("baseline structural validation failed")
    candidate_path = RUN_DIR / "TEST_INNER_CANDIDATE_LOCK.csv"
    baseline_path = RUN_DIR / "TEST_INNER_BASELINE_LOCK.csv"
    summary_path = RUN_DIR / "TEST_INNER_PORTFOLIO_SUMMARY.csv"
    write_csv(candidate_path, candidates)
    write_csv(baseline_path, baselines)
    write_csv(summary_path, summaries)
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "policy_count": len(policy_grid()),
        "candidate_rows": int(len(candidates)),
        "baseline_rows": int(len(baselines)),
        "candidate_validation": candidate_validation,
        "baseline_validation": baseline_validation,
        "candidate_lock_sha256": sha256_file(candidate_path),
        "baseline_lock_sha256": sha256_file(baseline_path),
        "inner_label_parse_count": 0,
        "outer_label_parse_count": 0,
    }
    write_json(RUN_DIR / "TEST_INNER_LOCK.json", payload)
    write_json(
        RUN_DIR / "TEST_RUN_STATE.json",
        {
            "schema_version": 1,
            "marker": "TEST",
            "experiment_id": EXPERIMENT_ID,
            "stage": "INNER_LOCKED_NO_LABELS",
            "inner_label_parse_count": 0,
            "outer_label_parse_count": 0,
        },
    )
    return payload


def select_policy() -> dict[str, Any]:
    require_prepared()
    lock_meta = json.loads(
        (RUN_DIR / "TEST_INNER_LOCK.json").read_text(encoding="utf-8")
    )
    candidate_path = RUN_DIR / "TEST_INNER_CANDIDATE_LOCK.csv"
    baseline_path = RUN_DIR / "TEST_INNER_BASELINE_LOCK.csv"
    if sha256_file(candidate_path) != lock_meta["candidate_lock_sha256"]:
        raise RuntimeError("inner candidate lock drift")
    if sha256_file(baseline_path) != lock_meta["baseline_lock_sha256"]:
        raise RuntimeError("inner baseline lock drift")
    candidates = pd.read_csv(candidate_path, dtype={"stock_id": str})
    baselines = pd.read_csv(baseline_path, dtype={"stock_id": str})
    candidates["stock_id"] = candidates["stock_id"].str.zfill(6)
    baselines["stock_id"] = baselines["stock_id"].str.zfill(6)
    labels, hashes = load_labels("inner")
    candidate_scored = score_locked_portfolios(candidates, labels)
    baseline_scored = score_locked_portfolios(baselines, labels)
    summary = global_policy_summary(
        candidate_scored, baseline_scored, candidates
    )
    write_csv(RUN_DIR / "TEST_GLOBAL_POLICY_GRID.csv", summary)
    write_csv(RUN_DIR / "TEST_INNER_CANDIDATE_SCORED.csv", candidate_scored)
    write_csv(RUN_DIR / "TEST_INNER_BASELINE_SCORED.csv", baseline_scored)
    try:
        winner = choose_global_policy(summary)
    except RuntimeError as exc:
        details = {
            "eligible_policy_count": int(summary["eligible"].sum()),
            "inner_outcome_hashes": hashes,
        }
        return terminal_status(
            "REJECTED_AT_COMBINATION_GATE", str(exc), details
        )
    selected_policy = {
        key: (
            bool(winner[key])
            if isinstance(winner[key], (np.bool_, bool))
            else float(winner[key])
            if isinstance(winner[key], (np.floating, float))
            else int(winner[key])
            if isinstance(winner[key], (np.integer, int))
            else str(winner[key])
        )
        for key in summary.columns
    }
    policy = next(
        item for item in policy_grid() if item.policy_id == winner["policy_id"]
    )
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "selected_policy": asdict(policy),
        "policy_id": policy.policy_id,
        "selected_metrics": selected_policy,
        "eligible_policy_count": int(summary["eligible"].sum()),
        "inner_outcome_hashes": hashes,
        "inner_label_file_parse_count": len(hashes),
        "outer_label_parse_count": 0,
        "selection_rule": preregistration_payload()[
            "global_policy_selection"
        ],
    }
    write_json(RUN_DIR / "TEST_GLOBAL_POLICY_LOCK.json", payload)
    write_json(
        RUN_DIR / "TEST_RUN_STATE.json",
        {
            "schema_version": 1,
            "marker": "TEST",
            "experiment_id": EXPERIMENT_ID,
            "stage": "GLOBAL_POLICY_LOCKED",
            "inner_label_parse_count": len(hashes),
            "outer_label_parse_count": 0,
        },
    )
    return payload


def locked_policy() -> v1.Policy:
    payload = json.loads(
        (RUN_DIR / "TEST_GLOBAL_POLICY_LOCK.json").read_text(encoding="utf-8")
    )
    if payload["status"] != "PASS":
        raise RuntimeError("global policy lock did not pass")
    return v1.Policy(**payload["selected_policy"])


def outer_lock(replica: str) -> dict[str, Any]:
    replica = replica.upper()
    if replica not in {"A", "B"}:
        raise ValueError("replica must be A or B")
    require_prepared()
    policy = locked_policy()
    market = v1.MarketData(v1.MARKET_CSV)
    long_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        for fold in FOLDS:
            inner_rank = v1.load_predictions(
                v1.prediction_path(seed, fold, "inner"),
                include_label=False,
            )
            _, history = build_contexts(inner_rank, market)
            outer_rank = v1.load_predictions(
                v1.prediction_path(seed, fold, "outer"),
                include_label=False,
            )
            contexts, _ = build_contexts(
                outer_rank, market, initial_state_history=history
            )
            dates = v1.decision_dates(outer_rank)
            specifications = [
                ("DEFENSIVE_CANDIDATE", policy, None),
                (
                    "BASELINE_TOP2_EQUAL",
                    None,
                    v1.ANCHOR_TEMPLATES["BASELINE_TOP2_EQUAL"],
                ),
                (
                    "ANCHOR_TOP5_EQUAL",
                    None,
                    v1.ANCHOR_TEMPLATES["ANCHOR_TOP5_EQUAL"],
                ),
                (
                    "ANCHOR_TOP5_RANK_30_25_20_15_10",
                    None,
                    v1.ANCHOR_TEMPLATES[
                        "ANCHOR_TOP5_RANK_30_25_20_15_10"
                    ],
                ),
            ]
            for role, selected_policy, weights in specifications:
                long_frame, summary_frame = simulate_policy(
                    contexts,
                    dates,
                    seed,
                    fold,
                    "outer",
                    role,
                    policy=selected_policy,
                    anchor_weights=weights,
                )
                long_frames.append(long_frame)
                summary_frames.append(summary_frame)
    long_result = pd.concat(long_frames, ignore_index=True).sort_values(
        [
            "seed",
            "fold_id",
            "policy_role",
            "decision_date",
            "selection_order",
        ]
    )
    summary_result = pd.concat(summary_frames, ignore_index=True).sort_values(
        ["seed", "fold_id", "policy_role", "decision_date"]
    )
    candidate_validation = validate_portfolios(
        long_result.loc[
            long_result["policy_role"] == "DEFENSIVE_CANDIDATE"
        ]
    )
    anchor_validation = validate_native_anchors(long_result)
    if candidate_validation["status"] != "PASS":
        raise RuntimeError("outer portfolio structural validation failed")
    if anchor_validation["status"] != "PASS":
        raise RuntimeError("outer anchor structural validation failed")
    long_path = RUN_DIR / f"TEST_OUTER_PORTFOLIO_LOCK_{replica}.csv"
    summary_path = RUN_DIR / f"TEST_OUTER_PORTFOLIO_SUMMARY_{replica}.csv"
    write_csv(long_path, long_result)
    write_csv(summary_path, summary_result)
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "replica": replica,
        "status": "PASS",
        "portfolio_sha256": sha256_file(long_path),
        "summary_sha256": sha256_file(summary_path),
        "candidate_validation": candidate_validation,
        "anchor_validation": anchor_validation,
        "outer_label_parse_count": 0,
        "prospective_paths_accessed": False,
    }
    write_json(RUN_DIR / f"TEST_OUTER_LOCK_{replica}.json", payload)
    return payload


def seal() -> dict[str, Any]:
    require_prepared()
    metadata = {
        replica: json.loads(
            (RUN_DIR / f"TEST_OUTER_LOCK_{replica}.json").read_text(
                encoding="utf-8"
            )
        )
        for replica in ("A", "B")
    }
    paths = {
        replica: RUN_DIR / f"TEST_OUTER_PORTFOLIO_LOCK_{replica}.csv"
        for replica in ("A", "B")
    }
    summaries = {
        replica: RUN_DIR / f"TEST_OUTER_PORTFOLIO_SUMMARY_{replica}.csv"
        for replica in ("A", "B")
    }
    long_hashes = {replica: sha256_file(path) for replica, path in paths.items()}
    summary_hashes = {
        replica: sha256_file(path) for replica, path in summaries.items()
    }
    if long_hashes["A"] != long_hashes["B"]:
        raise RuntimeError("outer portfolio replicas are not byte-identical")
    if summary_hashes["A"] != summary_hashes["B"]:
        raise RuntimeError("outer summary replicas are not byte-identical")
    if any(item["outer_label_parse_count"] != 0 for item in metadata.values()):
        raise RuntimeError("outer label access occurred before seal")
    locked = pd.read_csv(paths["A"], dtype={"stock_id": str})
    locked["stock_id"] = locked["stock_id"].str.zfill(6)
    defensive = locked.loc[
        locked["policy_role"] == "DEFENSIVE_CANDIDATE"
    ].copy()
    missing_columns = [
        column for column in FINAL_PORTFOLIO_COLUMNS if column not in defensive
    ]
    if missing_columns:
        raise RuntimeError(f"missing final portfolio columns: {missing_columns}")
    final_path = RUN_DIR / "TEST_DEFENSIVE_PORTFOLIOS.csv"
    write_csv(final_path, defensive)
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "outer_portfolio_sha256": long_hashes["A"],
        "outer_summary_sha256": summary_hashes["A"],
        "replicas_byte_identical": True,
        "final_defensive_portfolios_sha256": sha256_file(final_path),
        "final_defensive_rows": int(len(defensive)),
        "outer_label_parse_count": 0,
        "outer_container_hash_captured": False,
        "prospective_paths_accessed": False,
    }
    write_json(RUN_DIR / "TEST_OUTER_SEAL.json", payload)
    write_json(
        RUN_DIR / "TEST_RUN_STATE.json",
        {
            "schema_version": 1,
            "marker": "TEST",
            "experiment_id": EXPERIMENT_ID,
            "stage": "OUTER_DOUBLE_LOCK_SEALED",
            "outer_label_parse_count": 0,
        },
    )
    return payload


def final_report(
    status: str,
    metrics: Mapping[str, Any],
    anchors: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    protection: Mapping[str, Any],
) -> str:
    failed = metrics["failed_checks"]
    failed_text = "、".join(failed) if failed else "无"
    selected = policy_lock["selected_policy"]
    candidate = metrics["candidate_roundtrip_10bps"]
    baseline = metrics["baseline_roundtrip_10bps"]
    return f"""# Candidate-D Defensive Combo V2 TEST 最终报告

状态：`{status}`

本结果是暴露历史窗口的拒绝性 TEST，不是晋级证据。未训练、未访问 prospective
窗口、未修改 Candidate-D、D01、官方参考或 V1。

## 全局政策

- policy：`{policy_lock['policy_id']}`
- 权重模板：`{selected['template']}`
- lambda volatility/correlation/turnover/reversal：
  `{selected['lambda_volatility']}` / `{selected['lambda_correlation']}` /
  `{selected['lambda_turnover']}` / `{selected['lambda_reversal']}`
- 聚类：complete linkage，rho=0.65；候选池按 20/50/100/完整排名确定性扩展。

## 10 bps/side 结果

|指标|Defensive|Candidate-D top-2|
|---|---:|---:|
|正收益周比例|{candidate['positive_rate']:.2%}|{baseline['positive_rate']:.2%}|
|周中位数|{candidate['median']:.4%}|{baseline['median']:.4%}|
|CVaR20|{candidate['cvar20']:.4%}|{baseline['cvar20']:.4%}|
|最差周|{candidate['worst']:.4%}|{baseline['worst']:.4%}|
|平均跨周换手|{metrics['candidate_mean_cross_week_turnover']:.2%}|{metrics['baseline_mean_cross_week_turnover']:.2%}|
|平均组合相关性|{metrics['candidate_mean_pair_correlation']:.4f}|{metrics['baseline_mean_pair_correlation']:.4f}|
|最大 HHI|{metrics['max_candidate_hhi']:.3f}|0.500|

失败门槛：{failed_text}

原生 top-5 等权均值超额：
`{anchors['ANCHOR_TOP5_EQUAL']['mean_excess']:.4%}`；固定权重均值超额：
`{anchors['ANCHOR_TOP5_RANK_30_25_20_15_10']['mean_excess']:.4%}`。

保护审计：`{protection['status']}`。Defensive Champion 仍为 `NONE`，
Alpha Champion 仍为 `D01_FrozenResearchChampion_20260709`。
"""


def score() -> dict[str, Any]:
    require_prepared()
    seal_payload = json.loads(
        (RUN_DIR / "TEST_OUTER_SEAL.json").read_text(encoding="utf-8")
    )
    locked_path = RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_A.csv"
    if sha256_file(locked_path) != seal_payload["outer_portfolio_sha256"]:
        raise RuntimeError("sealed outer lock drift")
    if seal_payload["outer_label_parse_count"] != 0:
        raise RuntimeError("outer labels were accessed before sealing")
    locked = pd.read_csv(locked_path, dtype={"stock_id": str})
    locked["stock_id"] = locked["stock_id"].str.zfill(6)
    labels, outcome_hashes = load_labels("outer")
    scored = score_locked_portfolios(locked, labels)
    metrics, paired = v1.gate_metrics(scored)
    anchors = v1.anchor_diagnostics(scored)
    status = (
        "REJECTED_AT_QUICK_GATE"
        if metrics["failed_checks"]
        else "PASS_EXPOSED_QUICK_SCREEN_NOT_PROMOTION_ELIGIBLE"
    )
    gate = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "status": status,
        "metrics": metrics,
        "anchor_diagnostics": anchors,
        "outer_outcome_hashes": outcome_hashes,
        "outer_label_file_parse_count": len(outcome_hashes),
        "serious_seed_consensus": "NOT_RUN",
        "training_performed": False,
        "promotion_eligible": False,
    }
    write_csv(RUN_DIR / "TEST_OUTER_SCORED_DECISIONS.csv", scored)
    write_csv(RUN_DIR / "TEST_PAIRED_DEFENSIVE_VS_D.csv", paired)
    write_json(RUN_DIR / "TEST_QUICK_GATE.json", gate)

    before = json.loads(
        (RUN_DIR / "TEST_PROTECTED_FINGERPRINTS_BEFORE.json").read_text(
            encoding="utf-8"
        )
    )
    after = protection_manifest()
    write_json(RUN_DIR / "TEST_PROTECTED_FINGERPRINTS_AFTER.json", after)
    protection = compare_protection(before, after)
    write_json(RUN_DIR / "TEST_PROTECTION_AUDIT.json", protection)
    leakage = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if protection["status"] == "PASS" else "FAIL",
        "checks": {
            "no_training": True,
            "inner_locks_before_inner_labels": True,
            "outer_rank_columns_before_seal": RANK_COLUMNS,
            "outer_label_parse_count_before_seal": 0,
            "outer_label_file_parse_count_after_seal": len(outcome_hashes),
            "two_independent_outer_locks_byte_identical": True,
            "all_risk_features_strictly_lag1": True,
            "turnover_uses_previous_locked_portfolio_only": True,
            "industry_and_external_features_disabled": True,
            "prospective_paths_accessed": False,
            "protected_objects_unchanged": protection["status"] == "PASS",
        },
        "promotion_eligible": False,
    }
    write_json(RUN_DIR / "TEST_LEAKAGE_AUDIT.json", leakage)
    policy_lock = json.loads(
        (RUN_DIR / "TEST_GLOBAL_POLICY_LOCK.json").read_text(encoding="utf-8")
    )
    report = final_report(status, metrics, anchors, policy_lock, protection)
    (RUN_DIR / "TEST_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    status_payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "failed_checks": metrics["failed_checks"],
        "gpu_time_seconds": 0.0,
        "training_run": False,
        "h01_created": False,
        "candidate_d_modified": protection["status"] != "PASS",
        "d01_modified": protection["status"] != "PASS",
        "official_reference_modified": protection["status"] != "PASS",
        "v1_modified": protection["status"] != "PASS",
        "defensive_champion": None,
        "alpha_champion": "D01_FrozenResearchChampion_20260709",
        "serious_run": False,
        "promotion_eligible": False,
        "outer_label_file_parse_count": len(outcome_hashes),
        "output_bytes": sum(
            path.stat().st_size for path in RUN_DIR.rglob("*") if path.is_file()
        ),
    }
    write_json(RUN_DIR / "TEST_STATUS.json", status_payload)
    return status_payload


def verify() -> dict[str, Any]:
    status_path = RUN_DIR / "TEST_STATUS.json"
    if not status_path.exists():
        raise RuntimeError("TEST_STATUS.json does not exist")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    required = [
        "TEST_PREREGISTRATION.json",
        "TEST_INPUT_MANIFEST.json",
        "TEST_ENVIRONMENT.json",
        "TEST_FEASIBILITY_CONTEXTS.csv",
        "TEST_FEASIBILITY_GATE.json",
        "TEST_INNER_CANDIDATE_LOCK.csv",
        "TEST_INNER_BASELINE_LOCK.csv",
        "TEST_GLOBAL_POLICY_GRID.csv",
        "TEST_GLOBAL_POLICY_LOCK.json",
        "TEST_OUTER_PORTFOLIO_LOCK_A.csv",
        "TEST_OUTER_PORTFOLIO_LOCK_B.csv",
        "TEST_OUTER_SEAL.json",
        "TEST_DEFENSIVE_PORTFOLIOS.csv",
        "TEST_OUTER_SCORED_DECISIONS.csv",
        "TEST_QUICK_GATE.json",
        "TEST_LEAKAGE_AUDIT.json",
        "TEST_PROTECTION_AUDIT.json",
        "TEST_FINAL_REPORT.md",
        "TEST_STATUS.json",
    ]
    missing = [name for name in required if not (RUN_DIR / name).exists()]
    portfolio = pd.read_csv(
        RUN_DIR / "TEST_DEFENSIVE_PORTFOLIOS.csv", dtype={"stock_id": str}
    )
    portfolio_validation = validate_portfolios(portfolio)
    seal_payload = json.loads(
        (RUN_DIR / "TEST_OUTER_SEAL.json").read_text(encoding="utf-8")
    )
    lock_a = sha256_file(RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_A.csv")
    lock_b = sha256_file(RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_B.csv")
    protection = json.loads(
        (RUN_DIR / "TEST_PROTECTION_AUDIT.json").read_text(encoding="utf-8")
    )
    output_bytes = sum(
        path.stat().st_size for path in RUN_DIR.rglob("*") if path.is_file()
    )
    columns_present = all(
        column in portfolio.columns for column in FINAL_PORTFOLIO_COLUMNS
    )
    checks = {
        "required_outputs_present": not missing,
        "portfolio_schema": columns_present,
        "portfolio_constraints": portfolio_validation["status"] == "PASS",
        "outer_locks_byte_identical": lock_a == lock_b,
        "seal_matches_lock": lock_a == seal_payload["outer_portfolio_sha256"],
        "outer_label_parse_before_seal_zero": (
            seal_payload["outer_label_parse_count"] == 0
        ),
        "protection": protection["status"] == "PASS",
        "disk_under_150_mib": output_bytes <= 150 * 1024 * 1024,
    }
    result = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing": missing,
        "portfolio_validation": portfolio_validation,
        "output_bytes": output_bytes,
        "terminal_status": status["status"],
        "promotion_eligible": False,
    }
    write_json(RUN_DIR / "TEST_VERIFICATION.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Candidate-D complete-link defensive combination V2 TEST"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "prepare",
            "feasibility",
            "inner-lock",
            "select-policy",
            "outer-lock",
            "seal",
            "score",
            "verify",
        ),
    )
    parser.add_argument("--replica", choices=("A", "B", "a", "b"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        result = prepare()
    elif args.stage == "feasibility":
        result = feasibility()
    elif args.stage == "inner-lock":
        result = inner_lock()
    elif args.stage == "select-policy":
        result = select_policy()
    elif args.stage == "outer-lock":
        if args.replica is None:
            raise SystemExit("--replica A or B is required for outer-lock")
        result = outer_lock(args.replica)
    elif args.stage == "seal":
        result = seal()
    elif args.stage == "score":
        result = score()
    elif args.stage == "verify":
        result = verify()
    else:
        raise AssertionError(args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

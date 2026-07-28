from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


WORKSPACE = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "TEST_D_DEFENSIVE_COMBO_V1_20260728"
RUN_DIR = (
    WORKSPACE
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "automation_runs"
    / EXPERIMENT_ID
)
H0_ROOT = (
    WORKSPACE
    / "Experimental_WorkSpace"
    / "candidates_H"
    / "automation_runs"
    / "TEST_H_QUICK"
    / "TEST_H0_CONTROL"
)
MARKET_CSV = (
    WORKSPACE
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "D01_FrozenResearchChampion_20260709"
    / "data"
    / "train.csv"
)

SEEDS = (20260726, 20260727)
FOLDS = ("2025Q1", "2025Q2", "2025Q3", "2026Q1")
FOLD_SPECS = {
    "2025Q1": {
        "inner_start": "2024-10-01",
        "inner_end": "2024-12-24",
        "train_label_end": "2024-12-24",
        "outer_start": "2025-01-03",
        "outer_end": "2025-03-31",
    },
    "2025Q2": {
        "inner_start": "2025-01-03",
        "inner_end": "2025-03-24",
        "train_label_end": "2025-03-24",
        "outer_start": "2025-04-01",
        "outer_end": "2025-06-30",
    },
    "2025Q3": {
        "inner_start": "2025-04-01",
        "inner_end": "2025-06-23",
        "train_label_end": "2025-06-23",
        "outer_start": "2025-07-01",
        "outer_end": "2025-09-30",
    },
    "2026Q1": {
        "inner_start": "2025-07-01",
        "inner_end": "2025-09-23",
        "train_label_end": "2025-09-23",
        "outer_start": "2026-01-05",
        "outer_end": "2026-03-27",
    },
}

LAMBDA_VOLATILITY = (0.00, 0.05, 0.10, 0.15)
LAMBDA_CORRELATION = (0.00, 0.05, 0.10, 0.15)
LAMBDA_TURNOVER = (0.00, 0.025, 0.05)
LAMBDA_REVERSAL = (0.00, 0.05, 0.10)
WEIGHT_TEMPLATES = {
    "TOP5_EQUAL": (0.20, 0.20, 0.20, 0.20, 0.20),
    "TOP5_RANK_30_25_20_15_10": (0.30, 0.25, 0.20, 0.15, 0.10),
}
ANCHOR_TEMPLATES = {
    "BASELINE_TOP2_EQUAL": (0.50, 0.50),
    "ANCHOR_TOP5_EQUAL": WEIGHT_TEMPLATES["TOP5_EQUAL"],
    "ANCHOR_TOP5_RANK_30_25_20_15_10": WEIGHT_TEMPLATES[
        "TOP5_RANK_30_25_20_15_10"
    ],
}

CANDIDATE_POOL_SIZE = 20
PORTFOLIO_SIZE = 5
MAX_CLUSTER_HOLDINGS = 2
CORRELATION_LOOKBACK = 60
CORRELATION_MIN_OBS = 45
CORRELATION_MISSING_VALUE = 0.65
CORRELATION_SHRINKAGE = 0.25
CORRELATION_CLUSTER_CUT = 0.65
VOLATILITY_LOOKBACK = 20
VOLATILITY_MIN_OBS = 15
STATE_MIN_HISTORY = 20
STATE_MULTIPLIER_MIN = 0.75
STATE_MULTIPLIER_MAX = 1.25
NONINFERIORITY_MARGIN = 0.0020
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_BLOCK_LENGTH = 4
FLOAT_TOLERANCE = 1e-12
EXPOSED_ONLY = True
PROSPECTIVE_PATH_TOKEN = "TEST_D_PROSPECTIVE_8W_20260727"

PROTECTED_TREES = {
    "candidate_d": WORKSPACE / "candidate" / "Candidate-D",
    "d01": (
        WORKSPACE
        / "Experimental_WorkSpace"
        / "candidates_D"
        / "D01_FrozenResearchChampion_20260709"
    ),
    "official_reference": WORKSPACE / "candidate" / "THU-BDC2026-main",
}

STATE_FIELDS = (
    "breadth5_adverse",
    "breadth20_adverse",
    "volatility_ratio",
    "mean_top20_correlation",
    "leader_reversal_strength",
    "cross_sectional_dispersion",
    "market_proxy_trend_adverse",
)


@dataclass(frozen=True)
class Policy:
    template: str
    lambda_volatility: float
    lambda_correlation: float
    lambda_turnover: float
    lambda_reversal: float

    @property
    def policy_id(self) -> str:
        prefix = "EQ" if self.template == "TOP5_EQUAL" else "FX"
        return (
            f"{prefix}_V{self.lambda_volatility:.3f}"
            f"_C{self.lambda_correlation:.3f}"
            f"_T{self.lambda_turnover:.3f}"
            f"_R{self.lambda_reversal:.3f}"
        )

    @property
    def total_penalty(self) -> float:
        return (
            self.lambda_volatility
            + self.lambda_correlation
            + self.lambda_turnover
            + self.lambda_reversal
        )


@dataclass
class DayContext:
    decision_date: pd.Timestamp
    candidates: pd.DataFrame
    correlation: pd.DataFrame
    raw_state: dict[str, float]
    regime_multiplier: float


class NoEligibleInnerPolicy(RuntimeError):
    def __init__(
        self,
        seed: int,
        fold: str,
        summaries: list[dict[str, Any]],
    ) -> None:
        super().__init__(
            f"no preregistered inner policy passes for seed={seed}, fold={fold}"
        )
        self.seed = seed
        self.fold = fold
        self.summaries = summaries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        float_format="%.12g",
        lineterminator="\n",
    )


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def tree_manifest(path: Path) -> dict[str, Any]:
    root = path.resolve()
    records: list[dict[str, Any]] = []
    for file_path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        records.append(
            {
                "relative_path": file_path.relative_to(root).as_posix(),
                "bytes": int(file_path.stat().st_size),
                "sha256": sha256_file(file_path),
            }
        )
    canonical = "\n".join(
        f"{item['relative_path']}\t{item['bytes']}\t{item['sha256']}"
        for item in records
    ).encode("utf-8")
    return {
        "path": relative_path(root),
        "file_count": len(records),
        "bytes": int(sum(item["bytes"] for item in records)),
        "tree_sha256": sha256_bytes(canonical),
        "files": records,
    }


def protected_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "algorithm": "sorted(relative_path<TAB>bytes<TAB>sha256), UTF-8, SHA-256",
        "trees": {
            name: tree_manifest(path) for name, path in PROTECTED_TREES.items()
        },
    }


def compare_protected(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    mismatches: list[dict[str, str]] = []
    before_trees = before["trees"]
    after_trees = after["trees"]
    for name in PROTECTED_TREES:
        left = before_trees[name]["tree_sha256"]
        right = after_trees[name]["tree_sha256"]
        if left != right:
            mismatches.append(
                {"tree": name, "before_sha256": left, "after_sha256": right}
            )
    return {
        "status": "PASS" if not mismatches else "FAIL_PROTECTED_TREE_DRIFT",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def prediction_path(seed: int, fold: str, phase: str) -> Path:
    return (
        H0_ROOT
        / f"S{seed}"
        / fold
        / phase
        / "best_val_predictions.csv"
    )


def companion_path(seed: int, fold: str, phase: str, name: str) -> Path:
    return H0_ROOT / f"S{seed}" / fold / phase / name


def load_predictions(path: Path, include_label: bool) -> pd.DataFrame:
    columns = ["marker", "sample_end_date", "stock_id", "score"]
    if include_label:
        columns.append("label")
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype={"stock_id": str},
        parse_dates=["sample_end_date"],
    )
    frame["sample_end_date"] = frame["sample_end_date"].dt.normalize()
    frame["stock_id"] = frame["stock_id"].str.zfill(6)
    if frame.duplicated(["sample_end_date", "stock_id"]).any():
        raise ValueError(f"duplicate prediction keys in {path}")
    if set(frame["marker"].dropna().unique()) != {"TEST"}:
        raise ValueError(f"non-TEST prediction rows in {path}")
    return frame


def decision_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    return [
        pd.Timestamp(value).normalize()
        for value in sorted(frame["sample_end_date"].dropna().unique())[::5]
    ]


def percentile_risk(values: pd.Series, missing_worst: bool = True) -> pd.Series:
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


class MarketData:
    def __init__(self, path: Path) -> None:
        source = pd.read_csv(
            path,
            usecols=["股票代码", "日期", "收盘"],
            dtype={"股票代码": str},
            parse_dates=["日期"],
        )
        source["股票代码"] = source["股票代码"].str.zfill(6)
        source["日期"] = source["日期"].dt.normalize()
        if source.duplicated(["日期", "股票代码"]).any():
            raise ValueError("market data contains duplicate date/stock rows")
        close = (
            source.pivot(index="日期", columns="股票代码", values="收盘")
            .sort_index()
            .sort_index(axis=1)
        )
        log_returns = np.log(close.where(close > 0.0)).diff()
        lower = log_returns.quantile(0.01, axis=1)
        upper = log_returns.quantile(0.99, axis=1)
        clipped = log_returns.clip(lower=lower, upper=upper, axis=0)
        self.close = close
        self.returns = clipped
        self.sum5 = clipped.rolling(5, min_periods=4).sum()
        self.sum20 = clipped.rolling(20, min_periods=15).sum()
        self.std5 = clipped.rolling(5, min_periods=4).std(ddof=0)
        self.std20 = clipped.rolling(
            VOLATILITY_LOOKBACK, min_periods=VOLATILITY_MIN_OBS
        ).std(ddof=0)

    def _asof_position(self, decision_date: pd.Timestamp) -> int:
        position = int(self.returns.index.searchsorted(decision_date, side="left")) - 1
        if position < CORRELATION_LOOKBACK:
            raise ValueError(
                f"insufficient lag-1 market history before {decision_date.date()}"
            )
        return position

    def snapshot(
        self, decision_date: pd.Timestamp, stocks: Sequence[str]
    ) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict[str, float]]:
        position = self._asof_position(decision_date)
        available = [stock for stock in stocks if stock in self.returns.columns]
        window = self.returns.iloc[
            position - CORRELATION_LOOKBACK + 1 : position + 1
        ].reindex(columns=stocks)
        raw_corr = window.corr(min_periods=CORRELATION_MIN_OBS)
        corr_values = raw_corr.to_numpy(dtype=float)
        off_diagonal = corr_values[
            ~np.eye(len(stocks), dtype=bool) & np.isfinite(corr_values)
        ]
        median_corr = float(np.median(off_diagonal)) if off_diagonal.size else 0.0
        shrunk = (1.0 - CORRELATION_SHRINKAGE) * raw_corr + (
            CORRELATION_SHRINKAGE * median_corr
        )
        shrunk = shrunk.fillna(CORRELATION_MISSING_VALUE).clip(-1.0, 1.0)
        np.fill_diagonal(shrunk.values, 1.0)

        volatility = self.std20.iloc[position].reindex(stocks)
        recent5 = self.sum5.iloc[position].reindex(stocks)
        total20 = self.sum20.iloc[position].reindex(stocks)
        lead15 = total20 - recent5
        lead_rank = percentile_risk(lead15, missing_worst=False)
        reversal_down = (-recent5).clip(lower=0.0)
        reversal_down_rank = percentile_risk(reversal_down, missing_worst=True)
        reversal_raw = lead_rank * reversal_down_rank
        reversal_rank = percentile_risk(reversal_raw, missing_worst=True)

        market_sum5 = self.sum5.iloc[position].dropna()
        market_sum20 = self.sum20.iloc[position].dropna()
        market_std5 = self.std5.iloc[position]
        market_std20 = self.std20.iloc[position]
        volatility_ratio = (
            (market_std5 / market_std20.replace(0.0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        top_sum5 = recent5.dropna()
        top_sum20 = total20.dropna()
        dispersion = float(
            np.median(np.abs(market_sum5 - np.median(market_sum5)))
        )
        mean_corr = (
            float(np.mean(off_diagonal)) if off_diagonal.size else 0.0
        )
        raw_state = {
            "breadth5_adverse": 1.0
            - 0.5
            * (
                float((market_sum5 > 0.0).mean())
                + float((top_sum5 > 0.0).mean())
            ),
            "breadth20_adverse": 1.0
            - 0.5
            * (
                float((market_sum20 > 0.0).mean())
                + float((top_sum20 > 0.0).mean())
            ),
            "volatility_ratio": (
                float(volatility_ratio.median())
                if not volatility_ratio.empty
                else 1.0
            ),
            "mean_top20_correlation": mean_corr,
            "leader_reversal_strength": float(reversal_raw.mean()),
            "cross_sectional_dispersion": dispersion,
            "market_proxy_trend_adverse": -float(market_sum20.median()),
        }
        return volatility, reversal_rank, shrunk, raw_state


def canonical_clusters(correlation: pd.DataFrame) -> dict[str, str]:
    stocks = list(correlation.index.astype(str))
    if len(stocks) == 1:
        return {stocks[0]: stocks[0]}
    values = correlation.to_numpy(dtype=float)
    distance = np.sqrt(np.clip((1.0 - values) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=True)
    tree = linkage(condensed, method="average", optimal_ordering=True)
    threshold = math.sqrt((1.0 - CORRELATION_CLUSTER_CUT) / 2.0)
    raw_labels = fcluster(tree, t=threshold, criterion="distance")
    members: dict[int, list[str]] = {}
    for stock, label in zip(stocks, raw_labels, strict=True):
        members.setdefault(int(label), []).append(stock)
    canonical = {
        label: min(group) for label, group in members.items()
    }
    return {
        stock: canonical[int(label)]
        for stock, label in zip(stocks, raw_labels, strict=True)
    }


def state_percentiles(
    raw_state: Mapping[str, float],
    history: Mapping[str, list[float]],
) -> dict[str, float]:
    percentiles: dict[str, float] = {}
    for field in STATE_FIELDS:
        past = np.asarray(history[field], dtype=np.float64)
        past = past[np.isfinite(past)]
        current = float(raw_state[field])
        if past.size < STATE_MIN_HISTORY or not np.isfinite(current):
            percentiles[field] = 0.5
            continue
        less = float(np.mean(past < current))
        equal = float(np.mean(past == current))
        percentiles[field] = less + 0.5 * equal
    return percentiles


def build_contexts(
    rank_only: pd.DataFrame,
    market: MarketData,
    initial_state_history: Mapping[str, Sequence[float]] | None = None,
) -> tuple[dict[pd.Timestamp, DayContext], dict[str, list[float]]]:
    history = {
        field: list(initial_state_history[field]) if initial_state_history else []
        for field in STATE_FIELDS
    }
    contexts: dict[pd.Timestamp, DayContext] = {}
    for decision_date in sorted(rank_only["sample_end_date"].unique()):
        date = pd.Timestamp(decision_date).normalize()
        day = (
            rank_only.loc[rank_only["sample_end_date"] == date]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["score"])
            .sort_values(["score", "stock_id"], ascending=[False, True])
            .head(CANDIDATE_POOL_SIZE)
            .copy()
        )
        if len(day) < PORTFOLIO_SIZE:
            raise ValueError(f"fewer than five ranked stocks on {date.date()}")
        day = day.reset_index(drop=True)
        day["d_rank_score"] = np.linspace(1.0, 0.0, len(day))
        stocks = day["stock_id"].astype(str).tolist()
        volatility, reversal, correlation, raw_state = market.snapshot(
            date, stocks
        )
        day["volatility_rank"] = percentile_risk(
            volatility.reset_index(drop=True), missing_worst=True
        )
        day["reversal_risk_rank"] = reversal.to_numpy(dtype=float)
        clusters = canonical_clusters(correlation)
        day["cluster_id"] = day["stock_id"].map(clusters)
        percentiles = state_percentiles(raw_state, history)
        regime_multiplier = STATE_MULTIPLIER_MIN + (
            STATE_MULTIPLIER_MAX - STATE_MULTIPLIER_MIN
        ) * float(np.mean(list(percentiles.values())))
        contexts[date] = DayContext(
            decision_date=date,
            candidates=day,
            correlation=correlation,
            raw_state={key: float(value) for key, value in raw_state.items()},
            regime_multiplier=float(regime_multiplier),
        )
        for field in STATE_FIELDS:
            history[field].append(float(raw_state[field]))
    return contexts, history


def policy_grid() -> list[Policy]:
    return [
        Policy(template, vol, corr, turnover, reversal)
        for template, vol, corr, turnover, reversal in itertools.product(
            WEIGHT_TEMPLATES,
            LAMBDA_VOLATILITY,
            LAMBDA_CORRELATION,
            LAMBDA_TURNOVER,
            LAMBDA_REVERSAL,
        )
    ]


def select_defensive(
    context: DayContext,
    policy: Policy,
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
            if selected_stocks:
                max_corr = max(
                    float(context.correlation.loc[stock, prior])
                    for prior in selected_stocks
                )
            else:
                max_corr = 0.0
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
    weights = WEIGHT_TEMPLATES[policy.template]
    for item, weight in zip(selected, weights, strict=True):
        item["weight"] = float(weight)
    return selected


def select_anchor(context: DayContext, weights: Sequence[float]) -> list[dict[str, Any]]:
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


def transition_turnover(
    current: Mapping[str, float], previous: Mapping[str, float]
) -> float:
    if not previous:
        return 1.0
    stocks = set(current) | set(previous)
    return 0.5 * sum(
        abs(float(current.get(stock, 0.0)) - float(previous.get(stock, 0.0)))
        for stock in stocks
    )


def weighted_pair_correlation(
    selected: Sequence[Mapping[str, Any]], correlation: pd.DataFrame
) -> float:
    numerator = 0.0
    denominator = 0.0
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            left_stock = str(selected[left]["stock_id"])
            right_stock = str(selected[right]["stock_id"])
            product = float(selected[left]["weight"]) * float(
                selected[right]["weight"]
            )
            numerator += product * float(
                correlation.loc[left_stock, right_stock]
            )
            denominator += product
    return numerator / denominator if denominator else 0.0


def simulate_policy(
    contexts: Mapping[pd.Timestamp, DayContext],
    dates: Sequence[pd.Timestamp],
    seed: int,
    fold: str,
    phase: str,
    role: str,
    policy: Policy | None = None,
    anchor_weights: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    previous: dict[str, float] = {}
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for date in dates:
        context = contexts[pd.Timestamp(date).normalize()]
        if policy is not None:
            selected = select_defensive(context, policy, previous)
            policy_id = policy.policy_id
        elif anchor_weights is not None:
            selected = select_anchor(context, anchor_weights)
            policy_id = role
        else:
            raise ValueError("policy or anchor_weights is required")
        if len(selected) != len(anchor_weights or WEIGHT_TEMPLATES[policy.template]):
            raise ValueError(
                f"portfolio selection failed for {role} on {date.date()}"
            )
        current = {
            str(item["stock_id"]): float(item["weight"]) for item in selected
        }
        trade_turnover = transition_turnover(current, previous)
        cross_week_turnover = trade_turnover if previous else np.nan
        pair_correlation = weighted_pair_correlation(
            selected, context.correlation
        )
        hhi = float(sum(weight * weight for weight in current.values()))
        max_cluster_count = max(
            pd.Series([item["cluster_id"] for item in selected]).value_counts()
        )
        for item in selected:
            long_rows.append(
                {
                    "marker": "TEST",
                    "experiment_id": EXPERIMENT_ID,
                    "seed": seed,
                    "fold_id": fold,
                    "phase": phase,
                    "policy_role": role,
                    "policy_id": policy_id,
                    "decision_date": date.strftime("%Y-%m-%d"),
                    "regime_multiplier": context.regime_multiplier,
                    "trade_turnover": trade_turnover,
                    "cross_week_turnover": cross_week_turnover,
                    "weighted_pair_correlation": pair_correlation,
                    "hhi": hhi,
                    "max_cluster_count": int(max_cluster_count),
                    **item,
                }
            )
        summary_rows.append(
            {
                "marker": "TEST",
                "experiment_id": EXPERIMENT_ID,
                "seed": seed,
                "fold_id": fold,
                "phase": phase,
                "policy_role": role,
                "policy_id": policy_id,
                "decision_date": date.strftime("%Y-%m-%d"),
                "regime_multiplier": context.regime_multiplier,
                "trade_turnover": trade_turnover,
                "cross_week_turnover": cross_week_turnover,
                "weighted_pair_correlation": pair_correlation,
                "hhi": hhi,
                "max_weight": max(current.values()),
                "weight_sum": sum(current.values()),
                "stock_count": len(current),
                "max_cluster_count": int(max_cluster_count),
                "holdings": "|".join(current),
                "weights": "|".join(f"{value:.8f}" for value in current.values()),
            }
        )
        previous = current
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def score_locked_portfolios(
    locked_long: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    working = locked_long.merge(
        labels,
        on=["seed", "fold_id", "decision_date", "stock_id"],
        how="left",
        validate="one_to_one",
    )
    if working["label"].isna().any():
        missing = working.loc[
            working["label"].isna(),
            ["seed", "fold_id", "decision_date", "stock_id"],
        ]
        raise ValueError(f"missing labels for locked holdings: {missing.to_dict('records')}")
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


def weekly_metrics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("cannot compute metrics on an empty series")
    tail_count = max(1, int(np.ceil(array.size * 0.20)))
    ordered_desc = np.sort(array)[::-1]
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "positive_rate": float(np.mean(array > 0.0)),
        "std": float(array.std(ddof=0)),
        "cvar20": float(np.sort(array)[:tail_count].mean()),
        "worst": float(array.min()),
        "sum_remove_best1": float(ordered_desc[1:].sum()),
        "sum_remove_best2": float(ordered_desc[2:].sum()),
    }


def max_drawdown(frame: pd.DataFrame, return_column: str) -> float:
    drawdowns: list[float] = []
    for _, group in frame.groupby("seed", sort=True):
        ordered = group.assign(
            _date=pd.to_datetime(group["decision_date"])
        ).sort_values("_date")
        equity = (1.0 + ordered[return_column].astype(float)).cumprod()
        drawdown = 1.0 - equity / equity.cummax()
        drawdowns.append(float(drawdown.max()))
    return max(drawdowns)


def moving_block_bootstrap_lower(
    values: Sequence[float],
    reps: int = BOOTSTRAP_REPS,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < block_length:
        raise ValueError("insufficient values for moving block bootstrap")
    rng = np.random.default_rng(seed)
    starts = np.arange(array.size - block_length + 1)
    block_count = int(np.ceil(array.size / block_length))
    means = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        selected_starts = rng.choice(starts, size=block_count, replace=True)
        sample = np.concatenate(
            [array[start : start + block_length] for start in selected_starts]
        )[: array.size]
        means[index] = sample.mean()
    return float(np.quantile(means, 0.025))


def inner_candidate_metrics(
    candidate: pd.DataFrame, baseline: pd.DataFrame
) -> dict[str, float]:
    paired = candidate.merge(
        baseline[
            [
                "decision_date",
                "portfolio_return_gross",
            ]
        ],
        on="decision_date",
        how="inner",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    paired["excess"] = (
        paired["portfolio_return_gross_candidate"]
        - paired["portfolio_return_gross_baseline"]
    )
    candidate_stats = weekly_metrics(paired["portfolio_return_gross_candidate"])
    excess_stats = weekly_metrics(paired["excess"])
    return {
        "decision_count": int(len(paired)),
        "candidate_positive_rate": candidate_stats["positive_rate"],
        "candidate_median": candidate_stats["median"],
        "candidate_cvar20": candidate_stats["cvar20"],
        "candidate_worst": candidate_stats["worst"],
        "mean_excess": float(paired["excess"].mean()),
        "excess_sum_remove_best2": excess_stats["sum_remove_best2"],
        "mean_cross_week_turnover": float(
            candidate["cross_week_turnover"].dropna().mean()
        ),
        "mean_pair_correlation": float(
            candidate["weighted_pair_correlation"].mean()
        ),
        "max_hhi": float(candidate["hhi"].max()),
        "max_weight": 0.0,
        "all_fully_invested": True,
    }


def load_inner_labels(path: Path, seed: int, fold: str) -> pd.DataFrame:
    frame = load_predictions(path, include_label=True)
    return frame.assign(
        seed=seed,
        fold_id=fold,
        decision_date=frame["sample_end_date"].dt.strftime("%Y-%m-%d"),
    )[["seed", "fold_id", "decision_date", "stock_id", "label"]]


def input_manifest() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            for phase in ("inner", "outer"):
                for name in (
                    "best_val_predictions.csv",
                    "config.json",
                    "split_metadata.json",
                    "scalers.json",
                ):
                    path = companion_path(seed, fold, phase, name)
                    files.append(
                        {
                            "role": f"h0_{phase}_{name}",
                            "seed": seed,
                            "fold_id": fold,
                            "path": relative_path(path),
                            "bytes": int(path.stat().st_size),
                            "sha256": sha256_file(path),
                        }
                    )
    extra_paths = {
        "d01_market_csv": MARKET_CSV,
        "candidate_d_model": (
            WORKSPACE / "candidate" / "Candidate-D" / "model" / "final_model.pth"
        ),
        "candidate_d_scalers": (
            WORKSPACE / "candidate" / "Candidate-D" / "model" / "scalers.json"
        ),
        "implementation": Path(__file__).resolve(),
    }
    for role, path in extra_paths.items():
        files.append(
            {
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
        "files": files,
        "prospective_path_accessed": False,
    }


def preregistration_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "submission_eligible": False,
        "training_authorized": False,
        "training_performed": False,
        "identity": {
            "alpha_champion": "D01_FrozenResearchChampion_20260709",
            "submission_identity": "candidate/Candidate-D",
            "defensive_champion": None,
            "challenger": EXPERIMENT_ID,
            "h01_created": False,
        },
        "sources": {
            "rankings": (
                "Existing H0/D01-program nested Quick full rankings; outer "
                "labels are blocked until portfolio lock."
            ),
            "market_data": relative_path(MARKET_CSV),
            "industry_data_enabled": False,
            "external_data_enabled": False,
            "prospective_windows_accessed": False,
        },
        "folds": FOLD_SPECS,
        "seeds": list(SEEDS),
        "decision_frequency": "every fifth available prediction date, starting first",
        "portfolio": {
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "stock_count": PORTFOLIO_SIZE,
            "weight_templates": WEIGHT_TEMPLATES,
            "weight_sum": 1.0,
            "max_weight": 0.30,
            "max_hhi": 0.30,
            "cash_allowed": False,
        },
        "risk_features": {
            "raw_cutoff": "strictly before decision date (lag-1)",
            "volatility_lookback": VOLATILITY_LOOKBACK,
            "correlation_lookback": CORRELATION_LOOKBACK,
            "correlation_min_observations": CORRELATION_MIN_OBS,
            "correlation_shrinkage_to_cross_sectional_median": CORRELATION_SHRINKAGE,
            "correlation_missing_value": CORRELATION_MISSING_VALUE,
            "correlation_cluster_cut": CORRELATION_CLUSTER_CUT,
            "cluster_method": "average-linkage on sqrt((1-rho)/2)",
            "max_cluster_holdings": MAX_CLUSTER_HOLDINGS,
            "state_fields": STATE_FIELDS,
            "state_min_history": STATE_MIN_HISTORY,
            "state_multiplier_range": [
                STATE_MULTIPLIER_MIN,
                STATE_MULTIPLIER_MAX,
            ],
            "market_trend_note": (
                "Uses the lag-1 equal-universe market proxy because no verified "
                "point-in-time index series is present in the frozen D01 snapshot."
            ),
        },
        "grid": {
            "lambda_volatility": LAMBDA_VOLATILITY,
            "lambda_correlation": LAMBDA_CORRELATION,
            "lambda_turnover": LAMBDA_TURNOVER,
            "lambda_reversal": LAMBDA_REVERSAL,
            "weight_templates": list(WEIGHT_TEMPLATES),
            "variant_count": len(policy_grid()),
            "no_expansion_after_failure": True,
        },
        "tie_break": [
            "higher DefensiveScore",
            "higher D rank score",
            "lower volatility rank",
            "lower max correlation to selected names",
            "already held in previous locked portfolio",
            "lower stock code",
        ],
        "inner_selection_priority": [
            "positive-week rate",
            "weekly median",
            "CVaR20",
            "worst week",
            "excess after removing best two",
            "lower turnover",
            "lower correlation",
            "smaller total penalty",
            "equal-weight template",
        ],
        "noninferiority": {
            "margin_per_five_day_period": NONINFERIORITY_MARGIN,
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "moving_block_length": BOOTSTRAP_BLOCK_LENGTH,
            "outer_gate": "paired mean excess 95% lower bound > -0.0020",
        },
        "cost_bps_per_side": [0, 10, 20],
        "outcome_protocol": {
            "inner_labels": "selection only",
            "outer_labels": "score only after double-run byte-identical lock",
            "outer_label_parse_count_before_lock": 0,
            "known_outer_authority": "rejection only; never promotion",
            "blacklisted_exposed_dates": ["2026-07-06", "2026-07-24"],
        },
        "stop_rules": {
            "inner_no_eligible_policy": "REJECTED_AT_COMBINATION_GATE",
            "outer_any_key_gate_failure": "REJECTED_AT_QUICK_GATE",
            "pass_label": "PASS_EXPOSED_QUICK_SCREEN",
            "training_auto_start": False,
            "candidate_auto_promotion": False,
        },
    }


def policy_priority(metrics: Mapping[str, float], policy: Policy) -> tuple[Any, ...]:
    return (
        metrics["candidate_positive_rate"],
        metrics["candidate_median"],
        metrics["candidate_cvar20"],
        metrics["candidate_worst"],
        metrics["excess_sum_remove_best2"],
        -metrics["mean_cross_week_turnover"],
        -metrics["mean_pair_correlation"],
        -policy.total_penalty,
        1 if policy.template == "TOP5_EQUAL" else 0,
        policy.policy_id,
    )


def select_inner_policy(
    seed: int,
    fold: str,
    market: MarketData,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float]]]:
    path = prediction_path(seed, fold, "inner")
    predictions = load_predictions(path, include_label=True)
    rank_only = predictions.drop(columns=["label"])
    contexts, state_history = build_contexts(rank_only, market)
    dates = decision_dates(rank_only)
    labels = predictions.assign(
        seed=seed,
        fold_id=fold,
        decision_date=predictions["sample_end_date"].dt.strftime("%Y-%m-%d"),
    )[["seed", "fold_id", "decision_date", "stock_id", "label"]]
    baseline_long, _ = simulate_policy(
        contexts,
        dates,
        seed,
        fold,
        "inner",
        "BASELINE_TOP2_EQUAL",
        anchor_weights=ANCHOR_TEMPLATES["BASELINE_TOP2_EQUAL"],
    )
    baseline = score_locked_portfolios(baseline_long, labels)
    summaries: list[dict[str, Any]] = []
    eligible: list[tuple[tuple[Any, ...], Policy, dict[str, float]]] = []
    for policy in policy_grid():
        try:
            candidate_long, _ = simulate_policy(
                contexts,
                dates,
                seed,
                fold,
                "inner",
                "DEFENSIVE_CANDIDATE",
                policy=policy,
            )
            candidate = score_locked_portfolios(candidate_long, labels)
            metrics = inner_candidate_metrics(candidate, baseline)
            metrics["max_weight"] = max(WEIGHT_TEMPLATES[policy.template])
            metrics["all_fully_invested"] = bool(
                np.allclose(
                    candidate_long.groupby("decision_date")["weight"].sum(),
                    1.0,
                    atol=FLOAT_TOLERANCE,
                )
            )
            structural = (
                metrics["max_hhi"] <= 0.30 + FLOAT_TOLERANCE
                and metrics["max_weight"] <= 0.30 + FLOAT_TOLERANCE
                and metrics["all_fully_invested"]
                and int(candidate_long["max_cluster_count"].max())
                <= MAX_CLUSTER_HOLDINGS
            )
            noninferiority = metrics["mean_excess"] >= -NONINFERIORITY_MARGIN
            is_eligible = structural and noninferiority
            error = ""
        except Exception as exc:  # preserves failed grid evidence
            metrics = {
                "decision_count": 0,
                "candidate_positive_rate": float("-inf"),
                "candidate_median": float("-inf"),
                "candidate_cvar20": float("-inf"),
                "candidate_worst": float("-inf"),
                "mean_excess": float("-inf"),
                "excess_sum_remove_best2": float("-inf"),
                "mean_cross_week_turnover": float("inf"),
                "mean_pair_correlation": float("inf"),
                "max_hhi": float("inf"),
                "max_weight": float("inf"),
                "all_fully_invested": False,
            }
            structural = False
            noninferiority = False
            is_eligible = False
            error = f"{type(exc).__name__}: {exc}"
        summary = {
            "marker": "TEST",
            "experiment_id": EXPERIMENT_ID,
            "seed": seed,
            "fold_id": fold,
            **asdict(policy),
            "policy_id": policy.policy_id,
            **metrics,
            "structural_pass": structural,
            "inner_noninferiority_pass": noninferiority,
            "eligible": is_eligible,
            "error": error,
        }
        summaries.append(summary)
        if is_eligible:
            eligible.append((policy_priority(metrics, policy), policy, metrics))
    if not eligible:
        raise NoEligibleInnerPolicy(seed, fold, summaries)
    _, winner, winner_metrics = max(eligible, key=lambda item: item[0])
    lock = {
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "fold_id": fold,
        "selected_policy": asdict(winner),
        "policy_id": winner.policy_id,
        "inner_metrics": winner_metrics,
        "inner_prediction_path": relative_path(path),
        "inner_prediction_sha256": sha256_file(path),
        "selection_rule": preregistration_payload()["inner_selection_priority"],
        "outer_outcome_read": False,
    }
    return lock, summaries, state_history


def policy_from_lock(lock: Mapping[str, Any]) -> Policy:
    return Policy(**lock["selected_policy"])


def generate_outer_lock(
    policy_locks: Sequence[Mapping[str, Any]],
    market: MarketData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    lock_by_key = {
        (int(item["seed"]), str(item["fold_id"])): item for item in policy_locks
    }
    for seed in SEEDS:
        for fold in FOLDS:
            inner_rank = load_predictions(
                prediction_path(seed, fold, "inner"), include_label=False
            )
            _, state_history = build_contexts(inner_rank, market)
            outer_rank = load_predictions(
                prediction_path(seed, fold, "outer"), include_label=False
            )
            contexts, _ = build_contexts(
                outer_rank, market, initial_state_history=state_history
            )
            dates = decision_dates(outer_rank)
            selected_policy = policy_from_lock(lock_by_key[(seed, fold)])
            specifications = [
                (
                    "DEFENSIVE_CANDIDATE",
                    selected_policy,
                    None,
                ),
                (
                    "BASELINE_TOP2_EQUAL",
                    None,
                    ANCHOR_TEMPLATES["BASELINE_TOP2_EQUAL"],
                ),
                (
                    "ANCHOR_TOP5_EQUAL",
                    None,
                    ANCHOR_TEMPLATES["ANCHOR_TOP5_EQUAL"],
                ),
                (
                    "ANCHOR_TOP5_RANK_30_25_20_15_10",
                    None,
                    ANCHOR_TEMPLATES[
                        "ANCHOR_TOP5_RANK_30_25_20_15_10"
                    ],
                ),
            ]
            for role, policy, anchor_weights in specifications:
                long_frame, summary_frame = simulate_policy(
                    contexts,
                    dates,
                    seed,
                    fold,
                    "outer",
                    role,
                    policy=policy,
                    anchor_weights=anchor_weights,
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
    return long_result.reset_index(drop=True), summary_result.reset_index(drop=True)


def verify_input_manifest(manifest: Mapping[str, Any]) -> None:
    for item in manifest["files"]:
        path = WORKSPACE / item["path"]
        current = sha256_file(path)
        if current != item["sha256"]:
            raise RuntimeError(f"input drift: {item['path']}")


def finalize_combination_rejection(
    *,
    started: float,
    seed: int,
    fold: str,
    grid_rows: list[dict[str, Any]],
    policy_locks: list[dict[str, Any]],
    protected_before: Mapping[str, Any],
) -> dict[str, Any]:
    grid = pd.DataFrame(grid_rows)
    write_csv(RUN_DIR / "TEST_INNER_GRID_SUMMARY.csv", grid)
    locks_payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "locks": policy_locks,
        "stopped_at": {"seed": seed, "fold_id": fold},
        "stop_reason": "NO_PREREGISTERED_INNER_POLICY_ELIGIBLE",
        "outer_outcome_read_count": 0,
    }
    write_json(RUN_DIR / "TEST_INNER_POLICY_LOCKS.json", locks_payload)

    mean_excess = pd.to_numeric(grid["mean_excess"], errors="coerce")
    finite = grid.loc[np.isfinite(mean_excess)].copy()
    best_row: dict[str, Any] | None = None
    if not finite.empty:
        best = finite.loc[
            pd.to_numeric(finite["mean_excess"], errors="coerce").idxmax()
        ]
        best_row = {
            "policy_id": str(best["policy_id"]),
            "mean_excess": float(best["mean_excess"]),
            "structural_pass": bool(best["structural_pass"]),
            "inner_noninferiority_pass": bool(
                best["inner_noninferiority_pass"]
            ),
        }
    gate = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "REJECTED_AT_COMBINATION_GATE",
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "failed_seed": seed,
        "failed_fold": fold,
        "evaluated_policy_count": int(len(grid)),
        "eligible_policy_count": int(grid["eligible"].fillna(False).sum()),
        "structural_pass_count": int(
            grid["structural_pass"].fillna(False).sum()
        ),
        "inner_noninferiority_pass_count": int(
            grid["inner_noninferiority_pass"].fillna(False).sum()
        ),
        "frozen_inner_noninferiority_margin": -NONINFERIORITY_MARGIN,
        "best_mean_excess_policy": best_row,
        "outer_portfolio_lock_attempted": False,
        "outer_outcome_parsed": False,
        "quick_gate_run": False,
        "training_authorized": False,
        "training_performed": False,
        "promotion_eligible": False,
    }
    write_json(RUN_DIR / "TEST_COMBINATION_GATE.json", gate)

    pre_scoring_stop = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "preregistration_sha256": sha256_file(
            RUN_DIR / "TEST_PREREGISTRATION.json"
        ),
        "input_manifest_sha256": sha256_file(
            RUN_DIR / "TEST_INPUT_MANIFEST.json"
        ),
        "inner_policy_locks_sha256": sha256_file(
            RUN_DIR / "TEST_INNER_POLICY_LOCKS.json"
        ),
        "stopped_before_outer_portfolio_generation": True,
        "double_run_byte_identical": "NOT_ATTEMPTED_DUE_TO_INNER_STOP",
        "outer_label_parse_count": 0,
        "outer_prediction_container_hash_only_before_stop": True,
        "prospective_outcome_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(RUN_DIR / "TEST_PRE_SCORING_LOCK.json", pre_scoring_stop)

    protected_after = protected_manifest()
    write_json(
        RUN_DIR / "TEST_PROTECTED_FINGERPRINTS_AFTER.json", protected_after
    )
    protection = compare_protected(protected_before, protected_after)
    write_json(RUN_DIR / "TEST_PROTECTION_AUDIT.json", protection)
    leakage = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if protection["status"] == "PASS" else "FAIL",
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "checks": {
            "no_training": True,
            "inner_only_parameter_selection": True,
            "outer_label_parse_count": 0,
            "outer_portfolio_generation_skipped_after_inner_failure": True,
            "outer_prediction_container_sha256_only": True,
            "all_risk_features_strictly_lag1": True,
            "industry_features_disabled": True,
            "external_features_disabled": True,
            "prospective_paths_accessed": False,
            "protected_trees_unchanged": protection["status"] == "PASS",
        },
        "promotion_eligible": False,
    }
    write_json(RUN_DIR / "TEST_LEAKAGE_AUDIT.json", leakage)

    elapsed = time.perf_counter() - started
    best_text = (
        "无有限结果"
        if best_row is None
        else (
            f"`{best_row['policy_id']}`，内层平均超额 "
            f"`{best_row['mean_excess']:.4%}`"
        )
    )
    report = f"""# Candidate-D Defensive Combo V1 TEST 最终报告

状态：`REJECTED_AT_COMBINATION_GATE`

本 TEST 在首个失败的内层选择单元 seed `{seed}` / fold `{fold}` 停止。
冻结网格共评估 `{len(grid)}` 个策略，合格数为 `0`；最优均值超额策略为
{best_text}，未达到执行前冻结的内层非劣边界 `-0.20%` 或结构约束。

## 停止与权限

- 未生成外层组合，未解析外层 label，未执行 Quick 外层评分。
- 未训练模型，GPU 时间为 0；未下载新 outcome，未访问 prospective 窗口。
- 未改变网格、门槛、seed、epoch 或数据分支以挽救结果。
- Candidate-D、D01、官方参考保护审计：`{protection['status']}`。
- D Alpha Champion 仍为 `D01_FrozenResearchChampion_20260709`。
- Defensive Champion 仍为 `NONE`；未创建 H01，不具备晋级或替换权限。

完整失败证据见 `TEST_COMBINATION_GATE.json` 和
`TEST_INNER_GRID_SUMMARY.csv`。外层预测容器仅做 SHA-256 完整性散列，
没有解析 `label` 列；该散列不参与任何参数或门槛选择。
"""
    (RUN_DIR / "TEST_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    status_payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": "REJECTED_AT_COMBINATION_GATE",
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "failed_seed": seed,
        "failed_fold": fold,
        "gpu_time_seconds": 0.0,
        "elapsed_seconds": elapsed,
        "output_bytes": sum(
            path.stat().st_size for path in RUN_DIR.rglob("*") if path.is_file()
        ),
        "failed_checks": ["inner_no_eligible_policy"],
        "candidate_d_modified": protection["status"] != "PASS",
        "d01_modified": protection["status"] != "PASS",
        "official_reference_modified": protection["status"] != "PASS",
        "defensive_champion": None,
        "alpha_champion": "D01_FrozenResearchChampion_20260709",
        "serious_run": False,
        "training_run": False,
        "h01_created": False,
        "promotion_eligible": False,
    }
    write_json(RUN_DIR / "TEST_STATUS.json", status_payload)
    return status_payload


def prepare() -> dict[str, Any]:
    started = time.perf_counter()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    terminal_files = [
        RUN_DIR / "TEST_QUICK_GATE.json",
        RUN_DIR / "TEST_FINAL_REPORT.md",
    ]
    if any(path.exists() for path in terminal_files):
        raise RuntimeError("terminal run artifacts already exist; refusing overwrite")

    protected_before = protected_manifest()
    write_json(
        RUN_DIR / "TEST_PROTECTED_FINGERPRINTS_BEFORE.json",
        protected_before,
    )
    preregistration = preregistration_payload()
    write_json(RUN_DIR / "TEST_PREREGISTRATION.json", preregistration)
    manifest = input_manifest()
    write_json(RUN_DIR / "TEST_INPUT_MANIFEST.json", manifest)

    market = MarketData(MARKET_CSV)
    policy_locks: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for fold in FOLDS:
            try:
                lock, summaries, _ = select_inner_policy(seed, fold, market)
            except NoEligibleInnerPolicy as exc:
                grid_rows.extend(exc.summaries)
                return finalize_combination_rejection(
                    started=started,
                    seed=exc.seed,
                    fold=exc.fold,
                    grid_rows=grid_rows,
                    policy_locks=policy_locks,
                    protected_before=protected_before,
                )
            policy_locks.append(lock)
            grid_rows.extend(summaries)
    write_csv(RUN_DIR / "TEST_INNER_GRID_SUMMARY.csv", pd.DataFrame(grid_rows))
    locks_payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "locks": policy_locks,
        "outer_outcome_read_count": 0,
    }
    write_json(RUN_DIR / "TEST_INNER_POLICY_LOCKS.json", locks_payload)

    pass1_long, pass1_summary = generate_outer_lock(policy_locks, market)
    pass2_long, pass2_summary = generate_outer_lock(policy_locks, market)
    pass1_path = RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_PASS1.csv"
    pass2_path = RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_PASS2.csv"
    pass1_summary_path = RUN_DIR / "TEST_OUTER_PORTFOLIO_SUMMARY_PASS1.csv"
    pass2_summary_path = RUN_DIR / "TEST_OUTER_PORTFOLIO_SUMMARY_PASS2.csv"
    write_csv(pass1_path, pass1_long)
    write_csv(pass2_path, pass2_long)
    write_csv(pass1_summary_path, pass1_summary)
    write_csv(pass2_summary_path, pass2_summary)
    hashes = {
        "long_pass1": sha256_file(pass1_path),
        "long_pass2": sha256_file(pass2_path),
        "summary_pass1": sha256_file(pass1_summary_path),
        "summary_pass2": sha256_file(pass2_summary_path),
    }
    if hashes["long_pass1"] != hashes["long_pass2"]:
        raise RuntimeError("outer portfolio long locks are not byte-identical")
    if hashes["summary_pass1"] != hashes["summary_pass2"]:
        raise RuntimeError("outer portfolio summary locks are not byte-identical")
    pre_scoring_lock = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "preregistration_sha256": sha256_file(
            RUN_DIR / "TEST_PREREGISTRATION.json"
        ),
        "input_manifest_sha256": sha256_file(
            RUN_DIR / "TEST_INPUT_MANIFEST.json"
        ),
        "inner_policy_locks_sha256": sha256_file(
            RUN_DIR / "TEST_INNER_POLICY_LOCKS.json"
        ),
        "outer_portfolio_lock_hashes": hashes,
        "double_run_byte_identical": True,
        "outer_label_parse_count": 0,
        "prospective_outcome_accessed": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(RUN_DIR / "TEST_PRE_SCORING_LOCK.json", pre_scoring_lock)
    return pre_scoring_lock


def load_outer_labels() -> tuple[pd.DataFrame, int]:
    frames: list[pd.DataFrame] = []
    parse_count = 0
    for seed in SEEDS:
        for fold in FOLDS:
            path = prediction_path(seed, fold, "outer")
            predictions = load_predictions(path, include_label=True)
            parse_count += 1
            frames.append(
                predictions.assign(
                    seed=seed,
                    fold_id=fold,
                    decision_date=predictions[
                        "sample_end_date"
                    ].dt.strftime("%Y-%m-%d"),
                )[["seed", "fold_id", "decision_date", "stock_id", "label"]]
            )
    return pd.concat(frames, ignore_index=True), parse_count


def paired_role(
    scored: pd.DataFrame, role: str, return_column: str
) -> pd.DataFrame:
    baseline = scored.loc[scored["policy_role"] == "BASELINE_TOP2_EQUAL"].copy()
    candidate = scored.loc[scored["policy_role"] == role].copy()
    keys = ["seed", "fold_id", "decision_date"]
    left_columns = keys + [
        return_column,
        "trade_turnover",
        "cross_week_turnover",
        "weighted_pair_correlation",
        "hhi",
    ]
    paired = baseline[left_columns].merge(
        candidate[left_columns],
        on=keys,
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise ValueError(f"role {role} is not exactly paired to baseline")
    paired["excess"] = (
        paired[f"{return_column}_candidate"]
        - paired[f"{return_column}_baseline"]
    )
    return paired


def gate_metrics(scored: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    gross = paired_role(scored, "DEFENSIVE_CANDIDATE", "portfolio_return_gross")
    roundtrip10 = paired_role(
        scored, "DEFENSIVE_CANDIDATE", "return_roundtrip_10bps"
    )
    roundtrip20 = paired_role(
        scored, "DEFENSIVE_CANDIDATE", "return_roundtrip_20bps"
    )
    turnover10 = paired_role(
        scored, "DEFENSIVE_CANDIDATE", "return_turnover_10bps"
    )
    turnover20 = paired_role(
        scored, "DEFENSIVE_CANDIDATE", "return_turnover_20bps"
    )
    gross_candidate = weekly_metrics(
        gross["portfolio_return_gross_candidate"]
    )
    gross_baseline = weekly_metrics(gross["portfolio_return_gross_baseline"])
    roundtrip10_candidate = weekly_metrics(
        roundtrip10["return_roundtrip_10bps_candidate"]
    )
    roundtrip10_baseline = weekly_metrics(
        roundtrip10["return_roundtrip_10bps_baseline"]
    )
    roundtrip20_candidate = weekly_metrics(
        roundtrip20["return_roundtrip_20bps_candidate"]
    )
    roundtrip20_baseline = weekly_metrics(
        roundtrip20["return_roundtrip_20bps_baseline"]
    )
    excess_metrics = weekly_metrics(gross["excess"])
    seed_fold = (
        gross.groupby(["seed", "fold_id"], sort=True)["excess"].mean().reset_index()
    )
    positive_by_seed = (
        seed_fold.assign(positive=seed_fold["excess"] > 0.0)
        .groupby("seed")["positive"]
        .sum()
    )
    seed_means = (
        gross.groupby("seed", sort=True)["excess"].mean().astype(float).to_dict()
    )
    recent_excess = float(
        gross.loc[gross["fold_id"] == "2026Q1", "excess"].mean()
    )
    unique_date_excess = (
        gross.groupby(["fold_id", "decision_date"], sort=True)["excess"]
        .mean()
        .reset_index()
        .sort_values(["fold_id", "decision_date"])
    )
    bootstrap_lower = moving_block_bootstrap_lower(
        unique_date_excess["excess"].to_numpy(dtype=float)
    )
    candidate_scored = scored.loc[
        scored["policy_role"] == "DEFENSIVE_CANDIDATE"
    ]
    baseline_scored = scored.loc[
        scored["policy_role"] == "BASELINE_TOP2_EQUAL"
    ]
    metrics = {
        "paired_decision_count": int(len(gross)),
        "unique_market_decision_count": int(len(unique_date_excess)),
        "positive_folds": int(positive_by_seed.min()),
        "positive_folds_by_seed": {
            str(int(seed)): int(count) for seed, count in positive_by_seed.items()
        },
        "recent_fold_excess": recent_excess,
        "seed_mean_excess": {
            str(int(seed)): float(value) for seed, value in seed_means.items()
        },
        "strict_weekly_win_rate": float((gross["excess"] > 0.0).mean()),
        "excess_sum_remove_best1": excess_metrics["sum_remove_best1"],
        "excess_sum_remove_best2": excess_metrics["sum_remove_best2"],
        "candidate_gross": gross_candidate,
        "baseline_gross": gross_baseline,
        "candidate_roundtrip_10bps": roundtrip10_candidate,
        "baseline_roundtrip_10bps": roundtrip10_baseline,
        "candidate_roundtrip_20bps": roundtrip20_candidate,
        "baseline_roundtrip_20bps": roundtrip20_baseline,
        "candidate_max_drawdown_10bps": max_drawdown(
            candidate_scored, "return_roundtrip_10bps"
        ),
        "baseline_max_drawdown_10bps": max_drawdown(
            baseline_scored, "return_roundtrip_10bps"
        ),
        "candidate_max_drawdown_20bps": max_drawdown(
            candidate_scored, "return_roundtrip_20bps"
        ),
        "baseline_max_drawdown_20bps": max_drawdown(
            baseline_scored, "return_roundtrip_20bps"
        ),
        "candidate_mean_cross_week_turnover": float(
            candidate_scored["cross_week_turnover"].dropna().mean()
        ),
        "baseline_mean_cross_week_turnover": float(
            baseline_scored["cross_week_turnover"].dropna().mean()
        ),
        "candidate_mean_pair_correlation": float(
            candidate_scored["weighted_pair_correlation"].mean()
        ),
        "baseline_mean_pair_correlation": float(
            baseline_scored["weighted_pair_correlation"].mean()
        ),
        "max_candidate_hhi": float(candidate_scored["hhi"].max()),
        "max_candidate_cluster_count": int(
            candidate_scored["max_cluster_count"].max()
        ),
        "bootstrap_95pct_lower_mean_excess": bootstrap_lower,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "mean_excess_roundtrip_10bps": float(roundtrip10["excess"].mean()),
        "mean_excess_roundtrip_20bps": float(roundtrip20["excess"].mean()),
        "mean_excess_turnover_10bps": float(turnover10["excess"].mean()),
        "mean_excess_turnover_20bps": float(turnover20["excess"].mean()),
    }
    checks = {
        "positive_folds_at_least_3_of_4": metrics["positive_folds"] >= 3,
        "recent_fold_excess_positive": recent_excess > 0.0,
        "both_seed_means_positive": len(seed_means) == 2
        and all(value > 0.0 for value in seed_means.values()),
        "strict_weekly_win_rate_at_least_55pct": metrics[
            "strict_weekly_win_rate"
        ]
        >= 0.55,
        "positive_week_rate_strictly_better_10bps": (
            roundtrip10_candidate["positive_rate"]
            > roundtrip10_baseline["positive_rate"]
        ),
        "positive_week_rate_strictly_better_20bps": (
            roundtrip20_candidate["positive_rate"]
            > roundtrip20_baseline["positive_rate"]
        ),
        "median_not_worse_10bps": (
            roundtrip10_candidate["median"] >= roundtrip10_baseline["median"]
        ),
        "median_not_worse_20bps": (
            roundtrip20_candidate["median"] >= roundtrip20_baseline["median"]
        ),
        "cvar20_strictly_better_10bps": (
            roundtrip10_candidate["cvar20"] > roundtrip10_baseline["cvar20"]
        ),
        "cvar20_strictly_better_20bps": (
            roundtrip20_candidate["cvar20"] > roundtrip20_baseline["cvar20"]
        ),
        "worst_week_strictly_better_10bps": (
            roundtrip10_candidate["worst"] > roundtrip10_baseline["worst"]
        ),
        "worst_week_strictly_better_20bps": (
            roundtrip20_candidate["worst"] > roundtrip20_baseline["worst"]
        ),
        "max_drawdown_strictly_better_10bps": (
            metrics["candidate_max_drawdown_10bps"]
            < metrics["baseline_max_drawdown_10bps"]
        ),
        "max_drawdown_strictly_better_20bps": (
            metrics["candidate_max_drawdown_20bps"]
            < metrics["baseline_max_drawdown_20bps"]
        ),
        "excess_positive_after_best_week_removed": (
            excess_metrics["sum_remove_best1"] > 0.0
        ),
        "excess_positive_after_best_two_removed": (
            excess_metrics["sum_remove_best2"] > 0.0
        ),
        "hhi_at_most_0_30": metrics["max_candidate_hhi"] <= 0.30,
        "cluster_holdings_at_most_2": (
            metrics["max_candidate_cluster_count"] <= MAX_CLUSTER_HOLDINGS
        ),
        "pair_correlation_strictly_lower": (
            metrics["candidate_mean_pair_correlation"]
            < metrics["baseline_mean_pair_correlation"]
        ),
        "cross_week_turnover_strictly_lower": (
            metrics["candidate_mean_cross_week_turnover"]
            < metrics["baseline_mean_cross_week_turnover"]
        ),
        "noninferiority_95pct_lower_above_minus_20bps": (
            bootstrap_lower > -NONINFERIORITY_MARGIN
        ),
        "roundtrip_cost_conclusion_positive_at_10_20bps": (
            metrics["mean_excess_roundtrip_10bps"] > 0.0
            and metrics["mean_excess_roundtrip_20bps"] > 0.0
        ),
        "turnover_cost_conclusion_positive_at_10_20bps": (
            metrics["mean_excess_turnover_10bps"] > 0.0
            and metrics["mean_excess_turnover_20bps"] > 0.0
        ),
    }
    metrics["checks"] = checks
    metrics["failed_checks"] = [
        name for name, passed in checks.items() if not passed
    ]
    return metrics, gross


def anchor_diagnostics(scored: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for role in (
        "ANCHOR_TOP5_EQUAL",
        "ANCHOR_TOP5_RANK_30_25_20_15_10",
    ):
        paired = paired_role(scored, role, "portfolio_return_gross")
        candidate = weekly_metrics(paired["portfolio_return_gross_candidate"])
        baseline = weekly_metrics(paired["portfolio_return_gross_baseline"])
        excess = weekly_metrics(paired["excess"])
        role_rows = scored.loc[scored["policy_role"] == role]
        diagnostics[role] = {
            "authority": "DIAGNOSTIC_ANCHOR_ONLY",
            "candidate": candidate,
            "baseline": baseline,
            "mean_excess": float(paired["excess"].mean()),
            "strict_win_rate": float((paired["excess"] > 0.0).mean()),
            "excess_sum_remove_best2": excess["sum_remove_best2"],
            "mean_turnover": float(
                role_rows["cross_week_turnover"].dropna().mean()
            ),
            "mean_pair_correlation": float(
                role_rows["weighted_pair_correlation"].mean()
            ),
            "max_hhi": float(role_rows["hhi"].max()),
        }
    return diagnostics


def final_report(
    status: str,
    metrics: Mapping[str, Any],
    anchors: Mapping[str, Any],
    protection: Mapping[str, Any],
    runtime_seconds: float,
    output_bytes: int,
) -> str:
    failed = metrics["failed_checks"]
    failed_text = "、".join(failed) if failed else "无"
    candidate10 = metrics["candidate_roundtrip_10bps"]
    baseline10 = metrics["baseline_roundtrip_10bps"]
    seed_means = metrics["seed_mean_excess"]
    return f"""# Candidate-D Defensive Combo V1 TEST 最终报告

状态：`{status}`

分类：`RETROSPECTIVE_EXPOSED_REJECTION_SCREEN / TEST / NOT PROMOTION EVIDENCE`

## 执行边界

- 未训练模型，GPU 时间为 0。
- 只使用既有 H0/D01-program 两种 seed、四个开发折的冻结完整排名。
- outer 组合在不解析 label 的情况下独立生成两次并字节一致，之后才评分。
- 未访问 prospective 窗口、未下载 outcome、未使用行业或外部特征。
- Candidate-D、D01、官方参考保护审计：`{protection['status']}`。

## Defensive challenger 主结果

|指标|Defensive|Candidate-D Top-2|
|---|---:|---:|
|10bps 正收益周比例|{candidate10['positive_rate']:.2%}|{baseline10['positive_rate']:.2%}|
|10bps 周中位数|{candidate10['median']:.4%}|{baseline10['median']:.4%}|
|10bps CVaR20|{candidate10['cvar20']:.4%}|{baseline10['cvar20']:.4%}|
|10bps 最差周|{candidate10['worst']:.4%}|{baseline10['worst']:.4%}|
|10bps 最大回撤|{metrics['candidate_max_drawdown_10bps']:.4%}|{metrics['baseline_max_drawdown_10bps']:.4%}|
|平均跨周换手|{metrics['candidate_mean_cross_week_turnover']:.2%}|{metrics['baseline_mean_cross_week_turnover']:.2%}|
|平均组合内相关性|{metrics['candidate_mean_pair_correlation']:.4f}|{metrics['baseline_mean_pair_correlation']:.4f}|
|最大 HHI|{metrics['max_candidate_hhi']:.3f}|0.500|

- 正向折最小值：`{metrics['positive_folds']}/4`
- 最近折平均超额：`{metrics['recent_fold_excess']:.4%}`
- seed 平均超额：`{seed_means}`
- 严格逐周胜率：`{metrics['strict_weekly_win_rate']:.2%}`
- 去最佳两周后的超额和：`{metrics['excess_sum_remove_best2']:.4%}`
- 配对均值超额95% bootstrap 下界：`{metrics['bootstrap_95pct_lower_mean_excess']:.4%}`；冻结非劣边界 `-0.20%`
- 失败门槛：`{failed_text}`

## 原生 Top-5 锚点

- Top-5 等权：均值超额 `{anchors['ANCHOR_TOP5_EQUAL']['mean_excess']:.4%}`，
  严格胜率 `{anchors['ANCHOR_TOP5_EQUAL']['strict_win_rate']:.2%}`，
  HHI `{anchors['ANCHOR_TOP5_EQUAL']['max_hhi']:.3f}`。
- Top-5 固定权重：均值超额
  `{anchors['ANCHOR_TOP5_RANK_30_25_20_15_10']['mean_excess']:.4%}`，
  严格胜率
  `{anchors['ANCHOR_TOP5_RANK_30_25_20_15_10']['strict_win_rate']:.2%}`，
  HHI `{anchors['ANCHOR_TOP5_RANK_30_25_20_15_10']['max_hhi']:.3f}`。

锚点只有诊断权限，不能绕过 Defensive 主门槛。

## 身份与停止

- D Alpha Champion 仍为 `D01_FrozenResearchChampion_20260709`。
- Defensive Champion 仍为 `NONE`。
- Candidate-D 未替换、未修改。
- 未创建 H01；训练分支未启动。
- 本结果的 outer 折已暴露，即使全部通过也只能支持后续独立计划，不具备晋级权限。

运行耗时 `{runtime_seconds:.2f}` 秒；TEST 输出约 `{output_bytes / (1024 ** 2):.2f}` MiB。
"""


def score() -> dict[str, Any]:
    started = time.perf_counter()
    required = [
        RUN_DIR / "TEST_PREREGISTRATION.json",
        RUN_DIR / "TEST_INPUT_MANIFEST.json",
        RUN_DIR / "TEST_INNER_POLICY_LOCKS.json",
        RUN_DIR / "TEST_PRE_SCORING_LOCK.json",
        RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_PASS1.csv",
        RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_PASS2.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing pre-scoring artifacts: {missing}")
    prelock = json.loads(
        (RUN_DIR / "TEST_PRE_SCORING_LOCK.json").read_text(encoding="utf-8")
    )
    if prelock["outer_label_parse_count"] != 0:
        raise RuntimeError("pre-scoring lock already records outer label access")
    if not prelock["double_run_byte_identical"]:
        raise RuntimeError("outer portfolio double-run lock failed")
    if sha256_file(
        RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_PASS1.csv"
    ) != sha256_file(RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_PASS2.csv"):
        raise RuntimeError("outer portfolio lock drift")
    manifest = json.loads(
        (RUN_DIR / "TEST_INPUT_MANIFEST.json").read_text(encoding="utf-8")
    )
    verify_input_manifest(manifest)

    locked = pd.read_csv(
        RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_PASS1.csv",
        dtype={"stock_id": str},
    )
    locked["stock_id"] = locked["stock_id"].str.zfill(6)
    labels, parse_count = load_outer_labels()
    scored = score_locked_portfolios(locked, labels)
    metrics, paired = gate_metrics(scored)
    anchors = anchor_diagnostics(scored)
    status = (
        "REJECTED_AT_QUICK_GATE"
        if metrics["failed_checks"]
        else "PASS_EXPOSED_QUICK_SCREEN"
    )
    gate_payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "status": status,
        "metrics": metrics,
        "anchor_diagnostics": anchors,
        "serious_seed_consensus": "DEFERRED_NOT_RUN",
        "training_authorized": False,
        "training_performed": False,
        "serious_run": False,
        "h01_created": False,
        "promotion_eligible": False,
    }
    write_csv(RUN_DIR / "TEST_OUTER_SCORED_DECISIONS.csv", scored)
    write_csv(RUN_DIR / "TEST_PAIRED_DEFENSIVE_VS_D.csv", paired)
    write_json(RUN_DIR / "TEST_QUICK_GATE.json", gate_payload)

    before = json.loads(
        (RUN_DIR / "TEST_PROTECTED_FINGERPRINTS_BEFORE.json").read_text(
            encoding="utf-8"
        )
    )
    after = protected_manifest()
    write_json(RUN_DIR / "TEST_PROTECTED_FINGERPRINTS_AFTER.json", after)
    protection = compare_protected(before, after)
    write_json(RUN_DIR / "TEST_PROTECTION_AUDIT.json", protection)

    leakage = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "PASS" if protection["status"] == "PASS" else "FAIL"
        ),
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "checks": {
            "no_training": True,
            "inner_only_parameter_selection": True,
            "outer_rank_columns_loaded_before_lock": [
                "marker",
                "sample_end_date",
                "stock_id",
                "score",
            ],
            "outer_label_parse_count_before_lock": 0,
            "outer_label_file_parse_count_after_lock": parse_count,
            "double_run_portfolio_lock_byte_identical": True,
            "all_risk_features_strictly_lag1": True,
            "per_date_correlation_clustering": True,
            "turnover_uses_previous_locked_portfolio_only": True,
            "industry_features_disabled": True,
            "external_features_disabled": True,
            "prospective_paths_accessed": False,
            "blacklisted_2026_07_06_to_2026_07_24_used_for_selection": False,
            "protected_trees_unchanged": protection["status"] == "PASS",
        },
        "outer_outcome_hashes": [
            item
            for item in manifest["files"]
            if item["role"] == "h0_outer_best_val_predictions.csv"
        ],
        "promotion_eligible": False,
    }
    write_json(RUN_DIR / "TEST_LEAKAGE_AUDIT.json", leakage)

    runtime_seconds = time.perf_counter() - started + float(
        prelock["elapsed_seconds"]
    )
    output_bytes_before_report = sum(
        path.stat().st_size for path in RUN_DIR.rglob("*") if path.is_file()
    )
    report = final_report(
        status,
        metrics,
        anchors,
        protection,
        runtime_seconds,
        output_bytes_before_report,
    )
    (RUN_DIR / "TEST_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    status_payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "classification": "RETROSPECTIVE_EXPOSED_REJECTION_SCREEN",
        "gpu_time_seconds": 0.0,
        "elapsed_seconds": runtime_seconds,
        "output_bytes": sum(
            path.stat().st_size for path in RUN_DIR.rglob("*") if path.is_file()
        ),
        "failed_checks": metrics["failed_checks"],
        "candidate_d_modified": protection["status"] != "PASS",
        "d01_modified": protection["status"] != "PASS",
        "official_reference_modified": protection["status"] != "PASS",
        "defensive_champion": None,
        "alpha_champion": "D01_FrozenResearchChampion_20260709",
        "serious_run": False,
        "training_run": False,
        "h01_created": False,
        "promotion_eligible": False,
    }
    write_json(RUN_DIR / "TEST_STATUS.json", status_payload)
    return status_payload


def verify() -> dict[str, Any]:
    status_path = RUN_DIR / "TEST_STATUS.json"
    if not status_path.exists():
        raise RuntimeError("TEST_STATUS.json does not exist")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] == "REJECTED_AT_COMBINATION_GATE":
        required = [
            "TEST_PREREGISTRATION.json",
            "TEST_INPUT_MANIFEST.json",
            "TEST_INNER_GRID_SUMMARY.csv",
            "TEST_INNER_POLICY_LOCKS.json",
            "TEST_COMBINATION_GATE.json",
            "TEST_PRE_SCORING_LOCK.json",
            "TEST_LEAKAGE_AUDIT.json",
            "TEST_PROTECTED_FINGERPRINTS_BEFORE.json",
            "TEST_PROTECTED_FINGERPRINTS_AFTER.json",
            "TEST_PROTECTION_AUDIT.json",
            "TEST_FINAL_REPORT.md",
            "TEST_STATUS.json",
        ]
        missing = [name for name in required if not (RUN_DIR / name).exists()]
        protection = json.loads(
            (RUN_DIR / "TEST_PROTECTION_AUDIT.json").read_text(
                encoding="utf-8"
            )
        )
        pre_scoring = json.loads(
            (RUN_DIR / "TEST_PRE_SCORING_LOCK.json").read_text(
                encoding="utf-8"
            )
        )
        result = {
            "status": (
                "PASS"
                if (
                    not missing
                    and protection["status"] == "PASS"
                    and pre_scoring["outer_label_parse_count"] == 0
                    and pre_scoring[
                        "stopped_before_outer_portfolio_generation"
                    ]
                )
                else "FAIL"
            ),
            "missing": missing,
            "outer_label_parse_count": pre_scoring[
                "outer_label_parse_count"
            ],
            "outer_portfolio_lock": "NOT_ATTEMPTED_DUE_TO_INNER_STOP",
            "protection_status": protection["status"],
            "terminal_status": status["status"],
        }
        write_json(RUN_DIR / "TEST_VERIFICATION.json", result)
        return result
    required = [
        "TEST_PREREGISTRATION.json",
        "TEST_INPUT_MANIFEST.json",
        "TEST_INNER_GRID_SUMMARY.csv",
        "TEST_INNER_POLICY_LOCKS.json",
        "TEST_PRE_SCORING_LOCK.json",
        "TEST_OUTER_PORTFOLIO_LOCK_PASS1.csv",
        "TEST_OUTER_PORTFOLIO_LOCK_PASS2.csv",
        "TEST_OUTER_SCORED_DECISIONS.csv",
        "TEST_QUICK_GATE.json",
        "TEST_LEAKAGE_AUDIT.json",
        "TEST_PROTECTION_AUDIT.json",
        "TEST_FINAL_REPORT.md",
        "TEST_STATUS.json",
    ]
    missing = [name for name in required if not (RUN_DIR / name).exists()]
    pass1 = sha256_file(RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_PASS1.csv")
    pass2 = sha256_file(RUN_DIR / "TEST_OUTER_PORTFOLIO_LOCK_PASS2.csv")
    protection = json.loads(
        (RUN_DIR / "TEST_PROTECTION_AUDIT.json").read_text(encoding="utf-8")
    )
    result = {
        "status": (
            "PASS"
            if not missing and pass1 == pass2 and protection["status"] == "PASS"
            else "FAIL"
        ),
        "missing": missing,
        "portfolio_locks_byte_identical": pass1 == pass2,
        "protection_status": protection["status"],
        "terminal_status": status["status"],
    }
    write_json(RUN_DIR / "TEST_VERIFICATION.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Candidate-D defensive combination-only TEST"
    )
    parser.add_argument(
        "--stage", choices=("prepare", "score", "verify", "all"), required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in ("prepare", "all"):
        print(json.dumps(prepare(), ensure_ascii=False, indent=2))
    if args.stage in ("score", "all"):
        print(json.dumps(score(), ensure_ascii=False, indent=2))
    if args.stage in ("verify", "all"):
        print(json.dumps(verify(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

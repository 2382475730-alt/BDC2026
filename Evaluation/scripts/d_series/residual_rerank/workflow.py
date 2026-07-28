from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


EXPOSED_DIAGNOSTIC_RANGES = (
    (pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-24")),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_prediction_lock(
    files: Iterable[str | Path],
    lock_path: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    items = []
    for value in sorted({str(Path(path).resolve()) for path in files}):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        items.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "files": items,
        "metadata": dict(metadata or {}),
    }
    output = Path(lock_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_prediction_lock(lock_path: str | Path) -> dict[str, object]:
    path = Path(lock_path)
    if not path.is_file():
        raise ValueError(f"prediction lock does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("prediction lock contains no files")
    for item in files:
        artifact = Path(item["path"])
        if not artifact.is_file():
            raise ValueError(f"locked artifact is missing: {artifact}")
        if int(item["bytes"]) != artifact.stat().st_size:
            raise ValueError(f"locked hash/byte count differs: {artifact}")
        current = sha256_file(artifact)
        if current != item["sha256"]:
            raise ValueError(f"locked hash differs: {artifact}")
    return payload


def _ensure_not_exposed(decision_dates: Iterable[pd.Timestamp | str]) -> None:
    for value in decision_dates:
        date = pd.Timestamp(value).normalize()
        if any(start <= date <= end for start, end in EXPOSED_DIAGNOSTIC_RANGES):
            raise ValueError(f"decision date is in permanently exposed diagnostic range: {date.date()}")


def score_locked_portfolios(
    predictions_path: str | Path,
    outcomes: pd.DataFrame,
    lock_path: str | Path,
    cost_bps_per_side: float,
) -> pd.DataFrame:
    lock = verify_prediction_lock(lock_path)
    predictions = Path(predictions_path).resolve()
    locked_paths = {str(Path(item["path"]).resolve()) for item in lock["files"]}
    if str(predictions) not in locked_paths:
        raise ValueError("portfolio predictions are not covered by the supplied lock")
    required_predictions = {"decision_date", "stock_id", "weight"}
    required_outcomes = {"decision_date", "stock_id", "forward_return"}
    portfolio = pd.read_csv(predictions, dtype={"stock_id": str})
    if missing := required_predictions.difference(portfolio.columns):
        raise KeyError(f"missing portfolio columns: {sorted(missing)}")
    if missing := required_outcomes.difference(outcomes.columns):
        raise KeyError(f"missing outcome columns: {sorted(missing)}")

    portfolio["decision_date"] = pd.to_datetime(portfolio["decision_date"]).dt.normalize()
    portfolio["stock_id"] = portfolio["stock_id"].astype(str).str.zfill(6)
    _ensure_not_exposed(portfolio["decision_date"].unique())
    if portfolio.duplicated(["decision_date", "stock_id"]).any():
        raise ValueError("portfolio contains duplicate stocks for a decision")

    counts = portfolio.groupby("decision_date")["stock_id"].size()
    sums = portfolio.groupby("decision_date")["weight"].sum()
    if (counts > 5).any() or (counts < 1).any():
        raise ValueError("portfolio must contain between one and five stocks")
    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("portfolio must be fully invested with weight sum one")
    if (pd.to_numeric(portfolio["weight"], errors="coerce") < 0.0).any():
        raise ValueError("portfolio weights must be nonnegative")

    labels = outcomes.copy()
    labels["decision_date"] = pd.to_datetime(labels["decision_date"]).dt.normalize()
    labels["stock_id"] = labels["stock_id"].astype(str).str.zfill(6)
    if labels.duplicated(["decision_date", "stock_id"]).any():
        raise ValueError("outcomes contain duplicate stock/date rows")
    merged = portfolio.merge(
        labels,
        on=["decision_date", "stock_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["forward_return"].isna().any():
        raise ValueError("one or more locked holdings lack outcomes")

    cost = 2.0 * float(cost_bps_per_side) / 10_000.0
    rows = []
    for decision_date, day in merged.groupby("decision_date", sort=True):
        gross = float(
            (
                pd.to_numeric(day["weight"], errors="raise")
                * pd.to_numeric(day["forward_return"], errors="raise")
            ).sum()
        )
        rows.append(
            {
                "decision_date": decision_date,
                "gross_return": gross,
                "cost": cost,
                "net_return": gross - cost,
                "hhi": float(np.square(day["weight"].astype(float)).sum()),
                "stock_count": int(len(day)),
            }
        )
    return pd.DataFrame(rows)


def fingerprint_tree(root: str | Path) -> dict[str, object]:
    directory = Path(root).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = []
    for path in sorted(
        (item for item in directory.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(directory).as_posix(),
    ):
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    canonical = "\n".join(
        f"{item['path']}\0{item['bytes']}\0{item['sha256']}" for item in files
    ).encode("utf-8")
    return {
        "root": str(directory),
        "file_count": len(files),
        "total_bytes": int(sum(int(item["bytes"]) for item in files)),
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def compare_fingerprints(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    before_files = {item["path"]: item for item in before["files"]}
    after_files = {item["path"]: item for item in after["files"]}
    before_paths = set(before_files)
    after_paths = set(after_files)
    changed = sorted(
        path
        for path in before_paths & after_paths
        if before_files[path]["sha256"] != after_files[path]["sha256"]
        or before_files[path]["bytes"] != after_files[path]["bytes"]
    )
    added = sorted(after_paths - before_paths)
    removed = sorted(before_paths - after_paths)
    identical = (
        not changed
        and not added
        and not removed
        and before.get("aggregate_sha256") == after.get("aggregate_sha256")
    )
    return {
        "identical": identical,
        "changed_paths": changed,
        "added_paths": added,
        "removed_paths": removed,
        "before_aggregate_sha256": before.get("aggregate_sha256"),
        "after_aggregate_sha256": after.get("aggregate_sha256"),
    }


def _distribution_metrics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "q10": 0.0,
            "cvar20": 0.0,
            "sum_remove_best2": 0.0,
            "best_positive_contribution": 1.0,
        }
    tail_count = max(1, int(math.ceil(len(array) * 0.20)))
    positives = array[array > 0.0]
    positive_sum = float(positives.sum())
    ordered_desc = np.sort(array)[::-1]
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "q10": float(np.quantile(array, 0.10)),
        "cvar20": float(np.sort(array)[:tail_count].mean()),
        "sum_remove_best2": float(ordered_desc[2:].sum()),
        "best_positive_contribution": (
            float(positives.max() / positive_sum) if positive_sum > 0 else 1.0
        ),
    }


def quick_gate(paired_returns: pd.DataFrame) -> dict[str, object]:
    required = {
        "seed",
        "fold_id",
        "decision_date",
        "control_net_return",
        "candidate_net_return",
        "control_hhi",
        "candidate_hhi",
    }
    if missing := required.difference(paired_returns.columns):
        raise KeyError(f"missing Quick columns: {sorted(missing)}")
    frame = paired_returns.copy()
    if frame.duplicated(["seed", "fold_id", "decision_date"]).any():
        raise ValueError("Quick frame contains duplicate paired decisions")
    frame["excess"] = (
        frame["candidate_net_return"].astype(float)
        - frame["control_net_return"].astype(float)
    )
    seed_fold = frame.groupby(["seed", "fold_id"], sort=True)["excess"].mean()
    positive_counts = (
        seed_fold.gt(0.0).groupby(level="seed").sum().astype(int)
    )
    seed_means = frame.groupby("seed", sort=True)["excess"].mean().astype(float)
    latest_fold = (
        frame.assign(_date=pd.to_datetime(frame["decision_date"]))
        .groupby("fold_id")["_date"]
        .max()
        .idxmax()
    )
    excess_metrics = _distribution_metrics(frame["excess"])
    candidate_metrics = _distribution_metrics(frame["candidate_net_return"])
    control_metrics = _distribution_metrics(frame["control_net_return"])
    cost_consistent = (
        bool(frame["cost_direction_consistent"].all())
        if "cost_direction_consistent" in frame
        else True
    )
    metrics: dict[str, object] = {
        "positive_folds": int(positive_counts.min()),
        "positive_folds_by_seed": {
            str(int(seed)): int(value) for seed, value in positive_counts.items()
        },
        "recent_fold": str(latest_fold),
        "recent_fold_excess": float(
            frame.loc[frame["fold_id"] == latest_fold, "excess"].mean()
        ),
        "seed_mean_excess": {
            str(int(seed)): float(value) for seed, value in seed_means.items()
        },
        "strict_weekly_win_rate": float((frame["excess"] > 0.0).mean()),
        "excess_sum_remove_best2": excess_metrics["sum_remove_best2"],
        "candidate_cvar20": candidate_metrics["cvar20"],
        "control_cvar20": control_metrics["cvar20"],
        "max_candidate_hhi": float(frame["candidate_hhi"].max()),
        "best_positive_contribution": candidate_metrics[
            "best_positive_contribution"
        ],
        "cost_direction_consistent": cost_consistent,
        "paired_decision_count": int(len(frame)),
    }
    checks = {
        "positive_folds_at_least_3_of_4": metrics["positive_folds"] >= 3,
        "recent_fold_positive": metrics["recent_fold_excess"] > 0.0,
        "both_seed_means_positive": len(seed_means) == 2
        and bool((seed_means > 0.0).all()),
        "strict_weekly_win_rate_at_least_55pct": metrics[
            "strict_weekly_win_rate"
        ]
        >= 0.55,
        "excess_positive_after_best_two_removed": metrics[
            "excess_sum_remove_best2"
        ]
        > 0.0,
        "cvar20_not_worse_than_control": metrics["candidate_cvar20"]
        >= metrics["control_cvar20"],
        "hhi_at_most_0_30": metrics["max_candidate_hhi"] <= 0.30,
        "best_week_at_most_20pct_positive_sum": metrics[
            "best_positive_contribution"
        ]
        <= 0.20,
        "cost_conclusion_no_reversal": cost_consistent,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "metrics": metrics,
    }

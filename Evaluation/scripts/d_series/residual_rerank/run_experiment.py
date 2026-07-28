from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from checkpoint_scores import assert_byte_identical, sha256_file
from core import build_causal_factor_table, calibration_choice
from pipeline import (
    GRID_KEYS,
    build_forward_outcomes,
    build_grid_holdings,
    calibration_metrics,
    five_day_decision_dates,
    score_grid_locked,
)
from workflow import (
    compare_fingerprints,
    fingerprint_tree,
    quick_gate,
    verify_prediction_lock,
    write_prediction_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ID = "TEST_D_ALPHA_RESIDUAL_RERANK_20260728"
RUN_ROOT = (
    REPO_ROOT
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "automation_runs"
    / EXPERIMENT_ID
)
H0_ROOT = (
    REPO_ROOT
    / "Experimental_WorkSpace"
    / "candidates_H"
    / "automation_runs"
    / "TEST_H_QUICK"
    / "TEST_H0_CONTROL"
)
H_CODE_ROOT = (
    REPO_ROOT / "Experimental_WorkSpace" / "candidates_H" / "code" / "src"
)
D01_DATA = (
    REPO_ROOT
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "D01_FrozenResearchChampion_20260709"
    / "data"
    / "train.csv"
)
CANDIDATE_D = REPO_ROOT / "candidate" / "Candidate-D"
PROTECTED_ROOTS = {
    "candidate_d": CANDIDATE_D,
    "d01": (
        REPO_ROOT
        / "Experimental_WorkSpace"
        / "candidates_D"
        / "D01_FrozenResearchChampion_20260709"
    ),
    "official_reference": REPO_ROOT / "candidate" / "THU-BDC2026-main",
}
SEEDS = (20260726, 20260727)
OUTER_FOLDS = (
    ("2025Q1", "2025-01-03", "2025-03-31"),
    ("2025Q2", "2025-04-01", "2025-06-30"),
    ("2025Q3", "2025-07-01", "2025-09-30"),
    ("2026Q1", "2026-01-05", "2026-03-27"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )


def _status() -> dict[str, object]:
    return _read_json(RUN_ROOT / "TEST_STATUS.json")


def _update_status(**updates: object) -> None:
    payload = _status()
    payload.update(updates)
    payload["updated_at_utc"] = _utc_now()
    _write_json(RUN_ROOT / "TEST_STATUS.json", payload)


def _require_stage(*allowed: str) -> dict[str, object]:
    status = _status()
    if status.get("stage") not in allowed:
        raise RuntimeError(
            f"stage {status.get('stage')!r} is not one of the required stages {allowed}"
        )
    return status


def build_score_jobs() -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for seed in SEEDS:
        jobs.append(
            {
                "seed": seed,
                "phase": "inner",
                "fold_id": "CALIBRATION",
                "checkpoint_fold": "2025Q1",
                "start": "2024-10-08",
                "end": "2024-12-24",
            }
        )
        for fold_id, start, end in OUTER_FOLDS:
            jobs.append(
                {
                    "seed": seed,
                    "phase": "outer",
                    "fold_id": fold_id,
                    "checkpoint_fold": fold_id,
                    "start": start,
                    "end": end,
                }
            )
    return jobs


def _source_files() -> list[Path]:
    source_dir = Path(__file__).resolve().parent
    files = sorted(source_dir.glob("*.py"))
    files.extend(
        [
            RUN_ROOT / "PREREGISTRATION.md",
            RUN_ROOT / "EXPOSURE_LEDGER.json",
            RUN_ROOT / "config" / "RERANK_GRID.json",
            RUN_ROOT / "config" / "FOLDS.json",
            RUN_ROOT / "config" / "COST_AND_GATES.json",
        ]
    )
    return files


def initialize() -> None:
    _require_stage("PREREGISTERED", "INITIALIZED")
    started = time.perf_counter()
    manifest_dir = RUN_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    before = {
        name: fingerprint_tree(path) for name, path in PROTECTED_ROOTS.items()
    }
    _write_json(manifest_dir / "protected_before.json", before)
    write_prediction_lock(
        _source_files(),
        manifest_dir / "source_code.lock.json",
        {
            "marker": "TEST",
            "experiment_id": EXPERIMENT_ID,
            "purpose": "freeze code and preregistration before score export",
        },
    )

    sandbox = RUN_ROOT / "sandbox" / "Candidate-D"
    if sandbox.exists():
        source_fingerprint = fingerprint_tree(CANDIDATE_D)
        sandbox_fingerprint = fingerprint_tree(sandbox)
        comparison = compare_fingerprints(source_fingerprint, sandbox_fingerprint)
        if not comparison["identical"]:
            raise RuntimeError("existing Candidate-D sandbox differs from protected source")
    else:
        sandbox.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(CANDIDATE_D, sandbox, copy_function=shutil.copy2)
    sandbox_check = compare_fingerprints(
        before["candidate_d"],
        fingerprint_tree(sandbox),
    )
    if not sandbox_check["identical"]:
        raise RuntimeError("isolated Candidate-D copy is not byte-identical")
    _write_json(manifest_dir / "sandbox_copy_verification.json", sandbox_check)
    _write_json(
        manifest_dir / "initialize_timing.json",
        {"elapsed_seconds": time.perf_counter() - started, "completed_at_utc": _utc_now()},
    )
    _update_status(
        stage="INITIALIZED",
        allowed_next_stage="SCORE_EXPORT_AND_HOLDING_LOCK",
        protected_before_recorded=True,
        source_code_locked=True,
        sandbox_candidate_d_verified=True,
    )
    print("INITIALIZED")


def _run_subprocess(command: list[str], log_path: Path) -> None:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"elapsed_seconds={time.perf_counter() - started:.6f}",
                f"returncode={completed.returncode}",
                "[stdout]",
                completed.stdout,
                "[stderr]",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"subprocess failed; see {log_path}")


def _export_candidate_d_diagnostic() -> dict[str, object]:
    sandbox = RUN_ROOT / "sandbox" / "Candidate-D"
    output_dir = RUN_ROOT / "predictions" / "candidate_d_current_diagnostic"
    reports = []
    outputs = []
    for pass_id in (1, 2):
        output = output_dir / f"full_scores_pass{pass_id}.csv"
        report = output_dir / f"full_scores_pass{pass_id}.report.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "candidate_full_scores.py"),
            "--code-root",
            str(sandbox / "code" / "src"),
            "--config",
            str(sandbox / "model" / "config.json"),
            "--history",
            str(sandbox / "data" / "train.csv"),
            "--test-data",
            str(sandbox / "data" / "test.csv"),
            "--checkpoint",
            str(sandbox / "model" / "final_model.pth"),
            "--scalers",
            str(sandbox / "model" / "scalers.json"),
            "--output",
            str(output),
            "--report",
            str(report),
            "--experiment-id",
            f"{EXPERIMENT_ID}_CURRENT_DIAGNOSTIC",
            "--seed",
            "20260708",
        ]
        _run_subprocess(
            command,
            RUN_ROOT / "logs" / f"candidate_d_full_scores_pass{pass_id}.log",
        )
        outputs.append(output)
        reports.append(_read_json(report))
    identity = assert_byte_identical(outputs[0], outputs[1])
    frame = pd.read_csv(outputs[0], dtype={"stock_id": str})
    if len(frame) != 300 or frame["stock_id"].nunique() != 300:
        raise ValueError("isolated Candidate-D diagnostic did not export all 300 stocks")
    if set(frame.columns) != {
        "marker",
        "experiment_id",
        "decision_date",
        "stock_id",
        "score",
    }:
        raise ValueError("Candidate-D diagnostic score schema is not score-only")
    payload = {
        "role": "exposed_diagnostic_only_never_used_for_selection",
        "reports": reports,
        "byte_identity": identity,
    }
    _write_json(output_dir / "DOUBLE_RUN_VERIFICATION.json", payload)
    return payload


def _score_job_name(job: Mapping[str, object]) -> str:
    return (
        f"S{job['seed']}_{job['fold_id']}_{job['phase']}_"
        f"{job['start']}_{job['end']}"
    )


def historical_score_universe_check(frame: pd.DataFrame) -> dict[str, object]:
    required = {"decision_date", "stock_id"}
    if missing := required.difference(frame.columns):
        raise KeyError(f"missing historical score universe columns: {sorted(missing)}")
    counts = frame.groupby("decision_date")["stock_id"].agg(["size", "nunique"])
    checks = {
        "at_least_290_available_stocks": bool((counts["size"] >= 290).all()),
        "no_duplicate_stocks": bool((counts["size"] == counts["nunique"]).all()),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "min_stock_count": int(counts["size"].min()),
        "max_stock_count": int(counts["size"].max()),
    }


def _export_h0_scores(job: Mapping[str, object]) -> dict[str, object]:
    seed = int(job["seed"])
    checkpoint_fold = str(job["checkpoint_fold"])
    phase = str(job["phase"])
    source = H0_ROOT / f"S{seed}" / checkpoint_fold / phase
    output_dir = RUN_ROOT / "predictions" / _score_job_name(job)
    reports = []
    outputs = []
    for pass_id in (1, 2):
        output = output_dir / f"scores_pass{pass_id}.csv"
        report = output_dir / f"scores_pass{pass_id}.report.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "checkpoint_scores.py"),
            "--h-code-root",
            str(H_CODE_ROOT),
            "--config",
            str(source / "config.json"),
            "--data",
            str(D01_DATA),
            "--scalers",
            str(source / "scalers.json"),
            "--checkpoint",
            str(source / "best_model.pth"),
            "--start",
            str(job["start"]),
            "--end",
            str(job["end"]),
            "--output",
            str(output),
            "--report",
            str(report),
            "--experiment-id",
            f"{EXPERIMENT_ID}_{_score_job_name(job)}",
            "--seed",
            str(seed),
        ]
        _run_subprocess(
            command,
            RUN_ROOT
            / "logs"
            / f"{_score_job_name(job)}_scores_pass{pass_id}.log",
        )
        outputs.append(output)
        reports.append(_read_json(report))
    identity = assert_byte_identical(outputs[0], outputs[1])
    frame = pd.read_csv(outputs[0], dtype={"stock_id": str})
    universe = historical_score_universe_check(frame)
    if not universe["passed"]:
        raise ValueError(
            f"historical full-available score universe failed: {_score_job_name(job)}"
        )
    payload = {
        "job": dict(job),
        "source": str(source.resolve()),
        "reports": reports,
        "byte_identity": identity,
        "universe": universe,
        "canonical_output": str(outputs[0].resolve()),
    }
    _write_json(output_dir / "DOUBLE_RUN_VERIFICATION.json", payload)
    return payload


def _load_score_manifest(
    job_manifests: Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    frames = []
    for item in job_manifests:
        job = item["job"]
        frame = pd.read_csv(item["canonical_output"], dtype={"stock_id": str})
        frame["seed"] = int(job["seed"])
        frame["fold_id"] = str(job["fold_id"])
        frame["phase"] = str(job["phase"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _column_mapping(raw: pd.DataFrame) -> dict[str, str]:
    if len(raw.columns) < 8:
        raise ValueError("D01 market data has fewer than the required eight columns")
    return {
        "stock": str(raw.columns[0]),
        "date": str(raw.columns[1]),
        "open": str(raw.columns[2]),
        "close": str(raw.columns[3]),
        "high": str(raw.columns[4]),
        "low": str(raw.columns[5]),
        "volume": str(raw.columns[6]),
        "amount": str(raw.columns[7]),
    }


def _data_gate(scores: pd.DataFrame, factors: pd.DataFrame) -> dict[str, object]:
    selected_dates = []
    for _, group in scores.groupby(["seed", "fold_id", "phase"], sort=True):
        selected_dates.extend(five_day_decision_dates(group["decision_date"]))
    selected_dates = sorted(set(selected_dates))
    relevant = factors.loc[factors["decision_date"].isin(selected_dates)].copy()
    complete_counts = relevant.groupby("decision_date")["factor_complete"].sum()
    ranked_scores = scores.copy()
    ranked_scores["decision_date"] = pd.to_datetime(
        ranked_scores["decision_date"]
    ).dt.normalize()
    ranked_scores = ranked_scores.loc[
        ranked_scores["decision_date"].isin(selected_dates)
    ]
    top10 = (
        ranked_scores.sort_values(
            ["seed", "fold_id", "phase", "decision_date", "score", "stock_id"],
            ascending=[True, True, True, True, False, True],
        )
        .groupby(["seed", "fold_id", "phase", "decision_date"], sort=True)
        .head(10)
        .merge(
            relevant[["decision_date", "stock_id", "factor_complete"]],
            on=["decision_date", "stock_id"],
            how="left",
            validate="many_to_one",
        )
    )
    grouped_universe = scores.groupby(
        ["seed", "fold_id", "phase", "decision_date"]
    )["stock_id"].agg(["size", "nunique"])
    checks = {
        "score_full_available_universe_at_least_290": bool(
            (grouped_universe["size"] >= 290).all()
            and (grouped_universe["size"] == grouped_universe["nunique"]).all()
        ),
        "factor_complete_at_least_290": bool((complete_counts >= 290).all()),
        "candidate_d_top10_factors_complete": bool(
            top10["factor_complete"].fillna(False).all()
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "min_factor_complete_count": int(complete_counts.min()),
        "decision_date_count": int(len(selected_dates)),
        "top10_rows_checked": int(len(top10)),
    }


def _write_double_holdings(
    frame: pd.DataFrame,
    stem: str,
) -> tuple[Path, dict[str, object]]:
    output_dir = RUN_ROOT / "holdings"
    pass1 = output_dir / f"{stem}_pass1.csv"
    pass2 = output_dir / f"{stem}_pass2.csv"
    _write_csv(frame, pass1)
    _write_csv(frame.copy(), pass2)
    identity = assert_byte_identical(pass1, pass2)
    lock = write_prediction_lock(
        [pass1, pass2],
        output_dir / f"{stem}.lock.json",
        {
            "marker": "TEST",
            "experiment_id": EXPERIMENT_ID,
            "outcomes_read_before_lock": False,
            "byte_identical": True,
        },
    )
    return pass1, {"byte_identity": identity, "lock": lock}


def prepare() -> None:
    _require_stage("INITIALIZED")
    verify_prediction_lock(RUN_ROOT / "manifests" / "source_code.lock.json")
    started = time.perf_counter()
    current_diagnostic = _export_candidate_d_diagnostic()
    job_manifests = []
    for index, job in enumerate(build_score_jobs(), start=1):
        print(f"EXPORT {index}/10 {_score_job_name(job)}", flush=True)
        job_manifests.append(_export_h0_scores(job))
    _write_json(
        RUN_ROOT / "manifests" / "score_export_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "candidate_d_current_diagnostic": current_diagnostic,
            "jobs": job_manifests,
        },
    )
    scores = _load_score_manifest(job_manifests)
    raw = pd.read_csv(D01_DATA)
    mapping = _column_mapping(raw)
    selected_dates: set[pd.Timestamp] = set()
    for _, group in scores.groupby(["seed", "fold_id", "phase"], sort=True):
        selected_dates.update(five_day_decision_dates(group["decision_date"]))
    factors = build_causal_factor_table(raw, sorted(selected_dates), mapping)
    factors_path = RUN_ROOT / "factors" / "lag1_residual_factors.csv"
    _write_csv(factors, factors_path)
    write_prediction_lock(
        [factors_path],
        RUN_ROOT / "factors" / "lag1_residual_factors.lock.json",
        {
            "marker": "TEST",
            "asof_rule": "strictly earlier than decision date",
            "external_data_enabled": False,
        },
    )
    gate = _data_gate(scores, factors)
    _write_json(RUN_ROOT / "reports" / "DATA_GATE.json", gate)
    if not gate["passed"]:
        _update_status(
            stage="TERMINAL",
            status="REJECTED_AT_DATA_GATE",
            allowed_next_stage="FINALIZE_ONLY",
            calibration_outcomes_read=False,
            outer_outcomes_read=False,
        )
        print("REJECTED_AT_DATA_GATE")
        return

    holdings = build_grid_holdings(scores, factors)
    calibration = holdings.loc[holdings["phase"] == "inner"].copy()
    outer = holdings.loc[holdings["phase"] == "outer"].copy()
    calibration_path, calibration_lock = _write_double_holdings(
        calibration,
        "calibration_grid",
    )
    outer_path, outer_lock = _write_double_holdings(outer, "outer_grid")
    _write_json(
        RUN_ROOT / "manifests" / "holding_lock_manifest.json",
        {
            "calibration_path": str(calibration_path.resolve()),
            "calibration": calibration_lock,
            "outer_grid_path": str(outer_path.resolve()),
            "outer_grid": outer_lock,
            "calibration_outcomes_read_before_lock": False,
            "outer_outcomes_read_before_lock": False,
        },
    )
    _write_json(
        RUN_ROOT / "manifests" / "prepare_timing.json",
        {"elapsed_seconds": time.perf_counter() - started, "completed_at_utc": _utc_now()},
    )
    _update_status(
        stage="PREPARED_AND_LOCKED",
        allowed_next_stage="CALIBRATION_SCORER",
        data_gate_passed=True,
        calibration_outcomes_read=False,
        outer_outcomes_read=False,
    )
    print("PREPARED_AND_LOCKED")


def select_locked_outer_holdings(
    outer_grid: pd.DataFrame,
    selected_lambdas: Mapping[str, float],
) -> pd.DataFrame:
    frame = outer_grid.copy()
    selected_mask = np.ones(len(frame), dtype=bool)
    zero_mask = np.ones(len(frame), dtype=bool)
    for key in GRID_KEYS:
        selected_mask &= np.isclose(
            pd.to_numeric(frame[key]).to_numpy(dtype=float),
            float(selected_lambdas[key]),
            atol=1e-12,
            rtol=0.0,
        )
        zero_mask &= np.isclose(
            pd.to_numeric(frame[key]).to_numpy(dtype=float),
            0.0,
            atol=1e-12,
            rtol=0.0,
        )
    keep = (
        (zero_mask & frame["policy_id"].isin(
            [
                "TOP2_EQUAL",
                "TOP5_EQUAL",
                "TOP5_FIXED_30_25_20_15_10",
            ]
        ))
        | (
            selected_mask
            & frame["policy_id"].isin(
                ["TOP5_EQUAL", "TOP5_FIXED_30_25_20_15_10"]
            )
        )
    )
    columns = list(frame.columns)
    return (
        frame.loc[keep, columns]
        .sort_values(
            [
                "seed",
                "fold_id",
                "decision_date",
                "policy_id",
                *GRID_KEYS,
                "portfolio_rank",
                "stock_id",
            ]
        )
        .reset_index(drop=True)
    )


def calibrate() -> None:
    _require_stage("PREPARED_AND_LOCKED")
    started = time.perf_counter()
    holdings_path = RUN_ROOT / "holdings" / "calibration_grid_pass1.csv"
    lock_path = RUN_ROOT / "holdings" / "calibration_grid.lock.json"
    verify_prediction_lock(lock_path)
    holdings = pd.read_csv(holdings_path, dtype={"stock_id": str})
    decision_dates = sorted(
        pd.to_datetime(holdings["decision_date"]).dt.normalize().unique()
    )
    raw = pd.read_csv(D01_DATA)
    outcomes = build_forward_outcomes(raw, decision_dates, _column_mapping(raw))
    outcomes_path = RUN_ROOT / "outcomes" / "CALIBRATION_OUTCOMES.csv"
    _write_csv(outcomes, outcomes_path)
    write_prediction_lock(
        [outcomes_path],
        RUN_ROOT / "outcomes" / "CALIBRATION_OUTCOMES.lock.json",
        {
            "marker": "TEST",
            "decision_start": "2024-10-08",
            "decision_end": "2024-12-24",
            "purpose": "lambda calibration only",
        },
    )
    scored = score_grid_locked(
        holdings_path,
        outcomes,
        lock_path,
        cost_bps_per_side=10,
    )
    scored_path = RUN_ROOT / "scores" / "CALIBRATION_GRID_10BPS.csv"
    _write_csv(scored, scored_path)
    metrics = calibration_metrics(scored)
    metrics_path = RUN_ROOT / "scores" / "CALIBRATION_METRICS.csv"
    _write_csv(metrics, metrics_path)
    choice = calibration_choice(metrics)
    choice["calibration_outcome_sha256"] = sha256_file(outcomes_path)
    choice["calibration_holding_lock_sha256"] = sha256_file(lock_path)
    _write_json(RUN_ROOT / "reports" / "CALIBRATION_DECISION.json", choice)
    if choice["status"] != "PASS":
        _update_status(
            stage="TERMINAL",
            status="REJECTED_AT_CALIBRATION_GATE",
            allowed_next_stage="FINALIZE_ONLY",
            calibration_outcomes_read=True,
            outer_outcomes_read=False,
        )
        print("REJECTED_AT_CALIBRATION_GATE")
        return

    outer_grid = pd.read_csv(
        RUN_ROOT / "holdings" / "outer_grid_pass1.csv",
        dtype={"stock_id": str},
    )
    selected = select_locked_outer_holdings(
        outer_grid,
        choice["selected_lambdas"],
    )
    selected_path, selected_lock = _write_double_holdings(
        selected,
        "selected_outer",
    )
    _write_json(
        RUN_ROOT / "manifests" / "selected_outer_lock_manifest.json",
        {
            "selected_lambdas": choice["selected_lambdas"],
            "selected_path": str(selected_path.resolve()),
            "lock": selected_lock,
            "outer_outcomes_read_before_lock": False,
        },
    )
    _write_json(
        RUN_ROOT / "manifests" / "calibration_timing.json",
        {"elapsed_seconds": time.perf_counter() - started, "completed_at_utc": _utc_now()},
    )
    _update_status(
        stage="CALIBRATION_PASSED_AND_OUTER_LOCKED",
        status="IN_PROGRESS",
        allowed_next_stage="ONE_SHOT_OUTER_SCORER",
        calibration_outcomes_read=True,
        outer_outcomes_read=False,
        selected_lambdas=choice["selected_lambdas"],
    )
    print("CALIBRATION_PASSED_AND_OUTER_LOCKED")


def _lambda_mask(
    frame: pd.DataFrame,
    lambdas: Mapping[str, float],
) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for key in GRID_KEYS:
        mask &= np.isclose(
            pd.to_numeric(frame[key]).to_numpy(dtype=float),
            float(lambdas[key]),
            atol=1e-12,
            rtol=0.0,
        )
    return mask


def _pair_returns(
    scored: pd.DataFrame,
    *,
    candidate_policy: str,
    candidate_lambdas: Mapping[str, float],
    control_policy: str,
    control_lambdas: Mapping[str, float],
) -> pd.DataFrame:
    keys = ["seed", "fold_id", "decision_date"]
    candidate = scored.loc[
        (scored["policy_id"] == candidate_policy)
        & _lambda_mask(scored, candidate_lambdas)
    ][keys + ["net_return", "hhi"]].rename(
        columns={
            "net_return": "candidate_net_return",
            "hhi": "candidate_hhi",
        }
    )
    control = scored.loc[
        (scored["policy_id"] == control_policy)
        & _lambda_mask(scored, control_lambdas)
    ][keys + ["net_return", "hhi"]].rename(
        columns={
            "net_return": "control_net_return",
            "hhi": "control_hhi",
        }
    )
    paired = candidate.merge(control, on=keys, how="inner", validate="one_to_one")
    if len(paired) != len(candidate) or len(paired) != len(control):
        raise ValueError("candidate and control returns are not exactly paired")
    return paired.sort_values(keys).reset_index(drop=True)


def _cost_direction_consistent(
    pairs_by_cost: Mapping[int, pd.DataFrame],
) -> bool:
    reference = pairs_by_cost[10].copy()
    reference["excess"] = (
        reference["candidate_net_return"] - reference["control_net_return"]
    )
    for cost in (0, 20):
        other = pairs_by_cost[cost].copy()
        other["excess"] = (
            other["candidate_net_return"] - other["control_net_return"]
        )
        merged = reference.merge(
            other,
            on=["seed", "fold_id", "decision_date"],
            suffixes=("_10", f"_{cost}"),
            validate="one_to_one",
        )
        if not np.allclose(
            merged["excess_10"].to_numpy(dtype=float),
            merged[f"excess_{cost}"].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        ):
            return False
    return True


def attribution_gate(full_gate_result: Mapping[str, object]) -> dict[str, object]:
    source = full_gate_result["checks"]
    names = (
        "positive_folds_at_least_3_of_4",
        "both_seed_means_positive",
        "strict_weekly_win_rate_at_least_55pct",
        "excess_positive_after_best_two_removed",
    )
    checks = {name: bool(source[name]) for name in names}
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "metrics": full_gate_result.get("metrics", {}),
    }


def score_outer() -> None:
    status = _require_stage("CALIBRATION_PASSED_AND_OUTER_LOCKED")
    started = time.perf_counter()
    selected_lambdas = status["selected_lambdas"]
    zero = {key: 0.0 for key in GRID_KEYS}
    holdings_path = RUN_ROOT / "holdings" / "selected_outer_pass1.csv"
    lock_path = RUN_ROOT / "holdings" / "selected_outer.lock.json"
    verify_prediction_lock(lock_path)
    holdings = pd.read_csv(holdings_path, dtype={"stock_id": str})
    decision_dates = sorted(
        pd.to_datetime(holdings["decision_date"]).dt.normalize().unique()
    )
    raw = pd.read_csv(D01_DATA)
    outcomes = build_forward_outcomes(raw, decision_dates, _column_mapping(raw))
    outcomes_path = RUN_ROOT / "outcomes" / "OUTER_OUTCOMES_ONE_SHOT.csv"
    _write_csv(outcomes, outcomes_path)
    write_prediction_lock(
        [outcomes_path],
        RUN_ROOT / "outcomes" / "OUTER_OUTCOMES_ONE_SHOT.lock.json",
        {
            "marker": "TEST",
            "folds": [item[0] for item in OUTER_FOLDS],
            "read_count": 1,
            "purpose": "one-shot Quick scoring",
        },
    )
    scored_by_cost: dict[int, pd.DataFrame] = {}
    for cost in (0, 10, 20):
        scored = score_grid_locked(
            holdings_path,
            outcomes,
            lock_path,
            cost_bps_per_side=cost,
        )
        scored_by_cost[cost] = scored
        _write_csv(
            scored,
            RUN_ROOT / "scores" / f"OUTER_SELECTED_{cost}BPS.csv",
        )

    primary_pairs: dict[int, pd.DataFrame] = {}
    attribution_pairs: dict[int, pd.DataFrame] = {}
    fixed_pairs: dict[int, pd.DataFrame] = {}
    for cost, scored in scored_by_cost.items():
        primary_pairs[cost] = _pair_returns(
            scored,
            candidate_policy="TOP5_EQUAL",
            candidate_lambdas=selected_lambdas,
            control_policy="TOP2_EQUAL",
            control_lambdas=zero,
        )
        attribution_pairs[cost] = _pair_returns(
            scored,
            candidate_policy="TOP5_EQUAL",
            candidate_lambdas=selected_lambdas,
            control_policy="TOP5_EQUAL",
            control_lambdas=zero,
        )
        fixed_pairs[cost] = _pair_returns(
            scored,
            candidate_policy="TOP5_FIXED_30_25_20_15_10",
            candidate_lambdas=selected_lambdas,
            control_policy="TOP2_EQUAL",
            control_lambdas=zero,
        )
    primary_consistent = _cost_direction_consistent(primary_pairs)
    attribution_consistent = _cost_direction_consistent(attribution_pairs)
    primary = primary_pairs[10].copy()
    primary["cost_direction_consistent"] = primary_consistent
    attribution = attribution_pairs[10].copy()
    attribution["cost_direction_consistent"] = attribution_consistent
    fixed = fixed_pairs[10].copy()
    fixed["cost_direction_consistent"] = _cost_direction_consistent(fixed_pairs)
    _write_csv(primary, RUN_ROOT / "scores" / "PAIRED_PRIMARY_10BPS.csv")
    _write_csv(attribution, RUN_ROOT / "scores" / "PAIRED_ATTRIBUTION_10BPS.csv")
    _write_csv(fixed, RUN_ROOT / "scores" / "PAIRED_FIXED_DIAGNOSTIC_10BPS.csv")

    primary_gate = quick_gate(primary)
    attribution_full = quick_gate(attribution)
    attribution_result = attribution_gate(attribution_full)
    fixed_diagnostic = quick_gate(fixed)
    passed = bool(primary_gate["passed"] and attribution_result["passed"])
    result = {
        "experiment_id": EXPERIMENT_ID,
        "marker": "TEST",
        "selected_lambdas": selected_lambdas,
        "primary_vs_candidate_d_top2": primary_gate,
        "attribution_vs_zero_residual_top5": attribution_result,
        "fixed_30_25_20_15_10_diagnostic_only": fixed_diagnostic,
        "outer_outcome_sha256": sha256_file(outcomes_path),
        "selected_holding_lock_sha256": sha256_file(lock_path),
        "terminal_status": (
            "PASS_AS_TEST_ONLY" if passed else "REJECTED_AT_QUICK_GATE"
        ),
        "training_authorized": False,
        "promotion_authorized": False,
    }
    _write_json(RUN_ROOT / "reports" / "QUICK_GATE_RESULTS.json", result)
    _write_json(
        RUN_ROOT / "manifests" / "outer_score_timing.json",
        {"elapsed_seconds": time.perf_counter() - started, "completed_at_utc": _utc_now()},
    )
    _update_status(
        stage="TERMINAL",
        status=result["terminal_status"],
        allowed_next_stage="FINALIZE_ONLY",
        outer_outcomes_read=True,
        outer_score_count=1,
        training_performed=False,
        candidate_promotion_performed=False,
    )
    print(result["terminal_status"])


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def finalize() -> None:
    _require_stage("TERMINAL")
    before = _read_json(RUN_ROOT / "manifests" / "protected_before.json")
    after = {
        name: fingerprint_tree(path) for name, path in PROTECTED_ROOTS.items()
    }
    _write_json(RUN_ROOT / "manifests" / "protected_after.json", after)
    comparisons = {
        name: compare_fingerprints(before[name], after[name])
        for name in PROTECTED_ROOTS
    }
    protected = bool(all(item["identical"] for item in comparisons.values()))
    _write_json(
        RUN_ROOT / "reports" / "PROTECTED_TREE_VERIFICATION.json",
        {"passed": protected, "comparisons": comparisons},
    )
    status = _status()
    if not protected:
        _update_status(
            status="ABORTED_PROTECTION_FAILURE",
            protection_verification_passed=False,
            allowed_next_stage="NONE",
        )
        status = _status()
    else:
        _update_status(
            protection_verification_passed=True,
            allowed_next_stage="NONE",
        )
        status = _status()

    gate_path = RUN_ROOT / "reports" / "QUICK_GATE_RESULTS.json"
    calibration_path = RUN_ROOT / "reports" / "CALIBRATION_DECISION.json"
    gate = _read_json(gate_path) if gate_path.is_file() else None
    calibration = (
        _read_json(calibration_path) if calibration_path.is_file() else None
    )
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "marker": "TEST",
        "terminal_status": status["status"],
        "calibration": calibration,
        "quick_gate": gate,
        "protected_trees_unchanged": protected,
        "training_performed": False,
        "candidate_promotion_performed": False,
        "run_disk_bytes": _directory_size(RUN_ROOT),
        "completed_at_utc": _utc_now(),
    }
    _write_json(RUN_ROOT / "FINAL_REPORT.json", summary)
    lines = [
        f"# {EXPERIMENT_ID} final TEST report",
        "",
        f"- Terminal status: `{status['status']}`",
        f"- Protected trees unchanged: `{protected}`",
        "- Training performed: `false`",
        "- Candidate promotion performed: `false`",
        f"- Run disk bytes: `{summary['run_disk_bytes']}`",
    ]
    if calibration:
        lines.extend(
            [
                f"- Calibration status: `{calibration['status']}`",
                f"- Selected lambdas: `{calibration.get('selected_lambdas')}`",
            ]
        )
    if gate:
        primary = gate["primary_vs_candidate_d_top2"]
        attribution = gate["attribution_vs_zero_residual_top5"]
        lines.extend(
            [
                f"- Primary Quick gate passed: `{primary['passed']}`",
                f"- Primary failed checks: `{primary['failed_checks']}`",
                f"- Attribution gate passed: `{attribution['passed']}`",
                f"- Attribution failed checks: `{attribution['failed_checks']}`",
            ]
        )
    lines.extend(
        [
            "",
            "This is TEST evidence only. It does not authorize training, H01,",
            "promotion, packaging, or any write to a formal candidate.",
            "",
        ]
    )
    (RUN_ROOT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(status["status"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-gated Candidate-D inference-only residual rerank TEST"
    )
    parser.add_argument(
        "stage",
        choices=("initialize", "prepare", "calibrate", "score-outer", "finalize"),
    )
    return parser.parse_args()


def main() -> None:
    stage = parse_args().stage
    functions = {
        "initialize": initialize,
        "prepare": prepare,
        "calibrate": calibrate,
        "score-outer": score_outer,
        "finalize": finalize,
    }
    functions[stage]()


if __name__ == "__main__":
    main()

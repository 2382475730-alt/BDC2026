from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    ROOT
    / "Evaluation"
    / "configs"
    / "walk_forward"
    / "TEST_D_PROSPECTIVE_8W_20260727.json"
)
DEFAULT_DATA_PLAN = (
    ROOT
    / "Evaluation"
    / "configs"
    / "walk_forward"
    / "TEST_D_PROSPECTIVE_DATA_PLAN_20260727.json"
)
DEFAULT_EXPERIMENT_ROOT = (
    ROOT
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "automation_runs"
    / "TEST_D_PROSPECTIVE_8W_20260727"
)
DEFAULT_PUBLIC_RUN = ROOT / "Evaluation" / "runs" / "TEST_D_PROSPECTIVE_8W_20260727"
DEFAULT_PYTHON = (
    ROOT
    / "Experimental_WorkSpace"
    / "shared_envs"
    / "pytorch-cu128-py311"
    / "Scripts"
    / "python.exe"
)

COL_CODE = "股票代码"
COL_DATE = "日期"
COL_OPEN = "开盘"
PUBLIC_SCORE_FORBIDDEN_KEYS = {
    "weekly_return",
    "return_gross",
    "return_net",
    "candidate_d_net",
    "official_net",
    "sse_return",
    "hs300_return",
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    source_relative: str
    train_args: tuple[str, ...]
    predict_args: tuple[str, ...]
    checkpoint_paths: tuple[str, ...]
    config_paths: tuple[str, ...]


CANDIDATE_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="THU-BDC2026-main",
        source_relative="candidate/THU-BDC2026-main",
        train_args=("code/src/train.py",),
        predict_args=("code/src/predict.py",),
        checkpoint_paths=(
            "model/60_158+39/best_model.pth",
            "model/60_158+39/scaler.pkl",
        ),
        config_paths=(
            "code/src/config.py",
            "model/60_158+39/config.json",
        ),
    ),
    CandidateSpec(
        name="Candidate-D",
        source_relative="candidate/Candidate-D",
        train_args=("code/src/train.py", "--work-dir", "."),
        predict_args=(
            "code/src/test.py",
            "--work-dir",
            ".",
            "--history",
            "data/train.csv",
            "--test-data",
            "",
            "--output",
            "output/result.csv",
        ),
        checkpoint_paths=(
            "model/final_model.pth",
            "model/scalers.json",
        ),
        config_paths=("model/config.json",),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def local_now(timezone_name: str = "Asia/Shanghai") -> datetime:
    return datetime.now(ZoneInfo(timezone_name))


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include timezone offset: {value}")
    return parsed


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, *, refuse_overwrite: bool = False) -> None:
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"refusing to overwrite sealed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def tree_manifest(
    root: Path,
    *,
    excluded_names: set[str] | None = None,
) -> dict[str, Any]:
    excluded_names = excluded_names or set()
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if any(part in excluded_names for part in path.relative_to(root).parts):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    tree_hash = sha256_bytes(canonical_json_bytes(files))
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": int(sum(row["bytes"] for row in files)),
        "tree_sha256": tree_hash,
        "files": files,
    }


def validate_preregistration(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != "test_d_prospective_8w_v1":
        raise ValueError("invalid prospective protocol schema")
    if config.get("classification") != "PROSPECTIVE_SEQUENTIAL_DIAGNOSTIC":
        raise ValueError("classification must be PROSPECTIVE_SEQUENTIAL_DIAGNOSTIC")
    windows = list(config.get("windows") or [])
    if len(windows) < 8:
        raise ValueError("prospective protocol requires at least 8 preregistered windows")
    if int(config.get("minimum_valid_windows", 0)) < 8:
        raise ValueError("minimum_valid_windows must be at least 8")
    if config.get("performance_blinding") != "UNTIL_MINIMUM_WINDOW_COUNT":
        raise ValueError("performance must remain blinded until the minimum window count")
    if config.get("candidates") != ["THU-BDC2026-main", "Candidate-D"]:
        raise ValueError("core candidates must be official reference and Candidate-D")

    exposed = {
        (
            row["decision_date"],
            row["buy_date"],
            row["sell_date"],
        )
        for row in config.get("excluded_exposed_periods", [])
    }
    seen_ids: set[str] = set()
    previous_buy: str | None = None
    previous_sell: str | None = None
    for window in windows:
        window_id = str(window.get("window_id", ""))
        if not window_id or window_id in seen_ids:
            raise ValueError(f"duplicate or empty window_id: {window_id}")
        seen_ids.add(window_id)
        triple = (
            window["decision_date"],
            window["buy_date"],
            window["sell_date"],
        )
        if triple in exposed:
            raise ValueError(f"window {window_id} reuses an exposed period")
        decision_lock = parse_timestamp(window["decision_lock_time"])
        buy_deadline = parse_timestamp(window["buy_deadline"])
        score_not_before = parse_timestamp(window["score_not_before"])
        if not decision_lock < buy_deadline < score_not_before:
            raise ValueError(f"window {window_id} has invalid time gates")
        if decision_lock.date().isoformat() != window["decision_date"]:
            raise ValueError(f"window {window_id} decision timestamp/date mismatch")
        if buy_deadline.date().isoformat() != window["buy_date"]:
            raise ValueError(f"window {window_id} buy timestamp/date mismatch")
        if score_not_before.date().isoformat() != window["sell_date"]:
            raise ValueError(f"window {window_id} sell timestamp/date mismatch")
        if not window["decision_date"] < window["buy_date"] <= window["sell_date"]:
            raise ValueError(f"window {window_id} chronology is invalid")
        if previous_buy is not None and window["buy_date"] <= previous_buy:
            raise ValueError("windows are not strictly ordered")
        if previous_sell is not None and window["buy_date"] <= previous_sell:
            raise ValueError("investment windows overlap")
        previous_buy = window["buy_date"]
        previous_sell = window["sell_date"]

    frozen = config.get("frozen_protocol", {}).get("candidate_d", {})
    expected = {
        "seed": 20260708,
        "epochs": 12,
        "sequence_length": 60,
        "top_k": 2,
        "weights": [0.5, 0.5],
    }
    if frozen != expected:
        raise ValueError(f"Candidate-D frozen protocol mismatch: {frozen}")
    if config.get("frozen_protocol", {}).get("cost_bps_per_side") != [0, 10, 20]:
        raise ValueError("cost sensitivity must be frozen at 0/10/20 bps per side")
    gates = config.get("decision_gates", {})
    if int(gates.get("minimum_valid_windows", 0)) < 8:
        raise ValueError("decision gates require at least 8 valid windows")
    return config


def window_by_id(config: dict[str, Any], window_id: str) -> dict[str, Any]:
    matches = [row for row in config["windows"] if row["window_id"] == window_id]
    if len(matches) != 1:
        raise KeyError(f"unknown window_id: {window_id}")
    return matches[0]


def window_phase(
    window: dict[str, Any],
    now: datetime,
    has_lock: bool,
    has_score: bool,
) -> str:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    decision_lock = parse_timestamp(window["decision_lock_time"])
    buy_deadline = parse_timestamp(window["buy_deadline"])
    score_not_before = parse_timestamp(window["score_not_before"])
    if has_score:
        return "SCORED"
    if has_lock:
        if now >= score_not_before:
            return "READY_TO_SCORE"
        return "LOCKED"
    if now >= buy_deadline:
        return "MISSED"
    if now < decision_lock:
        return "WAIT_DECISION"
    return "PREPARE_AND_LOCK"


def validate_training_prefix(path: Path, decision_date: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={COL_CODE: str}, encoding="utf-8-sig")
    required = {COL_CODE, COL_DATE, COL_OPEN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"training prefix missing columns: {sorted(missing)}")
    frame[COL_CODE] = frame[COL_CODE].map(normalize_code)
    frame[COL_DATE] = pd.to_datetime(frame[COL_DATE], errors="raise").dt.normalize()
    if frame.empty:
        raise ValueError("training prefix is empty")
    cutoff = pd.Timestamp(decision_date).normalize()
    maximum = frame[COL_DATE].max()
    if maximum > cutoff:
        raise ValueError(
            f"training prefix contains future rows through {maximum.date()} after {decision_date}"
        )
    if maximum < cutoff:
        raise ValueError(
            f"training prefix does not contain the decision date {decision_date}"
        )
    if frame.duplicated([COL_CODE, COL_DATE]).any():
        raise ValueError("training prefix contains duplicate stock/date rows")
    return {
        "path": str(path.resolve()),
        "rows": int(len(frame)),
        "stocks": int(frame[COL_CODE].nunique()),
        "date_start": str(frame[COL_DATE].min().date()),
        "date_end": str(maximum.date()),
        "sha256": sha256_file(path),
        "future_rows": 0,
    }


def validate_scoring_prefix_dates(
    frame: pd.DataFrame,
    sell_date: str,
) -> dict[str, Any]:
    if COL_DATE not in frame.columns:
        raise ValueError(f"scoring prefix missing {COL_DATE}")
    dates = pd.to_datetime(frame[COL_DATE], errors="raise").dt.normalize()
    if dates.empty:
        raise ValueError("scoring prefix is empty")
    sell = pd.Timestamp(sell_date).normalize()
    maximum = dates.max()
    if maximum > sell:
        raise ValueError(
            f"scoring prefix contains later-window data through {maximum.date()}"
        )
    if maximum < sell:
        raise ValueError(f"scoring prefix does not contain sell date {sell_date}")
    return {
        "date_start": dates.min().date().isoformat(),
        "date_end": maximum.date().isoformat(),
        "later_window_rows": 0,
    }


def derive_candidate_d_causal_metadata(
    path: Path,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=[COL_CODE, COL_DATE], dtype={COL_CODE: str})
    frame[COL_DATE] = pd.to_datetime(frame[COL_DATE], errors="raise").dt.normalize()
    minimum_stocks = int(base_config.get("data", {}).get("min_stocks_per_day", 10))
    counts = frame.groupby(COL_DATE)[COL_CODE].nunique().sort_index()
    eligible_dates = counts[counts >= minimum_stocks].index
    sell_offset = int(base_config.get("label", {}).get("sell_offset", 5))
    if len(eligible_dates) <= sell_offset:
        raise ValueError("not enough eligible trading sessions for Candidate-D labels")
    sample_end = pd.Timestamp(eligible_dates[-(sell_offset + 1)]).normalize()
    sample_start = pd.Timestamp(base_config["submission"]["train_sample_start"]).normalize()
    sample_dates = eligible_dates[
        (eligible_dates >= sample_start) & (eligible_dates <= sample_end)
    ]
    if len(sample_dates) == 0:
        raise ValueError("no Candidate-D production samples in causal prefix")
    return {
        "train_sample_end": sample_end.date().isoformat(),
        "train_sample_count": int(len(sample_dates)),
        "label_end_not_after": pd.Timestamp(eligible_dates[-1]).date().isoformat(),
    }


def strip_candidate_d_runtime_fields(config: dict[str, Any]) -> dict[str, Any]:
    stripped = deepcopy(config)
    stripped["experiment_id"] = "<RUNTIME_EXPERIMENT_ID>"
    submission = stripped.setdefault("submission", {})
    for key in (
        "raw_data_end",
        "train_sample_end",
        "expected_train_sample_count",
    ):
        submission[key] = f"<RUNTIME_{key.upper()}>"
    return stripped


def build_candidate_d_runtime_config(
    base_config: dict[str, Any],
    *,
    experiment_id: str,
    decision_date: str,
    train_sample_end: str,
    train_sample_count: int,
) -> dict[str, Any]:
    runtime = deepcopy(base_config)
    runtime["experiment_id"] = experiment_id
    runtime["submission"]["raw_data_end"] = decision_date
    runtime["submission"]["train_sample_end"] = train_sample_end
    runtime["submission"]["expected_train_sample_count"] = int(train_sample_count)
    if strip_candidate_d_runtime_fields(runtime) != strip_candidate_d_runtime_fields(
        base_config
    ):
        raise ValueError("runtime patch changed fields outside the causal metadata allowlist")
    return runtime


def validate_result(path: Path, universe: set[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"stock_id": str}, encoding="utf-8-sig")
    if list(frame.columns) != ["stock_id", "weight"]:
        raise ValueError("result columns must be exactly stock_id,weight")
    frame["stock_id"] = frame["stock_id"].map(normalize_code)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if not 1 <= len(frame) <= 5:
        raise ValueError("result must contain one to five stocks")
    if frame["stock_id"].duplicated().any():
        raise ValueError("result contains duplicate stocks")
    if universe is not None and not set(frame["stock_id"]).issubset(universe):
        raise ValueError("result contains stock outside the causal training universe")
    if not np.isfinite(frame["weight"]).all() or (frame["weight"] < 0).any():
        raise ValueError("result contains invalid weight")
    if float(frame["weight"].sum()) > 1.0 + 1e-10:
        raise ValueError("result weight sum exceeds one")
    return frame


def build_prediction_lock(
    *,
    window: dict[str, Any],
    candidate: str,
    locked_at: str,
    training_sha256: str,
    checkpoint_hashes: dict[str, str],
    config_hashes: dict[str, str],
    result_paths: Sequence[Path],
) -> dict[str, Any]:
    timestamp = parse_timestamp(locked_at)
    if timestamp >= parse_timestamp(window["buy_deadline"]):
        raise ValueError("prediction lock missed the buy deadline")
    if len(result_paths) != 2:
        raise ValueError("prediction lock requires exactly two inference runs")
    hashes = [sha256_file(path) for path in result_paths]
    if len(set(hashes)) != 1:
        raise ValueError("two inference results are not byte-identical")
    parsed = validate_result(result_paths[0])
    if candidate == "Candidate-D":
        if len(parsed) != 2 or not np.allclose(
            parsed["weight"].to_numpy(dtype=float),
            np.asarray([0.5, 0.5], dtype=float),
            atol=1e-12,
        ):
            raise ValueError("Candidate-D must remain frozen at Top-2 equal weights")
    return {
        "schema_version": "test_d_prospective_prediction_lock_v1",
        "status": "LOCKED",
        "window_id": window["window_id"],
        "candidate": candidate,
        "decision_date": window["decision_date"],
        "buy_date": window["buy_date"],
        "sell_date": window["sell_date"],
        "locked_at": timestamp.isoformat(),
        "buy_deadline": window["buy_deadline"],
        "training_data_sha256": training_sha256,
        "checkpoint_hashes": checkpoint_hashes,
        "config_hashes": config_hashes,
        "result_path": str(result_paths[0].resolve()),
        "result_sha256": hashes[0],
        "result_rows": int(len(parsed)),
        "result_weight_sum": float(parsed["weight"].sum()),
        "selected_stocks": parsed["stock_id"].tolist(),
        "inference_result_hashes": hashes,
        "repeated_inference_byte_identical": True,
        "future_returns_accessed": False,
    }


def append_hash_chained_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    previous_hash = "0" * 64
    sequence = 1
    if path.exists():
        verified = validate_audit_chain(path)
        previous_hash = verified["head_sha256"]
        sequence = verified["event_count"] + 1
    core = {
        "sequence": sequence,
        "previous_sha256": previous_hash,
        "event": event,
    }
    row = {
        **core,
        "event_sha256": sha256_bytes(canonical_json_bytes(core)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return row


def validate_audit_chain(path: Path) -> dict[str, Any]:
    previous_hash = "0" * 64
    count = 0
    for count, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        row = json.loads(line)
        core = {
            "sequence": row.get("sequence"),
            "previous_sha256": row.get("previous_sha256"),
            "event": row.get("event"),
        }
        expected = sha256_bytes(canonical_json_bytes(core))
        if (
            row.get("sequence") != count
            or row.get("previous_sha256") != previous_hash
            or row.get("event_sha256") != expected
        ):
            raise ValueError(f"audit chain verification failed at event {count}")
        previous_hash = expected
    return {
        "status": "PASS",
        "event_count": count,
        "head_sha256": previous_hash,
    }


def audit_events_for_window(
    path: Path,
    *,
    event_name: str,
    window_id: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    validate_audit_chain(path)
    matches: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)["event"]
        if (
            event.get("event") == event_name
            and str(event.get("window_id")) == window_id
        ):
            matches.append(event)
    return matches


def _frozen_program_path(repository_root: Path, declared_path: str) -> Path:
    root = repository_root.resolve()
    candidate = (root / declared_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"frozen program escapes repository root: {declared_path}"
        ) from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def verify_execution_freeze(
    *,
    data_plan_path: Path,
    experiment_root: Path,
    repository_root: Path = ROOT,
    controller_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the pre-P01 plan and every executable against the audit chain."""

    audit_path = experiment_root / "audit" / "events.jsonl"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    chain = validate_audit_chain(audit_path)
    activation_rows: list[dict[str, Any]] = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (
            row["event"].get("event")
            == "EXECUTION_FREEZE_ACTIVATED_BEFORE_P01"
        ):
            activation_rows.append(row)
    if len(activation_rows) != 1:
        raise ValueError(
            "execution freeze requires exactly one activation event; "
            f"found {len(activation_rows)}"
        )
    activation_row = activation_rows[0]
    activation = activation_row["event"]
    if (
        activation.get("performance_values_present") is not False
        or activation.get("source_policy")
        != "baostock_then_tencent_qfq"
        or activation.get("provider_mix_within_window_allowed") is not False
    ):
        raise ValueError("execution freeze activation has unsafe policy fields")

    resolved_plan = data_plan_path.resolve()
    if not resolved_plan.is_file():
        raise FileNotFoundError(resolved_plan)
    if activation.get("data_plan_path") != str(resolved_plan):
        raise ValueError("execution freeze data plan path mismatch")
    actual_plan_hash = sha256_file(resolved_plan)
    if activation.get("data_plan_sha256") != actual_plan_hash:
        raise ValueError("execution freeze data plan hash mismatch")
    plan = read_json(resolved_plan)
    if (
        plan.get("schema_version") != "test_d_prospective_data_plan_v2"
        or not str(plan.get("finalization_status", "")).startswith(
            "FROZEN_BEFORE_P01"
        )
    ):
        raise ValueError("execution data plan is not frozen before P01")

    acquisition = plan.get("acquisition_program", {})
    outcome = plan.get("outcome_acquisition_program", {})
    evaluation = plan.get("evaluation_program", {})
    if (
        acquisition.get("source_policy")
        != "baostock_then_tencent_qfq"
        or outcome.get("source_policy")
        != "baostock_then_tencent_qfq"
        or acquisition.get("provider_mix_within_window_allowed") is not False
        or outcome.get("provider_mix_within_window_allowed") is not False
        or outcome.get("stocks_and_benchmark_must_switch_together") is not True
        or acquisition.get("future_query_allowed") is not False
        or acquisition.get("historical_overlap_replacement_allowed") is not False
        or outcome.get("later_window_query_allowed") is not False
        or outcome.get("training_workspace_write_allowed") is not False
        or outcome.get("candidate_package_write_allowed") is not False
    ):
        raise ValueError("execution data plan leakage/source policy is unsafe")
    acquisition_helper = acquisition.get("fallback_helper", {})
    outcome_helper = outcome.get("fallback_helper", {})
    if (
        acquisition_helper.get("path") != outcome_helper.get("path")
        or acquisition_helper.get("sha256")
        != outcome_helper.get("sha256")
    ):
        raise ValueError("execution data plan fallback helper declarations differ")

    actual_controller = (
        controller_path.resolve()
        if controller_path is not None
        else Path(__file__).resolve()
    )
    declared_programs = {
        "causal_prefix": acquisition,
        "outcome_bundle": outcome,
        "tencent_fallback": acquisition_helper,
        "controller": evaluation,
    }
    event_hashes = activation.get("program_sha256")
    if not isinstance(event_hashes, dict):
        raise ValueError("execution freeze activation lacks program hashes")
    actual_hashes: dict[str, str] = {}
    resolved_paths: dict[str, str] = {}
    for key, declaration in declared_programs.items():
        if key == "controller":
            path = actual_controller
            expected_declared_path = _frozen_program_path(
                repository_root,
                str(declaration.get("path", "")),
            )
            if path != expected_declared_path:
                raise ValueError("execution freeze controller path mismatch")
        else:
            path = _frozen_program_path(
                repository_root,
                str(declaration.get("path", "")),
            )
        actual_hash = sha256_file(path)
        if (
            declaration.get("sha256") != actual_hash
            or event_hashes.get(key) != actual_hash
        ):
            raise ValueError(f"execution freeze program hash mismatch: {key}")
        actual_hashes[key] = actual_hash
        resolved_paths[key] = str(path)
    if set(event_hashes) != set(declared_programs):
        raise ValueError("execution freeze program hash set mismatch")

    return {
        "status": "PASS",
        "event_sequence": activation_row["sequence"],
        "audit_chain": chain,
        "data_plan_path": str(resolved_plan),
        "data_plan_sha256": actual_plan_hash,
        "program_paths": resolved_paths,
        "program_sha256": actual_hashes,
        "source_policy": "baostock_then_tencent_qfq",
        "provider_mix_within_window_allowed": False,
        "performance_values_present": False,
    }


def _assert_public_receipt_has_no_performance(payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False).lower()
    for token in PUBLIC_SCORE_FORBIDDEN_KEYS:
        if token in text:
            raise ValueError(f"public receipt leaks performance field: {token}")


def store_sealed_score(
    *,
    vault_root: Path,
    public_receipt_path: Path,
    window_id: str,
    scored_at: str,
    score_not_before: str,
    outcome_sha256: str,
    secret_payload: dict[str, Any],
) -> dict[str, Any]:
    timestamp = parse_timestamp(scored_at)
    if timestamp < parse_timestamp(score_not_before):
        raise ValueError("score attempted before the sell-open time gate")
    secret_path = vault_root / "sealed_scores" / f"{window_id}.json"
    if public_receipt_path.exists() and not secret_path.exists():
        raise ValueError(
            f"sealed score conflict: public receipt exists without secret for {window_id}"
        )
    requested_secret = {
        "schema_version": "test_d_sealed_score_v1",
        "window_id": window_id,
        "scored_at": timestamp.isoformat(),
        "outcome_sha256": outcome_sha256,
        "payload": secret_payload,
    }
    if secret_path.exists():
        secret_wrapper = read_json(secret_path)
        comparable_existing = {
            key: value
            for key, value in secret_wrapper.items()
            if key != "scored_at"
        }
        comparable_requested = {
            key: value
            for key, value in requested_secret.items()
            if key != "scored_at"
        }
        if canonical_json_bytes(comparable_existing) != canonical_json_bytes(
            comparable_requested
        ):
            raise ValueError(
                f"sealed score conflict with existing payload for {window_id}"
            )
        existing_timestamp = parse_timestamp(str(secret_wrapper["scored_at"]))
        if existing_timestamp < parse_timestamp(score_not_before):
            raise ValueError(
                f"sealed score conflict: existing timestamp violates gate for {window_id}"
            )
    else:
        secret_wrapper = requested_secret
        write_json(secret_path, secret_wrapper, refuse_overwrite=True)
    secret_sha256 = sha256_file(secret_path)
    public = {
        "schema_version": "test_d_public_score_receipt_v1",
        "window_id": window_id,
        "status": "SEALED",
        "scored_at": secret_wrapper["scored_at"],
        "outcome_sha256": outcome_sha256,
        "sealed_payload_sha256": secret_sha256,
        "sealed_payload_location_class": "EXTERNAL_VAULT_NOT_TRAINING_WORKSPACE",
        "performance_fields_disclosed": False,
    }
    _assert_public_receipt_has_no_performance(public)
    if public_receipt_path.exists():
        existing_public = read_json(public_receipt_path)
        if canonical_json_bytes(existing_public) != canonical_json_bytes(public):
            raise ValueError(
                f"sealed score conflict with existing public receipt for {window_id}"
            )
    else:
        write_json(public_receipt_path, public, refuse_overwrite=True)
    return public


def compound_return(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.prod(1.0 + array) - 1.0)


def evaluate_final_gates(
    rows: Sequence[dict[str, Any]],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    minimum = int(gate_config["minimum_valid_windows"])
    if len(rows) < minimum:
        return {
            "decision": "INCONCLUSIVE_EXTEND",
            "valid_windows": len(rows),
            "minimum_valid_windows": minimum,
            "gates": {},
        }
    d0 = np.asarray([row["candidate_d_net_0bps"] for row in rows], dtype=float)
    d10 = np.asarray([row["candidate_d_net_10bps"] for row in rows], dtype=float)
    d20 = np.asarray([row["candidate_d_net_20bps"] for row in rows], dtype=float)
    official10 = np.asarray([row["official_net_10bps"] for row in rows], dtype=float)
    sse = np.asarray([row["sse_return"] for row in rows], dtype=float)
    excess = d10 - official10
    required_wins = int(gate_config["minimum_official_excess_wins"])
    if "minimum_official_excess_win_rate_for_extension" in gate_config:
        required_wins = max(
            required_wins,
            math.ceil(
                len(rows)
                * float(
                    gate_config[
                        "minimum_official_excess_win_rate_for_extension"
                    ]
                )
            ),
        )
    kept = np.argsort(excess)[: max(0, len(excess) - 2)]
    leave_two_positive = (
        len(kept) > 0
        and compound_return(d10[kept])
        > compound_return(official10[kept])
    )
    gates = {
        "minimum_valid_windows": len(rows) >= minimum,
        "minimum_official_excess_wins": int(np.sum(excess > 0))
        >= required_wins,
        "mean_official_excess_positive": float(np.mean(excess)) > 0,
        "median_official_excess_positive": float(np.median(excess)) > 0,
        "compound_net10_beats_official": compound_return(d10)
        > compound_return(official10),
        "compound_net10_beats_sse": compound_return(d10) > compound_return(sse),
        "leave_best_two_out_official_excess_positive": bool(leave_two_positive),
        "minimum_single_week_net10": float(np.min(d10))
        >= float(gate_config["minimum_single_week_net10"]),
        "cost_direction_must_hold": bool(
            np.all(d0 + 1e-12 >= d10) and np.all(d10 + 1e-12 >= d20)
        ),
    }
    failed_gates = [name for name, passed in gates.items() if not passed]
    decision = "PASS" if not failed_gates else "FAIL"
    extension = gate_config.get("extension_at_eight")
    next_required_window_count: int | None = None
    if decision == "FAIL" and len(rows) == minimum and extension:
        hard_failures = set(extension.get("hard_fail_gates", [])).intersection(
            failed_gates
        )
        non_safety_failures = [
            name for name in failed_gates if name not in hard_failures
        ]
        if (
            not hard_failures
            and len(non_safety_failures)
            <= int(
                extension[
                    "eligible_if_failed_non_safety_gate_count_is_at_most"
                ]
            )
        ):
            decision = "INCONCLUSIVE_EXTEND"
            next_required_window_count = 12
    return {
        "decision": decision,
        "valid_windows": len(rows),
        "gates": gates,
        "failed_gates": failed_gates,
        "required_official_excess_wins": required_wins,
        **(
            {"next_required_window_count": next_required_window_count}
            if next_required_window_count is not None
            else {}
        ),
        "diagnostics": {
            "official_excess_wins": int(np.sum(excess > 0)),
            "mean_official_excess": float(np.mean(excess)),
            "median_official_excess": float(np.median(excess)),
            "candidate_d_compound_net10": compound_return(d10),
            "official_compound_net10": compound_return(official10),
            "sse_compound": compound_return(sse),
            "candidate_d_min_week_net10": float(np.min(d10)),
        },
    }


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _candidate_source(spec: CandidateSpec) -> Path:
    return ROOT / Path(spec.source_relative)


def _artifact_hashes(
    candidate_dir: Path,
    relative_paths: Iterable[str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in relative_paths:
        path = candidate_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = sha256_file(path)
    return hashes


def environment_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python": sys.version,
        "platform": sys.platform,
        "captured_at": utc_now(),
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_runtime": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        payload["torch_probe_error"] = repr(exc)
    payload.update(
        {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        }
    )
    return payload


def initialize_protocol(
    config_path: Path,
    experiment_root: Path,
    public_run: Path,
) -> dict[str, Any]:
    config = validate_preregistration(read_json(config_path))
    experiment_root.mkdir(parents=True, exist_ok=True)
    public_run.mkdir(parents=True, exist_ok=True)
    audit_path = experiment_root / "audit" / "events.jsonl"
    identity_path = experiment_root / "manifests" / "source_identity.json"
    config_snapshot = experiment_root / "manifests" / "preregistration.json"
    state_path = experiment_root / "state.json"

    identities = {
        spec.name: tree_manifest(_candidate_source(spec))
        for spec in CANDIDATE_SPECS
    }
    identity_payload = {
        "schema_version": "test_d_protected_source_identity_v1",
        "captured_at": utc_now(),
        "candidates": identities,
        "candidate_sources_are_read_only_inputs": True,
    }
    if identity_path.exists():
        existing = read_json(identity_path)
        for name, current in identities.items():
            if (
                existing["candidates"][name]["tree_sha256"]
                != current["tree_sha256"]
            ):
                raise ValueError(f"protected candidate source changed: {name}")
    else:
        write_json(identity_path, identity_payload, refuse_overwrite=True)
        shutil.copy2(config_path, config_snapshot)

    config_hash = sha256_file(config_snapshot)
    state = {
        "schema_version": "test_d_prospective_state_v1",
        "experiment_id": config["experiment_id"],
        "classification": config["classification"],
        "status": "PREREGISTERED_NOT_STARTED",
        "created_at": utc_now(),
        "preregistration_sha256": config_hash,
        "minimum_valid_windows": config["minimum_valid_windows"],
        "performance_blinding": config["performance_blinding"],
        "windows": {
            row["window_id"]: {
                "status": "WAIT_DECISION",
                "decision_date": row["decision_date"],
                "buy_date": row["buy_date"],
                "sell_date": row["sell_date"],
            }
            for row in config["windows"]
        },
        "sealed_score_count": 0,
        "performance_values_present": False,
    }
    if state_path.exists():
        state = read_json(state_path)
        if state["preregistration_sha256"] != config_hash:
            raise ValueError("preregistration snapshot changed after initialization")
    else:
        write_json(state_path, state, refuse_overwrite=True)
        append_hash_chained_event(
            audit_path,
            {
                "event": "PROTOCOL_INITIALIZED",
                "at": utc_now(),
                "preregistration_sha256": config_hash,
                "source_tree_sha256": {
                    name: item["tree_sha256"]
                    for name, item in identities.items()
                },
                "performance_values_present": False,
            },
        )
        write_json(
            public_run / "protocol_status.json",
            {
                "experiment_id": config["experiment_id"],
                "status": state["status"],
                "preregistration_sha256": config_hash,
                "performance_blinding": config["performance_blinding"],
                "performance_values_present": False,
            },
        )
        write_json(experiment_root / "manifests" / "environment.json", environment_manifest())
    return state


def _load_initialized(
    config_path: Path,
    experiment_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = validate_preregistration(read_json(config_path))
    snapshot = experiment_root / "manifests" / "preregistration.json"
    state_path = experiment_root / "state.json"
    if not snapshot.is_file() or not state_path.is_file():
        raise FileNotFoundError("protocol is not initialized")
    if sha256_file(snapshot) != read_json(state_path)["preregistration_sha256"]:
        raise ValueError("initialized preregistration hash mismatch")
    if canonical_json_bytes(read_json(snapshot)) != canonical_json_bytes(config):
        raise ValueError("active config differs from frozen preregistration")
    return config, read_json(state_path)


def _validate_prepared_round(
    *,
    round_dir: Path,
    window: dict[str, Any],
    prefix_report: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = round_dir / "round_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"existing round is incomplete and not recoverable: {round_dir}")
    manifest = read_json(manifest_path)
    shared_train = round_dir / "data" / "shared_train.csv"
    prepared_at = parse_timestamp(str(manifest.get("prepared_at")))
    if (
        manifest.get("status") != "PREPARED"
        or canonical_json_bytes(manifest.get("window")) != canonical_json_bytes(window)
        or manifest.get("shared_train_path") != str(shared_train.resolve())
        or not shared_train.is_file()
        or sha256_file(shared_train) != prefix_report["sha256"]
        or manifest.get("shared_train_sha256") != prefix_report["sha256"]
        or manifest.get("outcome_path_present") is not False
        or manifest.get("future_returns_accessed") is not False
        or window_phase(window, prepared_at, False, False) != "PREPARE_AND_LOCK"
    ):
        raise ValueError(f"existing prepared round conflicts with protocol: {round_dir}")
    copies = {
        str(item.get("candidate")): item
        for item in manifest.get("candidate_copies", [])
    }
    if set(copies) != {spec.name for spec in CANDIDATE_SPECS}:
        raise ValueError(f"prepared round candidate set is incomplete: {round_dir}")
    for spec in CANDIDATE_SPECS:
        candidate_dir = round_dir / "candidates" / spec.name
        train_path = candidate_dir / "data" / "train.csv"
        copy_record = copies[spec.name]
        if (
            copy_record.get("destination") != str(candidate_dir.resolve())
            or copy_record.get("ordinary_copy_no_hardlinks") is not True
            or copy_record.get("train_sha256") != prefix_report["sha256"]
            or copy_record.get("test_csv_present") is not False
            or not train_path.is_file()
            or sha256_file(train_path) != prefix_report["sha256"]
            or (candidate_dir / "data" / "test.csv").exists()
        ):
            raise ValueError(
                f"prepared candidate copy is not recoverable: {spec.name}"
            )
    return manifest


def _commit_prepared_round(
    *,
    experiment_root: Path,
    state: dict[str, Any],
    window_id: str,
    manifest: dict[str, Any],
) -> None:
    audit_path = experiment_root / "audit" / "events.jsonl"
    existing_events = audit_events_for_window(
        audit_path,
        event_name="WINDOW_PREPARED",
        window_id=window_id,
    )
    if len(existing_events) > 1:
        raise ValueError(f"duplicate WINDOW_PREPARED audit event: {window_id}")
    expected_fields = {
        "window_id": window_id,
        "training_sha256": manifest["shared_train_sha256"],
        "future_returns_accessed": False,
    }
    if existing_events:
        if any(
            existing_events[0].get(key) != value
            for key, value in expected_fields.items()
        ):
            raise ValueError(f"WINDOW_PREPARED audit conflict for {window_id}")
    else:
        append_hash_chained_event(
            audit_path,
            {
                "event": "WINDOW_PREPARED",
                "at": manifest["prepared_at"],
                **expected_fields,
            },
        )
    current_status = state["windows"][window_id]["status"]
    if current_status not in {"WAIT_DECISION", "PREPARED"}:
        raise ValueError(
            f"cannot reconcile prepared round from state {current_status}: {window_id}"
        )
    state["windows"][window_id]["status"] = "PREPARED"
    state["status"] = "ACTIVE_BLINDED"
    write_json(experiment_root / "state.json", state)


def _materialize_prepared_round(
    *,
    config: dict[str, Any],
    window: dict[str, Any],
    window_id: str,
    market_prefix: Path,
    prefix_report: dict[str, Any],
    experiment_root: Path,
    now: datetime,
) -> Path:
    rounds_dir = experiment_root / "rounds"
    round_dir = rounds_dir / window_id
    rounds_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = rounds_dir / f".{window_id}.preparing"
    if staging_dir.exists():
        if (
            staging_dir.parent.resolve() != rounds_dir.resolve()
            or staging_dir.name != f".{window_id}.preparing"
        ):
            raise ValueError(f"unsafe preparation staging path: {staging_dir}")
        shutil.rmtree(staging_dir)
    try:
        shared_train = staging_dir / "data" / "shared_train.csv"
        shared_train.parent.mkdir(parents=True)
        shutil.copy2(market_prefix, shared_train)
        if sha256_file(shared_train) != prefix_report["sha256"]:
            raise ValueError("ordinary training snapshot copy changed bytes")

        copies: list[dict[str, Any]] = []
        for spec in CANDIDATE_SPECS:
            source = _candidate_source(spec)
            staging_destination = staging_dir / "candidates" / spec.name
            final_destination = round_dir / "candidates" / spec.name
            shutil.copytree(
                source,
                staging_destination,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            _clear_directory(staging_destination / "output")
            _clear_directory(staging_destination / "temp")
            _remove_if_exists(staging_destination / "data" / "test.csv")
            shutil.copy2(
                shared_train,
                staging_destination / "data" / "train.csv",
            )
            if (
                sha256_file(staging_destination / "data" / "train.csv")
                != prefix_report["sha256"]
            ):
                raise ValueError(f"training bytes differ for {spec.name}")
            if spec.name == "THU-BDC2026-main":
                _clear_directory(staging_destination / "model" / "60_158+39")
            else:
                _remove_if_exists(
                    staging_destination / "model" / "final_model.pth"
                )
                _remove_if_exists(
                    staging_destination / "model" / "scalers.json"
                )
                config_file = staging_destination / "model" / "config.json"
                base_d_config = read_json(config_file)
                causal = derive_candidate_d_causal_metadata(
                    shared_train,
                    base_d_config,
                )
                runtime = build_candidate_d_runtime_config(
                    base_d_config,
                    experiment_id=f"{config['experiment_id']}_{window_id}",
                    decision_date=window["decision_date"],
                    train_sample_end=causal["train_sample_end"],
                    train_sample_count=causal["train_sample_count"],
                )
                write_json(config_file, runtime)
            copies.append(
                {
                    "candidate": spec.name,
                    "source": str(source.resolve()),
                    "destination": str(final_destination.resolve()),
                    "ordinary_copy_no_hardlinks": True,
                    "train_sha256": sha256_file(
                        staging_destination / "data" / "train.csv"
                    ),
                    "test_csv_present": (
                        staging_destination / "data" / "test.csv"
                    ).exists(),
                }
            )

        final_shared_train = round_dir / "data" / "shared_train.csv"
        manifest = {
            "schema_version": "test_d_prospective_round_v1",
            "window": window,
            "status": "PREPARED",
            "prepared_at": now.isoformat(),
            "training_prefix": prefix_report,
            "shared_train_path": str(final_shared_train.resolve()),
            "shared_train_sha256": sha256_file(shared_train),
            "candidate_copies": copies,
            "outcome_path_present": False,
            "future_returns_accessed": False,
        }
        write_json(
            staging_dir / "round_manifest.json",
            manifest,
            refuse_overwrite=True,
        )
        staging_dir.replace(round_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    return round_dir


def prepare_window(
    *,
    config_path: Path,
    experiment_root: Path,
    window_id: str,
    market_prefix: Path,
    now: datetime,
) -> Path:
    config, state = _load_initialized(config_path, experiment_root)
    window = window_by_id(config, window_id)
    prefix_report = validate_training_prefix(market_prefix, window["decision_date"])
    identity = read_json(experiment_root / "manifests" / "source_identity.json")
    for spec in CANDIDATE_SPECS:
        current = tree_manifest(_candidate_source(spec))
        if (
            current["tree_sha256"]
            != identity["candidates"][spec.name]["tree_sha256"]
        ):
            raise ValueError(f"protected candidate source changed: {spec.name}")

    round_dir = experiment_root / "rounds" / window_id
    if round_dir.exists():
        manifest = _validate_prepared_round(
            round_dir=round_dir,
            window=window,
            prefix_report=prefix_report,
        )
        _commit_prepared_round(
            experiment_root=experiment_root,
            state=state,
            window_id=window_id,
            manifest=manifest,
        )
        return round_dir
    if window_phase(window, now, False, False) != "PREPARE_AND_LOCK":
        raise ValueError(f"window {window_id} is not inside its preparation time gate")
    round_dir = _materialize_prepared_round(
        config=config,
        window=window,
        window_id=window_id,
        market_prefix=market_prefix,
        prefix_report=prefix_report,
        experiment_root=experiment_root,
        now=now,
    )
    manifest = _validate_prepared_round(
        round_dir=round_dir,
        window=window,
        prefix_report=prefix_report,
    )
    _commit_prepared_round(
        experiment_root=experiment_root,
        state=state,
        window_id=window_id,
        manifest=manifest,
    )
    return round_dir


def _sanitized_training_environment() -> dict[str, str]:
    allow = {
        "PATH",
        "PATHEXT",
        "SystemRoot",
        "WINDIR",
        "TEMP",
        "TMP",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NUMBER_OF_PROCESSORS",
    }
    environment = {key: value for key, value in os.environ.items() if key in allow}
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "20260708",
            "BDC2026_EXECUTION_ROLE": "TRAINING_NO_OUTCOME_ACCESS",
        }
    )
    return environment


def run_process(
    args: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    lowered = " ".join(args).lower()
    for forbidden in ("sealed_scores", "outcome", "validation_csv", "score_vault"):
        if forbidden in lowered:
            raise ValueError(f"training/inference command contains forbidden token: {forbidden}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    started = time.perf_counter()
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=_sanitized_training_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(
        "COMMAND=" + json.dumps(args, ensure_ascii=False) + "\n" + completed.stdout,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, args)
    return {
        "command": args,
        "cwd": str(cwd.resolve()),
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "return_code": completed.returncode,
        "log_path": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "environment_keys": sorted(_sanitized_training_environment()),
        "outcome_paths_in_environment": False,
    }


def _validate_candidate_lock(
    *,
    path: Path,
    candidate_dir: Path,
    spec: CandidateSpec,
    window: dict[str, Any],
    training_hash: str,
    universe: set[str],
) -> dict[str, Any]:
    lock = read_json(path)
    expected_result = (candidate_dir / "locked" / "result.csv").resolve()
    result = Path(str(lock.get("result_path", ""))).resolve()
    if (
        lock.get("status") != "LOCKED"
        or lock.get("window_id") != window["window_id"]
        or lock.get("candidate") != spec.name
        or lock.get("training_data_sha256") != training_hash
        or not lock.get("repeated_inference_byte_identical")
        or result != expected_result
        or not result.is_file()
        or sha256_file(result) != lock.get("result_sha256")
        or parse_timestamp(str(lock.get("locked_at")))
        >= parse_timestamp(window["buy_deadline"])
    ):
        raise ValueError(f"invalid recoverable prediction lock: {path}")
    if lock.get("checkpoint_hashes") != _artifact_hashes(
        candidate_dir,
        spec.checkpoint_paths,
    ):
        raise ValueError(f"checkpoint changed after prediction lock: {path}")
    if lock.get("config_hashes") != _artifact_hashes(
        candidate_dir,
        spec.config_paths,
    ):
        raise ValueError(f"config changed after prediction lock: {path}")
    parsed = validate_result(result, universe)
    if spec.name == "Candidate-D" and (
        len(parsed) != 2
        or not np.allclose(
            parsed["weight"].to_numpy(dtype=float),
            np.asarray([0.5, 0.5], dtype=float),
            atol=1e-12,
        )
    ):
        raise ValueError(f"Candidate-D lock changed its frozen portfolio: {path}")
    return lock


def train_and_lock_window(
    *,
    config_path: Path,
    experiment_root: Path,
    window_id: str,
    now: datetime,
    python_executable: Path = DEFAULT_PYTHON,
) -> list[dict[str, Any]]:
    config, state = _load_initialized(config_path, experiment_root)
    window = window_by_id(config, window_id)
    round_dir = experiment_root / "rounds" / window_id
    manifest_path = round_dir / "round_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    shared_train = Path(manifest["shared_train_path"])
    training_hash = sha256_file(shared_train)
    universe_frame = pd.read_csv(
        shared_train,
        usecols=[COL_CODE],
        dtype={COL_CODE: str},
    )
    universe = set(universe_frame[COL_CODE].map(normalize_code))
    locks: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        candidate_dir = round_dir / "candidates" / spec.name
        if (candidate_dir / "data" / "test.csv").exists():
            raise ValueError(f"future test.csv present in training sandbox: {spec.name}")
        if sha256_file(candidate_dir / "data" / "train.csv") != training_hash:
            raise ValueError(f"training hash mismatch for {spec.name}")
        final_lock_path = candidate_dir / "prediction_lock.json"
        pending_lock_path = candidate_dir / "prediction_lock.pending.json"
        if final_lock_path.exists() and pending_lock_path.exists():
            raise ValueError(
                f"both final and pending prediction locks exist for {spec.name}"
            )
        existing_lock_path = (
            final_lock_path
            if final_lock_path.exists()
            else pending_lock_path
            if pending_lock_path.exists()
            else None
        )
        if existing_lock_path is not None:
            locks.append(
                _validate_candidate_lock(
                    path=existing_lock_path,
                    candidate_dir=candidate_dir,
                    spec=spec,
                    window=window,
                    training_hash=training_hash,
                    universe=universe,
                )
            )
            continue
        if now >= parse_timestamp(window["buy_deadline"]):
            raise ValueError("cannot train or create a missing lock after the buy deadline")

        training = run_process(
            [str(python_executable), *spec.train_args],
            cwd=candidate_dir,
            log_path=candidate_dir / "logs" / "training.log",
            timeout_seconds=8 * 60 * 60,
        )
        checkpoint_hashes = _artifact_hashes(candidate_dir, spec.checkpoint_paths)
        config_hashes = _artifact_hashes(candidate_dir, spec.config_paths)
        write_json(
            candidate_dir / "training_execution.json",
            {
                "schema_version": "test_d_training_execution_v1",
                "window_id": window_id,
                "candidate": spec.name,
                "training_data_sha256": training_hash,
                "training_data_date_end": manifest["training_prefix"]["date_end"],
                "future_returns_accessed": False,
                "execution": training,
                "checkpoint_hashes": checkpoint_hashes,
                "config_hashes": config_hashes,
                "training_time_passed": training["elapsed_seconds"] < 8 * 60 * 60,
            },
        )
        locked_dir = candidate_dir / "locked"
        locked_dir.mkdir(parents=True, exist_ok=True)
        result_paths: list[Path] = []
        inference_manifests: list[dict[str, Any]] = []
        for run_number in (1, 2):
            execution = run_process(
                [str(python_executable), *spec.predict_args],
                cwd=candidate_dir,
                log_path=candidate_dir / "logs" / f"inference_{run_number}.log",
                timeout_seconds=5 * 60,
            )
            output = candidate_dir / "output" / "result.csv"
            validate_result(output, universe)
            frozen = locked_dir / f"result_run{run_number}.csv"
            shutil.copy2(output, frozen)
            result_paths.append(frozen)
            execution["result_sha256"] = sha256_file(frozen)
            execution["inference_time_passed"] = execution["elapsed_seconds"] < 5 * 60
            inference_manifests.append(execution)
        lock = build_prediction_lock(
            window=window,
            candidate=spec.name,
            locked_at=local_now(config["timezone"]).isoformat(),
            training_sha256=training_hash,
            checkpoint_hashes=checkpoint_hashes,
            config_hashes=config_hashes,
            result_paths=result_paths,
        )
        final_result = locked_dir / "result.csv"
        shutil.copy2(result_paths[0], final_result)
        lock["result_path"] = str(final_result.resolve())
        lock["result_sha256"] = sha256_file(final_result)
        lock["inference_runs"] = inference_manifests
        write_json(pending_lock_path, lock, refuse_overwrite=True)
        locks.append(
            _validate_candidate_lock(
                path=pending_lock_path,
                candidate_dir=candidate_dir,
                spec=spec,
                window=window,
                training_hash=training_hash,
                universe=universe,
            )
        )

    if len(locks) != len(CANDIDATE_SPECS):
        raise ValueError("not all candidate prediction locks are recoverable")
    for spec in CANDIDATE_SPECS:
        candidate_dir = round_dir / "candidates" / spec.name
        pending_lock_path = candidate_dir / "prediction_lock.pending.json"
        final_lock_path = candidate_dir / "prediction_lock.json"
        if pending_lock_path.exists():
            if final_lock_path.exists():
                raise ValueError(
                    f"cannot promote duplicate prediction lock for {spec.name}"
                )
            pending_lock_path.replace(final_lock_path)
    locks = verify_window_locks(experiment_root, window_id)
    manifest["status"] = "PREDICTIONS_LOCKED"
    manifest["prediction_locks"] = [
        {
            "candidate": lock["candidate"],
            "locked_at": lock["locked_at"],
            "result_sha256": lock["result_sha256"],
        }
        for lock in locks
    ]
    write_json(manifest_path, manifest)

    audit_path = experiment_root / "audit" / "events.jsonl"
    result_hashes = {
        lock["candidate"]: lock["result_sha256"] for lock in locks
    }
    existing_events = audit_events_for_window(
        audit_path,
        event_name="PREDICTIONS_LOCKED",
        window_id=window_id,
    )
    if len(existing_events) > 1:
        raise ValueError(f"duplicate PREDICTIONS_LOCKED audit event: {window_id}")
    if existing_events:
        if (
            existing_events[0].get("candidate_result_sha256") != result_hashes
            or existing_events[0].get("future_returns_accessed") is not False
        ):
            raise ValueError(
                f"PREDICTIONS_LOCKED audit conflict for window {window_id}"
            )
    else:
        append_hash_chained_event(
            audit_path,
            {
                "event": "PREDICTIONS_LOCKED",
                "at": local_now(config["timezone"]).isoformat(),
                "window_id": window_id,
                "candidate_result_sha256": result_hashes,
                "future_returns_accessed": False,
            },
        )
    state["windows"][window_id]["status"] = "LOCKED"
    write_json(experiment_root / "state.json", state)
    return locks


def verify_window_locks(
    experiment_root: Path,
    window_id: str,
) -> list[dict[str, Any]]:
    locks: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        path = (
            experiment_root
            / "rounds"
            / window_id
            / "candidates"
            / spec.name
            / "prediction_lock.json"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        lock = read_json(path)
        result = Path(lock["result_path"])
        if (
            lock.get("status") != "LOCKED"
            or not lock.get("repeated_inference_byte_identical")
            or not result.is_file()
            or sha256_file(result) != lock["result_sha256"]
        ):
            raise ValueError(f"invalid prediction lock: {path}")
        locks.append(lock)
    return locks


def _portfolio_score(
    result: pd.DataFrame,
    outcome: pd.DataFrame,
    *,
    buy_date: str,
    sell_date: str,
) -> dict[str, float]:
    start = (
        outcome[outcome[COL_DATE] == pd.Timestamp(buy_date)]
        .set_index(COL_CODE)[COL_OPEN]
        .astype(float)
    )
    end = (
        outcome[outcome[COL_DATE] == pd.Timestamp(sell_date)]
        .set_index(COL_CODE)[COL_OPEN]
        .astype(float)
    )
    returns: list[float] = []
    weights: list[float] = []
    for row in result.itertuples(index=False):
        stock = normalize_code(row.stock_id)
        if stock not in start or stock not in end:
            raise ValueError(f"missing outcome open for selected stock {stock}")
        returns.append(float(end[stock] / start[stock] - 1.0))
        weights.append(float(row.weight))
    gross = float(np.dot(np.asarray(weights), np.asarray(returns)))
    invested = float(sum(weights))
    return {
        "gross": gross,
        "net_0bps": gross,
        "net_10bps": gross - invested * 2.0 * 10.0 / 10000.0,
        "net_20bps": gross - invested * 2.0 * 20.0 / 10000.0,
    }


def _benchmark_return(path: Path, buy_date: str, sell_date: str) -> float:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if COL_DATE not in frame or COL_OPEN not in frame:
        raise ValueError("benchmark CSV must contain 日期 and 开盘")
    frame[COL_DATE] = pd.to_datetime(frame[COL_DATE]).dt.normalize()
    validate_scoring_prefix_dates(frame, sell_date)
    lookup = frame.set_index(COL_DATE)[COL_OPEN].astype(float)
    buy = pd.Timestamp(buy_date)
    sell = pd.Timestamp(sell_date)
    if buy not in lookup or sell not in lookup:
        raise ValueError("benchmark CSV does not contain buy/sell dates")
    return float(lookup[sell] / lookup[buy] - 1.0)


def score_window(
    *,
    config_path: Path,
    experiment_root: Path,
    public_run: Path,
    vault_root: Path,
    window_id: str,
    outcome_market: Path,
    sse_open_csv: Path,
    now: datetime,
) -> dict[str, Any]:
    config, state = _load_initialized(config_path, experiment_root)
    window = window_by_id(config, window_id)
    locks = verify_window_locks(experiment_root, window_id)
    if now < parse_timestamp(window["score_not_before"]):
        raise ValueError("cannot score before the sell-open time gate")
    outcome_hash = sha256_file(outcome_market)
    outcome = pd.read_csv(
        outcome_market,
        dtype={COL_CODE: str},
        encoding="utf-8-sig",
    )
    required = {COL_CODE, COL_DATE, COL_OPEN}
    if not required.issubset(outcome.columns):
        raise ValueError("outcome market CSV is missing required columns")
    outcome[COL_CODE] = outcome[COL_CODE].map(normalize_code)
    outcome[COL_DATE] = pd.to_datetime(outcome[COL_DATE]).dt.normalize()
    validate_scoring_prefix_dates(outcome, window["sell_date"])
    allowed_dates = {
        pd.Timestamp(window["buy_date"]),
        pd.Timestamp(window["sell_date"]),
    }
    if not allowed_dates.issubset(set(outcome[COL_DATE].unique())):
        raise ValueError("outcome market CSV lacks buy/sell opens")

    candidate_scores: dict[str, dict[str, float]] = {}
    for lock in locks:
        result = validate_result(Path(lock["result_path"]))
        candidate_scores[lock["candidate"]] = _portfolio_score(
            result,
            outcome,
            buy_date=window["buy_date"],
            sell_date=window["sell_date"],
        )
    buy_rows = outcome[outcome[COL_DATE] == pd.Timestamp(window["buy_date"])]
    sell_rows = outcome[outcome[COL_DATE] == pd.Timestamp(window["sell_date"])]
    merged = buy_rows[[COL_CODE, COL_OPEN]].merge(
        sell_rows[[COL_CODE, COL_OPEN]],
        on=COL_CODE,
        suffixes=("_buy", "_sell"),
        validate="one_to_one",
    )
    hs300_equal_weight = float(
        (merged[f"{COL_OPEN}_sell"] / merged[f"{COL_OPEN}_buy"] - 1.0).mean()
    )
    secret = {
        "window_id": window_id,
        "buy_date": window["buy_date"],
        "sell_date": window["sell_date"],
        "candidate_scores": candidate_scores,
        "sse_return": _benchmark_return(
            sse_open_csv,
            window["buy_date"],
            window["sell_date"],
        ),
        "hs300_equal_weight_return": hs300_equal_weight,
        "lock_hashes": {
            lock["candidate"]: lock["result_sha256"] for lock in locks
        },
    }
    receipt = store_sealed_score(
        vault_root=vault_root,
        public_receipt_path=public_run / "sealed_receipts" / f"{window_id}.json",
        window_id=window_id,
        scored_at=now.isoformat(),
        score_not_before=window["score_not_before"],
        outcome_sha256=outcome_hash,
        secret_payload=secret,
    )
    audit_path = experiment_root / "audit" / "events.jsonl"
    existing_events = audit_events_for_window(
        audit_path,
        event_name="SCORE_SEALED",
        window_id=window_id,
    )
    if len(existing_events) > 1:
        raise ValueError(f"duplicate SCORE_SEALED audit event: {window_id}")
    expected_event_fields = {
        "window_id": window_id,
        "outcome_sha256": outcome_hash,
        "sealed_payload_sha256": receipt["sealed_payload_sha256"],
        "performance_values_present": False,
    }
    if existing_events:
        existing = existing_events[0]
        if any(
            existing.get(key) != value
            for key, value in expected_event_fields.items()
        ):
            raise ValueError(f"SCORE_SEALED audit conflict for window {window_id}")
    else:
        append_hash_chained_event(
            audit_path,
            {
                "event": "SCORE_SEALED",
                "at": now.isoformat(),
                **expected_event_fields,
            },
        )

    score_event_ids: set[str] = set()
    validate_audit_chain(audit_path)
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)["event"]
        if event.get("event") != "SCORE_SEALED":
            continue
        scored_window = str(event["window_id"])
        if scored_window in score_event_ids:
            raise ValueError(f"duplicate SCORE_SEALED audit event: {scored_window}")
        score_event_ids.add(scored_window)
    state["windows"][window_id]["status"] = "SCORED_SEALED"
    state["sealed_score_count"] = len(score_event_ids)
    state["performance_values_present"] = False
    write_json(experiment_root / "state.json", state)
    return receipt


def finalize_scores(
    *,
    config_path: Path,
    experiment_root: Path,
    public_run: Path,
    vault_root: Path,
) -> dict[str, Any]:
    config, state = _load_initialized(config_path, experiment_root)
    minimum = int(config["minimum_valid_windows"])
    audit_path = experiment_root / "audit" / "events.jsonl"
    validate_audit_chain(audit_path)
    score_events: dict[str, dict[str, Any]] = {}
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)["event"]
        if event.get("event") != "SCORE_SEALED":
            continue
        window_id = str(event["window_id"])
        if window_id in score_events:
            raise ValueError(f"duplicate SCORE_SEALED audit event: {window_id}")
        score_events[window_id] = event
    paths = {
        path.stem: path
        for path in (vault_root / "sealed_scores").glob("P*.json")
    }
    if set(paths) != set(score_events):
        raise ValueError(
            "sealed score files are not represented exactly once in the audit chain"
        )
    if int(state.get("sealed_score_count", -1)) != len(score_events):
        raise ValueError("state sealed_score_count differs from the audit chain")
    if len(score_events) < minimum:
        raise ValueError(
            f"cannot unseal before {minimum} valid windows; found {len(score_events)}"
        )
    rows: list[dict[str, Any]] = []
    private_windows: list[dict[str, Any]] = []
    configured_order = [row["window_id"] for row in config["windows"]]
    unrecognized = set(score_events).difference(configured_order)
    if unrecognized:
        raise ValueError(f"audit contains unregistered windows: {sorted(unrecognized)}")
    ordered_window_ids = [
        window_id for window_id in configured_order if window_id in score_events
    ]
    for window_id in ordered_window_ids:
        path = paths[window_id]
        receipt_path = public_run / "sealed_receipts" / f"{window_id}.json"
        if not receipt_path.is_file():
            raise ValueError(f"missing public sealed receipt: {window_id}")
        receipt = read_json(receipt_path)
        event = score_events[window_id]
        if (
            receipt.get("sealed_payload_sha256") != sha256_file(path)
            or event.get("sealed_payload_sha256")
            != receipt.get("sealed_payload_sha256")
            or event.get("outcome_sha256") != receipt.get("outcome_sha256")
        ):
            raise ValueError(
                f"sealed file, public receipt, and audit hash mismatch: {window_id}"
            )
        wrapper = read_json(path)
        if (
            wrapper.get("window_id") != window_id
            or wrapper.get("outcome_sha256") != receipt.get("outcome_sha256")
        ):
            raise ValueError(f"sealed score metadata mismatch: {window_id}")
        secret = wrapper["payload"]
        d = secret["candidate_scores"]["Candidate-D"]
        official = secret["candidate_scores"]["THU-BDC2026-main"]
        rows.append(
            {
                "window_id": secret["window_id"],
                "candidate_d_net_0bps": d["net_0bps"],
                "candidate_d_net_10bps": d["net_10bps"],
                "candidate_d_net_20bps": d["net_20bps"],
                "official_net_10bps": official["net_10bps"],
                "sse_return": secret["sse_return"],
            }
        )
        private_windows.append(secret)
    decision = evaluate_final_gates(rows, config["decision_gates"])
    report = {
        "schema_version": "test_d_prospective_final_report_v1",
        "experiment_id": config["experiment_id"],
        "classification": config["classification"],
        "unsealed_at": utc_now(),
        "window_count": len(rows),
        "decision": decision,
        "windows": private_windows,
        "promotion_authorized": False,
        "candidate_packages_modified": False,
    }
    if decision["decision"] == "INCONCLUSIVE_EXTEND":
        output_path = public_run / "interim_decision_8w.json"
        state["status"] = "EXTENSION_REQUIRED_RULES_FROZEN"
        audit_event = "SCORES_UNSEALED_INTERIM_EXTENSION"
    else:
        output_path = public_run / "final_results.json"
        state["status"] = (
            "COMPLETE_PASS_NO_PROMOTION"
            if decision["decision"] == "PASS"
            else "COMPLETE_FAIL"
        )
        audit_event = "SCORES_UNSEALED_FINAL"
    write_json(output_path, report, refuse_overwrite=True)
    state["performance_values_present"] = True
    write_json(experiment_root / "state.json", state)
    append_hash_chained_event(
        audit_path,
        {
            "event": audit_event,
            "at": utc_now(),
            "window_count": len(rows),
            "decision": decision["decision"],
        },
    )
    return report


def integrity_audit(
    *,
    config_path: Path,
    experiment_root: Path,
    public_run: Path,
) -> dict[str, Any]:
    config, state = _load_initialized(config_path, experiment_root)
    issues: list[str] = []
    identity = read_json(experiment_root / "manifests" / "source_identity.json")
    source_checks: dict[str, Any] = {}
    for spec in CANDIDATE_SPECS:
        current = tree_manifest(_candidate_source(spec))
        expected = identity["candidates"][spec.name]["tree_sha256"]
        passed = current["tree_sha256"] == expected
        source_checks[spec.name] = {
            "expected_tree_sha256": expected,
            "current_tree_sha256": current["tree_sha256"],
            "unchanged": passed,
        }
        if not passed:
            issues.append(f"protected source changed: {spec.name}")
    try:
        chain = validate_audit_chain(experiment_root / "audit" / "events.jsonl")
    except Exception as exc:
        issues.append(str(exc))
        chain = {"status": "FAIL", "error": repr(exc)}
    round_checks: dict[str, Any] = {}
    for window in config["windows"]:
        round_dir = experiment_root / "rounds" / window["window_id"]
        if not round_dir.exists():
            continue
        manifest = read_json(round_dir / "round_manifest.json")
        shared_hash = sha256_file(Path(manifest["shared_train_path"]))
        candidate_hashes: dict[str, str] = {}
        test_files: list[str] = []
        for spec in CANDIDATE_SPECS:
            candidate_dir = round_dir / "candidates" / spec.name
            train_path = candidate_dir / "data" / "train.csv"
            candidate_hashes[spec.name] = sha256_file(train_path)
            if (candidate_dir / "data" / "test.csv").exists():
                test_files.append(spec.name)
        if set(candidate_hashes.values()) != {shared_hash}:
            issues.append(f"training bytes differ in {window['window_id']}")
        if test_files:
            issues.append(f"test.csv leaked into {window['window_id']}: {test_files}")
        round_checks[window["window_id"]] = {
            "shared_train_sha256": shared_hash,
            "candidate_train_sha256": candidate_hashes,
            "same_training_bytes": set(candidate_hashes.values()) == {shared_hash},
            "test_csv_present": test_files,
            "future_returns_accessed": False,
        }
    payload = {
        "schema_version": "test_d_prospective_integrity_audit_v1",
        "captured_at": utc_now(),
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "source_checks": source_checks,
        "audit_chain": chain,
        "round_checks": round_checks,
        "state_performance_values_present": state["performance_values_present"],
        "candidate_packages_modified": bool(
            any(not row["unchanged"] for row in source_checks.values())
        ),
    }
    write_json(public_run / "integrity_audit.json", payload)
    if issues:
        raise ValueError("integrity audit failed: " + "; ".join(issues))
    return payload


def protocol_status(
    *,
    config_path: Path,
    experiment_root: Path,
    public_run: Path,
    now: datetime,
) -> dict[str, Any]:
    config, state = _load_initialized(config_path, experiment_root)
    status_rows: list[dict[str, Any]] = []
    for window in config["windows"]:
        window_id = window["window_id"]
        try:
            has_lock = len(verify_window_locks(experiment_root, window_id)) == 2
        except (FileNotFoundError, ValueError):
            has_lock = False
        has_score = (public_run / "sealed_receipts" / f"{window_id}.json").is_file()
        status_rows.append(
            {
                "window_id": window_id,
                "phase": window_phase(window, now, has_lock, has_score),
                "decision_date": window["decision_date"],
                "buy_date": window["buy_date"],
                "sell_date": window["sell_date"],
            }
        )
    payload = {
        "experiment_id": config["experiment_id"],
        "at": now.isoformat(),
        "state_status": state["status"],
        "performance_blinded": not state["performance_values_present"],
        "sealed_score_count": state["sealed_score_count"],
        "windows": status_rows,
    }
    write_json(public_run / "protocol_status.json", payload)
    return payload


def synthetic_dry_run(config: dict[str, Any]) -> dict[str, Any]:
    validate_preregistration(config)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prefix = root / "prefix.csv"
        pd.DataFrame(
            {
                COL_CODE: ["000001", "000002"],
                COL_DATE: ["2026-07-31", "2026-07-31"],
                COL_OPEN: [10.0, 20.0],
            }
        ).to_csv(prefix, index=False, encoding="utf-8-sig")
        prefix_report = validate_training_prefix(prefix, "2026-07-31")
        result1 = root / "run1.csv"
        result2 = root / "run2.csv"
        content = "stock_id,weight\n000001,0.5\n000002,0.5\n"
        result1.write_text(content, encoding="utf-8")
        result2.write_text(content, encoding="utf-8")
        lock = build_prediction_lock(
            window=config["windows"][0],
            candidate="Candidate-D",
            locked_at="2026-08-03T09:20:00+08:00",
            training_sha256=prefix_report["sha256"],
            checkpoint_hashes={"checkpoint": "A" * 64},
            config_hashes={"config": "B" * 64},
            result_paths=[result1, result2],
        )
        audit = root / "audit.jsonl"
        append_hash_chained_event(audit, {"event": "INIT"})
        append_hash_chained_event(audit, {"event": "LOCK", "window_id": "P01"})
        chain = validate_audit_chain(audit)
        receipt = store_sealed_score(
            vault_root=root / "external_vault",
            public_receipt_path=root / "public" / "P01.json",
            window_id="P01",
            scored_at="2026-08-07T09:35:00+08:00",
            score_not_before=config["windows"][0]["score_not_before"],
            outcome_sha256="C" * 64,
            secret_payload={"synthetic": True, "weekly_return_net_10bps": 0.01},
        )
        return {
            "status": "PASS",
            "uses_real_future_data": False,
            "training_prefix": prefix_report,
            "prediction_lock": {
                "repeated_inference_byte_identical": lock[
                    "repeated_inference_byte_identical"
                ],
                "result_sha256": lock["result_sha256"],
            },
            "audit_chain": chain,
            "sealed_receipt": receipt,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-resistant prospective Candidate-D stability controller."
    )
    parser.add_argument(
        "stage",
        choices=[
            "init",
            "verify-freeze",
            "status",
            "prepare",
            "train-lock",
            "score",
            "audit",
            "finalize",
            "dry-run",
        ],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-plan", type=Path, default=DEFAULT_DATA_PLAN)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--public-run", type=Path, default=DEFAULT_PUBLIC_RUN)
    parser.add_argument("--window")
    parser.add_argument("--market-prefix", type=Path)
    parser.add_argument("--outcome-market", type=Path)
    parser.add_argument("--sse-open-csv", type=Path)
    parser.add_argument("--vault-root", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--as-of")
    return parser.parse_args()


def _required(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"{name} is required for this stage")
    return value


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    data_plan_path = args.data_plan.resolve()
    experiment_root = args.experiment_root.resolve()
    public_run = args.public_run.resolve()
    timezone_name = read_json(config_path).get("timezone", "Asia/Shanghai")
    now = (
        parse_timestamp(args.as_of)
        if args.as_of
        else local_now(timezone_name)
    )
    if args.stage in {"prepare", "train-lock", "score", "finalize"}:
        verify_execution_freeze(
            data_plan_path=data_plan_path,
            experiment_root=experiment_root,
        )
    if args.stage == "init":
        payload = initialize_protocol(config_path, experiment_root, public_run)
    elif args.stage == "verify-freeze":
        payload = verify_execution_freeze(
            data_plan_path=data_plan_path,
            experiment_root=experiment_root,
        )
    elif args.stage == "status":
        payload = protocol_status(
            config_path=config_path,
            experiment_root=experiment_root,
            public_run=public_run,
            now=now,
        )
    elif args.stage == "prepare":
        payload = {
            "round_dir": str(
                prepare_window(
                    config_path=config_path,
                    experiment_root=experiment_root,
                    window_id=_required(args.window, "--window"),
                    market_prefix=_required(
                        args.market_prefix,
                        "--market-prefix",
                    ).resolve(),
                    now=now,
                )
            )
        }
    elif args.stage == "train-lock":
        payload = {
            "locks": train_and_lock_window(
                config_path=config_path,
                experiment_root=experiment_root,
                window_id=_required(args.window, "--window"),
                now=now,
                python_executable=args.python.resolve(),
            )
        }
    elif args.stage == "score":
        payload = score_window(
            config_path=config_path,
            experiment_root=experiment_root,
            public_run=public_run,
            vault_root=_required(args.vault_root, "--vault-root").resolve(),
            window_id=_required(args.window, "--window"),
            outcome_market=_required(
                args.outcome_market,
                "--outcome-market",
            ).resolve(),
            sse_open_csv=_required(
                args.sse_open_csv,
                "--sse-open-csv",
            ).resolve(),
            now=now,
        )
    elif args.stage == "audit":
        payload = integrity_audit(
            config_path=config_path,
            experiment_root=experiment_root,
            public_run=public_run,
        )
    elif args.stage == "finalize":
        payload = finalize_scores(
            config_path=config_path,
            experiment_root=experiment_root,
            public_run=public_run,
            vault_root=_required(args.vault_root, "--vault-root").resolve(),
        )
    else:
        payload = synthetic_dry_run(validate_preregistration(read_json(config_path)))
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

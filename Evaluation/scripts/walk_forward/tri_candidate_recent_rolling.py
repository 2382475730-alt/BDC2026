from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = ROOT / "Experimental_WorkSpace" / "tri_candidate_rolling_validation_20260722"
STAGING_CANDIDATES = EXPERIMENT_ROOT / "candidates"
ROUND_ROOT = EXPERIMENT_ROOT / "rounds"
BASE_TRAIN = EXPERIMENT_ROOT / "data" / "shared_train_through_20260703.csv"
PLAN_PATH = EXPERIMENT_ROOT / "manifests" / "training_round_plan.json"
PYTHON_EXE = ROOT / "Experimental_WorkSpace" / "shared_envs" / "pytorch-cu128-py311" / "Scripts" / "python.exe"
VALIDATION_ROOT = Path(
    r"D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本"
    r"\BDC2026_Validation_20260706_20260718_Isolated"
)
VALIDATION_CSV = VALIDATION_ROOT / "validation_20260706_20260717.csv"
VALIDATION_REPORT = VALIDATION_ROOT / "validation_download_report.json"
UNIVERSE_REPORT = VALIDATION_ROOT / "hs300_universe_check_20260722.json"
EVAL_RUN = ROOT / "Evaluation" / "runs" / "tri_candidate_recent_rolling_20260722"
REPORT_PATH = ROOT / "Evaluation" / "reports" / "tri_candidate_recent_rolling_20260722.md"

COL_CODE = "股票代码"
COL_DATE = "日期"
COL_OPEN = "开盘"

ROUND_DEFINITIONS: dict[str, dict[str, Any]] = {
    "W1_20260703": {
        "decision_date": "2026-07-03",
        "buy_date": "2026-07-06",
        "sell_date": "2026-07-10",
        "cutoff": "2026-07-03",
        "expected_rows": 337852,
        "expected_sample_count": 1083,
    },
    "W2_20260710": {
        "decision_date": "2026-07-10",
        "buy_date": "2026-07-13",
        "sell_date": "2026-07-17",
        "cutoff": "2026-07-10",
        "expected_rows": 339352,
        "expected_sample_count": 1088,
    },
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    train_args: tuple[str, ...]
    predict_args: tuple[str, ...]
    checkpoint_paths: tuple[str, ...]
    config_paths: tuple[str, ...]


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="THU-BDC2026-main",
        train_args=("code/src/train.py",),
        predict_args=("code/src/predict.py",),
        checkpoint_paths=("model/60_158+39/best_model.pth", "model/60_158+39/scaler.pkl"),
        config_paths=("code/src/config.py", "model/60_158+39/config.json"),
    ),
    CandidateSpec(
        name="Candidate-C",
        train_args=("code/src/train.py",),
        predict_args=("code/src/test.py", "--work-dir", ".", "--output", "output/result.csv"),
        checkpoint_paths=(
            "model/60_158+39/best_model.pth",
            "model/60_158+39/scaler.pkl",
        ),
        config_paths=(
            "code/src/config.py",
            "model/60_158+39/config.json",
            "model/60_158+39/rerank_config.json",
        ),
    ),
    CandidateSpec(
        name="Candidate-D",
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
        checkpoint_paths=("model/final_model.pth", "model/scalers.json"),
        config_paths=("model/config.json",),
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_tree(path: Path, *, excluded_names: set[str] | None = None) -> str:
    excluded_names = excluded_names or set()
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file() and not any(part in excluded_names for part in p.parts)):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def load_market_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={COL_CODE: str}, encoding="utf-8-sig")
    frame[COL_CODE] = frame[COL_CODE].map(normalize_code)
    frame[COL_DATE] = pd.to_datetime(frame[COL_DATE]).dt.normalize()
    return frame.sort_values([COL_DATE, COL_CODE]).reset_index(drop=True)


def load_market_through_date(path: Path, cutoff: str) -> pd.DataFrame:
    """Parse complete market rows only when their date is inside the unsealed prefix."""
    allowed_lines: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"empty market file: {path}")
        allowed_lines.append(header)
        for line in handle:
            prefix = line.split(",", 2)
            if len(prefix) < 3:
                raise ValueError(f"malformed market row in {path}")
            if prefix[1] <= cutoff:
                allowed_lines.append(line)
    frame = pd.read_csv(io.StringIO("".join(allowed_lines)), dtype={COL_CODE: str})
    frame[COL_CODE] = frame[COL_CODE].map(normalize_code)
    frame[COL_DATE] = pd.to_datetime(frame[COL_DATE]).dt.normalize()
    return frame.sort_values([COL_DATE, COL_CODE]).reset_index(drop=True)


def inspect_train(path: Path) -> dict[str, Any]:
    frame = load_market_csv(path)
    return {
        "path": str(path.resolve()),
        "rows": int(len(frame)),
        "stocks": int(frame[COL_CODE].nunique()),
        "date_start": str(frame[COL_DATE].min().date()),
        "date_end": str(frame[COL_DATE].max().date()),
        "sha256": sha256_file(path),
    }


def environment_manifest() -> dict[str, Any]:
    probe = (
        "import json,sys,torch,pandas,numpy,sklearn,joblib;"
        "print(json.dumps({'python':sys.version,'torch':torch.__version__,"
        "'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda,"
        "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
        "'pandas':pandas.__version__,'numpy':numpy.__version__,"
        "'sklearn':sklearn.__version__,'joblib':joblib.__version__},ensure_ascii=False))"
    )
    completed = subprocess.run(
        [str(PYTHON_EXE), "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    payload = json.loads(completed.stdout.strip())
    payload.update({"python_executable": str(PYTHON_EXE.resolve()), "captured_at": utc_now()})
    return payload


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


def _replace_with_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _remove_if_exists(destination)
    os.link(source, destination)


def _prepare_candidate_copy(spec: CandidateSpec, round_dir: Path, shared_train: Path, round_id: str) -> dict[str, Any]:
    source = STAGING_CANDIDATES / spec.name
    destination = round_dir / "candidates" / spec.name
    if destination.exists():
        raise FileExistsError(f"round candidate already exists: {destination}")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _clear_directory(destination / "output")
    _clear_directory(destination / "temp")
    _replace_with_hardlink(shared_train, destination / "data" / "train.csv")
    _remove_if_exists(destination / "data" / "test.csv")

    if spec.name in {"THU-BDC2026-main", "Candidate-C"}:
        _clear_directory(destination / "model" / "60_158+39")
    else:
        _remove_if_exists(destination / "model" / "final_model.pth")
        _remove_if_exists(destination / "model" / "scalers.json")

    if spec.name == "Candidate-D" and round_id == "W2_20260710":
        config_path = destination / "model" / "config.json"
        config = read_json(config_path)
        config["experiment_id"] = "Candidate-D_Rolling_W2_20260710"
        config["submission"]["raw_data_end"] = "2026-07-10"
        config["submission"]["train_sample_end"] = "2026-07-03"
        config["submission"]["expected_train_sample_count"] = 1088
        write_json(config_path, config)

    return {
        "candidate": spec.name,
        "source_path": str(source.resolve()),
        "source_tree_sha256": sha256_tree(source, excluded_names={"output", "temp", "__pycache__"}),
        "round_path": str(destination.resolve()),
        "train_sha256": sha256_file(destination / "data" / "train.csv"),
    }


def preflight() -> dict[str, Any]:
    required = [PYTHON_EXE, BASE_TRAIN, PLAN_PATH, VALIDATION_CSV, VALIDATION_REPORT, UNIVERSE_REPORT]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    for spec in CANDIDATES:
        if not (STAGING_CANDIDATES / spec.name).is_dir():
            raise FileNotFoundError(STAGING_CANDIDATES / spec.name)
    plan = read_json(PLAN_PATH)
    validation_report = read_json(VALIDATION_REPORT)
    universe_report = read_json(UNIVERSE_REPORT)
    base = inspect_train(BASE_TRAIN)
    issues: list[str] = []
    if base["sha256"].lower() != plan["shared_base"]["sha256"].lower():
        issues.append("base training hash differs from frozen plan")
    reported_validation_hash = validation_report["files"][VALIDATION_CSV.name]["sha256"]
    if reported_validation_hash.lower() != plan["isolated_validation"]["sha256"].lower():
        issues.append("validation report hash differs from frozen plan")
    if not validation_report["w2_feature_training_gate"]["passed"]:
        issues.append("W2 feature gate failed")
    if not universe_report["exact_match"]:
        issues.append("local universe no longer matches frozen official universe audit")
    if base["rows"] != ROUND_DEFINITIONS["W1_20260703"]["expected_rows"]:
        issues.append("unexpected W1 training row count")
    if issues:
        raise ValueError("; ".join(issues))
    payload = {
        "status": "passed",
        "captured_at": utc_now(),
        "environment": environment_manifest(),
        "base_train": base,
        "validation": {
            "path": str(VALIDATION_CSV.resolve()),
            "reported_sha256": reported_validation_hash,
            "rows": validation_report["row_count"],
            "stocks": validation_report["stock_count"],
            "w2_feature_gate_passed": validation_report["w2_feature_training_gate"]["passed"],
            "price_bytes_accessed_before_w1_lock": False,
        },
        "universe_exact_match": universe_report["exact_match"],
        "candidate_sources": [str((STAGING_CANDIDATES / spec.name).resolve()) for spec in CANDIDATES],
    }
    write_json(EXPERIMENT_ROOT / "manifests" / "execution_preflight.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def prepare_w1() -> Path:
    preflight()
    round_id = "W1_20260703"
    round_dir = ROUND_ROOT / round_id
    if round_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing round: {round_dir}")
    (round_dir / "data").mkdir(parents=True)
    shared_train = round_dir / "data" / "shared_train.csv"
    _replace_with_hardlink(BASE_TRAIN, shared_train)
    copies = [_prepare_candidate_copy(spec, round_dir, shared_train, round_id) for spec in CANDIDATES]
    hashes = {item["train_sha256"] for item in copies}
    train_info = inspect_train(shared_train)
    if len(hashes) != 1 or train_info["date_end"] != ROUND_DEFINITIONS[round_id]["cutoff"]:
        raise ValueError("W1 shared input invariant failed")
    manifest = {
        "round_id": round_id,
        **ROUND_DEFINITIONS[round_id],
        "created_at": utc_now(),
        "shared_train": train_info,
        "candidate_copies": copies,
        "future_validation_materialized": False,
        "status": "prepared",
    }
    write_json(round_dir / "round_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return round_dir


def require_all_locks(round_id: str) -> list[dict[str, Any]]:
    locks = []
    for spec in CANDIDATES:
        path = ROUND_ROOT / round_id / "candidates" / spec.name / "prediction_lock.json"
        if not path.exists():
            raise FileNotFoundError(f"missing prediction lock: {path}")
        lock = read_json(path)
        if lock.get("status") != "locked" or not lock.get("repeated_inference_byte_identical"):
            raise ValueError(f"invalid prediction lock: {path}")
        result_path = Path(lock["result_path"])
        if not result_path.exists() or sha256_file(result_path) != lock.get("result_sha256"):
            raise ValueError(f"locked result hash mismatch: {result_path}")
        locks.append(lock)
    return locks


def materialize_w2() -> Path:
    require_all_locks("W1_20260703")
    round_id = "W2_20260710"
    round_dir = ROUND_ROOT / round_id
    if round_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing round: {round_dir}")
    base = load_market_csv(BASE_TRAIN)
    validation = load_market_through_date(VALIDATION_CSV, cutoff="2026-07-10")
    if len(validation) != 1500 or validation[COL_DATE].max() != pd.Timestamp("2026-07-10"):
        raise ValueError("chronological W1 validation prefix is not exactly 2026-07-06 through 2026-07-10")
    week1 = validation[
        (validation[COL_DATE] >= pd.Timestamp("2026-07-06"))
        & (validation[COL_DATE] <= pd.Timestamp("2026-07-10"))
    ].copy()
    if len(week1) != 1500 or week1[COL_CODE].nunique() != 300:
        raise ValueError("W2 append slice must contain 1500 rows and 300 stocks")
    combined = pd.concat([base, week1], ignore_index=True)
    combined = combined.drop_duplicates([COL_CODE, COL_DATE], keep="last").sort_values([COL_DATE, COL_CODE])
    if len(combined) != ROUND_DEFINITIONS[round_id]["expected_rows"]:
        raise ValueError(f"unexpected W2 row count: {len(combined)}")
    if combined[COL_DATE].max() != pd.Timestamp("2026-07-10"):
        raise ValueError("W2 contains an invalid maximum date")

    (round_dir / "data").mkdir(parents=True)
    shared_train = round_dir / "data" / "shared_train.csv"
    output = combined.copy()
    output[COL_DATE] = output[COL_DATE].dt.strftime("%Y-%m-%d")
    output[COL_CODE] = output[COL_CODE].astype(int)
    output.to_csv(shared_train, index=False, encoding="utf-8")
    copies = [_prepare_candidate_copy(spec, round_dir, shared_train, round_id) for spec in CANDIDATES]
    hashes = {item["train_sha256"] for item in copies}
    train_info = inspect_train(shared_train)
    if len(hashes) != 1 or train_info["date_end"] != ROUND_DEFINITIONS[round_id]["cutoff"]:
        raise ValueError("W2 shared input invariant failed")
    manifest = {
        "round_id": round_id,
        **ROUND_DEFINITIONS[round_id],
        "created_at": utc_now(),
        "shared_train": train_info,
        "candidate_copies": copies,
        "future_validation_materialized": True,
        "materialized_validation_dates": ["2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10"],
        "excluded_validation_dates": ["2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16", "2026-07-17"],
        "w1_prediction_locks": [lock["result_sha256"] for lock in require_all_locks("W1_20260703")],
        "status": "prepared",
    }
    write_json(round_dir / "round_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return round_dir


def run_process(args: list[str], cwd: Path, log_path: Path, timeout_seconds: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_at = utc_now()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write("COMMAND=" + json.dumps(args, ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if time.perf_counter() - started > timeout_seconds:
                    process.kill()
                    raise TimeoutError(f"command exceeded {timeout_seconds}s: {args}")
            return_code = process.wait(timeout=max(1, timeout_seconds - int(time.perf_counter() - started)))
        except BaseException:
            process.kill()
            process.wait()
            raise
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, args)
    return {
        "command": args,
        "cwd": str(cwd.resolve()),
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "return_code": return_code,
        "log_path": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
    }


def _restore_frozen_c_policy(candidate_dir: Path) -> None:
    source = STAGING_CANDIDATES / "Candidate-C" / "model" / "60_158+39" / "rerank_config.json"
    destination = candidate_dir / "model" / "60_158+39" / "rerank_config.json"
    shutil.copy2(source, destination)


def _artifact_hashes(candidate_dir: Path, relative_paths: Iterable[str], *, require_all: bool = True) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in relative_paths:
        path = candidate_dir / relative
        if path.exists():
            output[relative] = sha256_file(path)
        elif require_all:
            raise FileNotFoundError(path)
    return output


def validate_result(path: Path, universe: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"stock_id": str}, encoding="utf-8-sig")
    if list(frame.columns) != ["stock_id", "weight"]:
        raise ValueError(f"invalid result columns: {frame.columns.tolist()}")
    frame["stock_id"] = frame["stock_id"].map(normalize_code)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    if not 1 <= len(frame) <= 5:
        raise ValueError("result must contain one to five rows")
    if frame["stock_id"].duplicated().any():
        raise ValueError("result contains duplicate stock IDs")
    if not set(frame["stock_id"]).issubset(universe):
        raise ValueError("result contains stocks outside the frozen HS300 universe")
    if (frame["weight"] < 0).any() or not np.isfinite(frame["weight"]).all():
        raise ValueError("result contains invalid weights")
    if float(frame["weight"].sum()) > 1.0 + 1e-10:
        raise ValueError("result weight sum exceeds one")
    return frame


def train_and_lock_round(round_id: str) -> None:
    round_dir = ROUND_ROOT / round_id
    manifest_path = round_dir / "round_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    shared_train = round_dir / "data" / "shared_train.csv"
    expected_hash = sha256_file(shared_train)
    universe = set(load_market_csv(shared_train)[COL_CODE].unique())
    for spec in CANDIDATES:
        candidate_dir = round_dir / "candidates" / spec.name
        if sha256_file(candidate_dir / "data" / "train.csv") != expected_hash:
            raise ValueError(f"train hash mismatch before training: {spec.name}")
        lock_path = candidate_dir / "prediction_lock.json"
        if lock_path.exists():
            lock = read_json(lock_path)
            locked_result = Path(lock.get("result_path", ""))
            if (
                lock.get("status") == "locked"
                and lock.get("repeated_inference_byte_identical") is True
                and lock.get("training_data_sha256") == expected_hash
                and locked_result.exists()
                and sha256_file(locked_result) == lock.get("result_sha256")
            ):
                print(f"SKIP already locked {round_id} {spec.name}", flush=True)
                continue
            raise ValueError(f"existing but invalid prediction lock: {lock_path}")
        print(f"\n===== TRAIN {round_id} {spec.name} =====", flush=True)
        training = run_process(
            [str(PYTHON_EXE), *spec.train_args],
            cwd=candidate_dir,
            log_path=candidate_dir / "logs" / "training.log",
            timeout_seconds=8 * 60 * 60,
        )
        if spec.name == "Candidate-C":
            _restore_frozen_c_policy(candidate_dir)
        checkpoint_hashes = _artifact_hashes(candidate_dir, spec.checkpoint_paths)
        training_manifest = {
            "round_id": round_id,
            "candidate": spec.name,
            "status": "completed",
            "train_data": inspect_train(candidate_dir / "data" / "train.csv"),
            "training": training,
            "checkpoint_hashes": checkpoint_hashes,
            "config_hashes": _artifact_hashes(candidate_dir, spec.config_paths),
            "training_time_limit_seconds": 8 * 60 * 60,
            "training_time_passed": training["elapsed_seconds"] < 8 * 60 * 60,
        }
        write_json(candidate_dir / "training_execution.json", training_manifest)

        locked_dir = candidate_dir / "locked"
        locked_dir.mkdir(parents=True, exist_ok=True)
        inference_runs = []
        result_hashes = []
        for run_number in (1, 2):
            print(f"\n===== INFERENCE {run_number} {round_id} {spec.name} =====", flush=True)
            execution = run_process(
                [str(PYTHON_EXE), *spec.predict_args],
                cwd=candidate_dir,
                log_path=candidate_dir / "logs" / f"inference_{run_number}.log",
                timeout_seconds=5 * 60,
            )
            result_path = candidate_dir / "output" / "result.csv"
            validate_result(result_path, universe)
            frozen_path = locked_dir / f"result_run{run_number}.csv"
            shutil.copy2(result_path, frozen_path)
            result_hashes.append(sha256_file(frozen_path))
            execution["result_path"] = str(frozen_path.resolve())
            execution["result_sha256"] = result_hashes[-1]
            execution["inference_time_passed"] = execution["elapsed_seconds"] < 5 * 60
            inference_runs.append(execution)
        if result_hashes[0] != result_hashes[1]:
            raise ValueError(f"non-deterministic repeated inference: {round_id} {spec.name}")
        final_result = locked_dir / "result.csv"
        shutil.copy2(locked_dir / "result_run1.csv", final_result)
        result_frame = validate_result(final_result, universe)
        lock = {
            "schema_version": "tri_candidate_prediction_lock_v1",
            "status": "locked",
            "locked_at": utc_now(),
            "round_id": round_id,
            "candidate": spec.name,
            "decision_date": ROUND_DEFINITIONS[round_id]["decision_date"],
            "training_data_sha256": expected_hash,
            "training_data_date_end": inspect_train(shared_train)["date_end"],
            "checkpoint_hashes": checkpoint_hashes,
            "config_hashes": _artifact_hashes(candidate_dir, spec.config_paths),
            "result_path": str(final_result.resolve()),
            "result_sha256": sha256_file(final_result),
            "result_rows": int(len(result_frame)),
            "result_weight_sum": float(result_frame["weight"].sum()),
            "selected_stocks": result_frame["stock_id"].tolist(),
            "repeated_inference_byte_identical": True,
            "inference_runs": inference_runs,
            "training_execution_path": str((candidate_dir / "training_execution.json").resolve()),
            "future_returns_accessed": False,
        }
        write_json(candidate_dir / "prediction_lock.json", lock)
        print(json.dumps(lock, ensure_ascii=False, indent=2), flush=True)
    locks = require_all_locks(round_id)
    manifest["status"] = "predictions_locked"
    manifest["prediction_locks"] = [
        {
            "candidate": lock["candidate"],
            "result_sha256": lock["result_sha256"],
            "locked_at": lock["locked_at"],
        }
        for lock in locks
    ]
    write_json(manifest_path, manifest)


def portfolio_week_score(
    result: pd.DataFrame,
    market: pd.DataFrame,
    *,
    buy_date: str,
    sell_date: str,
    estimated_codes: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    buy = market[market[COL_DATE] == pd.Timestamp(buy_date)][[COL_CODE, COL_OPEN]].rename(columns={COL_OPEN: "open_buy"})
    sell = market[market[COL_DATE] == pd.Timestamp(sell_date)][[COL_CODE, COL_OPEN]].rename(columns={COL_OPEN: "open_sell"})
    prices = buy.merge(sell, on=COL_CODE, how="inner")
    day = result.merge(prices, left_on="stock_id", right_on=COL_CODE, how="left")
    if day[["open_buy", "open_sell"]].isna().any().any():
        raise ValueError("selected portfolio has missing evaluation prices")
    day["stock_return"] = day["open_sell"] / day["open_buy"] - 1.0
    day["weighted_return"] = day["weight"] * day["stock_return"]
    invested = float(day["weight"].sum())
    gross = float(day["weighted_return"].sum())
    net = {bps: gross - 2.0 * bps / 10000.0 * invested for bps in (0, 10, 20)}
    universe_returns = prices["open_sell"] / prices["open_buy"] - 1.0
    benchmark = float(universe_returns.mean())
    weights = day["weight"].to_numpy(dtype=float)
    returns = day["stock_return"].to_numpy(dtype=float)
    hhi = float(np.sum(np.square(weights)))
    normalized = weights / invested if invested > 0 else np.zeros_like(weights)
    normalized_mean = float(np.sum(normalized * returns)) if invested > 0 else 0.0
    dispersion = float(np.sqrt(np.sum(normalized * np.square(returns - normalized_mean)))) if invested > 0 else 0.0
    equal_weight = invested / len(day) if len(day) else 0.0
    equal_selected_gross = float(equal_weight * returns.sum())
    summary = {
        "weekly_return_gross": gross,
        "weekly_return_net_0bps": net[0],
        "weekly_return_net_10bps": net[10],
        "weekly_return_net_20bps": net[20],
        "benchmark_equal_weight_return": benchmark,
        "gross_excess_vs_benchmark": gross - benchmark,
        "net_10bps_excess_vs_benchmark": net[10] - benchmark,
        "weight_sum": invested,
        "cash_ratio": max(0.0, 1.0 - invested),
        "selected_count": int(len(day)),
        "selected_stocks": ",".join(day["stock_id"].tolist()),
        "max_weight": float(weights.max()) if len(weights) else 0.0,
        "hhi": hhi,
        "effective_holdings": 1.0 / hhi if hhi > 0 else 0.0,
        "worst_holding_return": float(returns.min()) if len(returns) else 0.0,
        "holding_return_dispersion": dispersion,
        "estimated_feature_exposure_weight": float(
            day.loc[day["stock_id"].isin(estimated_codes), "weight"].sum()
        ),
        "equal_selected_gross_return": equal_selected_gross,
    }
    details = [
        {
            "stock_id": row.stock_id,
            "weight": float(row.weight),
            "open_buy": float(row.open_buy),
            "open_sell": float(row.open_sell),
            "stock_return": float(row.stock_return),
            "weighted_return": float(row.weighted_return),
            "estimated_feature_stock": row.stock_id in estimated_codes,
        }
        for row in day.itertuples(index=False)
    ]
    return summary, details


def max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))
    values = np.concatenate(([1.0], equity))
    drawdown = values / np.maximum(peak, 1e-12) - 1.0
    return float(abs(drawdown.min()))


def portfolio_turnover(previous: dict[str, float], current: dict[str, float]) -> float:
    """One-way portfolio turnover, including any change in the cash weight."""
    stock_ids = set(previous) | set(current)
    traded_weight = sum(abs(current.get(stock_id, 0.0) - previous.get(stock_id, 0.0)) for stock_id in stock_ids)
    previous_cash = max(0.0, 1.0 - sum(previous.values()))
    current_cash = max(0.0, 1.0 - sum(current.values()))
    return float(0.5 * (traded_weight + abs(current_cash - previous_cash)))


def summarize_candidate(rows: pd.DataFrame) -> dict[str, Any]:
    rows = rows.sort_values("buy_date")
    gross = rows["weekly_return_gross"].to_numpy(dtype=float)
    net10 = rows["weekly_return_net_10bps"].to_numpy(dtype=float)
    net20 = rows["weekly_return_net_20bps"].to_numpy(dtype=float)
    worst = float(net10.min())
    return {
        "week_count": int(len(rows)),
        "mean_weekly_return_gross": float(gross.mean()),
        "median_weekly_return_gross": float(np.median(gross)),
        "mean_weekly_return_net_10bps": float(net10.mean()),
        "median_weekly_return_net_10bps": float(np.median(net10)),
        "mean_weekly_return_net_20bps": float(net20.mean()),
        "cumulative_return_gross": float(np.prod(1.0 + gross) - 1.0),
        "cumulative_return_net_10bps": float(np.prod(1.0 + net10) - 1.0),
        "positive_week_rate_net_10bps": float(np.mean(net10 > 0.0)),
        "worst_week_net_10bps": worst,
        "weekly_volatility_net_10bps": float(np.std(net10, ddof=0)),
        "max_drawdown_net_10bps": max_drawdown(net10),
        "recent_robust_score": float(0.7 * net10.mean() + 0.3 * worst),
        "mean_gross_excess_vs_benchmark": float(rows["gross_excess_vs_benchmark"].mean()),
        "mean_net_10bps_excess_vs_benchmark": float(rows["net_10bps_excess_vs_benchmark"].mean()),
        "mean_hhi": float(rows["hhi"].mean()),
        "mean_effective_holdings": float(rows["effective_holdings"].mean()),
        "mean_max_weight": float(rows["max_weight"].mean()),
        "mean_estimated_feature_exposure_weight": float(rows["estimated_feature_exposure_weight"].mean()),
        "cross_week_jaccard": float(rows["jaccard_prev_week"].dropna().mean()) if rows["jaccard_prev_week"].notna().any() else None,
        "cross_week_turnover": float(rows["turnover_prev_week"].dropna().mean()) if rows["turnover_prev_week"].notna().any() else None,
        "all_rounds_valid": bool(rows["validation_ok"].all()),
        "formal_robustness_gate_active": len(rows) >= 8,
        "minimum_weeks_for_formal_gate": 8,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.{digits}%}"


def provisional_labels(summaries: dict[str, dict[str, Any]]) -> dict[str, str]:
    baseline = summaries["THU-BDC2026-main"]
    highest = max(item["mean_weekly_return_net_10bps"] for item in summaries.values())
    labels = {"THU-BDC2026-main": "官方基准"}
    for name, item in summaries.items():
        if name == "THU-BDC2026-main":
            continue
        mean_highest = item["mean_weekly_return_net_10bps"] >= highest - 1e-12
        cumulative_beats = item["cumulative_return_gross"] > baseline["cumulative_return_gross"]
        risk_passes = item["worst_week_net_10bps"] >= baseline["worst_week_net_10bps"] - 0.01
        valid = item["all_rounds_valid"]
        if mean_highest and cumulative_beats and risk_passes and valid:
            labels[name] = "临时领先"
        elif mean_highest and (not risk_passes or not cumulative_beats):
            labels[name] = "收益领先但风险未通过"
        elif item["worst_week_net_10bps"] >= baseline["worst_week_net_10bps"] and not mean_highest:
            labels[name] = "风险较低但收益未领先"
        else:
            labels[name] = "未领先"
    return labels


def build_report(
    weekly: pd.DataFrame,
    summaries: dict[str, dict[str, Any]],
    labels: dict[str, str],
    leakage: dict[str, Any],
    runtime: dict[str, Any],
) -> str:
    ranked = sorted(summaries, key=lambda name: summaries[name]["mean_weekly_return_net_10bps"], reverse=True)
    risk_ranked = sorted(
        summaries,
        key=lambda name: (
            summaries[name]["worst_week_net_10bps"],
            -summaries[name]["max_drawdown_net_10bps"],
        ),
        reverse=True,
    )
    lines = [
        "# 三候选近期滚动验证报告",
        "",
        "## 结论",
        "",
        "当前仅有两个独立周窗口，以下结论属于近期观察，不构成候选晋级或未来收益保证。正式稳健门将在累计至少8周后启用。",
        "",
        "| 排名 | 候选 | 周均毛收益 | 周均净收益(单边10bps) | 两周累计毛收益 | 最差周净收益 | 近期稳健分 | 结论 |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for index, name in enumerate(ranked, start=1):
        item = summaries[name]
        lines.append(
            f"| {index} | {name} | {_fmt(item['mean_weekly_return_gross'])} | "
            f"{_fmt(item['mean_weekly_return_net_10bps'])} | {_fmt(item['cumulative_return_gross'])} | "
            f"{_fmt(item['worst_week_net_10bps'])} | {_fmt(item['recent_robust_score'])} | {labels[name]} |"
        )
    baseline = summaries["THU-BDC2026-main"]
    lines.extend(
        [
            "",
            "## 相对THU差值",
            "",
            "| 候选 | 周均净收益差(10bps) | 两周累计毛收益差 | 最差周净收益差 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("Candidate-C", "Candidate-D"):
        item = summaries[name]
        lines.append(
            f"| {name} | {_fmt(item['mean_weekly_return_net_10bps'] - baseline['mean_weekly_return_net_10bps'])} | "
            f"{_fmt(item['cumulative_return_gross'] - baseline['cumulative_return_gross'])} | "
            f"{_fmt(item['worst_week_net_10bps'] - baseline['worst_week_net_10bps'])} |"
        )
    lines.extend(
        [
            "",
            "## 分周结果",
            "",
            "| 周次 | 候选 | 买入-卖出 | 毛收益 | 净收益10bps | 沪深300等权 | 毛超额 | 持仓 | 最大权重 | 有效持仓 | 估算字段暴露 |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in weekly.sort_values(["round_id", "candidate"]).itertuples(index=False):
        lines.append(
            f"| {row.round_id} | {row.candidate} | {row.buy_date} 至 {row.sell_date} | "
            f"{_fmt(row.weekly_return_gross)} | {_fmt(row.weekly_return_net_10bps)} | "
            f"{_fmt(row.benchmark_equal_weight_return)} | {_fmt(row.gross_excess_vs_benchmark)} | "
            f"{row.selected_stocks} | {_fmt(row.max_weight)} | {row.effective_holdings:.2f} | "
            f"{_fmt(row.estimated_feature_exposure_weight)} |"
        )
    lines.extend(
        [
            "",
            "## 所选股票等权诊断",
            "",
            "该诊断固定选股集合、仅把组合内部改为等权，用于区分选股效果与原生权重策略效果。",
            "",
            "| 周次 | 候选 | 原生组合毛收益 | 所选股票等权毛收益 | 原生权重贡献差 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in weekly.sort_values(["round_id", "candidate"]).itertuples(index=False):
        lines.append(
            f"| {row.round_id} | {row.candidate} | {_fmt(row.weekly_return_gross)} | "
            f"{_fmt(row.equal_selected_gross_return)} | "
            f"{_fmt(row.weekly_return_gross - row.equal_selected_gross_return)} |"
        )
    lines.extend(
        [
            "",
            "## 风险与稳定性",
            "",
            "| 风险排名 | 候选 | 最差周净收益 | 周波动 | 最大回撤 | HHI | 平均有效持仓 | 跨周换手 | 平均基准超额 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for risk_rank, name in enumerate(risk_ranked, start=1):
        item = summaries[name]
        lines.append(
            f"| {risk_rank} | {name} | {_fmt(item['worst_week_net_10bps'])} | "
            f"{_fmt(item['weekly_volatility_net_10bps'])} | {_fmt(item['max_drawdown_net_10bps'])} | "
            f"{item['mean_hhi']:.4f} | {item['mean_effective_holdings']:.2f} | "
            f"{_fmt(item['cross_week_turnover'])} | {_fmt(item['mean_gross_excess_vs_benchmark'])} |"
        )
    environment = runtime["environment"]
    lines.extend(
        [
            "",
            "## 执行环境与耗时",
            "",
            f"- GPU：{environment['gpu']}；Python：{environment['python'].split()[0]}；"
            f"PyTorch：{environment['torch']}；CUDA：{environment['cuda_runtime']}。",
            "",
            "| 周次 | 候选 | 训练分钟 | 推理1秒 | 推理2秒 | 训练<8小时 | 推理<5分钟 |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in runtime["executions"]:
        lines.append(
            f"| {row['round_id']} | {row['candidate']} | {row['training_seconds'] / 60.0:.2f} | "
            f"{row['inference_1_seconds']:.2f} | {row['inference_2_seconds']:.2f} | "
            f"{'是' if row['training_under_8h'] else '否'} | {'是' if row['inference_under_5m'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 协议与限制",
            "",
            f"- 泄露审计：`{leakage['status']}`；W1训练数据截至2026-07-03，W2截至2026-07-10。",
            "- 三候选每轮使用字节一致的训练表，各自保留原生训练、内部验证和仓位规则。",
            "- 收益按T+1开盘至T+5开盘计算；现金收益为0；实务主指标扣除单边10bps。",
            "- 验证期开盘价用于评分，不使用估算值；估算成交额/换手率只可能影响W2特征，并单独披露持仓暴露。",
            "- 当前只有两周，CVaR和最差20%周均值不用于决策；累计到8周后再启用正式稳健门。",
            "- 本次不会修改或晋级正式candidate目录。",
            "",
        ]
    )
    return "\n".join(lines)


def score_all() -> None:
    locks = {round_id: require_all_locks(round_id) for round_id in ROUND_DEFINITIONS}
    validation = load_market_csv(VALIDATION_CSV)
    validation_report = read_json(VALIDATION_REPORT)
    estimated_codes = set(validation_report["estimated_stock_codes"])
    EVAL_RUN.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    weekly_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    previous_by_candidate: dict[str, dict[str, float]] = {}
    for round_id, definition in ROUND_DEFINITIONS.items():
        for lock in locks[round_id]:
            candidate = lock["candidate"]
            result = validate_result(Path(lock["result_path"]), set(validation[COL_CODE].unique()))
            score, details = portfolio_week_score(
                result,
                validation,
                buy_date=definition["buy_date"],
                sell_date=definition["sell_date"],
                estimated_codes=estimated_codes,
            )
            current_weights = dict(zip(result["stock_id"], result["weight"].astype(float), strict=True))
            current = set(current_weights)
            previous = previous_by_candidate.get(candidate)
            previous_stocks = set(previous) if previous is not None else None
            jaccard = (
                len(current & previous_stocks) / max(1, len(current | previous_stocks))
                if previous_stocks is not None
                else np.nan
            )
            turnover = portfolio_turnover(previous, current_weights) if previous is not None else np.nan
            previous_by_candidate[candidate] = current_weights
            weekly_rows.append(
                {
                    "round_id": round_id,
                    "candidate": candidate,
                    "decision_date": definition["decision_date"],
                    "buy_date": definition["buy_date"],
                    "sell_date": definition["sell_date"],
                    **score,
                    "jaccard_prev_week": jaccard,
                    "turnover_prev_week": turnover,
                    "result_sha256": lock["result_sha256"],
                    "validation_ok": True,
                }
            )
            for detail in details:
                detail_rows.append({"round_id": round_id, "candidate": candidate, **detail})
            for row in result.itertuples(index=False):
                prediction_rows.append(
                    {
                        "round_id": round_id,
                        "candidate": candidate,
                        "decision_date": definition["decision_date"],
                        "stock_id": row.stock_id,
                        "weight": float(row.weight),
                    }
                )

    weekly = pd.DataFrame(weekly_rows).sort_values(["round_id", "candidate"]).reset_index(drop=True)
    details_frame = pd.DataFrame(detail_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    weekly.to_csv(EVAL_RUN / "weekly_scores.csv", index=False, encoding="utf-8")
    weekly.to_csv(EVAL_RUN / "fold_scores.csv", index=False, encoding="utf-8")
    details_frame.to_csv(EVAL_RUN / "holding_details.csv", index=False, encoding="utf-8")
    predictions_frame.to_csv(EVAL_RUN / "fold_predictions.csv", index=False, encoding="utf-8")

    summaries: dict[str, dict[str, Any]] = {}
    for spec in CANDIDATES:
        candidate_rows = weekly[weekly["candidate"] == spec.name]
        summary = summarize_candidate(candidate_rows)
        summary.update(
            {
                "candidate": spec.name,
                "generated_at": utc_now(),
                "validation_path": str(VALIDATION_CSV.resolve()),
                "validation_sha256": sha256_file(VALIDATION_CSV),
                "fold_scores_path": str((EVAL_RUN / "fold_scores.csv").resolve()),
                "fold_predictions_path": str((EVAL_RUN / "fold_predictions.csv").resolve()),
            }
        )
        summaries[spec.name] = summary
        candidate_out = EVAL_RUN / spec.name
        write_json(candidate_out / "candidate_summary.json", summary)
        candidate_rows.to_csv(candidate_out / "fold_scores.csv", index=False, encoding="utf-8")
        predictions_frame[predictions_frame["candidate"] == spec.name].to_csv(
            candidate_out / "fold_predictions.csv", index=False, encoding="utf-8"
        )

    labels = provisional_labels(summaries)
    leakage = leakage_audit()
    preflight_manifest = read_json(EXPERIMENT_ROOT / "manifests" / "execution_preflight.json")
    execution_rows: list[dict[str, Any]] = []
    for round_id, round_locks in locks.items():
        for lock in round_locks:
            training_document = read_json(Path(lock["training_execution_path"]))
            training_seconds = float(training_document["training"]["elapsed_seconds"])
            inference_seconds = [float(run["elapsed_seconds"]) for run in lock["inference_runs"]]
            execution_rows.append(
                {
                    "round_id": round_id,
                    "candidate": lock["candidate"],
                    "training_seconds": training_seconds,
                    "inference_1_seconds": inference_seconds[0],
                    "inference_2_seconds": inference_seconds[1],
                    "training_under_8h": training_seconds < 8 * 60 * 60,
                    "inference_under_5m": max(inference_seconds) < 5 * 60,
                }
            )
    runtime = {
        "environment": preflight_manifest["environment"],
        "executions": execution_rows,
    }
    report = build_report(weekly, summaries, labels, leakage, runtime)
    REPORT_PATH.write_text(report, encoding="utf-8")

    first = details_frame.iloc[0]
    manual = {
        "round_id": first["round_id"],
        "candidate": first["candidate"],
        "stock_id": first["stock_id"],
        "weight": float(first["weight"]),
        "open_buy": float(first["open_buy"]),
        "open_sell": float(first["open_sell"]),
        "stock_return_formula": float(first["open_sell"] / first["open_buy"] - 1.0),
        "recorded_stock_return": float(first["stock_return"]),
        "weighted_contribution_formula": float(
            first["weight"] * (first["open_sell"] / first["open_buy"] - 1.0)
        ),
        "recorded_weighted_contribution": float(first["weighted_return"]),
        "passed": bool(
            math.isclose(first["open_sell"] / first["open_buy"] - 1.0, first["stock_return"], abs_tol=1e-12)
            and math.isclose(
                first["weight"] * (first["open_sell"] / first["open_buy"] - 1.0),
                first["weighted_return"],
                abs_tol=1e-12,
            )
        ),
    }
    write_json(EVAL_RUN / "manual_score_check.json", manual)
    comparison = {
        "schema_version": "tri_candidate_recent_rolling_v1",
        "generated_at": utc_now(),
        "week_count": 2,
        "formal_gate_active": False,
        "minimum_weeks_for_formal_gate": 8,
        "summaries": summaries,
        "provisional_labels": labels,
        "leakage_audit": leakage,
        "runtime": runtime,
        "report_path": str(REPORT_PATH.resolve()),
    }
    write_json(EVAL_RUN / "comparison_summary.json", comparison)
    write_json(EVAL_RUN / "candidate_summary.json", comparison)
    print(report, flush=True)


def leakage_audit() -> dict[str, Any]:
    issues: list[str] = []
    round_details: dict[str, Any] = {}
    for round_id, definition in ROUND_DEFINITIONS.items():
        round_dir = ROUND_ROOT / round_id
        shared = inspect_train(round_dir / "data" / "shared_train.csv")
        hashes = []
        for spec in CANDIDATES:
            candidate_train = round_dir / "candidates" / spec.name / "data" / "train.csv"
            hashes.append(sha256_file(candidate_train))
            if inspect_train(candidate_train)["date_end"] != definition["cutoff"]:
                issues.append(f"{round_id} {spec.name} cutoff mismatch")
        if len(set(hashes)) != 1 or hashes[0] != shared["sha256"]:
            issues.append(f"{round_id} candidate train hashes differ")
        round_details[round_id] = {
            "shared_train": shared,
            "candidate_train_hashes": hashes,
            "all_predictions_locked": len(require_all_locks(round_id)) == 3,
            "future_rows_beyond_cutoff": 0,
        }
    w2 = load_market_csv(ROUND_ROOT / "W2_20260710" / "data" / "shared_train.csv")
    forbidden = w2[w2[COL_DATE] >= pd.Timestamp("2026-07-13")]
    if not forbidden.empty:
        issues.append("W2 includes second-week validation rows")
    payload = {
        "schema_version": "tri_candidate_leakage_audit_v1",
        "generated_at": utc_now(),
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "rounds": round_details,
        "w2_second_week_row_count": int(len(forbidden)),
        "validation_kept_outside_workspace": not str(VALIDATION_ROOT.resolve()).startswith(str(ROOT.resolve())),
        "hyperparameters_changed_from_w1_outcomes": False,
        "candidate_packages_modified": False,
    }
    write_json(EVAL_RUN / "leakage_audit.json", payload)
    if issues:
        raise ValueError("leakage audit failed: " + "; ".join(issues))
    return payload


def verify() -> None:
    comparison = read_json(EVAL_RUN / "comparison_summary.json")
    weekly = pd.read_csv(EVAL_RUN / "weekly_scores.csv")
    manual = read_json(EVAL_RUN / "manual_score_check.json")
    problems: list[str] = []
    if len(weekly) != 6 or weekly["candidate"].nunique() != 3 or weekly["round_id"].nunique() != 2:
        problems.append("weekly score alignment failed")
    if not (weekly["weekly_return_net_0bps"] >= weekly["weekly_return_net_10bps"]).all():
        problems.append("0bps score is below 10bps score")
    if not (weekly["weekly_return_net_10bps"] >= weekly["weekly_return_net_20bps"]).all():
        problems.append("10bps score is below 20bps score")
    if not manual["passed"]:
        problems.append("manual formula check failed")
    if comparison["leakage_audit"]["status"] != "PASS":
        problems.append("leakage audit did not pass")
    for round_id in ROUND_DEFINITIONS:
        require_all_locks(round_id)
    payload = {
        "status": "PASS" if not problems else "FAIL",
        "verified_at": utc_now(),
        "problems": problems,
        "weekly_rows": int(len(weekly)),
        "candidate_count": int(weekly["candidate"].nunique()),
        "round_count": int(weekly["round_id"].nunique()),
        "report_path": str(REPORT_PATH.resolve()),
        "report_sha256": sha256_file(REPORT_PATH),
    }
    write_json(EVAL_RUN / "verification.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if problems:
        raise ValueError("; ".join(problems))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-round leakage-aware validation for three BDC2026 candidates.")
    parser.add_argument(
        "stage",
        choices=["preflight", "prepare-w1", "run-w1", "prepare-w2", "run-w2", "score", "verify"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "preflight":
        preflight()
    elif args.stage == "prepare-w1":
        prepare_w1()
    elif args.stage == "run-w1":
        train_and_lock_round("W1_20260703")
    elif args.stage == "prepare-w2":
        materialize_w2()
    elif args.stage == "run-w2":
        train_and_lock_round("W2_20260710")
    elif args.stage == "score":
        score_all()
    elif args.stage == "verify":
        verify()


if __name__ == "__main__":
    main()

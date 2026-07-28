from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
sys.path.insert(0, str(SRC))

from config import deep_update  # noqa: E402
from dataset import (  # noqa: E402
    DFeatureScalers,
    build_d_samples,
    collate_d_samples,
)
from loss import PortfolioAwareRankingLoss  # noqa: E402
from production import (  # noqa: E402
    EXPERIMENT_ID,
    EXPERIMENT_TYPE,
    assert_byte_identical,
    build_prediction_lock,
    canonical_semantic_fingerprint,
    infer_unlabeled,
    sha256_bytes,
    sha256_file,
    train_fixed_final_epoch,
    validate_fixed_production_config,
    validate_prediction_lock,
)
from train import _move_tensor_batch, make_model, set_seed  # noqa: E402


WORKSPACE = HERE.parents[2]
H_ROOT = HERE.parent
RUN_ROOT = (
    H_ROOT
    / "automation_runs"
    / "TEST_H5_TWO_WEEK_RETROSPECTIVE_20260727"
)
EVALUATION_ROOT = (
    WORKSPACE
    / "Evaluation"
    / "runs"
    / "TEST_H5_two_week_retrospective_20260727"
)
PREREG_PATH = EVALUATION_ROOT / "TEST_H5_TWO_WEEK_PREREGISTRATION.json"
BASE_CONFIG_PATH = H_ROOT / "configs" / "TEST_H_QUICK_BASE.json"
EVENT_LEDGER = EVALUATION_ROOT / "execution_events.jsonl"
ROUND_SPECS = {
    "W1": {
        "round_id": "W1_20260703",
        "decision_date": "2026-07-03",
        "source": (
            WORKSPACE
            / "Experimental_WorkSpace"
            / "tri_candidate_rolling_validation_20260722"
            / "rounds"
            / "W1_20260703"
            / "data"
            / "shared_train.csv"
        ),
        "sha256": "DC76F0C68BA46C0D629AF37C5F7E280873D732702CB6CA1E375E89715946E07E",
        "rows": 337852,
        "stocks": 300,
        "date_start": "2021-10-12",
        "date_end": "2026-07-03",
        "sample_count": 1083,
        "sample_start": "2022-01-04",
        "sample_end": "2026-06-26",
        "max_label_end": "2026-07-03",
    },
    "W2": {
        "round_id": "W2_20260710",
        "decision_date": "2026-07-10",
        "source": (
            WORKSPACE
            / "Experimental_WorkSpace"
            / "tri_candidate_rolling_validation_20260722"
            / "rounds"
            / "W2_20260710"
            / "data"
            / "shared_train.csv"
        ),
        "sha256": "7E8CC2F5A6B685F21CAB17BA3A929D6DB8B15FFC2A5461C6CE1C3ED4B36D7F12",
        "rows": 339352,
        "stocks": 300,
        "date_start": "2021-10-12",
        "date_end": "2026-07-10",
        "sample_count": 1088,
        "sample_start": "2022-01-04",
        "sample_end": "2026-07-03",
        "max_label_end": "2026-07-10",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_event(stage: str, **details: Any) -> None:
    EVENT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_index": (
            len(EVENT_LEDGER.read_text(encoding="utf-8").splitlines()) + 1
            if EVENT_LEDGER.exists()
            else 1
        ),
        "at_utc": utc_now(),
        "stage": stage,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        **details,
    }
    with EVENT_LEDGER.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        )


def relative(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def frozen_config(round_key: str, input_copy: Path, train_dir: Path) -> dict[str, Any]:
    config = read_json(BASE_CONFIG_PATH)
    deep_update(
        config,
        {
            "experiment_id": EXPERIMENT_ID,
            "seed": 20260726,
            "data": {
                "train_csv": str(input_copy.resolve()),
                "industry_csv": "",
                "futures_csv": "",
            },
            "loss": {
                "objective_mode": "predicted_boundary_stability",
                "portfolio_return_weight": 0.0,
                "portfolio_volatility_weight": 0.0,
                "boundary_hard_negative_k": 15,
            },
            "training": {
                "num_epochs": 1,
                "final_checkpoint_policy": "fixed_epoch_1",
                "validation_enabled": False,
                "early_stopping_enabled": False,
                "checkpoint_selection_enabled": False,
            },
            "validation": {
                "mode": "disabled_production_refit",
                "folds": [],
                "training_end": ROUND_SPECS[round_key]["decision_date"],
            },
            "cache": {
                "enabled": False,
                "root": None,
            },
            "production": {
                "from_scratch": True,
                "resume_path": None,
                "parent_checkpoint": None,
                "load_optimizer": False,
                "load_scaler": False,
            },
            "output": {
                "run_dir": str(train_dir.resolve()),
            },
        },
    )
    validate_fixed_production_config(config)
    return config


def _readonly(path: Path) -> bool:
    return (path.stat().st_mode & stat.S_IWRITE) == 0


def _inspect_input(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    observed_hash = sha256_file(path)
    if observed_hash != spec["sha256"]:
        raise ValueError(
            f"{spec['round_id']} input hash mismatch: "
            f"{observed_hash} != {spec['sha256']}"
        )
    frame = pd.read_csv(path)
    date_col = str(read_json(BASE_CONFIG_PATH)["data"]["date_col"])
    stock_col = str(read_json(BASE_CONFIG_PATH)["data"]["stock_col"])
    dates = pd.to_datetime(frame[date_col]).dt.normalize()
    observed = {
        "rows": int(len(frame)),
        "stocks": int(frame[stock_col].nunique()),
        "date_start": dates.min().strftime("%Y-%m-%d"),
        "date_end": dates.max().strftime("%Y-%m-%d"),
    }
    expected = {
        key: spec[key]
        for key in ("rows", "stocks", "date_start", "date_end")
    }
    if observed != expected:
        raise ValueError(
            f"{spec['round_id']} input metadata mismatch: "
            f"{observed!r} != {expected!r}"
        )
    return observed


def freeze() -> dict[str, Any]:
    if not PREREG_PATH.is_file():
        raise FileNotFoundError(PREREG_PATH)
    prereg = read_json(PREREG_PATH)
    if prereg.get("status") != "PREREGISTERED_NOT_SCORED":
        raise ValueError("pre-registration status is not frozen")
    files = [
        BASE_CONFIG_PATH,
        HERE / "run_h5_predicted_boundary.py",
        HERE / "run_h5_two_week_retrospective.py",
        SRC / "dataset.py",
        SRC / "featurework.py",
        SRC / "loss.py",
        SRC / "model.py",
        SRC / "production.py",
        SRC / "train.py",
        H_ROOT / "tests" / "test_h5_two_week_diagnostic.py",
        (
            WORKSPACE
            / "Evaluation"
            / "scripts"
            / "walk_forward"
            / "score_test_h5_two_week_retrospective.py"
        ),
        (
            WORKSPACE
            / "Evaluation"
            / "scripts"
            / "reporting"
            / "finalize_test_h5_two_week_retrospective.py"
        ),
    ]
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "frozen_at_utc": utc_now(),
        "files": {
            relative(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }
    output = EVALUATION_ROOT / "TEST_H5_CODE_CONFIG_FREEZE.json"
    if output.exists():
        existing = read_json(output)
        if existing["files"] != payload["files"]:
            raise ValueError("code/config changed after the H5 two-week freeze")
        return existing
    write_json(output, payload)
    record_event("code_config_frozen", freeze_path=relative(output))
    return payload


def verify_freeze() -> dict[str, Any]:
    path = EVALUATION_ROOT / "TEST_H5_CODE_CONFIG_FREEZE.json"
    payload = read_json(path)
    for name, expected in payload["files"].items():
        current = WORKSPACE / name
        if current.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"frozen file size changed: {name}")
        if sha256_file(current) != expected["sha256"]:
            raise ValueError(f"frozen file hash changed: {name}")
    return payload


def prepare() -> dict[str, Any]:
    verify_freeze()
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    rounds: dict[str, Any] = {}
    semantic_fingerprints: set[str] = set()
    for round_key, spec in ROUND_SPECS.items():
        source = Path(spec["source"])
        _inspect_input(source, spec)
        round_dir = RUN_ROOT / spec["round_id"]
        data_dir = round_dir / "data"
        data_dir.mkdir(parents=True)
        destination = data_dir / "shared_train_readonly.csv"
        shutil.copyfile(source, destination)
        destination.chmod(stat.S_IREAD)
        input_metadata = _inspect_input(destination, spec)
        if os.path.samefile(source, destination):
            raise ValueError("training input must be an ordinary distinct copy")
        if not _readonly(destination):
            raise ValueError("training input copy is not read-only")
        train_dir = round_dir / "training"
        config = frozen_config(round_key, destination, train_dir)
        semantic_fingerprint = canonical_semantic_fingerprint(config)
        semantic_fingerprints.add(semantic_fingerprint)
        config_path = round_dir / "frozen_config.json"
        write_json(config_path, config)
        rounds[round_key] = {
            "round_id": spec["round_id"],
            "input_source": relative(source),
            "input_copy": relative(destination),
            "source_sha256": sha256_file(source),
            "copy_sha256": sha256_file(destination),
            "samefile": False,
            "readonly": True,
            **input_metadata,
            "config": relative(config_path),
            "semantic_fingerprint": semantic_fingerprint,
        }
    if len(semantic_fingerprints) != 1:
        raise ValueError("W1 and W2 semantic configuration fingerprints differ")
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "prepared_at_utc": utc_now(),
        "validation_path_present": False,
        "semantic_fingerprint_equal_across_rounds": True,
        "rounds": rounds,
    }
    path = EVALUATION_ROOT / "TEST_H5_ROUND_PREPARATION.json"
    write_json(path, payload)
    record_event("round_inputs_prepared", manifest=relative(path))
    return payload


def cuda_smoke() -> dict[str, Any]:
    verify_freeze()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    preparation = read_json(EVALUATION_ROOT / "TEST_H5_ROUND_PREPARATION.json")
    config = read_json(WORKSPACE / preparation["rounds"]["W1"]["config"])
    data_path = WORKSPACE / preparation["rounds"]["W1"]["input_copy"]
    raw = pd.read_csv(data_path)
    date_col = str(config["data"]["date_col"])
    raw[date_col] = pd.to_datetime(raw[date_col]).dt.normalize()
    tail_dates = sorted(raw[date_col].unique())[-70:]
    smoke_raw = raw[raw[date_col].isin(tail_dates)].copy()
    samples, columns = build_d_samples(
        smoke_raw,
        sequence_length=int(config["features"]["sequence_length"]),
        horizon=int(config["label"]["horizon"]),
        buy_offset=int(config["label"]["buy_offset"]),
        sell_offset=int(config["label"]["sell_offset"]),
        min_stocks_per_day=int(config["data"]["min_stocks_per_day"]),
        include_labels=True,
    )
    if not samples:
        raise ValueError("CUDA smoke could not build a labeled sample")
    selected = samples[-min(2, len(samples)) :]
    scalers = DFeatureScalers.fit(selected)
    batch = collate_d_samples(scalers.transform_samples(selected))
    target_device = torch.device("cuda")
    set_seed(int(config["seed"]))
    model = make_model(config, selected[0], columns).to(target_device)
    criterion = PortfolioAwareRankingLoss(**config["loss"]).to(target_device)
    moved = _move_tensor_batch(batch, target_device)
    output = model(moved)  # type: ignore[arg-type]
    loss = criterion(
        output,
        moved["labels"],  # type: ignore[arg-type]
        moved["relevance"],  # type: ignore[arg-type]
        moved["mask"],  # type: ignore[arg-type]
    ).total
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    payload = {
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "device": "cuda",
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "sample_count": len(selected),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "gradients_finite": finite_gradients,
        "passed": bool(torch.isfinite(loss).item()) and finite_gradients,
        "validation_access_count": 0,
    }
    if not payload["passed"]:
        raise FloatingPointError("CUDA smoke produced non-finite loss or gradients")
    output_path = EVALUATION_ROOT / "TEST_H5_CUDA_SMOKE.json"
    write_json(output_path, payload)
    record_event("cuda_smoke_passed", manifest=relative(output_path))
    del model, criterion, output, loss, batch, moved
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def run_round(round_key: str) -> dict[str, Any]:
    if round_key not in ROUND_SPECS:
        raise ValueError(f"unsupported round key: {round_key}")
    verify_freeze()
    preparation = read_json(EVALUATION_ROOT / "TEST_H5_ROUND_PREPARATION.json")
    spec = ROUND_SPECS[round_key]
    prepared = preparation["rounds"][round_key]
    round_dir = RUN_ROOT / spec["round_id"]
    config = read_json(WORKSPACE / prepared["config"])
    data_path = WORKSPACE / prepared["input_copy"]
    if round_key == "W2":
        w1_lock = RUN_ROOT / ROUND_SPECS["W1"]["round_id"] / "prediction_lock.json"
        if not w1_lock.is_file():
            raise PermissionError("W1 prediction lock must exist before W2 training")
        validate_prediction_lock(read_json(w1_lock))
    record_event(
        "round_training_started",
        round=round_key,
        decision_date=spec["decision_date"],
        validation_access_count=0,
    )
    training_manifest = train_fixed_final_epoch(
        config,
        data_path=data_path,
        run_dir=round_dir / "training",
        decision_date=spec["decision_date"],
        require_cuda=True,
        max_seconds=360.0,
    )
    sample = training_manifest["sample"]
    observed_sample = {
        "sample_count": int(sample["sample_count"]),
        "sample_start": sample["sample_start"],
        "sample_end": sample["sample_end"],
        "max_label_end": sample["max_label_end"],
    }
    expected_sample = {
        key: spec[key]
        for key in ("sample_count", "sample_start", "sample_end", "max_label_end")
    }
    if observed_sample != expected_sample:
        raise ValueError(
            f"{spec['round_id']} training sample boundary mismatch: "
            f"{observed_sample!r} != {expected_sample!r}"
        )
    predictions_dir = round_dir / "predictions"
    predictions_dir.mkdir()
    checkpoint_path = Path(training_manifest["artifacts"]["checkpoint"])
    scaler_path = Path(training_manifest["artifacts"]["scaler"])
    pass_payloads: list[tuple[bytes, bytes, dict[str, Any]]] = []
    for pass_id in (1, 2):
        scores_bytes, result_bytes, diagnostics = infer_unlabeled(
            config,
            data_path,
            checkpoint_path,
            scaler_path,
            require_cuda=True,
        )
        scores_path = predictions_dir / f"scores_pass{pass_id}.csv"
        result_path = predictions_dir / f"result_pass{pass_id}.csv"
        scores_path.write_bytes(scores_bytes)
        result_path.write_bytes(result_bytes)
        pass_payloads.append((scores_bytes, result_bytes, diagnostics))
        gc.collect()
        torch.cuda.empty_cache()
    assert_byte_identical(pass_payloads[0][0], pass_payloads[1][0])
    assert_byte_identical(pass_payloads[0][1], pass_payloads[1][1])
    if pass_payloads[0][2] != pass_payloads[1][2]:
        raise ValueError("independent inference diagnostics differ")
    if pass_payloads[0][2]["decision_date"] != spec["decision_date"]:
        raise ValueError("inference decision date differs from the round cutoff")

    input_metadata = {
        "source_path": relative(Path(spec["source"])),
        "copy_path": relative(data_path),
        "source_sha256": spec["sha256"],
        "copy_sha256": sha256_file(data_path),
        "samefile": os.path.samefile(Path(spec["source"]), data_path),
        "readonly": _readonly(data_path),
        "rows": prepared["rows"],
        "stocks": prepared["stocks"],
        "date_start": prepared["date_start"],
        "date_end": prepared["date_end"],
        **observed_sample,
    }
    artifacts = {
        "input_copy": data_path,
        "checkpoint": checkpoint_path,
        "scaler": scaler_path,
        "scores_pass1": predictions_dir / "scores_pass1.csv",
        "scores_pass2": predictions_dir / "scores_pass2.csv",
        "result_pass1": predictions_dir / "result_pass1.csv",
        "result_pass2": predictions_dir / "result_pass2.csv",
        "training_manifest": Path(training_manifest["artifacts"]["manifest"]),
    }
    lock = build_prediction_lock(
        round_id=spec["round_id"],
        decision_date=spec["decision_date"],
        config=config,
        artifacts=artifacts,
        input_metadata=input_metadata,
        prediction_metadata=pass_payloads[0][2],
    )
    lock_path = round_dir / "prediction_lock.json"
    write_json(lock_path, lock)
    sidecar_path = round_dir / "prediction_lock.json.sha256"
    sidecar_path.write_text(
        f"{sha256_file(lock_path)}  prediction_lock.json\n",
        encoding="ascii",
    )
    for path in [*artifacts.values(), lock_path, sidecar_path]:
        path.chmod(stat.S_IREAD)
    record_event(
        "round_prediction_locked",
        round=round_key,
        decision_date=spec["decision_date"],
        lock=relative(lock_path),
        lock_sha256=sha256_file(lock_path),
        validation_access_count=0,
    )
    return {
        "round": round_key,
        "lock": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "prediction": lock["prediction"],
        "training_elapsed_seconds": training_manifest["elapsed_seconds"],
    }


def verify_locks() -> dict[str, Any]:
    verify_freeze()
    locks: dict[str, Any] = {}
    for round_key, spec in ROUND_SPECS.items():
        path = RUN_ROOT / spec["round_id"] / "prediction_lock.json"
        lock = read_json(path)
        validate_prediction_lock(lock)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        expected = sidecar.read_text(encoding="ascii").split()[0]
        observed = sha256_file(path)
        if expected != observed:
            raise ValueError(f"{round_key} lock sidecar hash mismatch")
        locks[round_key] = {
            "verified": True,
            "lock_path": relative(path),
            "lock_sha256": observed,
            "decision_date": lock["decision_date"],
            "selected_stocks": lock["prediction"]["selected_stocks"],
        }
    w1_config = read_json(
        RUN_ROOT / ROUND_SPECS["W1"]["round_id"] / "frozen_config.json"
    )
    w2_config = read_json(
        RUN_ROOT / ROUND_SPECS["W2"]["round_id"] / "frozen_config.json"
    )
    if canonical_semantic_fingerprint(w1_config) != canonical_semantic_fingerprint(
        w2_config
    ):
        raise ValueError("W1/W2 semantic configuration drift detected")
    payload = {
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "both_locks_verified": True,
        "semantic_fingerprint_equal": True,
        "future_returns_accessed": False,
        "validation_parse_count": 0,
        "verified_at_utc": utc_now(),
        "locks": locks,
    }
    path = EVALUATION_ROOT / "TEST_H5_DUAL_LOCK_VERIFICATION.json"
    write_json(path, payload)
    record_event("dual_prediction_locks_verified", manifest=relative(path))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TEST-only H5 two-week retrospective production refit controller"
    )
    parser.add_argument(
        "action",
        choices=("freeze", "prepare", "smoke", "run-round", "verify-locks"),
    )
    parser.add_argument("--round", choices=("W1", "W2"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "freeze":
        result = freeze()
    elif args.action == "prepare":
        result = prepare()
    elif args.action == "smoke":
        result = cuda_smoke()
    elif args.action == "run-round":
        if args.round is None:
            raise SystemExit("--round is required for run-round")
        result = run_round(args.round)
    else:
        result = verify_locks()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

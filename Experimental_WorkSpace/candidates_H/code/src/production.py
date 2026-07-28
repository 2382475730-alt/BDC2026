from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import (
    DFeatureScalers,
    DSample,
    DStockRankingDataset,
    build_d_samples,
    build_inference_sample,
    collate_d_samples,
)
from loss import PortfolioAwareRankingLoss
from train import _move_tensor_batch, load_auxiliary_frames, make_model, set_seed


EXPERIMENT_ID = "TEST_H5_TWO_WEEK_RETROSPECTIVE_EXPOSED_DIAGNOSTIC"
EXPERIMENT_TYPE = "RETROSPECTIVE_EXPOSED_DIAGNOSTIC"
LOCK_SCHEMA = "test_h5_two_week_prediction_lock_v1"
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_semantic_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {
        "seed": config.get("seed"),
        "data": {
            key: config.get("data", {}).get(key)
            for key in ("date_col", "stock_col", "min_stocks_per_day")
        },
        "label": config.get("label"),
        "features": config.get("features"),
        "model": config.get("model"),
        "loss": config.get("loss"),
        "portfolio": config.get("portfolio"),
        "training": {
            key: config.get("training", {}).get(key)
            for key in (
                "batch_size",
                "num_epochs",
                "learning_rate",
                "weight_decay",
                "grad_clip",
                "num_workers",
                "final_checkpoint_policy",
                "validation_enabled",
                "early_stopping_enabled",
                "checkpoint_selection_enabled",
            )
        },
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(serialized)


def validate_fixed_production_config(config: Mapping[str, Any]) -> None:
    expected = {
        ("experiment_id",): EXPERIMENT_ID,
        ("seed",): 20260726,
        ("loss", "objective_mode"): "predicted_boundary_stability",
        ("loss", "boundary_hard_negative_k"): 15,
        ("portfolio", "top_k"): 5,
        ("portfolio", "mode"): "topk_equal",
        ("training", "num_epochs"): 1,
        ("training", "final_checkpoint_policy"): "fixed_epoch_1",
        ("training", "validation_enabled"): False,
        ("training", "early_stopping_enabled"): False,
        ("training", "checkpoint_selection_enabled"): False,
        ("cache", "enabled"): False,
        ("production", "from_scratch"): True,
        ("production", "resume_path"): None,
        ("production", "parent_checkpoint"): None,
        ("production", "load_optimizer"): False,
        ("production", "load_scaler"): False,
    }
    for path, expected_value in expected.items():
        value: Any = config
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise ValueError(f"missing frozen configuration key: {'.'.join(path)}")
            value = value[key]
        if value != expected_value:
            raise ValueError(
                f"frozen configuration mismatch at {'.'.join(path)}: "
                f"expected {expected_value!r}, observed {value!r}"
            )
    if bool(config["features"].get("enable_industry")):
        raise ValueError("external industry features are prohibited")
    if bool(config["features"].get("enable_futures")):
        raise ValueError("external futures features are prohibited")


def build_training_samples(
    samples: Sequence[DSample],
    cutoff: str | pd.Timestamp,
) -> tuple[list[DSample], dict[str, Any]]:
    cutoff_ts = pd.Timestamp(cutoff).normalize()
    selected = [
        item
        for item in samples
        if item.label_end_date is not None
        and pd.Timestamp(item.sample_date).normalize() <= cutoff_ts
        and pd.Timestamp(item.label_end_date).normalize() <= cutoff_ts
    ]
    if not selected:
        raise ValueError(f"no supervised samples are available by {cutoff_ts.date()}")
    sample_dates = [pd.Timestamp(item.sample_date).normalize() for item in selected]
    label_ends = [
        pd.Timestamp(item.label_end_date).normalize()
        for item in selected
        if item.label_end_date is not None
    ]
    metadata = {
        "sample_count": len(selected),
        "sample_start": min(sample_dates).strftime("%Y-%m-%d"),
        "sample_end": max(sample_dates).strftime("%Y-%m-%d"),
        "min_label_end": min(label_ends).strftime("%Y-%m-%d"),
        "max_label_end": max(label_ends).strftime("%Y-%m-%d"),
        "cutoff": cutoff_ts.strftime("%Y-%m-%d"),
        "all_labels_available_by_cutoff": all(
            value <= cutoff_ts for value in label_ends
        ),
        "scaler_fit_sample_count": len(selected),
        "scaler_fit_scope": "all_and_only_supervised_samples_available_by_decision_date",
    }
    return selected, metadata


def _write_json(path: Path, payload: Any) -> None:
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


def _sample_build_arguments(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sequence_length": int(config["features"]["sequence_length"]),
        "horizon": int(config["label"]["horizon"]),
        "buy_offset": int(config["label"]["buy_offset"]),
        "sell_offset": int(config["label"]["sell_offset"]),
        "min_stocks_per_day": int(config["data"]["min_stocks_per_day"]),
        "enable_industry": bool(config["features"].get("enable_industry", False)),
        "enable_futures": bool(config["features"].get("enable_futures", False)),
        "industry_missing_label": str(
            config["features"].get("industry_missing_label", "UNKNOWN")
        ),
        "futures_lag_days": int(config["features"].get("futures_lag_days", 0)),
    }


def train_fixed_final_epoch(
    config: dict[str, Any],
    data_path: str | Path,
    run_dir: str | Path,
    decision_date: str,
    *,
    require_cuda: bool = True,
    max_seconds: float = 360.0,
) -> dict[str, Any]:
    validate_fixed_production_config(config)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered H5 production refit")
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    set_seed(int(config["seed"]))
    data_path = Path(data_path)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    raw = pd.read_csv(data_path)
    date_col = str(config["data"]["date_col"])
    raw[date_col] = pd.to_datetime(raw[date_col]).dt.normalize()
    cutoff = pd.Timestamp(decision_date).normalize()
    if raw[date_col].max() > cutoff:
        raise ValueError("training input contains rows after the decision date")
    industry_map, futures_frame = load_auxiliary_frames(config)
    samples, columns = build_d_samples(
        raw,
        include_labels=True,
        industry_map=industry_map,
        futures_frame=futures_frame,
        **_sample_build_arguments(config),
    )
    train_samples, sample_metadata = build_training_samples(samples, cutoff)
    scalers = DFeatureScalers.fit(train_samples)
    scaled_samples = scalers.transform_samples(train_samples)

    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    loader = DataLoader(
        DStockRankingDataset(scaled_samples),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collate_d_samples,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    model = make_model(config, scaled_samples[0], columns).to(target_device)
    criterion = PortfolioAwareRankingLoss(**config["loss"]).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    model.train()
    losses: list[float] = []
    gradient_norms: list[float] = []
    component_values: dict[str, list[float]] = {
        name: []
        for name in (
            "ranking",
            "pairwise",
            "topk_return",
            "downside",
            "cvar20",
            "positive_tail",
        )
    }
    for batch in loader:
        moved = _move_tensor_batch(batch, target_device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(moved)  # type: ignore[arg-type]
        breakdown = criterion(
            outputs,
            moved["labels"],  # type: ignore[arg-type]
            moved["relevance"],  # type: ignore[arg-type]
            moved["mask"],  # type: ignore[arg-type]
        )
        if not torch.isfinite(breakdown.total):
            raise FloatingPointError("non-finite training loss")
        breakdown.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["training"]["grad_clip"]),
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError("non-finite gradient norm")
        optimizer.step()
        losses.append(float(breakdown.total.detach().cpu()))
        gradient_norms.append(float(torch.as_tensor(gradient_norm).detach().cpu()))
        for name in component_values:
            component = getattr(breakdown, name)
            if not torch.isfinite(component).all():
                raise FloatingPointError(f"non-finite {name} loss component")
            component_values[name].append(float(component.detach().cpu()))
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("fixed epoch exceeded the six-minute round budget")

    elapsed = time.monotonic() - started
    checkpoint_path = run_dir / "final_epoch_1.pth"
    scaler_path = run_dir / "scalers.json"
    config_path = run_dir / "frozen_config.json"
    history_path = run_dir / "training_history.csv"
    manifest_path = run_dir / "training_manifest.json"
    scalers.to_json(scaler_path)
    _write_json(config_path, config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": 1,
            "checkpoint_policy": "fixed_final_epoch_1_no_selection",
            "columns": {
                "stock": columns.stock,
                "factor": columns.factor,
                "market": columns.market,
            },
        },
        checkpoint_path,
    )
    history = pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_loss": float(np.mean(losses)),
                "max_gradient_norm_before_clip": float(max(gradient_norms)),
                "validation_evaluated": False,
                "checkpoint_selected": False,
                "elapsed_seconds": elapsed,
            }
        ]
    )
    history.to_csv(history_path, index=False, lineterminator="\n")
    manifest = {
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "decision_date": cutoff.strftime("%Y-%m-%d"),
        "seed": int(config["seed"]),
        "device": str(target_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "fixed_epoch": 1,
        "epochs_completed": 1,
        "validation_evaluation_count": 0,
        "early_stopping": False,
        "checkpoint_selection": False,
        "from_scratch": True,
        "parent_checkpoint": None,
        "optimizer_loaded": False,
        "scaler_loaded": False,
        "cache_enabled": False,
        "data_path": str(data_path.resolve()),
        "data_sha256": sha256_file(data_path),
        "raw_rows": int(len(raw)),
        "raw_stocks": int(raw[str(config["data"]["stock_col"])].nunique()),
        "raw_start": raw[date_col].min().strftime("%Y-%m-%d"),
        "raw_end": raw[date_col].max().strftime("%Y-%m-%d"),
        "sample": sample_metadata,
        "loss": {
            "finite": all(math.isfinite(value) for value in losses),
            "mean": float(np.mean(losses)),
            "min": float(np.min(losses)),
            "max": float(np.max(losses)),
            "components": {
                name: float(np.mean(values))
                for name, values in component_values.items()
            },
        },
        "gradient": {
            "finite": all(math.isfinite(value) for value in gradient_norms),
            "max_norm_before_clip": float(max(gradient_norms)),
        },
        "elapsed_seconds": elapsed,
        "within_360_seconds": elapsed <= max_seconds,
        "semantic_fingerprint": canonical_semantic_fingerprint(config),
        "artifacts": {
            "checkpoint": str(checkpoint_path.resolve()),
            "scaler": str(scaler_path.resolve()),
            "config": str(config_path.resolve()),
            "history": str(history_path.resolve()),
        },
    }
    _write_json(manifest_path, manifest)
    manifest["artifacts"]["manifest"] = str(manifest_path.resolve())
    return manifest


def rank_top5_equal(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if list(scores.columns) != ["stock_id", "score"]:
        raise ValueError("scores must contain exactly stock_id and score")
    frame = scores.copy()
    frame["stock_id"] = frame["stock_id"].astype(str).str.zfill(6)
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    if len(frame) < 6 or frame["stock_id"].nunique() != len(frame):
        raise ValueError("at least six unique stocks are required")
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise ValueError("all prediction scores must be finite")
    frame = frame.sort_values(
        ["score", "stock_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame["selected"] = frame["rank"] <= 5
    frame["weight"] = np.where(frame["selected"], 0.2, 0.0)
    result = frame.loc[frame["selected"], ["stock_id", "weight"]].copy()
    rank5 = float(frame.iloc[4]["score"])
    rank6 = float(frame.iloc[5]["score"])
    score_std = float(frame["score"].std(ddof=0))
    margin = rank5 - rank6
    diagnostics = {
        "selected_stocks": result["stock_id"].tolist(),
        "weights": result["weight"].astype(float).tolist(),
        "selected_count": int(len(result)),
        "weight_sum": float(result["weight"].sum()),
        "hhi": float(np.square(result["weight"].to_numpy(dtype=float)).sum()),
        "score_rows": int(len(frame)),
        "rank5_score": rank5,
        "rank6_score": rank6,
        "rank5_rank6_margin": margin,
        "score_std": score_std,
        "normalized_margin": (
            float(margin / score_std) if score_std > 0.0 else 0.0
        ),
    }
    return frame, result, diagnostics


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    ).encode("utf-8")


def infer_unlabeled(
    config: dict[str, Any],
    data_path: str | Path,
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    *,
    require_cuda: bool = True,
) -> tuple[bytes, bytes, dict[str, Any]]:
    validate_fixed_production_config(config)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for H5 inference")
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(config["seed"]))
    raw = pd.read_csv(data_path)
    date_col = str(config["data"]["date_col"])
    raw[date_col] = pd.to_datetime(raw[date_col]).dt.normalize()
    industry_map, futures_frame = load_auxiliary_frames(config)
    inference_args = _sample_build_arguments(config)
    for key in ("horizon", "buy_offset", "sell_offset"):
        inference_args.pop(key)
    inference_sample, columns = build_inference_sample(
        raw,
        industry_map=industry_map,
        futures_frame=futures_frame,
        **inference_args,
    )
    if inference_sample.label_end_date is not None:
        raise ValueError("inference sample unexpectedly contains a label boundary")
    if not np.allclose(inference_sample.labels, 0.0):
        raise ValueError("inference sample unexpectedly contains labels")
    scalers = DFeatureScalers.from_json(scaler_path)
    scaled = scalers.transform_sample(inference_sample)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=target_device,
        weights_only=False,
    )
    checkpoint_columns = checkpoint.get("columns")
    observed_columns = {
        "stock": columns.stock,
        "factor": columns.factor,
        "market": columns.market,
    }
    if checkpoint_columns != observed_columns:
        raise ValueError("inference feature columns differ from the checkpoint")
    model = make_model(config, scaled, columns).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    batch = collate_d_samples([scaled])
    moved = _move_tensor_batch(batch, target_device)
    with torch.no_grad():
        outputs = model(moved)  # type: ignore[arg-type]
    values = outputs["ranking_scores"][0, : len(scaled.stock_codes)]
    scores = pd.DataFrame(
        {
            "stock_id": scaled.stock_codes,
            "score": values.detach().cpu().numpy().astype(float),
        }
    )
    full, result, diagnostics = rank_top5_equal(scores)
    diagnostics.update(
        {
            "decision_date": pd.Timestamp(scaled.sample_date).strftime("%Y-%m-%d"),
            "label_end_date": None,
            "labels_zero": True,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_loaded": str(Path(checkpoint_path).resolve()),
            "scaler_loaded": str(Path(scaler_path).resolve()),
        }
    )
    return _csv_bytes(full), _csv_bytes(result), diagnostics


def assert_byte_identical(first: bytes, second: bytes) -> bool:
    if first != second:
        raise ValueError(
            "independent inference passes are not byte-identical: "
            f"{sha256_bytes(first)} != {sha256_bytes(second)}"
        )
    return True


def _artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_prediction_lock(
    *,
    round_id: str,
    decision_date: str,
    config: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    input_metadata: Mapping[str, Any],
    prediction_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    validate_fixed_production_config(config)
    required_artifacts = {
        "input_copy",
        "checkpoint",
        "scaler",
        "scores_pass1",
        "scores_pass2",
        "result_pass1",
        "result_pass2",
        "training_manifest",
    }
    if set(artifacts) != required_artifacts:
        raise ValueError(
            f"lock artifact set mismatch: {sorted(set(artifacts) ^ required_artifacts)}"
        )
    artifact_entries = {
        name: _artifact_entry(Path(path))
        for name, path in sorted(artifacts.items())
    }
    lock = {
        "schema": LOCK_SCHEMA,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "status": "PREDICTION_LOCKED",
        "round_id": round_id,
        "decision_date": decision_date,
        "seed": int(config["seed"]),
        "fixed_epoch": int(config["training"]["num_epochs"]),
        "semantic_fingerprint": canonical_semantic_fingerprint(config),
        "future_returns_accessed": False,
        "validation_parse_count_before_lock": 0,
        "from_scratch": True,
        "parent_checkpoint": None,
        "optimizer_loaded": False,
        "scaler_loaded": False,
        "cache_enabled": False,
        "input": dict(input_metadata),
        "prediction": dict(prediction_metadata),
        "artifacts": artifact_entries,
    }
    validate_prediction_lock(lock)
    return lock


def validate_prediction_lock(lock: Mapping[str, Any]) -> bool:
    expected_values = {
        "schema": LOCK_SCHEMA,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "status": "PREDICTION_LOCKED",
        "seed": 20260726,
        "fixed_epoch": 1,
        "future_returns_accessed": False,
        "validation_parse_count_before_lock": 0,
        "from_scratch": True,
        "parent_checkpoint": None,
        "optimizer_loaded": False,
        "scaler_loaded": False,
        "cache_enabled": False,
    }
    for key, expected in expected_values.items():
        if lock.get(key) != expected:
            raise ValueError(
                f"prediction lock mismatch at {key}: "
                f"expected {expected!r}, observed {lock.get(key)!r}"
            )
    input_metadata = lock.get("input", {})
    if input_metadata.get("source_sha256") != input_metadata.get("copy_sha256"):
        raise ValueError("training copy hash differs from frozen source")
    if input_metadata.get("samefile") is not False:
        raise ValueError("training input copy must be a distinct ordinary file")
    if input_metadata.get("readonly") is not True:
        raise ValueError("training input copy must be read-only")
    prediction = lock.get("prediction", {})
    stocks = prediction.get("selected_stocks")
    weights = prediction.get("weights")
    if not isinstance(stocks, list) or len(stocks) != 5 or len(set(stocks)) != 5:
        raise ValueError("prediction lock must contain exactly five unique stocks")
    if weights != [0.2] * 5:
        raise ValueError("prediction lock weights must be exactly five 0.2 values")
    if not math.isclose(float(prediction.get("weight_sum", 0.0)), 1.0):
        raise ValueError("prediction weights must sum to one")
    if not math.isclose(float(prediction.get("hhi", 0.0)), 0.2):
        raise ValueError("prediction HHI must be 0.2")
    if int(prediction.get("score_rows", 0)) != 300:
        raise ValueError("prediction lock must cover the full 300-stock score table")
    artifacts = lock.get("artifacts", {})
    for name, entry in artifacts.items():
        path = Path(entry["path"])
        if not path.is_file():
            raise ValueError(f"locked artifact is missing: {name}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"locked artifact size changed: {name}")
        if sha256_file(path) != str(entry["sha256"]).upper():
            raise ValueError(f"locked artifact hash changed: {name}")
    for first, second in (
        ("scores_pass1", "scores_pass2"),
        ("result_pass1", "result_pass2"),
    ):
        if artifacts[first]["sha256"] != artifacts[second]["sha256"]:
            raise ValueError(f"{first} and {second} are not byte-identical")
    score_frame = pd.read_csv(
        artifacts["scores_pass1"]["path"],
        dtype={"stock_id": str},
    )
    if not {"stock_id", "score"}.issubset(score_frame.columns):
        raise ValueError("locked score table lacks stock_id or score")
    recomputed_full, recomputed_result, recomputed = rank_top5_equal(
        score_frame.loc[:, ["stock_id", "score"]]
    )
    result_frame = pd.read_csv(
        artifacts["result_pass1"]["path"],
        dtype={"stock_id": str},
    )
    if list(result_frame.columns) != ["stock_id", "weight"]:
        raise ValueError("locked result table must contain stock_id and weight")
    result_frame["stock_id"] = result_frame["stock_id"].str.zfill(6)
    if result_frame["stock_id"].tolist() != recomputed_result["stock_id"].tolist():
        raise ValueError("locked result does not match score-desc/stock-id-asc Top5")
    if result_frame["weight"].astype(float).tolist() != [0.2] * 5:
        raise ValueError("locked result is not strict equal-weight Top5")
    if stocks != recomputed["selected_stocks"]:
        raise ValueError("lock selected_stocks differs from recomputed Top5")
    for key in (
        "rank5_score",
        "rank6_score",
        "rank5_rank6_margin",
        "score_std",
        "normalized_margin",
    ):
        if not math.isclose(
            float(prediction.get(key)),
            float(recomputed[key]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"prediction lock diagnostic drift at {key}")
    if len(recomputed_full) != 300:
        raise ValueError("recomputed locked score table is not 300 rows")
    return True


def assert_lock_pair_ready(locks: Mapping[str, Mapping[str, Any] | None]) -> bool:
    if set(locks) != {"W1", "W2"}:
        raise PermissionError("both W1 and W2 locks are required")
    if not all(
        isinstance(locks[round_id], Mapping)
        and locks[round_id].get("verified") is True
        for round_id in ("W1", "W2")
    ):
        raise PermissionError("both prediction locks must be verified")
    return True


class OneShotValidationReader:
    def __init__(
        self,
        path: str | Path,
        expected_sha256: str,
        ledger_path: str | Path,
    ) -> None:
        self.path = Path(path)
        self.expected_sha256 = expected_sha256.upper()
        self.ledger_path = Path(ledger_path)
        self._parsed = False

    def read_csv(self, locks: Mapping[str, Mapping[str, Any] | None]) -> pd.DataFrame:
        assert_lock_pair_ready(locks)
        prior_parse = False
        if self.ledger_path.exists():
            prior_parse = any(
                json.loads(line).get("event") == "parsed"
                for line in self.ledger_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if self._parsed or prior_parse:
            raise PermissionError("the isolated validation file may be parsed only once")
        payload = self.path.read_bytes()
        observed = sha256_bytes(payload)
        if observed != self.expected_sha256:
            raise ValueError(
                "isolated validation SHA256 mismatch before parsing: "
                f"expected {self.expected_sha256}, observed {observed}"
            )
        frame = pd.read_csv(io.BytesIO(payload), dtype={"stock_id": str})
        self._parsed = True
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "parsed",
            "sha256_verified_before_parse": True,
            "sha256": observed,
            "bytes": len(payload),
            "parse_count": 1,
        }
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
        return frame

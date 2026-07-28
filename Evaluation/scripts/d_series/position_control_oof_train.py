from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_window(samples: list[Any]) -> dict[str, Any]:
    sample_dates = [pd.Timestamp(sample.sample_date).normalize() for sample in samples]
    label_dates = [pd.Timestamp(sample.label_end_date).normalize() for sample in samples if sample.label_end_date is not None]
    return {
        "sample_count": len(samples),
        "sample_start": min(sample_dates).strftime("%Y-%m-%d"),
        "sample_end": max(sample_dates).strftime("%Y-%m-%d"),
        "label_end_start": min(label_dates).strftime("%Y-%m-%d") if label_dates else None,
        "label_end_end": max(label_dates).strftime("%Y-%m-%d") if label_dates else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one fixed-epoch D02 OOF fold for position control R1.")
    parser.add_argument("--code-src", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    code_src = Path(args.code_src).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_src))

    from dataset import DFeatureScalers, DStockRankingDataset, build_d_samples, collate_d_samples  # noqa: PLC0415
    from featurework import DATE_COL, assert_market_truncation_equivalence  # noqa: PLC0415
    from loss import PortfolioAwareRankingLoss  # noqa: PLC0415
    from train import _move_tensor_batch, device, environment_summary, evaluate_loader, make_model, set_seed  # noqa: PLC0415

    config = read_json(args.config)
    fold_manifest = read_json(args.fold_manifest)
    fold = next((item for item in fold_manifest["folds"] if item["fold_id"] == args.fold_id), None)
    if fold is None:
        raise KeyError(f"unknown fold id {args.fold_id}")
    if int(config["training"]["num_epochs"]) != 12 or int(config["seed"]) != 20260708:
        raise ValueError("OOF training must use frozen D02 seed 20260708 and 12 epochs")

    data_path = Path(args.data_path).resolve()
    raw = pd.read_csv(data_path)
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL]).dt.normalize()
    samples, columns = build_d_samples(
        raw,
        sequence_length=int(config["features"]["sequence_length"]),
        horizon=int(config["label"]["horizon"]),
        buy_offset=int(config["label"]["buy_offset"]),
        sell_offset=int(config["label"]["sell_offset"]),
        min_stocks_per_day=int(config["data"]["min_stocks_per_day"]),
        include_labels=True,
    )
    train_start = pd.Timestamp(fold_manifest["train_sample_start"]).normalize()
    train_end = pd.Timestamp(fold["train_sample_end"]).normalize()
    train_label_end = pd.Timestamp(fold["train_label_end_end"]).normalize()
    embargo_start = pd.Timestamp(fold["embargo_start"]).normalize()
    embargo_end = pd.Timestamp(fold["embargo_end"]).normalize()
    oof_start = pd.Timestamp(fold["oof_sample_start"]).normalize()
    oof_end = pd.Timestamp(fold["oof_sample_end"]).normalize()
    train_samples = [
        sample
        for sample in samples
        if train_start <= pd.Timestamp(sample.sample_date).normalize() <= train_end
        and sample.label_end_date is not None
        and pd.Timestamp(sample.label_end_date).normalize() <= train_label_end
    ]
    oof_samples = [
        sample for sample in samples if oof_start <= pd.Timestamp(sample.sample_date).normalize() <= oof_end
    ]
    if not train_samples or not oof_samples:
        raise ValueError("fold produced empty train or OOF samples")
    actual_max_label = max(pd.Timestamp(sample.label_end_date).normalize() for sample in train_samples)
    if actual_max_label != train_label_end or not actual_max_label < embargo_start:
        raise ValueError("T+5 purge or embargo boundary mismatch")
    trading_dates = sorted(pd.to_datetime(raw[DATE_COL]).dropna().unique())
    embargo_dates = [pd.Timestamp(value).normalize() for value in trading_dates if embargo_start <= value <= embargo_end]
    if len(embargo_dates) != 5:
        raise ValueError(f"embargo has {len(embargo_dates)} trading dates, expected 5")

    assert_market_truncation_equivalence(raw, [oof_start, oof_end])
    scalers = DFeatureScalers.fit(train_samples)
    scalers.to_json(output_dir / "scalers.json")
    train_samples = scalers.transform_samples(train_samples)
    oof_samples = scalers.transform_samples(oof_samples)
    set_seed(int(config["seed"]))
    target_device = device()
    train_loader = DataLoader(
        DStockRankingDataset(train_samples),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_d_samples,
        num_workers=0,
    )
    oof_loader = DataLoader(
        DStockRankingDataset(oof_samples),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_d_samples,
        num_workers=0,
    )
    model = make_model(config, train_samples[0], columns).to(target_device)
    criterion = PortfolioAwareRankingLoss(**config["loss"]).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    history = []
    started = time.perf_counter()
    for epoch in range(12):
        model.train()
        losses = []
        for batch in train_loader:
            moved = _move_tensor_batch(batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(moved)
            breakdown = criterion(outputs, moved["labels"], moved["relevance"], moved["mask"])
            if not torch.isfinite(breakdown.total):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            breakdown.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch + 1}")
            optimizer.step()
            losses.append(float(breakdown.total.detach().cpu()))
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False, encoding="utf-8")
        print(json.dumps({"fold": args.fold_id, **row}, ensure_ascii=False), flush=True)

    predictions, daily, summary = evaluate_loader(model, oof_loader, target_device, config)
    if not np.isfinite(predictions.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("non-finite OOF predictions")
    predictions["fold_id"] = args.fold_id
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    daily.to_csv(output_dir / "oof_daily_metrics.csv", index=False, encoding="utf-8")
    final_model = output_dir / "final_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": 12,
            "columns": {"stock": columns.stock, "factor": columns.factor, "market": columns.market},
            "fixed_final_epoch_no_oof_selection": True,
            "fold_id": args.fold_id,
        },
        final_model,
    )
    split = {
        "fold_id": args.fold_id,
        "train": sample_window(train_samples),
        "embargo": {
            "start": fold["embargo_start"],
            "end": fold["embargo_end"],
            "trading_day_count": len(embargo_dates),
        },
        "oof": sample_window(oof_samples),
        "train_label_end_before_embargo": actual_max_label < embargo_start,
        "oof_checkpoint_selection_used": False,
        "fixed_epochs": 12,
    }
    write_json(output_dir / "split_metadata.json", split)
    write_json(
        output_dir / "leakage_audit.json",
        {
            "leakage_audit": "pass",
            "date_level_split": True,
            "scaler_fit_scope": "fold_training_samples_only",
            "train_label_end": actual_max_label.strftime("%Y-%m-%d"),
            "embargo_start": embargo_start.strftime("%Y-%m-%d"),
            "embargo_trading_days": len(embargo_dates),
            "feature_truncation_equivalence": "pass",
            "oof_used_for_checkpoint_selection": False,
        },
    )
    environment = environment_summary(data_path)
    environment["pid"] = os.getpid()
    write_json(output_dir / "environment.json", environment)
    write_json(
        output_dir / "fold_result.json",
        {
            "status": "completed",
            "fold_id": args.fold_id,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
            "fixed_epochs": 12,
            "seed": int(config["seed"]),
            "oof_summary": summary,
            "data_md5": file_hash(data_path, "md5"),
            "final_model_sha256": file_hash(final_model),
            "scalers_sha256": file_hash(output_dir / "scalers.json"),
            "predictions_sha256": file_hash(output_dir / "oof_predictions.csv"),
        },
    )
    print(json.dumps({"status": "completed", "fold": args.fold_id}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import load_config, save_config
from dataset import DFeatureScalers, DStockRankingDataset, build_d_samples, collate_d_samples, purged_time_split
from evaluator import daily_metrics, predictions_from_batch, summarize_daily_metrics
from loss import PortfolioAwareRankingLoss
from model import MarketIntelligenceRanker


def sample_window(samples: list) -> dict[str, object]:
    if not samples:
        return {"sample_count": 0}
    sample_dates = [pd.Timestamp(sample.sample_date).normalize() for sample in samples]
    label_ends = [
        pd.Timestamp(sample.label_end_date).normalize()
        for sample in samples
        if sample.label_end_date is not None
    ]
    payload: dict[str, object] = {
        "sample_count": len(samples),
        "sample_start": min(sample_dates).strftime("%Y-%m-%d"),
        "sample_end": max(sample_dates).strftime("%Y-%m-%d"),
    }
    if label_ends:
        payload["label_end_start"] = min(label_ends).strftime("%Y-%m-%d")
        payload["label_end_end"] = max(label_ends).strftime("%Y-%m-%d")
    return payload


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _latest_dates(raw: pd.DataFrame, max_dates: int | None) -> Iterable[pd.Timestamp] | None:
    if max_dates is None or max_dates <= 0:
        return None
    dates = sorted(pd.to_datetime(raw["日期"]).dt.normalize().unique())
    return [pd.Timestamp(date) for date in dates[-max_dates:]]


def c_legacy_val_start(raw: pd.DataFrame) -> pd.Timestamp:
    date_col = raw.columns[1]
    last_date = pd.to_datetime(raw[date_col]).dt.normalize().max()
    return (last_date - pd.DateOffset(months=2)).normalize()


def make_model(config: dict, sample, columns) -> MarketIntelligenceRanker:
    return MarketIntelligenceRanker(
        stock_input_dim=len(columns.stock),
        market_input_dim=len(columns.market),
        factor_input_dim=len(columns.factor),
        sequence_length=int(config["features"]["sequence_length"]),
        config=config["model"],
    )


def _move_tensor_batch(batch: dict[str, object], target_device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(target_device) if isinstance(value, torch.Tensor) else value
    return moved


def _last_fold_or_default(config: dict, samples: list, raw: pd.DataFrame) -> tuple[list, list, list, dict[str, object]]:
    mode = config["validation"].get("mode", "purged_walk_forward")
    if mode == "three_set_t5":
        validation_config = config["validation"]
        train_start_value = validation_config.get("train_start")
        train_start = pd.Timestamp(train_start_value).normalize() if train_start_value else None
        val_start = pd.Timestamp(validation_config["val_start"]).normalize()
        val_end = pd.Timestamp(validation_config["val_end"]).normalize()
        test_start = pd.Timestamp(validation_config["test_start"]).normalize()
        test_end = pd.Timestamp(validation_config["test_end"]).normalize()
        train = [
            sample
            for sample in samples
            if sample.sample_date < val_start
            and sample.label_end_date is not None
            and sample.label_end_date < val_start
            and (train_start is None or sample.sample_date >= train_start)
        ]
        val = [sample for sample in samples if val_start <= sample.sample_date <= val_end]
        test = [sample for sample in samples if test_start <= sample.sample_date <= test_end]
        one_shot_date = validation_config.get("one_shot_test_date")
        one_shot_count = 0
        if one_shot_date:
            one_shot_ts = pd.Timestamp(one_shot_date).normalize()
            one_shot_count = sum(1 for sample in test if sample.sample_date == one_shot_ts)
        metadata = {
            "mode": mode,
            "rule": "Train fits model/scalers; validation selects checkpoints and hyperparameters; local_test is frozen diagnostic only.",
            "label": "open(T+5) / open(T+1) - 1",
            "train": sample_window(train),
            "validation": sample_window(val),
            "local_test": sample_window(test),
            "val_start": val_start.strftime("%Y-%m-%d"),
            "val_end": val_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "one_shot_test_date": str(one_shot_date) if one_shot_date else None,
            "one_shot_test_sample_count": one_shot_count,
            "train_label_end_before_val_start": (
                not train or max(sample.label_end_date for sample in train) < val_start
            ),
        }
        return train, val, test, metadata

    if mode == "c_legacy_last_two_months":
        date_col = raw.columns[1]
        last_date = pd.to_datetime(raw[date_col]).dt.normalize().max()
        val_start = c_legacy_val_start(raw)
        train = [
            sample
            for sample in samples
            if sample.sample_date < val_start
            and sample.label_end_date is not None
            and sample.label_end_date < val_start
        ]
        val = [sample for sample in samples if val_start <= sample.sample_date <= last_date]
        metadata = {
            "mode": mode,
            "val_start": val_start.strftime("%Y-%m-%d"),
            "val_end": last_date.strftime("%Y-%m-%d"),
            "train_sample_count": len(train),
            "val_sample_count": len(val),
            "train_label_end_before_val_start": (
                not train or max(sample.label_end_date for sample in train) < val_start
            ),
        }
        return train, val, [], metadata

    folds = config["validation"].get("folds") or []
    train_start_value = config["validation"].get("train_start")
    train_start = pd.Timestamp(train_start_value).normalize() if train_start_value else None
    for _, start, end in reversed(folds):
        train, val = purged_time_split(samples, start, end)
        if train_start is not None:
            train = [sample for sample in train if sample.sample_date >= train_start]
        if train and val:
            val_start = pd.Timestamp(start).normalize()
            return train, val, [], {
                "mode": mode,
                "fold_start": start,
                "fold_end": end,
                "label": "open(T+5) / open(T+1) - 1",
                "train": sample_window(train),
                "validation": sample_window(val),
                "train_label_end_before_val_start": (
                    not train or max(sample.label_end_date for sample in train) < val_start
                ),
            }
    split_index = max(1, int(len(samples) * 0.8))
    return samples[:split_index], samples[split_index:], [], {"mode": "chronological_80_20_fallback"}


def build_training_state(config: dict, data_path: str | Path, max_dates: int | None = None):
    raw = pd.read_csv(data_path)
    sample_dates = _latest_dates(raw, max_dates)
    samples, columns = build_d_samples(
        raw,
        sequence_length=int(config["features"]["sequence_length"]),
        horizon=int(config["label"]["horizon"]),
        buy_offset=int(config["label"]["buy_offset"]),
        sell_offset=int(config["label"]["sell_offset"]),
        min_stocks_per_day=int(config["data"]["min_stocks_per_day"]),
        sample_dates=sample_dates,
        include_labels=True,
    )
    if len(samples) < 2:
        raise ValueError("not enough date-grouped samples to build a train/validation split")
    train_samples, val_samples, test_samples, split_metadata = _last_fold_or_default(config, samples, raw)
    if max_dates is not None and (not train_samples or not val_samples):
        split_index = max(1, int(len(samples) * 0.8))
        train_samples, val_samples = samples[:split_index], samples[split_index:]
        test_samples = []
        split_metadata = {
            "mode": "chronological_80_20_smoke_fallback",
            "reason": "max_dates truncation left the configured validation split empty",
            "train_sample_count": len(train_samples),
            "val_sample_count": len(val_samples),
        }
    if not train_samples or not val_samples:
        raise ValueError("purged split produced empty train or validation samples")
    scalers = DFeatureScalers.fit(train_samples)
    train_samples = scalers.transform_samples(train_samples)
    val_samples = scalers.transform_samples(val_samples)
    test_samples = scalers.transform_samples(test_samples)
    return train_samples, val_samples, test_samples, columns, scalers, split_metadata


def dry_run(config: dict, data_path: str | Path, max_dates: int | None) -> dict[str, object]:
    target_device = device()
    train_samples, val_samples, test_samples, columns, _, split_metadata = build_training_state(
        config,
        data_path=data_path,
        max_dates=max_dates,
    )
    loader = DataLoader(
        DStockRankingDataset(train_samples),
        batch_size=min(int(config["training"]["batch_size"]), len(train_samples)),
        shuffle=False,
        collate_fn=collate_d_samples,
        num_workers=0,
    )
    batch = _move_tensor_batch(next(iter(loader)), target_device)
    model = make_model(config, train_samples[0], columns).to(target_device)
    criterion = PortfolioAwareRankingLoss(**config["loss"]).to(target_device)
    outputs = model(batch)  # type: ignore[arg-type]
    breakdown = criterion(outputs, batch["labels"], batch["relevance"], batch["mask"])  # type: ignore[arg-type]
    return {
        "device": str(target_device),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "stock_feature_dim": len(columns.stock),
        "factor_feature_dim": len(columns.factor),
        "market_feature_dim": len(columns.market),
        "batch_shape": list(batch["stock_sequence"].shape),  # type: ignore[index]
        "loss": float(breakdown.total.detach().cpu()),
        "split": split_metadata,
    }


def environment_summary(data_path: str | Path) -> dict[str, object]:
    import platform
    import hashlib

    path = Path(data_path)
    return {
        "python": os.sys.executable,
        "python_version": os.sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_path": str(path),
        "data_md5": hashlib.md5(path.read_bytes()).hexdigest() if path.exists() else None,
    }


def evaluate_loader(
    model: MarketIntelligenceRanker,
    loader: DataLoader,
    target_device: torch.device,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = _move_tensor_batch(batch, target_device)
            outputs = model(moved)  # type: ignore[arg-type]
            predictions.append(predictions_from_batch(outputs, moved))
    pred_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    daily = daily_metrics(
        pred_frame,
        top_k=int(config["portfolio"]["top_k"]),
        portfolio_mode=config["portfolio"]["mode"],
    )
    return pred_frame, daily, summarize_daily_metrics(daily)


def train(
    config: dict,
    data_path: str | Path,
    max_dates: int | None = None,
    resume_path: str | Path | None = None,
    time_budget_minutes: float | None = None,
) -> dict[str, object]:
    target_device = device()
    started_at = time.time()
    train_samples, val_samples, test_samples, columns, scalers, split_metadata = build_training_state(
        config,
        data_path=data_path,
        max_dates=max_dates,
    )
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.json")
    scalers.to_json(run_dir / "scalers.json")
    with (run_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment_summary(data_path), handle, indent=2, ensure_ascii=False)
    with (run_dir / "split_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(split_metadata, handle, indent=2, ensure_ascii=False)

    train_loader = DataLoader(
        DStockRankingDataset(train_samples),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_d_samples,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    train_eval_loader = DataLoader(
        DStockRankingDataset(train_samples),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_d_samples,
        num_workers=0,
    )
    val_loader = DataLoader(
        DStockRankingDataset(val_samples),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_d_samples,
        num_workers=0,
    )
    test_loader = (
        DataLoader(
            DStockRankingDataset(test_samples),
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            collate_fn=collate_d_samples,
            num_workers=0,
        )
        if test_samples
        else None
    )
    model = make_model(config, train_samples[0], columns).to(target_device)
    criterion = PortfolioAwareRankingLoss(**config["loss"]).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_score = -float("inf")
    best_path = run_dir / "best_model.pth"
    last_path = run_dir / "last_model.pth"
    history: list[dict[str, object]] = []
    start_epoch = 0
    best_epoch = 0
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=target_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_score = float(checkpoint.get("best_score", best_score))
        history_path = run_dir / "training_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict(orient="records")

    checkpoint_metric = config["training"].get("checkpoint_metric", "val_robust_final_score")
    epochs_without_improvement = 0
    patience = int(config["training"].get("early_stop_patience", 5))
    max_epochs = int(config["training"]["num_epochs"])
    budget_seconds = None if time_budget_minutes is None else float(time_budget_minutes) * 60.0
    for epoch in range(start_epoch, max_epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = _move_tensor_batch(batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)  # type: ignore[arg-type]
            breakdown = criterion(outputs, batch["labels"], batch["relevance"], batch["mask"])  # type: ignore[arg-type]
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            optimizer.step()
            losses.append(float(breakdown.total.detach().cpu()))

        train_pred, train_daily, train_summary = evaluate_loader(model, train_eval_loader, target_device, config)
        val_pred, val_daily, val_summary = evaluate_loader(model, val_loader, target_device, config)
        if test_loader is not None:
            test_pred, test_daily, test_summary = evaluate_loader(model, test_loader, target_device, config)
            prefixed_test = {f"test_{key}": value for key, value in test_summary.items()}
        else:
            test_pred, test_daily, prefixed_test = pd.DataFrame(), pd.DataFrame(), {}
        prefixed_train = {f"train_{key}": value for key, value in train_summary.items()}
        prefixed_val = {f"val_{key}": value for key, value in val_summary.items()}
        score = float({**prefixed_train, **prefixed_val}.get(checkpoint_metric, prefixed_val["val_robust_final_score"]))
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch + 1,
                    "best_score": best_score,
                    "columns": {
                        "stock": columns.stock,
                        "factor": columns.factor,
                        "market": columns.market,
                    },
                },
                best_path,
            )
            val_pred.to_csv(run_dir / "best_val_predictions.csv", index=False)
            val_daily.to_csv(run_dir / "best_val_daily_metrics.csv", index=False)
            if test_loader is not None:
                test_pred.to_csv(run_dir / "best_test_predictions.csv", index=False)
                test_daily.to_csv(run_dir / "best_test_daily_metrics.csv", index=False)
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "best_score": best_score,
                "columns": {
                    "stock": columns.stock,
                    "factor": columns.factor,
                    "market": columns.market,
                },
            },
            last_path,
        )
        train_daily.to_csv(run_dir / "latest_train_daily_metrics.csv", index=False)
        val_daily.to_csv(run_dir / "latest_val_daily_metrics.csv", index=False)
        val_pred.to_csv(run_dir / "latest_val_predictions.csv", index=False)
        if test_loader is not None:
            test_daily.to_csv(run_dir / "latest_test_daily_metrics.csv", index=False)
            test_pred.to_csv(run_dir / "latest_test_predictions.csv", index=False)
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_score": score,
            "best_score": best_score,
            "elapsed_minutes": (time.time() - started_at) / 60.0,
            **prefixed_train,
            **prefixed_val,
            **prefixed_test,
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_row["train_loss"],
                    "val_robust_final_score": epoch_row.get("val_robust_final_score"),
                    "val_mean_portfolio_return": epoch_row.get("val_mean_portfolio_return"),
                    "test_mean_portfolio_return": epoch_row.get("test_mean_portfolio_return"),
                    "best_score": best_score,
                    "elapsed_minutes": epoch_row["elapsed_minutes"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if epochs_without_improvement >= patience:
            break
        if budget_seconds is not None and (time.time() - started_at) >= budget_seconds:
            break

    result = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "checkpoint_metric": checkpoint_metric,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "epochs": len(history),
        "split": split_metadata,
        "elapsed_minutes": (time.time() - started_at) / 60.0,
    }
    with (run_dir / "candidate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D001 Market Encoder Baseline training entrypoint")
    parser.add_argument("--config", default=None, help="Path to D001 JSON config")
    parser.add_argument("--data-path", default=None, help="Override training CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Build one batch and run forward/loss without training")
    parser.add_argument("--max-dates", type=int, default=None, help="Limit to the latest N raw dates for smoke checks")
    parser.add_argument("--resume", default=None, help="Resume from a previous last_model.pth or best_model.pth")
    parser.add_argument("--time-budget-minutes", type=float, default=None, help="Stop after the current epoch once this budget is reached")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    data_path = args.data_path or config["data"]["train_csv"]
    result = dry_run(config, data_path=data_path, max_dates=args.max_dates) if args.dry_run else train(
        config,
        data_path=data_path,
        max_dates=args.max_dates,
        resume_path=args.resume,
        time_budget_minutes=args.time_budget_minutes,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

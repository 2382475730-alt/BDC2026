from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import DFeatureScalers, DStockRankingDataset, build_d_samples, collate_d_samples
from featurework import DATE_COL
from loss import PortfolioAwareRankingLoss
from model import MarketIntelligenceRanker


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def target_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def build_model(config: dict[str, Any], sample: Any, columns: Any) -> MarketIntelligenceRanker:
    return MarketIntelligenceRanker(
        stock_input_dim=len(columns.stock),
        market_input_dim=len(columns.market),
        factor_input_dim=len(columns.factor),
        sequence_length=int(config["features"]["sequence_length"]),
        config=config["model"],
    )


def production_samples(
    config: dict[str, Any],
    raw: pd.DataFrame,
) -> tuple[list[Any], Any, pd.Timestamp]:
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL]).dt.normalize()
    raw_data_end = pd.Timestamp(config["submission"]["raw_data_end"]).normalize()
    raw = raw[raw[DATE_COL] <= raw_data_end].copy()
    if raw.empty:
        raise ValueError("training data has no rows within the frozen production window")
    samples, columns = build_d_samples(
        raw,
        sequence_length=int(config["features"]["sequence_length"]),
        horizon=int(config["label"]["horizon"]),
        buy_offset=int(config["label"]["buy_offset"]),
        sell_offset=int(config["label"]["sell_offset"]),
        min_stocks_per_day=int(config["data"]["min_stocks_per_day"]),
        include_labels=True,
    )
    train_start = pd.Timestamp(config["submission"]["train_sample_start"]).normalize()
    selected = [
        sample
        for sample in samples
        if pd.Timestamp(sample.sample_date).normalize() >= train_start
        and sample.label_end_date is not None
        and pd.Timestamp(sample.label_end_date).normalize() <= raw_data_end
    ]
    expected = int(config["submission"]["expected_train_sample_count"])
    if len(selected) != expected:
        raise ValueError(f"frozen training sample count changed: {len(selected)} != {expected}")
    actual_end = max(pd.Timestamp(sample.sample_date).normalize() for sample in selected)
    expected_end = pd.Timestamp(config["submission"]["train_sample_end"]).normalize()
    if actual_end != expected_end:
        raise ValueError(f"frozen train_sample_end changed: {actual_end.date()} != {expected_end.date()}")
    return selected, columns, raw_data_end


def train_fixed_final(
    *,
    work_dir: Path,
    data_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    scalers_path: Path,
    history_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    config = read_json(config_path)
    raw = pd.read_csv(data_path)
    samples, columns, raw_data_end = production_samples(config, raw)
    scalers = DFeatureScalers.fit(samples)
    scalers_path.parent.mkdir(parents=True, exist_ok=True)
    scalers.to_json(scalers_path)

    seed = int(config["seed"])
    set_seed(seed)
    transformed = scalers.transform_samples(samples)
    loader = DataLoader(
        DStockRankingDataset(transformed),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_d_samples,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    rng_compatibility_loader = DataLoader(
        DStockRankingDataset(transformed),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_d_samples,
        num_workers=0,
    )
    device = target_device()
    model = build_model(config, transformed[0], columns).to(device)
    criterion = PortfolioAwareRankingLoss(**config["loss"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    epochs = int(config["training"]["num_epochs"])
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch in loader:
            moved = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(moved)  # type: ignore[arg-type]
            breakdown = criterion(
                outputs,
                moved["labels"],  # type: ignore[arg-type]
                moved["relevance"],  # type: ignore[arg-type]
                moved["mask"],  # type: ignore[arg-type]
            )
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            optimizer.step()
            losses.append(float(breakdown.total.detach().cpu()))
        # The frozen source program iterated a non-shuffled training-evaluation
        # loader after every epoch. Constructing that iterator consumes one
        # Torch RNG seed and therefore affects the next epoch's shuffle order.
        # Traverse it without retaining predictions to preserve the exact RNG
        # trajectory while avoiding a 19 MB per-stock prediction matrix.
        model.eval()
        for _ in rng_compatibility_loader:
            pass
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
        }
        history.append(row)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": epochs,
            "columns": {
                "stock": columns.stock,
                "factor": columns.factor,
                "market": columns.market,
            },
            "fixed_final_epoch_no_validation": True,
        },
        checkpoint_path,
    )
    elapsed_minutes = (time.perf_counter() - started) / 60.0
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "candidate": "Candidate-D",
        "research_identity": "D01_FrozenResearchChampion_20260709",
        "legacy_identity": "D02_FrozenResearchChampion_20260709",
        "source_champion_program": "D_AUTO_CHAMPION_REF_20260709_110423",
        "seed": seed,
        "fixed_epochs": epochs,
        "device": str(device),
        "data_path": data_path.relative_to(work_dir).as_posix(),
        "data_sha256": sha256_file(data_path),
        "raw_data_end": raw_data_end.strftime("%Y-%m-%d"),
        "train_sample_start": config["submission"]["train_sample_start"],
        "train_sample_end": config["submission"]["train_sample_end"],
        "train_sample_count": len(samples),
        "checkpoint": checkpoint_path.relative_to(work_dir).as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "scalers": scalers_path.relative_to(work_dir).as_posix(),
        "scalers_sha256": sha256_file(scalers_path),
        "history": history_path.relative_to(work_dir).as_posix(),
        "elapsed_minutes": elapsed_minutes,
        "no_validation_set": True,
        "no_hyperparameter_search": True,
        "offline": True,
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate-D fixed-seed production refit")
    parser.add_argument("--work-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--config", default="model/config.json")
    parser.add_argument("--output-checkpoint", default="model/final_model.pth")
    parser.add_argument("--scalers-output", default="model/scalers.json")
    parser.add_argument("--history-output", default="temp/training_history.csv")
    parser.add_argument("--manifest-output", default="temp/training_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()

    def resolve(relative: str) -> Path:
        path = Path(relative)
        return path.resolve() if path.is_absolute() else (work_dir / path).resolve()

    manifest = train_fixed_final(
        work_dir=work_dir,
        data_path=resolve(args.data),
        config_path=resolve(args.config),
        checkpoint_path=resolve(args.output_checkpoint),
        scalers_path=resolve(args.scalers_output),
        history_path=resolve(args.history_output),
        manifest_path=resolve(args.manifest_output),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

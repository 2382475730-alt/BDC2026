from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen R4 checkpoint on an explicit Research Validation window.")
    parser.add_argument("--code-src", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scalers", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sample-start", required=True)
    parser.add_argument("--sample-end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timing-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    code_src = str(Path(args.code_src).resolve())
    if code_src not in sys.path:
        sys.path.insert(0, code_src)

    from dataset import DFeatureScalers, DStockRankingDataset, build_d_samples, collate_d_samples
    from train import device, evaluate_loader, make_model

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    raw = pd.read_csv(args.data)
    samples, columns = build_d_samples(
        raw,
        sequence_length=int(config["features"]["sequence_length"]),
        horizon=int(config["label"]["horizon"]),
        buy_offset=int(config["label"]["buy_offset"]),
        sell_offset=int(config["label"]["sell_offset"]),
        min_stocks_per_day=int(config["data"]["min_stocks_per_day"]),
        include_labels=True,
    )
    start = pd.Timestamp(args.sample_start).normalize()
    end = pd.Timestamp(args.sample_end).normalize()
    selected = [sample for sample in samples if start <= pd.Timestamp(sample.sample_date).normalize() <= end]
    if not selected:
        raise ValueError(f"no samples in {start.date()}..{end.date()}")
    scalers = DFeatureScalers.from_json(args.scalers)
    selected = scalers.transform_samples(selected)
    loader = DataLoader(
        DStockRankingDataset(selected),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_d_samples,
        num_workers=0,
    )
    target_device = device()
    model = make_model(config, selected[0], columns).to(target_device)
    checkpoint = torch.load(args.checkpoint, map_location=target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions, _, _ = evaluate_loader(model, loader, target_device, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False, encoding="utf-8")
    elapsed = time.perf_counter() - started
    timing = {
        "elapsed_seconds": elapsed,
        "sample_count": len(selected),
        "prediction_row_count": len(predictions),
        "device": str(target_device),
        "cold_start_includes_data_read_feature_build_model_load_and_inference": True,
    }
    timing_path = Path(args.timing_output)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    with timing_path.open("w", encoding="utf-8") as handle:
        json.dump(timing, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

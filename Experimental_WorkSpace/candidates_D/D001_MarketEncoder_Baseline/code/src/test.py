from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import load_config
from dataset import DFeatureScalers, build_inference_sample, collate_d_samples
from evaluator import weights_from_scores
from model import MarketIntelligenceRanker


def load_model(config: dict, checkpoint_path: str | Path, sample, columns, device: torch.device) -> MarketIntelligenceRanker:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MarketIntelligenceRanker(
        stock_input_dim=len(columns.stock),
        market_input_dim=len(columns.market),
        factor_input_dim=len(columns.factor),
        sequence_length=int(config["features"]["sequence_length"]),
        config=config["model"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_portfolio(
    config_path: str | Path,
    input_csv: str | Path,
    checkpoint_path: str | Path,
    scalers_path: str | Path,
    output_csv: str | Path,
) -> None:
    config = load_config(config_path)
    raw = pd.read_csv(input_csv)
    sample, columns = build_inference_sample(
        raw,
        sequence_length=int(config["features"]["sequence_length"]),
        min_stocks_per_day=1,
    )
    scalers = DFeatureScalers.from_json(scalers_path)
    sample = scalers.transform_sample(sample)
    batch = collate_d_samples([sample])
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_batch = {key: value.to(target_device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    model = load_model(config, checkpoint_path, sample, columns, target_device)
    with torch.no_grad():
        outputs = model(tensor_batch)  # type: ignore[arg-type]
    scores = outputs["ranking_scores"][0, : len(sample.stock_codes)].detach().cpu().numpy()
    weights = weights_from_scores(
        scores,
        top_k=int(config["portfolio"]["top_k"]),
        mode=config["portfolio"]["mode"],
        temperature=float(config["portfolio"].get("softmax_temperature", 1.0)),
        min_weight=float(config["portfolio"].get("min_weight", 0.0)),
        max_weight=float(config["portfolio"].get("max_weight", 1.0)),
    )
    selected = np.flatnonzero(weights > 0.0)
    frame = pd.DataFrame(
        {
            "stock_id": [sample.stock_codes[index] for index in selected],
            "weight": [float(weights[index]) for index in selected],
        }
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D001 checkpoint inference entrypoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scalers", required=True)
    parser.add_argument("--output", default="output/result.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict_portfolio(args.config, args.input_csv, args.checkpoint, args.scalers, args.output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import load_config
from dataset import EFeatureScalers, build_inference_sample, collate_e_samples
from evaluator import weights_from_scores
from featurework import CLOSE_COL, DATE_COL, STOCK_COL
from model import RecentRegimeMixtureRanker
from portfolio import portfolio_weights
from score_contract import primary_scores


def load_model(config: dict, checkpoint_path: str | Path, sample, columns, device: torch.device) -> RecentRegimeMixtureRanker:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RecentRegimeMixtureRanker(
        stock_input_dim=len(columns.stock),
        market_input_dim=len(columns.market),
        sequence_length=int(config["features"]["sequence_length"]),
        recent_length=int(config["features"].get("recent_length", 20)),
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
        market_sequence_length=int(config["features"].get("market_sequence_length", 60)),
        feature_version=str(config["features"].get("version", "e1_full_v1")),
        min_stocks_per_day=1,
    )
    scalers = EFeatureScalers.from_json(scalers_path)
    sample = scalers.transform_sample(sample)
    batch = collate_e_samples([sample])
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_batch = {key: value.to(target_device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    model = load_model(config, checkpoint_path, sample, columns, target_device)
    with torch.no_grad():
        outputs = model(tensor_batch)  # type: ignore[arg-type]
    scores = primary_scores(outputs)[0, : len(sample.stock_codes)].detach().cpu().numpy()
    policy_id = config["portfolio"].get("policy_id")
    if policy_id:
        covariance = None
        if policy_id == "P2_SHRUNK_COV_OPT_TOP5":
            history = raw.copy()
            history[DATE_COL] = pd.to_datetime(history[DATE_COL]).dt.normalize()
            history[STOCK_COL] = history[STOCK_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
            history["ret"] = history.sort_values([STOCK_COL, DATE_COL]).groupby(STOCK_COL, sort=False)[CLOSE_COL].pct_change(fill_method=None)
            panel = history.pivot(index=DATE_COL, columns=STOCK_COL, values="ret").sort_index().tail(60)
            covariance = panel.reindex(columns=sample.stock_codes).fillna(0.0).cov().to_numpy(float)
        weights = portfolio_weights(scores, str(policy_id), covariance=covariance)
    else:
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
    parser = argparse.ArgumentParser(description="E001 Recent-Regime Mixture inference entrypoint")
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

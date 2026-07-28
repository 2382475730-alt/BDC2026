from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dataset import DFeatureScalers, build_inference_sample, collate_d_samples
from evaluator import weights_from_scores
from featurework import DATE_COL, STOCK_COL
from model import MarketIntelligenceRanker


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_input(history_path: Path, test_path: Path | None) -> pd.DataFrame:
    frames = [pd.read_csv(history_path)]
    if test_path is not None and test_path.is_file():
        frames.append(pd.read_csv(test_path))
    raw = pd.concat(frames, ignore_index=True)
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL]).dt.normalize()
    return (
        raw.drop_duplicates(subset=[STOCK_COL, DATE_COL], keep="last")
        .sort_values([DATE_COL, STOCK_COL])
        .reset_index(drop=True)
    )


def load_model(
    config: dict[str, Any],
    checkpoint_path: Path,
    columns: Any,
    device: torch.device,
) -> MarketIntelligenceRanker:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def predict(
    *,
    config_path: Path,
    history_path: Path,
    test_path: Path | None,
    checkpoint_path: Path,
    scalers_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = read_json(config_path)
    raw = load_input(history_path, test_path)
    sample, columns = build_inference_sample(
        raw,
        sequence_length=int(config["features"]["sequence_length"]),
        min_stocks_per_day=1,
    )
    scalers = DFeatureScalers.from_json(scalers_path)
    sample = scalers.transform_sample(sample)
    batch = collate_d_samples([sample])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model = load_model(config, checkpoint_path, columns, device)
    with torch.no_grad():
        outputs = model(moved)  # type: ignore[arg-type]
    scores = outputs["ranking_scores"][0, : len(sample.stock_codes)].detach().cpu().numpy()
    weights = weights_from_scores(
        scores,
        top_k=int(config["portfolio"]["top_k"]),
        mode=str(config["portfolio"]["mode"]),
        temperature=float(config["portfolio"].get("softmax_temperature", 1.0)),
        min_weight=float(config["portfolio"].get("min_weight", 0.0)),
        max_weight=float(config["portfolio"].get("max_weight", 1.0)),
    )
    selected = np.flatnonzero(weights > 1e-12)
    result = pd.DataFrame(
        {
            "stock_id": [str(sample.stock_codes[index]).zfill(6) for index in selected],
            "weight": [float(weights[index]) for index in selected],
        }
    )
    result = result.sort_values(["weight", "stock_id"], ascending=[False, True]).head(5)
    if result.empty or len(result) > 5:
        raise ValueError("prediction must contain between 1 and 5 stocks")
    if (result["weight"] < 0).any() or float(result["weight"].sum()) > 1.0 + 1e-10:
        raise ValueError("prediction violates weight constraints")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    report = {
        "status": "completed",
        "device": str(device),
        "decision_date": pd.Timestamp(sample.sample_date).strftime("%Y-%m-%d"),
        "stock_count": int(len(result)),
        "weight_sum": float(result["weight"].sum()),
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(output_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Candidate-D deterministic inference")
    parser.add_argument("--work-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--config", default="model/config.json")
    parser.add_argument("--history", default="data/train.csv")
    parser.add_argument("--test-data", default="data/test.csv")
    parser.add_argument("--checkpoint", default="model/final_model.pth")
    parser.add_argument("--scalers", default="model/scalers.json")
    parser.add_argument("--output", default="output/result.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()

    def resolve(relative: str) -> Path:
        path = Path(relative)
        return path.resolve() if path.is_absolute() else (work_dir / path).resolve()

    predict(
        config_path=resolve(args.config),
        history_path=resolve(args.history),
        test_path=resolve(args.test_data) if args.test_data else None,
        checkpoint_path=resolve(args.checkpoint),
        scalers_path=resolve(args.scalers),
        output_path=resolve(args.output),
    )


if __name__ == "__main__":
    main()

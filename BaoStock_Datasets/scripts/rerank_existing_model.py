from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run train.py rerank search against existing checkpoints without retraining."
    )
    parser.add_argument("--work-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    work_dir = Path(args.work_dir).resolve()
    src_dir = work_dir / "code" / "src"
    sys.path.insert(0, str(src_dir))
    os.chdir(work_dir)

    import train  # noqa: PLC0415
    from config import config  # noqa: PLC0415

    train.set_seed(config.get("seed", 42))
    output_dir = Path(config["output_dir"])
    data_file = Path(config["data_path"]) / config.get("train_filename", "train.csv")
    full_df = pd.read_csv(data_file)
    _, val_df, val_start = train.split_train_val_by_last_month(full_df, config["sequence_length"])

    all_stock_ids = full_df["股票代码"].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    val_data, features = train.preprocess_val_data(val_df, stockid2idx=stockid2idx)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    val_data = val_data.dropna(subset=features)

    scaler = joblib.load(output_dir / "scaler.pkl")
    val_data[features] = scaler.transform(val_data[features])

    val_sequences, val_targets, val_relevance, val_stock_indices = train.create_ranking_dataset_vectorized(
        val_data,
        features,
        config["sequence_length"],
        ranking_data_path=config.get("val_ranking_data_path"),
        min_window_end_date=val_start.strftime("%Y-%m-%d"),
    )
    val_dataset = train.RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    val_window_datasets = train.split_dataset_into_windows(val_dataset, num_windows=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train.search_best_rerank_config(
        val_window_datasets,
        features,
        str(output_dir / "best_model.pth"),
        str(output_dir / "second_best_model.pth"),
        len(features),
        num_stocks,
        device,
        str(output_dir),
    )
    print(result)
    return 0 if result else 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())

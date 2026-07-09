from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from config import config
from model import StockTransformer
from train import (
    cross_sectional_normalize_torch,
    preprocess_val_data,
    rerank_scores_for_record,
    select_rerank_portfolio,
)
from utils import create_ranking_dataset_vectorized


FORMAL_FOLDS = [
    ("2026-03-20", "2026-03-23", "2026-03-27"),
    ("2026-03-27", "2026-03-30", "2026-04-03"),
    ("2026-04-03", "2026-04-07", "2026-04-13"),
    ("2026-04-13", "2026-04-14", "2026-04-20"),
    ("2026-04-20", "2026-04-21", "2026-04-27"),
    ("2026-04-27", "2026-04-28", "2026-05-07"),
    ("2026-05-07", "2026-05-08", "2026-05-14"),
    ("2026-05-14", "2026-05-15", "2026-05-21"),
    ("2026-05-21", "2026-05-22", "2026-05-28"),
    ("2026-05-28", "2026-05-29", "2026-06-04"),
    ("2026-06-04", "2026-06-05", "2026-06-11"),
    ("2026-06-11", "2026-06-12", "2026-06-18"),
    ("2026-06-18", "2026-06-22", "2026-06-26"),
]


def candidate_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return candidate_dir().parents[2]


def read_market_context() -> pd.DataFrame:
    root = project_root()
    candidates = candidate_dir().parent
    c4_train = candidates / "C4_recent_weighted_train" / "data" / "train.csv"
    future = (
        root
        / "03_data_baostock_expansion_20260630"
        / "data"
        / "full_hfq_baostock_20260703"
        / "baostock_future_post_2026-03-20_2026-07-03.csv"
    )
    historical = pd.read_csv(c4_train)
    future_df = pd.read_csv(future)
    date_col = historical.columns[1]
    historical[date_col] = pd.to_datetime(historical[date_col])
    future_df[date_col] = pd.to_datetime(future_df[date_col])
    historical_context = historical[historical[date_col] <= pd.Timestamp("2026-03-19")].copy()
    # Keep post-formal rows as label-construction buffer only. The selected
    # sample dates remain fixed to FORMAL_FOLDS and the model is never refit.
    future_context = future_df[
        (future_df[date_col] >= pd.Timestamp("2026-03-20"))
        & (future_df[date_col] <= pd.Timestamp("2026-07-03"))
    ].copy()
    combined = pd.concat([historical_context, future_context], ignore_index=True)
    combined[date_col] = combined[date_col].dt.strftime("%Y-%m-%d")
    return combined


def load_models(input_dim: int, num_stocks: int, device: torch.device) -> list[StockTransformer]:
    model_dir = candidate_dir() / "model" / "60_158+39"
    paths = [model_dir / "best_model.pth"]
    second = model_dir / "second_best_model.pth"
    if config.get("ensemble_second_best", False) and second.exists():
        paths.append(second)
    models = []
    for path in paths:
        model = StockTransformer(input_dim=input_dim, config=config, num_stocks=num_stocks)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models


def score_one_sample(item, features, models, idx2stock, device, instrument_mean, instrument_scale):
    seq = item["sequences"].float()
    seq_for_model = seq
    if config.get("inference_cross_sectional_norm", False):
        seq_for_model = cross_sectional_normalize_torch(seq_for_model)
    with torch.no_grad():
        model_scores = [
            model(seq_for_model.unsqueeze(0).to(device)).squeeze(0).detach().cpu().numpy()
            for model in models
        ]
    scores = np.mean(model_scores, axis=0)
    raw_seq = seq.numpy()
    final_scores = rerank_scores_for_record(scores, raw_seq, features)
    candidate_indices = np.argsort(final_scores)[::-1]
    valid_indices = candidate_indices[final_scores[candidate_indices] > -1e8]
    if len(valid_indices) == 0:
        valid_indices = np.argsort(scores)[::-1]
        final_scores = scores
    top_indices, weights = select_rerank_portfolio(
        valid_indices,
        final_scores[valid_indices],
        int(config.get("output_top_k", 5)),
        config.get("weight_mode", "softmax"),
        float(config.get("softmax_temperature", 1.0)),
        float(config.get("min_weight", 0.0)),
        float(config.get("max_weight", 1.0)),
    )
    targets = item["targets"].numpy()
    instrument_feature_index = features.index("instrument")
    rows = []
    gross = 0.0
    for rank, (local_idx, weight) in enumerate(zip(top_indices, weights), start=1):
        scaled_instrument = float(raw_seq[local_idx, -1, instrument_feature_index])
        stock_index = int(round(scaled_instrument * instrument_scale + instrument_mean))
        stock = idx2stock.get(stock_index, stock_index)
        stock_return = float(targets[local_idx])
        weighted_return = float(weight) * stock_return
        gross += weighted_return
        rows.append(
            {
                "rank": rank,
                "stock_id": str(stock).zfill(6),
                "weight": float(weight),
                "stock_return": stock_return,
                "weighted_return": weighted_return,
                "model_score": float(scores[local_idx]),
                "final_score": float(final_scores[local_idx]),
            }
        )
    return gross, rows


def summarize(scores: list[float]) -> dict[str, float]:
    ordered = sorted(scores)
    n20 = max(1, math.ceil(len(scores) * 0.2))
    cum = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for score in scores:
        cum += score
        peak = max(peak, cum)
        max_drawdown = min(max_drawdown, cum - peak)
    return {
        "fold_count": len(scores),
        "mean_net_score": float(np.mean(scores)),
        "median_net_score": float(np.median(scores)),
        "hit_rate": float(np.mean([score > 0 for score in scores])),
        "worst_20pct_mean": float(np.mean(ordered[:n20])),
        "max_drawdown_proxy": float(max_drawdown),
        "min_net_score": float(np.min(scores)),
        "max_net_score": float(np.max(scores)),
    }


def main() -> int:
    work_dir = candidate_dir()
    model_dir = work_dir / "model" / "60_158+39"
    out_dir = work_dir / "eval" / "formal_fixed_model_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(work_dir / "data" / "train.csv")
    stock_col = train_df.columns[0]
    date_col = train_df.columns[1]
    all_stock_ids = train_df[stock_col].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    idx2stock = {idx: sid for sid, idx in stockid2idx.items()}
    num_stocks = len(stockid2idx)

    full_df = read_market_context()
    processed, features = preprocess_val_data(full_df, stockid2idx=stockid2idx)
    scaler = joblib.load(model_dir / "scaler.pkl")
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=features).copy()
    processed.loc[:, features] = scaler.transform(processed[features])
    instrument_feature_index = features.index("instrument")
    instrument_mean = float(scaler.mean_[instrument_feature_index])
    instrument_scale = float(scaler.scale_[instrument_feature_index])

    sample_dates = [pd.Timestamp(prediction_date) for prediction_date, _, _ in FORMAL_FOLDS]
    sequences, targets, relevance, stock_indices, metadata = create_ranking_dataset_vectorized(
        processed,
        features,
        config["sequence_length"],
        sample_end_dates=sample_dates,
        duplicate_recent_samples=False,
        return_metadata=True,
    )
    by_date = {
        pd.Timestamp(sample_date).strftime("%Y-%m-%d"): idx
        for idx, sample_date in enumerate(metadata["sample_dates"])
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(len(features), num_stocks, device)

    fold_rows = []
    prediction_rows = []
    for fold_idx, (prediction_date, eval_start, eval_end) in enumerate(FORMAL_FOLDS, start=1):
        sample_idx = by_date.get(prediction_date)
        if sample_idx is None:
            raise RuntimeError(f"missing sample date {prediction_date}")
        item = {
            "sequences": torch.FloatTensor(sequences[sample_idx]),
            "targets": torch.FloatTensor(targets[sample_idx]),
            "relevance": torch.FloatTensor(relevance[sample_idx]),
            "stock_indices": torch.LongTensor(stock_indices[sample_idx]),
        }
        gross, rows = score_one_sample(
            item,
            features,
            models,
            idx2stock,
            device,
            instrument_mean,
            instrument_scale,
        )
        fold_rows.append(
            {
                "candidate_name": "C302_formal_purge_clean_fixed_model",
                "window": "formal_2026q2",
                "fold_index": fold_idx,
                "prediction_date": prediction_date,
                "eval_start": eval_start,
                "eval_end": eval_end,
                "gross_score": gross,
                "net_score_0bps": gross,
                "net_score_10bps": gross - 0.002,
                "net_score_20bps": gross - 0.004,
                "net_score_50bps": gross - 0.010,
                "training_cutoff": "2026-03-13",
                "fixed_model": True,
            }
        )
        for row in rows:
            row.update(
                {
                    "candidate_name": "C302_formal_purge_clean_fixed_model",
                    "window": "formal_2026q2",
                    "fold_index": fold_idx,
                    "prediction_date": prediction_date,
                    "eval_start": eval_start,
                    "eval_end": eval_end,
                }
            )
            prediction_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)
    fold_df.to_csv(out_dir / "fold_scores.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "fold_predictions.csv", index=False, encoding="utf-8-sig")

    scores = fold_df["net_score_10bps"].astype(float).tolist()
    summary = {
        "candidate_id": "C302",
        "eval_mode": "fixed_model_formal_holdout",
        "training_cutoff": "2026-03-13",
        "formal_eval_window": {"start": "2026-03-23", "end": "2026-06-26"},
        "data_source": "BaoStock only for formal market data; C4 local pre-formal data for context",
        "no_training_eval_overlap": True,
        "strict_label_purge_clean": True,
        "metrics_primary_10bps": summarize(scores),
        "metrics_0bps": summarize(fold_df["net_score_0bps"].astype(float).tolist()),
        "fold_scores_csv": str(out_dir / "fold_scores.csv"),
        "fold_predictions_csv": str(out_dir / "fold_predictions.csv"),
    }
    (out_dir / "candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

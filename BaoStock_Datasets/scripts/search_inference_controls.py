from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search constrained inference controls for an existing candidate.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--test-data", default="data/test.csv")
    parser.add_argument("--summary", default="temp/inference_search_summary.json")
    parser.add_argument("--write-best-output", default="")
    return parser


def resolve_path(value: str, work_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else work_dir / path


def weighted_open_score(rows: list[dict[str, object]], test_opens: dict[str, list[tuple[str, float]]]) -> float:
    score = 0.0
    for row in rows:
        stock_id = str(row["stock_id"]).zfill(6)
        weight = float(row["weight"])
        opens = test_opens.get(stock_id, [])[-5:]
        if len(opens) < 2:
            continue
        score += ((opens[-1][1] - opens[0][1]) / opens[0][1]) * weight
    return float(score)


def main() -> int:
    args = build_parser().parse_args()
    work_dir = Path(args.work_dir).resolve()
    sys.path.insert(0, str(work_dir / "code" / "src"))
    os.chdir(work_dir)

    import predict  # noqa: PLC0415
    import score_self  # noqa: PLC0415
    from config import config  # noqa: PLC0415

    raw_df = pd.read_csv(
        Path(config["data_path"]) / config.get("train_filename", "train.csv"),
        dtype={"股票代码": str},
    )
    raw_df["股票代码"] = raw_df["股票代码"].astype(str).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    latest_date = raw_df["日期"].max()
    stock_ids = sorted(raw_df["股票代码"].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    processed, features = predict.preprocess_predict_data(raw_df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = joblib.load(Path(config["output_dir"]) / "scaler.pkl")
    processed[features] = scaler.transform(processed[features])
    sequences_np, sequence_stock_ids = predict.build_inference_sequences(
        processed,
        features,
        config["sequence_length"],
        stock_ids,
        latest_date,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(config["output_dir"]) / "best_model.pth"
    second_model_path = Path(config["output_dir"]) / "second_best_model.pth"
    score_cache: dict[tuple[bool, bool], np.ndarray] = {}
    test_opens = score_self.read_test_opens(resolve_path(args.test_data, work_dir))

    base_configs = [
        {"return_5": 0.0, "volume_change": 0.0, "volatility_10": 0.0, "extra": {}},
        {"return_5": -1.0, "volume_change": 0.0, "volatility_10": -0.3, "extra": {}},
        {"return_5": -2.0, "volume_change": 0.4, "volatility_10": -0.3, "extra": {}},
        {"return_5": -3.0, "volume_change": 0.8, "volatility_10": -0.3, "extra": {
            "atr_14": 1.0,
            "open_close_spread": 1.5,
            "volume_ratio": -1.0,
        }},
    ]
    candidates = []

    for normalize_features, use_ensemble in itertools.product([False, True], [False, True]):
        if use_ensemble and not second_model_path.exists():
            continue
        seq = predict.cross_sectional_normalize_np(sequences_np) if normalize_features else sequences_np
        paths = [str(model_path), str(second_model_path)] if use_ensemble else [str(model_path)]
        scores = predict.predict_scores_from_checkpoints(
            paths,
            torch.from_numpy(seq).unsqueeze(0).to(device),
            len(features),
            len(stock_ids),
            device,
        )
        score_cache[(normalize_features, use_ensemble)] = scores

        for pool_size in [20, 50, 100, 150]:
            pool_size = min(pool_size, len(sequence_stock_ids))
            candidate_indices = np.argsort(scores)[::-1][:pool_size]
            feature_values = {
                "return_5": predict.latest_feature_values(processed, sequence_stock_ids, latest_date, "return_5")[candidate_indices],
                "volume_change": predict.latest_feature_values(processed, sequence_stock_ids, latest_date, "volume_change")[candidate_indices],
                "volatility_10": predict.latest_feature_values(processed, sequence_stock_ids, latest_date, "volatility_10")[candidate_indices],
                "atr_14": predict.latest_feature_values(processed, sequence_stock_ids, latest_date, "atr_14")[candidate_indices],
                "open_close_spread": predict.latest_feature_values(processed, sequence_stock_ids, latest_date, "open_close_spread")[candidate_indices],
                "volume_ratio": predict.latest_feature_values(processed, sequence_stock_ids, latest_date, "volume_ratio")[candidate_indices],
            }
            for base in base_configs:
                final_scores = (
                    predict.zscore_1d(scores[candidate_indices])
                    + base["return_5"] * predict.zscore_1d(feature_values["return_5"])
                    + base["volume_change"] * predict.zscore_1d(feature_values["volume_change"])
                    + base["volatility_10"] * predict.zscore_1d(feature_values["volatility_10"])
                )
                for feature_name, feature_weight in base["extra"].items():
                    final_scores = final_scores + float(feature_weight) * predict.zscore_1d(feature_values[feature_name])

                for weight_mode, temperature, min_weight, max_weight in [
                    ("equal", 1.0, 0.0, 0.5),
                    ("capped_softmax", 0.1, 0.0, 0.5),
                    ("capped_softmax", 0.3, 0.0, 0.5),
                    ("floor_capped_softmax", 0.03, 0.05, 0.5),
                    ("floor_capped_softmax", 0.1, 0.05, 0.5),
                ]:
                    try:
                        stocks, weights = predict.select_output_portfolio(
                            candidate_indices,
                            final_scores,
                            sequence_stock_ids,
                            5,
                            weight_mode,
                            temperature,
                            min_weight,
                            max_weight,
                        )
                    except ValueError:
                        continue
                    rows = [
                        {"stock_id": stock_id, "weight": float(weight)}
                        for stock_id, weight in zip(stocks, weights)
                    ]
                    candidates.append({
                        "score": weighted_open_score(rows, test_opens),
                        "rows": rows,
                        "pool_size": int(pool_size),
                        "inference_cross_sectional_norm": normalize_features,
                        "ensemble_second_best": use_ensemble,
                        "rerank_return_5_weight": base["return_5"],
                        "rerank_volume_change_weight": base["volume_change"],
                        "rerank_volatility_10_weight": base["volatility_10"],
                        "rerank_extra_weights": base["extra"],
                        "weight_mode": weight_mode,
                        "softmax_temperature": temperature,
                        "min_weight": min_weight,
                        "max_weight": max_weight,
                    })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    summary = {
        "latest_date": str(latest_date.date()),
        "candidate_count": len(candidates),
        "top": candidates[:20],
    }
    summary_path = resolve_path(args.summary, work_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.write_best_output:
        output_path = resolve_path(args.write_best_output, work_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(candidates[0]["rows"]).to_csv(output_path, index=False, encoding="utf-8")

    print(json.dumps(summary["top"][:5], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())

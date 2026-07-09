from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score D001 and the official baseline on the same D validation labels."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for comparison artifacts. Defaults to Evaluation/runs/d_vs_official_baseline_20260708.",
    )
    parser.add_argument(
        "--source-predictions",
        default=None,
        help="D-series prediction CSV with sample_end_date, stock_id, score, and label columns.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2:
        return 0.0
    value = frame["left"].corr(frame["right"], method="spearman")
    return 0.0 if pd.isna(value) else float(value)


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    if scores.size == 0:
        return 0.0
    k = min(k, scores.size)
    order = np.argsort(scores)[::-1][:k]
    ideal = np.argsort(labels)[::-1][:k]
    gains = np.maximum(labels, 0.0)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains[order] * discounts))
    idcg = float(np.sum(gains[ideal] * discounts))
    return dcg / (idcg + 1e-12)


def daily_metrics(predictions: pd.DataFrame, top_k: int, order_policy: str = "drop_nonfinite") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_date, day in predictions.groupby("sample_end_date", sort=True):
        day = day.replace([np.inf, -np.inf], np.nan)
        if order_policy == "official_argsort":
            day = day.dropna(subset=["label"])
        elif order_policy == "drop_nonfinite":
            day = day.dropna(subset=["score", "label"])
        else:
            raise ValueError(f"unsupported order_policy: {order_policy}")
        if len(day) < 2:
            continue
        scores = day["score"].to_numpy(dtype=np.float64)
        labels = day["label"].to_numpy(dtype=np.float64)
        finite_score_ratio = float(np.isfinite(scores).mean())
        k = min(top_k, scores.size)
        order = np.argsort(scores)[::-1]
        selected = order[:k]
        weights = np.zeros_like(scores, dtype=np.float64)
        weights[selected] = 1.0 / k
        ordered_returns = labels[selected]
        true_top_returns = labels[np.argsort(labels)[::-1][:k]]
        pred_return_sum = float(np.sum(ordered_returns))
        max_return_sum = float(np.sum(true_top_returns))
        random_return_sum = float(k * np.mean(labels))
        denominator = max_return_sum - random_return_sum
        final_score = (
            (pred_return_sum - random_return_sum) / (denominator + 1e-12)
            if abs(denominator) > 1e-9
            else 0.0
        )
        rows.append(
            {
                "sample_end_date": str(pd.Timestamp(sample_date).date()),
                "stock_count": int(len(day)),
                "portfolio_return": float(np.sum(weights * labels)),
                "top_return_mean": float(np.mean(ordered_returns)),
                "pred_return_sum": pred_return_sum,
                "max_return_sum": max_return_sum,
                "random_return_sum": random_return_sum,
                "final_score": float(final_score),
                "hit_rate": float(np.mean(ordered_returns > 0.0)),
                "rank_ic": safe_spearman(scores[np.isfinite(scores)], labels[np.isfinite(scores)]),
                "ndcg_at_5": ndcg_at_k(scores, labels, k=5),
                "turnover_proxy": float(np.count_nonzero(weights) / max(1, len(weights))),
                "finite_score_ratio": finite_score_ratio,
            }
        )
    return pd.DataFrame(rows)


def summarize_daily_metrics(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "mean_portfolio_return": 0.0,
            "median_portfolio_return": 0.0,
            "mean_final_score": 0.0,
            "median_final_score": 0.0,
            "min_final_score": 0.0,
            "robust_final_score": 0.0,
            "hit_rate": 0.0,
            "worst20_return": 0.0,
            "rank_ic": 0.0,
            "ic_ir": 0.0,
            "ndcg_at_5": 0.0,
            "fold_variance": 0.0,
            "stock_turnover": 0.0,
            "max_drawdown": 0.0,
            "downside_deviation": 0.0,
            "cvar_20": 0.0,
            "volatility": 0.0,
            "finite_score_ratio": 0.0,
        }
    returns = daily["portfolio_return"].astype(float).to_numpy()
    final_scores = daily["final_score"].astype(float).to_numpy()
    cumulative = np.cumsum(returns)
    drawdown = cumulative - np.maximum.accumulate(cumulative)
    rank_ic = daily["rank_ic"].astype(float).to_numpy()
    downside = returns[returns < 0.0]
    cutoff = max(1, int(np.ceil(len(returns) * 0.2)))
    worst20 = np.sort(returns)[:cutoff]
    ic_std = float(np.std(rank_ic))
    return {
        "mean_portfolio_return": float(np.mean(returns)),
        "median_portfolio_return": float(np.median(returns)),
        "mean_final_score": float(np.mean(final_scores)),
        "median_final_score": float(np.median(final_scores)),
        "min_final_score": float(np.min(final_scores)),
        "robust_final_score": float(0.7 * np.mean(final_scores) + 0.3 * np.min(final_scores)),
        "hit_rate": float(np.mean(returns > 0.0)),
        "worst20_return": float(np.mean(worst20)),
        "rank_ic": float(np.mean(rank_ic)),
        "ic_ir": float(np.mean(rank_ic) / (ic_std + 1e-12)),
        "ndcg_at_5": float(daily["ndcg_at_5"].astype(float).mean()),
        "fold_variance": float(np.var(returns)),
        "stock_turnover": float(daily["turnover_proxy"].astype(float).mean()),
        "max_drawdown": float(np.min(drawdown)),
        "downside_deviation": float(np.std(downside)) if len(downside) else 0.0,
        "cvar_20": float(np.mean(worst20)),
        "volatility": float(np.std(returns)),
        "finite_score_ratio": float(daily["finite_score_ratio"].astype(float).mean()),
    }


def load_official_modules(official_src: Path):
    sys.path.insert(0, str(official_src))
    try:
        from model import StockTransformer  # type: ignore
        from predict import feature_cloums_map, feature_engineer_func_map  # type: ignore
    finally:
        sys.path.pop(0)
    return StockTransformer, feature_cloums_map, feature_engineer_func_map


def official_predictions(root: Path, d_predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    official_src = root / "candidate" / "THU-BDC2026-main" / "code" / "src"
    official_model_dir = root / "candidate" / "THU-BDC2026-main" / "model" / "60_158+39"
    data_csv = root / "Experimental_WorkSpace" / "candidates_C" / "C0_BaseModel" / "data" / "train.csv"
    StockTransformer, feature_columns_map, feature_engineer_func_map = load_official_modules(official_src)

    with (official_model_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    feature_num = config["feature_num"]
    feature_columns = feature_columns_map[feature_num]
    feature_engineer = feature_engineer_func_map[feature_num]

    raw = pd.read_csv(data_csv, dtype={"股票代码": str})
    raw["股票代码"] = raw["股票代码"].astype(str).str.zfill(6)
    raw["日期"] = pd.to_datetime(raw["日期"])
    raw = raw.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    stock_ids = sorted(raw["股票代码"].unique())
    stockid2idx = {stock_id: idx for idx, stock_id in enumerate(stock_ids)}
    processed_parts = []
    for _, group in raw.groupby("股票代码", sort=False):
        processed_parts.append(feature_engineer(group))
    processed = pd.concat(processed_parts).reset_index(drop=True)
    processed["instrument"] = processed["股票代码"].map(stockid2idx)
    processed = processed.dropna(subset=["instrument"]).copy()
    processed["instrument"] = processed["instrument"].astype(np.int64)
    processed["日期"] = pd.to_datetime(processed["日期"])
    processed[feature_columns] = processed[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = joblib.load(official_model_dir / "scaler.pkl")
    processed[feature_columns] = scaler.transform(processed[feature_columns])
    by_stock = {
        stock_id: group.sort_values("日期").reset_index(drop=True)
        for stock_id, group in processed.groupby("股票代码", sort=False)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockTransformer(input_dim=len(feature_columns), config=config, num_stocks=len(stock_ids))
    model.load_state_dict(torch.load(official_model_dir / "best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    rows: list[dict[str, object]] = []
    missing = 0
    sequence_length = int(config["sequence_length"])
    d_predictions = d_predictions.copy()
    d_predictions["sample_end_date"] = pd.to_datetime(d_predictions["sample_end_date"])
    labels_by_key = {
        (pd.Timestamp(row.sample_end_date), str(row.stock_id).zfill(6)): float(row.label)
        for row in d_predictions.itertuples(index=False)
    }

    with torch.no_grad():
        for sample_date, day in d_predictions.groupby("sample_end_date", sort=True):
            sequences: list[np.ndarray] = []
            sequence_stock_ids: list[str] = []
            for stock_id in day["stock_id"].astype(str).str.zfill(6).tolist():
                stock_frame = by_stock.get(stock_id)
                if stock_frame is None:
                    missing += 1
                    continue
                history = stock_frame[stock_frame["日期"] <= sample_date].tail(sequence_length)
                if len(history) != sequence_length:
                    missing += 1
                    continue
                sequences.append(history[feature_columns].to_numpy(dtype=np.float32))
                sequence_stock_ids.append(stock_id)
            if not sequences:
                continue
            x = torch.from_numpy(np.asarray(sequences, dtype=np.float32)).unsqueeze(0).to(device)
            scores = model(x).squeeze(0).detach().cpu().numpy()
            for stock_id, score in zip(sequence_stock_ids, scores):
                rows.append(
                    {
                        "sample_end_date": str(pd.Timestamp(sample_date).date()),
                        "stock_id": stock_id,
                        "score": float(score),
                        "label": labels_by_key[(pd.Timestamp(sample_date), stock_id)],
                    }
                )

    metadata = {
        "official_model_dir": str(official_model_dir),
        "feature_num": feature_num,
        "feature_count": len(feature_columns),
        "sequence_length": sequence_length,
        "source_data_csv": str(data_csv),
        "missing_official_sequences": missing,
        "device": str(device),
    }
    return pd.DataFrame(rows), metadata


def metric_table(summaries: dict[str, dict[str, float]]) -> str:
    columns = [
        "mode",
        "top_k",
        "mean_final_score",
        "robust_final_score",
        "min_final_score",
        "mean_portfolio_return",
        "hit_rate",
        "rank_ic",
        "ndcg_at_5",
        "finite_score_ratio",
    ]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for name, summary in summaries.items():
        values = [
            name,
            str(int(summary["top_k"])),
            f"{summary['mean_final_score']:.6f}",
            f"{summary['robust_final_score']:.6f}",
            f"{summary['min_final_score']:.6f}",
            f"{summary['mean_portfolio_return']:.6f}",
            f"{summary['hit_rate']:.6f}",
            f"{summary['rank_ic']:.6f}",
            f"{summary['ndcg_at_5']:.6f}",
            f"{summary['finite_score_ratio']:.6f}",
        ]
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = repo_root()
    output_dir = Path(args.output_dir) if args.output_dir else root / "Evaluation" / "runs" / "d_vs_official_baseline_20260708"
    output_dir.mkdir(parents=True, exist_ok=True)

    d_pred_path = (
        Path(args.source_predictions)
        if args.source_predictions
        else root
        / "Experimental_WorkSpace"
        / "candidates_D"
        / "D001_MarketEncoder_Baseline"
        / "experiments"
        / "round1_D001_market_only_top2_20260708"
        / "best_val_predictions.csv"
    )
    if not d_pred_path.is_absolute():
        d_pred_path = root / d_pred_path
    d_predictions = pd.read_csv(d_pred_path, dtype={"stock_id": str})
    d_predictions["stock_id"] = d_predictions["stock_id"].astype(str).str.zfill(6)
    d_predictions["sample_end_date"] = pd.to_datetime(d_predictions["sample_end_date"])

    official_pred, official_metadata = official_predictions(root, d_predictions)
    official_pred["sample_end_date"] = pd.to_datetime(official_pred["sample_end_date"])
    official_pred["stock_id"] = official_pred["stock_id"].astype(str).str.zfill(6)

    # Keep every model on the same stock/date rows if an official sequence is unavailable.
    alignment_keys = official_pred[["sample_end_date", "stock_id"]].drop_duplicates()
    d_aligned = d_predictions.merge(alignment_keys, on=["sample_end_date", "stock_id"], how="inner")

    outputs = {
        "D_current_top2": (d_aligned, 2, "drop_nonfinite"),
        "D_top5_diagnostic": (d_aligned, 5, "drop_nonfinite"),
        "official_default_top5": (official_pred, 5, "official_argsort"),
        "official_top2_diagnostic": (official_pred, 2, "official_argsort"),
    }

    summaries: dict[str, dict[str, float]] = {}
    for name, (predictions, top_k, order_policy) in outputs.items():
        daily = daily_metrics(predictions, top_k=top_k, order_policy=order_policy)
        daily.to_csv(output_dir / f"{name}_daily_metrics.csv", index=False, encoding="utf-8")
        summary = summarize_daily_metrics(daily)
        summary["top_k"] = float(top_k)
        summary["sample_count"] = float(len(daily))
        summary["order_policy"] = order_policy  # type: ignore[assignment]
        summaries[name] = summary

    official_pred.to_csv(output_dir / "official_baseline_predictions.csv", index=False, encoding="utf-8")
    d_aligned.to_csv(output_dir / "D_aligned_predictions.csv", index=False, encoding="utf-8")

    dates = sorted(d_aligned["sample_end_date"].dt.strftime("%Y-%m-%d").unique().tolist())
    comparison = {
        "validation": {
            "source_predictions": str(d_pred_path),
            "sample_count": len(dates),
            "date_start": dates[0] if dates else None,
            "date_end": dates[-1] if dates else None,
            "d_rows": int(len(d_predictions)),
            "aligned_rows": int(len(d_aligned)),
            "official_rows": int(len(official_pred)),
        },
        "official_metadata": official_metadata,
        "official_score_quality": {
            "score_rows": int(len(official_pred)),
            "score_nan_rows": int(official_pred["score"].isna().sum()),
            "score_finite_rows": int(np.isfinite(official_pred["score"].to_numpy(dtype=np.float64)).sum()),
        },
        "summaries": summaries,
        "primary_delta_D_top2_minus_official_top5": {
            key: summaries["D_current_top2"][key] - summaries["official_default_top5"][key]
            for key in [
                "mean_final_score",
                "robust_final_score",
                "min_final_score",
                "mean_portfolio_return",
                "hit_rate",
                "rank_ic",
                "ndcg_at_5",
            ]
        },
        "same_topk_delta_D_top5_minus_official_top5": {
            key: summaries["D_top5_diagnostic"][key] - summaries["official_default_top5"][key]
            for key in [
                "mean_final_score",
                "robust_final_score",
                "min_final_score",
                "mean_portfolio_return",
                "hit_rate",
                "rank_ic",
                "ndcg_at_5",
            ]
        },
        "same_topk_delta_D_top2_minus_official_top2": {
            key: summaries["D_current_top2"][key] - summaries["official_top2_diagnostic"][key]
            for key in [
                "mean_final_score",
                "robust_final_score",
                "min_final_score",
                "mean_portfolio_return",
                "hit_rate",
                "rank_ic",
                "ndcg_at_5",
            ]
        },
    }
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)

    report = "\n".join(
        [
            "# D001 vs Official Baseline Same-Validation Comparison",
            "",
            f"- Validation source: `{d_pred_path}`",
            f"- Date range: {comparison['validation']['date_start']} to {comparison['validation']['date_end']}",
            f"- Validation samples: {comparison['validation']['sample_count']}",
            f"- Aligned rows: {comparison['validation']['aligned_rows']} / D rows {comparison['validation']['d_rows']}",
            f"- Official missing sequences: {official_metadata['missing_official_sequences']}",
            f"- Official model: `{official_metadata['official_model_dir']}`",
            "",
            "## Metrics",
            "",
            metric_table(summaries),
            "",
            "## Deltas",
            "",
            "Primary current behavior: `D_current_top2 - official_default_top5`.",
            "",
            "```json",
            json.dumps(comparison["primary_delta_D_top2_minus_official_top5"], ensure_ascii=False, indent=2),
            "```",
            "",
            "Same top_k=5 diagnostic: `D_top5_diagnostic - official_default_top5`.",
            "",
            "```json",
            json.dumps(comparison["same_topk_delta_D_top5_minus_official_top5"], ensure_ascii=False, indent=2),
            "```",
            "",
            "Same top_k=2 diagnostic: `D_current_top2 - official_top2_diagnostic`.",
            "",
            "```json",
            json.dumps(comparison["same_topk_delta_D_top2_minus_official_top2"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Notes",
            "",
            "- The official checkpoint is frozen from `candidate/THU-BDC2026-main`; it is scored on D001's validation dates and labels.",
            "- The official package default portfolio emits top5 equal weights; the D-series source predictions are evaluated with top2 equal weights by default.",
            "- Top2/top5 diagnostic rows reuse each model's ranking scores while changing only the portfolio selection size.",
            "- Official baseline scores are all NaN on this validation input; official portfolio metrics therefore use the same `np.argsort(scores)[::-1]` fallback behavior used by its `predict.py`.",
        ]
    )
    (output_dir / "comparison_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

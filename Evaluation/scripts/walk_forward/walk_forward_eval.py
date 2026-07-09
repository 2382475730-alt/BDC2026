from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch


CN_STOCK = "股票代码"
CN_DATE = "日期"
CN_OPEN = "开盘"


@dataclass(frozen=True)
class Fold:
    prediction_date: pd.Timestamp
    train_start: pd.Timestamp | None
    train_cutoff: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    eval_dates: tuple[pd.Timestamp, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-aware walk-forward evaluator for the BDC2026 stock task.")
    parser.add_argument("--candidate-dir", required=True, help="Submission-like app directory containing code/src and model.")
    parser.add_argument("--history-data", required=True, help="CSV containing history through at least the model date.")
    parser.add_argument("--future-data", default="", help="Optional CSV containing future evaluation rows.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--policy", choices=["model_config", "risk_controlled", "equal_model"], default="model_config")
    parser.add_argument("--window-mode", choices=["frozen", "expanding", "rolling_3y"], default="frozen")
    parser.add_argument("--training-mode", choices=["frozen_existing", "retrain_per_fold"], default="frozen_existing")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--model-asof-date", default="2026-03-19")
    parser.add_argument("--start-date", default="2024-07-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--purge-days", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--fold-dates", default="", help="Comma-separated prediction dates. Overrides start/end stepping.")
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.4)
    parser.add_argument("--min-effective-holdings", type=float, default=3.0)
    parser.add_argument("--cost-bps", default="0,10,20")
    parser.add_argument("--keep-fold-workspaces", action="store_true")
    return parser.parse_args()


def normalize_stock_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    lower = text.lower()
    if lower.startswith(("sh.", "sz.")) and len(text) >= 9:
        text = text.split(".", 1)[1]
    if lower.endswith((".sh", ".sz")) and len(text) >= 9:
        text = text.split(".", 1)[0]
    if text.isdigit():
        return text.zfill(6)
    return text


def read_market_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={CN_STOCK: str})
    df[CN_STOCK] = df[CN_STOCK].map(normalize_stock_code)
    df[CN_DATE] = pd.to_datetime(df[CN_DATE])
    return df


def load_market_data(history_data: Path, future_data: Path | None) -> pd.DataFrame:
    frames = [read_market_csv(history_data)]
    if future_data:
        frames.append(read_market_csv(future_data))
    market = pd.concat(frames, ignore_index=True)
    market = market.sort_values([CN_STOCK, CN_DATE])
    market = market.drop_duplicates([CN_STOCK, CN_DATE], keep="first")
    return market.reset_index(drop=True)


def load_official_universe(candidate_dir: Path, market: pd.DataFrame) -> list[str]:
    train_path = candidate_dir / "data" / "train.csv"
    if train_path.exists():
        train = pd.read_csv(train_path, dtype={CN_STOCK: str}, usecols=[CN_STOCK])
        universe = sorted(train[CN_STOCK].map(normalize_stock_code).unique().tolist())
    else:
        universe = sorted(market[CN_STOCK].unique().tolist())
    if len(universe) < 5:
        raise ValueError("universe has fewer than five stocks")
    return universe


def build_folds(
    dates: list[pd.Timestamp],
    start_date: str,
    end_date: str,
    fold_dates: str,
    step_days: int,
    horizon_days: int,
    purge_days: int,
    window_mode: str,
    max_folds: int,
) -> list[Fold]:
    date_index = {date: idx for idx, date in enumerate(dates)}
    selected_indices: list[int] = []
    if fold_dates:
        wanted = [pd.Timestamp(item.strip()) for item in fold_dates.split(",") if item.strip()]
        for date in wanted:
            if date not in date_index:
                raise ValueError(f"fold date is not a trading date in supplied data: {date.date()}")
            selected_indices.append(date_index[date])
    else:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) if end_date else dates[-1]
        first_idx = next(idx for idx, date in enumerate(dates) if date >= start_ts)
        idx = first_idx
        while idx + horizon_days < len(dates) and dates[idx] <= end_ts:
            selected_indices.append(idx)
            idx += max(step_days, 1)

    folds: list[Fold] = []
    for idx in selected_indices:
        if idx - purge_days < 0 or idx + horizon_days >= len(dates):
            continue
        prediction_date = dates[idx]
        train_cutoff = dates[idx - purge_days]
        train_start = None
        if window_mode == "rolling_3y":
            train_start = prediction_date - pd.DateOffset(years=3)
        eval_dates = tuple(dates[idx + 1 : idx + horizon_days + 1])
        folds.append(
            Fold(
                prediction_date=prediction_date,
                train_start=train_start,
                train_cutoff=train_cutoff,
                eval_start=eval_dates[0],
                eval_end=eval_dates[-1],
                eval_dates=eval_dates,
            )
        )
        if max_folds and len(folds) >= max_folds:
            break
    return folds


def prepare_candidate_import(candidate_dir: Path) -> None:
    src_dir = candidate_dir / "code" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    os.chdir(candidate_dir)


def load_candidate_modules(candidate_dir: Path):
    prepare_candidate_import(candidate_dir)
    import predict  # noqa: PLC0415
    from config import config  # noqa: PLC0415

    return predict, config


def zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if values.size == 0 or std < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.mean()) / (std + eps)).astype(np.float32)


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-6)
    values = values - values.max()
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def effective_holdings(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(np.square(weights)))
    if denom <= 0:
        return 0.0
    return 1.0 / denom


def constrained_weights(
    scores: np.ndarray,
    min_weight: float,
    max_weight: float,
    min_effective: float,
    temperature: float = 0.25,
) -> np.ndarray:
    n = len(scores)
    if n != 5:
        raise ValueError("constrained portfolio expects exactly five stocks")
    equal = np.full(n, 1.0 / n, dtype=np.float64)
    raw = softmax(scores, temperature)
    weights = np.maximum(raw, min_weight)
    weights = weights / weights.sum()

    for _ in range(20):
        over = weights > max_weight
        if not over.any():
            break
        excess = float(np.sum(weights[over] - max_weight))
        weights[over] = max_weight
        under = ~over
        if not under.any():
            break
        room = np.maximum(max_weight - weights[under], 0.0)
        if room.sum() <= 1e-12:
            break
        weights[under] += excess * room / room.sum()
    weights = np.maximum(weights, min_weight)
    weights = weights / weights.sum()

    if effective_holdings(weights) < min_effective:
        for alpha in np.linspace(0.95, 0.0, 20):
            blended = alpha * weights + (1.0 - alpha) * equal
            if blended.max() <= max_weight + 1e-12 and effective_holdings(blended) >= min_effective:
                weights = blended
                break
    if weights.max() > max_weight + 1e-10 or effective_holdings(weights) < min_effective - 1e-10:
        weights = equal
    return weights / weights.sum()


def apply_policy(
    policy: str,
    predict_module: Any,
    rerank_config: dict[str, Any],
    config: dict[str, Any],
    scores: np.ndarray,
    processed: pd.DataFrame,
    sequence_stock_ids: list[str],
    latest_date: pd.Timestamp,
    min_weight: float,
    max_weight: float,
    min_effective: float,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    if policy == "model_config":
        pool_size = min(
            int(predict_module.get_inference_setting(rerank_config, "pool_size", config.get("rerank_pool_size", 20))),
            len(sequence_stock_ids),
        )
        candidate_indices = np.argsort(scores)[::-1][:pool_size]
        return_5 = predict_module.latest_feature_values(processed, sequence_stock_ids, latest_date, "return_5")[candidate_indices]
        volume_change = predict_module.latest_feature_values(processed, sequence_stock_ids, latest_date, "volume_change")[candidate_indices]
        volatility_10 = predict_module.latest_feature_values(processed, sequence_stock_ids, latest_date, "volatility_10")[candidate_indices]
        final_scores = (
            zscore(scores[candidate_indices])
            + float(predict_module.get_inference_setting(rerank_config, "rerank_return_5_weight", 0.3)) * zscore(return_5)
            + float(predict_module.get_inference_setting(rerank_config, "rerank_volume_change_weight", 0.2)) * zscore(volume_change)
            + float(predict_module.get_inference_setting(rerank_config, "rerank_volatility_10_weight", 0.2)) * zscore(volatility_10)
        )
        for feature_name, feature_weight in predict_module.get_inference_setting(rerank_config, "rerank_extra_weights", {}).items():
            extra_values = predict_module.latest_feature_values(processed, sequence_stock_ids, latest_date, feature_name)[candidate_indices]
            final_scores = final_scores + float(feature_weight) * zscore(extra_values)
        stocks, weights = predict_module.select_output_portfolio(
            candidate_indices,
            final_scores,
            sequence_stock_ids,
            predict_module.get_inference_setting(rerank_config, "output_top_k", config.get("output_top_k", 5)),
            predict_module.get_inference_setting(rerank_config, "weight_mode", config.get("weight_mode", "softmax")),
            predict_module.get_inference_setting(rerank_config, "softmax_temperature", config.get("softmax_temperature", 1.0)),
            predict_module.get_inference_setting(rerank_config, "min_weight", config.get("min_weight", 0.0)),
            predict_module.get_inference_setting(rerank_config, "max_weight", config.get("max_weight", 1.0)),
        )
        return stocks, np.asarray(weights, dtype=np.float64), {"pool_size": pool_size, "policy_detail": "model_config"}

    if policy == "equal_model":
        candidate_indices = np.argsort(scores)[::-1][:5]
        stocks = [sequence_stock_ids[idx] for idx in candidate_indices]
        return stocks, np.full(5, 0.2, dtype=np.float64), {"pool_size": 5, "policy_detail": "equal_top5_model_score"}

    pool_size = min(50, len(sequence_stock_ids))
    candidate_indices = np.argsort(scores)[::-1][:pool_size]
    return_5 = predict_module.latest_feature_values(processed, sequence_stock_ids, latest_date, "return_5")[candidate_indices]
    volume_change = predict_module.latest_feature_values(processed, sequence_stock_ids, latest_date, "volume_change")[candidate_indices]
    volatility_10 = predict_module.latest_feature_values(processed, sequence_stock_ids, latest_date, "volatility_10")[candidate_indices]
    final_scores = (
        zscore(scores[candidate_indices])
        - 0.6 * zscore(return_5)
        + 0.15 * zscore(volume_change)
        - 0.2 * zscore(volatility_10)
    )
    order = np.argsort(final_scores)[::-1][:5]
    selected = candidate_indices[order]
    selected_scores = final_scores[order]
    stocks = [sequence_stock_ids[idx] for idx in selected]
    weights = constrained_weights(selected_scores, min_weight, max_weight, min_effective)
    return stocks, weights, {
        "pool_size": pool_size,
        "policy_detail": "risk_controlled_model_score_light_rerank",
        "rerank_return_5_weight": -0.6,
        "rerank_volume_change_weight": 0.15,
        "rerank_volatility_10_weight": -0.2,
    }


def predict_for_date(
    candidate_dir: Path,
    market: pd.DataFrame,
    universe: list[str],
    prediction_date: pd.Timestamp,
    policy: str,
    min_weight: float,
    max_weight: float,
    min_effective: float,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    ensemble_config = candidate_dir / "ensemble_config.json"
    if ensemble_config.exists():
        prepare_candidate_import(candidate_dir)
        import ensemble_predict  # noqa: PLC0415

        raw_df = market[market[CN_DATE] <= prediction_date].copy()
        raw_df = raw_df[raw_df[CN_STOCK].isin(universe)]
        stocks, weights, policy_info = ensemble_predict.predict_ensemble_for_frame(
            str(ensemble_config),
            raw_df,
            prediction_date,
            universe,
        )
        return stocks, weights, policy_info

    predict_module, config = load_candidate_modules(candidate_dir)
    raw_df = market[market[CN_DATE] <= prediction_date].copy()
    raw_df = raw_df[raw_df[CN_STOCK].isin(universe)]
    stockid2idx = {sid: idx for idx, sid in enumerate(universe)}

    processed, features = predict_module.preprocess_predict_data(raw_df, stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = joblib.load(Path(config["output_dir"]) / "scaler.pkl")
    processed[features] = scaler.transform(processed[features])

    sequences_np, sequence_stock_ids = predict_module.build_inference_sequences(
        processed,
        features,
        int(config["sequence_length"]),
        universe,
        prediction_date,
    )
    if len(sequence_stock_ids) < 5:
        raise ValueError(f"fewer than five stocks have complete sequences on {prediction_date.date()}")

    rerank_config = predict_module.load_rerank_config(config["output_dir"])
    if predict_module.get_inference_setting(rerank_config, "inference_cross_sectional_norm", True):
        sequences_np = predict_module.cross_sectional_normalize_np(sequences_np)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
    model_path = Path(config["output_dir"]) / "best_model.pth"
    model_paths = [str(model_path)]
    second_model_path = Path(config["output_dir"]) / "second_best_model.pth"
    if predict_module.get_inference_setting(rerank_config, "ensemble_second_best", True) and second_model_path.exists():
        model_paths.append(str(second_model_path))
    scores = predict_module.predict_scores_from_checkpoints(model_paths, x, len(features), len(universe), device)

    stocks, weights, policy_info = apply_policy(
        policy,
        predict_module,
        rerank_config,
        config,
        scores,
        processed,
        sequence_stock_ids,
        prediction_date,
        min_weight,
        max_weight,
        min_effective,
    )
    return stocks, weights, {**policy_info, "model_count": len(model_paths), "available_stocks": len(sequence_stock_ids)}


def score_portfolio(
    market: pd.DataFrame,
    stocks: list[str],
    weights: np.ndarray,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    cost_bps_values: list[float],
) -> dict[str, Any]:
    details = []
    gross = 0.0
    for stock, weight in zip(stocks, weights):
        rows = market[(market[CN_STOCK] == stock) & (market[CN_DATE].isin([eval_start, eval_end]))].sort_values(CN_DATE)
        if len(rows) < 2:
            stock_return = 0.0
            start_open = math.nan
            end_open = math.nan
        else:
            start_open = float(rows.iloc[0][CN_OPEN])
            end_open = float(rows.iloc[-1][CN_OPEN])
            stock_return = (end_open - start_open) / start_open if start_open > 0 else 0.0
        weighted = float(weight) * stock_return
        gross += weighted
        details.append(
            {
                "stock_id": stock,
                "weight": float(weight),
                "start_open": start_open,
                "end_open": end_open,
                "return": stock_return,
                "weighted_return": weighted,
            }
        )
    invested = float(np.sum(weights))
    nets = {
        f"net_score_{int(cost_bps)}bps": gross - (2.0 * float(cost_bps) / 10000.0) * invested
        for cost_bps in cost_bps_values
    }
    return {"gross_score": gross, **nets, "details": details}


def classify_market_state(market: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp, universe: list[str]) -> str:
    start = market[(market[CN_DATE] == eval_start) & (market[CN_STOCK].isin(universe))][[CN_STOCK, CN_OPEN]]
    end = market[(market[CN_DATE] == eval_end) & (market[CN_STOCK].isin(universe))][[CN_STOCK, CN_OPEN]]
    merged = start.merge(end, on=CN_STOCK, suffixes=("_start", "_end"))
    if merged.empty:
        return "unknown"
    returns = (merged[f"{CN_OPEN}_end"] - merged[f"{CN_OPEN}_start"]) / merged[f"{CN_OPEN}_start"]
    mean_return = float(returns.mean())
    volatility = float(returns.std(ddof=0))
    direction = "up" if mean_return > 0.01 else "down" if mean_return < -0.01 else "sideways"
    vol_state = "high_vol" if volatility > 0.06 else "normal_vol"
    return f"{direction}_{vol_state}"


def validate_portfolio(stocks: list[str], weights: np.ndarray, max_weight: float, min_effective: float) -> list[str]:
    issues = []
    if len(stocks) != 5:
        issues.append("portfolio must contain exactly five stocks")
    if len(set(stocks)) != len(stocks):
        issues.append("portfolio contains duplicate stocks")
    if np.any(np.asarray(weights) < -1e-12):
        issues.append("portfolio contains negative weights")
    if float(np.sum(weights)) > 1.0 + 1e-10:
        issues.append("portfolio weight sum exceeds 1")
    if float(np.max(weights)) > max_weight + 1e-10:
        issues.append(f"portfolio max weight exceeds {max_weight}")
    if effective_holdings(weights) < min_effective - 1e-10:
        issues.append(f"effective holdings below {min_effective}")
    return issues


def run_retrain_fold(
    source_candidate_dir: Path,
    fold_workspace: Path,
    train_rows: pd.DataFrame,
    predict_rows: pd.DataFrame,
    python_exe: str,
    keep_workspace: bool,
) -> Path:
    if fold_workspace.exists():
        shutil.rmtree(fold_workspace)
    ignore = shutil.ignore_patterns("__pycache__", ".venv", "temp", "output", "model")
    shutil.copytree(source_candidate_dir, fold_workspace, ignore=ignore)
    (fold_workspace / "data").mkdir(exist_ok=True)
    (fold_workspace / "output").mkdir(exist_ok=True)
    (fold_workspace / "temp").mkdir(exist_ok=True)
    (fold_workspace / "model").mkdir(exist_ok=True)
    train_rows.to_csv(fold_workspace / "data" / "train.csv", index=False, encoding="utf-8")
    subprocess.run([python_exe, "code/src/train.py"], cwd=fold_workspace, check=True)
    predict_rows.to_csv(fold_workspace / "data" / "train.csv", index=False, encoding="utf-8")
    if not keep_workspace:
        # Keep trained model and data for this fold; caller may remove the parent after inference if desired.
        pass
    return fold_workspace


def summarize_scores(fold_rows: list[dict[str, Any]], primary_net_field: str) -> dict[str, Any]:
    values = np.asarray([float(row[primary_net_field]) for row in fold_rows], dtype=np.float64)
    if values.size == 0:
        return {}
    sorted_values = np.sort(values)
    worst_count = max(1, int(math.ceil(0.2 * len(sorted_values))))
    cumulative = np.cumsum(values)
    running_peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_peak
    wins = values[values > 0]
    losses = values[values < 0]
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    metrics = {
        "fold_count": int(values.size),
        "mean_net_score": float(values.mean()),
        "median_net_score": float(np.median(values)),
        "hit_rate": float(np.mean(values > 0)),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_loss_ratio": float(avg_win / abs(avg_loss)) if avg_loss < 0 else None,
        "worst_20pct_mean": float(sorted_values[:worst_count].mean()),
        "min_net_score": float(values.min()),
        "max_net_score": float(values.max()),
        "max_drawdown_proxy": float(drawdown.min()),
        "single_best_fold_contribution": float(values.max() / values.sum()) if abs(values.sum()) > 1e-12 else None,
        "single_worst_fold_loss": float(values.min()),
    }
    for key in sorted(fold_rows[0]):
        if key.startswith("net_score_") and key.endswith("bps"):
            cost_name = "cost_" + key.removeprefix("net_score_")
            metrics[cost_name] = float(np.mean([float(row[key]) for row in fold_rows]))
    return metrics


def read_official_self_score(candidate_dir: Path) -> float | None:
    path = candidate_dir / "temp" / "tmp.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                return float(row.get("Final Score", "nan"))
            except ValueError:
                return None
    return None


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    folds: list[Fold],
    fold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    primary_net_field: str,
    official_self_score: float | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "fold_predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_scores.csv", index=False, encoding="utf-8")

    summary = {
        "candidate_name": args.candidate_name,
        "candidate_dir": str(Path(args.candidate_dir).resolve()),
        "policy": args.policy,
        "window_mode": args.window_mode,
        "training_mode": args.training_mode,
        "model_asof_date": args.model_asof_date,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "step_days": args.step_days,
        "horizon_days": args.horizon_days,
        "purge_days": args.purge_days,
        "min_weight": args.min_weight,
        "max_weight": args.max_weight,
        "min_effective_holdings": args.min_effective_holdings,
        "primary_net_field": primary_net_field,
        "official_self_score": official_self_score,
        "fold_plan_count": len(folds),
        "metrics": summarize_scores(fold_rows, primary_net_field),
    }
    (output_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = summary["metrics"]
    lines = [
        f"# Walk-forward Report: {args.candidate_name}",
        "",
        f"- Policy: `{args.policy}`",
        f"- Window mode: `{args.window_mode}`",
        f"- Training mode: `{args.training_mode}`",
        f"- Primary metric: `{primary_net_field}`",
        f"- Official self-score: `{official_self_score}`",
        f"- Folds evaluated: `{metrics.get('fold_count', 0)}`",
        f"- Mean net score: `{metrics.get('mean_net_score')}`",
        f"- Median net score: `{metrics.get('median_net_score')}`",
        f"- Hit rate: `{metrics.get('hit_rate')}`",
        f"- Worst 20% mean: `{metrics.get('worst_20pct_mean')}`",
        f"- Max drawdown proxy: `{metrics.get('max_drawdown_proxy')}`",
    ]
    (output_dir / "walk_forward_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    history_data = Path(args.history_data).resolve()
    future_data = Path(args.future_data).resolve() if args.future_data else None
    cost_bps_values = [float(item.strip()) for item in args.cost_bps.split(",") if item.strip()]
    primary_net_field = "net_score_10bps" if 10.0 in cost_bps_values else f"net_score_{int(cost_bps_values[0])}bps"

    market = load_market_data(history_data, future_data)
    universe = load_official_universe(candidate_dir, market)
    dates = [pd.Timestamp(date) for date in sorted(market[CN_DATE].unique())]
    folds = build_folds(
        dates,
        args.start_date,
        args.end_date,
        args.fold_dates,
        args.step_days,
        args.horizon_days,
        args.purge_days,
        args.window_mode,
        args.max_folds,
    )
    model_asof = pd.Timestamp(args.model_asof_date)

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_workspace_root = output_dir / "fold_workspaces"
    for fold in folds:
        if args.training_mode == "frozen_existing" and fold.prediction_date < model_asof:
            continue

        active_candidate_dir = candidate_dir
        if args.training_mode == "retrain_per_fold":
            train_rows = market[market[CN_DATE] <= fold.train_cutoff].copy()
            if fold.train_start is not None:
                train_rows = train_rows[train_rows[CN_DATE] >= fold.train_start].copy()
            predict_rows = market[market[CN_DATE] <= fold.prediction_date].copy()
            if fold.train_start is not None:
                predict_rows = predict_rows[predict_rows[CN_DATE] >= fold.train_start].copy()
            active_candidate_dir = run_retrain_fold(
                candidate_dir,
                fold_workspace_root / fold.prediction_date.strftime("%Y%m%d"),
                train_rows,
                predict_rows,
                args.python_exe,
                args.keep_fold_workspaces,
            )

        stocks, weights, policy_info = predict_for_date(
            active_candidate_dir,
            market,
            universe,
            fold.prediction_date,
            args.policy,
            args.min_weight,
            args.max_weight,
            args.min_effective_holdings,
        )
        validation_issues = validate_portfolio(stocks, weights, args.max_weight, args.min_effective_holdings)
        score_info = score_portfolio(market, stocks, weights, fold.eval_start, fold.eval_end, cost_bps_values)
        market_state = classify_market_state(market, fold.eval_start, fold.eval_end, universe)

        fold_row = {
            "candidate_name": args.candidate_name,
            "policy": args.policy,
            "prediction_date": fold.prediction_date.date().isoformat(),
            "train_start": fold.train_start.date().isoformat() if fold.train_start else "",
            "train_cutoff": fold.train_cutoff.date().isoformat(),
            "eval_start": fold.eval_start.date().isoformat(),
            "eval_end": fold.eval_end.date().isoformat(),
            "gross_score": score_info["gross_score"],
            "effective_holdings": effective_holdings(weights),
            "max_weight": float(np.max(weights)),
            "weight_sum": float(np.sum(weights)),
            "market_state": market_state,
            "validation_ok": not validation_issues,
            "validation_issues": ";".join(validation_issues),
            **{key: value for key, value in score_info.items() if key.startswith("net_score_")},
            **policy_info,
        }
        fold_rows.append(fold_row)

        detail_by_stock = {item["stock_id"]: item for item in score_info["details"]}
        for rank, (stock, weight) in enumerate(zip(stocks, weights), start=1):
            detail = detail_by_stock[stock]
            prediction_rows.append(
                {
                    "candidate_name": args.candidate_name,
                    "prediction_date": fold.prediction_date.date().isoformat(),
                    "rank": rank,
                    "stock_id": stock,
                    "weight": float(weight),
                    "eval_start": fold.eval_start.date().isoformat(),
                    "eval_end": fold.eval_end.date().isoformat(),
                    "stock_return": detail["return"],
                    "weighted_return": detail["weighted_return"],
                }
            )

        if args.training_mode == "retrain_per_fold" and not args.keep_fold_workspaces:
            # Keep outputs compact once fold inference has finished.
            shutil.rmtree(active_candidate_dir, ignore_errors=True)

    write_outputs(output_dir, args, folds, fold_rows, prediction_rows, primary_net_field, read_official_self_score(candidate_dir))
    print(json.dumps({"candidate": args.candidate_name, "folds": len(fold_rows), "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())

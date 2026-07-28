from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _d02_dir() -> Path:
    return Path(__file__).resolve().parents[1]


D02_DIR = _d02_dir()
CODE_SRC = D02_DIR / "code_snapshot" / "src"
import sys

sys.path.insert(0, str(CODE_SRC))

from dataset import DFeatureScalers, DStockRankingDataset, build_d_samples, build_inference_sample, collate_d_samples  # noqa: E402
from evaluator import predictions_from_batch, weights_from_scores  # noqa: E402
from featurework import DATE_COL, STOCK_COL, assert_market_truncation_equivalence  # noqa: E402
from loss import PortfolioAwareRankingLoss  # noqa: E402
from train import _move_tensor_batch, device, environment_summary, evaluate_loader, make_model, set_seed  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_window(samples: list[Any]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}
    sample_dates = [pd.Timestamp(sample.sample_date).normalize() for sample in samples]
    label_ends = [
        pd.Timestamp(sample.label_end_date).normalize()
        for sample in samples
        if sample.label_end_date is not None
    ]
    payload: dict[str, Any] = {
        "sample_count": len(samples),
        "sample_start": min(sample_dates).strftime("%Y-%m-%d"),
        "sample_end": max(sample_dates).strftime("%Y-%m-%d"),
    }
    if label_ends:
        payload["label_end_start"] = min(label_ends).strftime("%Y-%m-%d")
        payload["label_end_end"] = max(label_ends).strftime("%Y-%m-%d")
    return payload


def next_weekday_trading_days(decision_date: pd.Timestamp, count: int) -> list[pd.Timestamp]:
    days: list[pd.Timestamp] = []
    current = decision_date
    while len(days) < count:
        current = current + timedelta(days=1)
        if current.weekday() < 5:
            days.append(pd.Timestamp(current).normalize())
    return days


def prepare_production_config(d02_dir: Path, output_dir: Path, data_path: Path) -> dict[str, Any]:
    config = read_json(d02_dir / "generated_config.json")
    champion_summary = read_json(d02_dir / "candidate_summary.json")
    fixed_epochs = int(champion_summary.get("best_epoch") or config["training"]["num_epochs"])
    config["experiment_id"] = "D02_PRODUCTION_REFIT_20260703"
    config["data"]["train_csv"] = str(data_path)
    config["training"]["num_epochs"] = fixed_epochs
    config["training"]["early_stop_patience"] = fixed_epochs
    config["training"]["checkpoint_metric"] = "fixed_final_epoch_no_validation"
    config["validation"] = {
        "mode": "production_refit_no_validation",
        "source_champion_program": "D_AUTO_CHAMPION_REF_20260709_110423",
        "fixed_epochs_from_research_best_epoch": fixed_epochs,
        "no_validation_set": True,
        "no_future_model_selection": True,
    }
    config["output"]["run_dir"] = str(output_dir)
    return config


def build_production_samples(config: dict[str, Any], raw: pd.DataFrame, train_start: pd.Timestamp, raw_data_end: pd.Timestamp):
    samples, columns = build_d_samples(
        raw,
        sequence_length=int(config["features"]["sequence_length"]),
        horizon=int(config["label"]["horizon"]),
        buy_offset=int(config["label"]["buy_offset"]),
        sell_offset=int(config["label"]["sell_offset"]),
        min_stocks_per_day=int(config["data"]["min_stocks_per_day"]),
        include_labels=True,
    )
    train_samples = [
        sample
        for sample in samples
        if pd.Timestamp(sample.sample_date).normalize() >= train_start
        and sample.label_end_date is not None
        and pd.Timestamp(sample.label_end_date).normalize() <= raw_data_end
    ]
    if not train_samples:
        raise ValueError("production refit produced no supervised training samples")
    return train_samples, columns


def build_split_metadata(
    *,
    raw: pd.DataFrame,
    data_path: Path,
    data_md5: str,
    train_samples: list[Any],
    train_start: pd.Timestamp,
    raw_data_end: pd.Timestamp,
    decision_date: pd.Timestamp,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
    config: dict[str, Any],
) -> dict[str, Any]:
    dates = sorted(pd.to_datetime(raw[DATE_COL]).dt.normalize().unique())
    train_window = sample_window(train_samples)
    return {
        "mode": "production_refit_no_validation",
        "split_version": "D02_production_refit_20260703_t5_v1",
        "source_champion_program": "D_AUTO_CHAMPION_REF_20260709_110423",
        "data_path": str(data_path),
        "data_md5": data_md5,
        "raw_data_start": pd.Timestamp(min(dates)).strftime("%Y-%m-%d"),
        "raw_data_end": raw_data_end.strftime("%Y-%m-%d"),
        "train_sample_start": train_window.get("sample_start"),
        "train_sample_end": train_window.get("sample_end"),
        "train_label_end_start": train_window.get("label_end_start"),
        "train_label_end_end": train_window.get("label_end_end"),
        "train_sample_count": train_window.get("sample_count"),
        "feature_end_for_inference": decision_date.strftime("%Y-%m-%d"),
        "decision_date": decision_date.strftime("%Y-%m-%d"),
        "buy_date": buy_date.strftime("%Y-%m-%d"),
        "planned_sell_date": sell_date.strftime("%Y-%m-%d"),
        "label": "open(T+5) / open(T+1) - 1",
        "sell_offset": int(config["label"]["sell_offset"]),
        "buy_offset": int(config["label"]["buy_offset"]),
        "fixed_epochs": int(config["training"]["num_epochs"]),
        "no_validation_set": True,
        "scaler_fit_scope": "production_train_samples_only",
        "trading_dates": {
            "first": pd.Timestamp(min(dates)).strftime("%Y-%m-%d"),
            "last": pd.Timestamp(max(dates)).strftime("%Y-%m-%d"),
            "count": len(dates),
            "tail": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in dates[-8:]],
        },
    }


def run_leakage_audit(
    *,
    raw: pd.DataFrame,
    train_samples: list[Any],
    split_metadata: dict[str, Any],
    raw_data_end: pd.Timestamp,
    decision_date: pd.Timestamp,
    scalers: DFeatureScalers,
) -> dict[str, Any]:
    issues: list[str] = []
    label_ends = [
        pd.Timestamp(sample.label_end_date).normalize()
        for sample in train_samples
        if sample.label_end_date is not None
    ]
    max_label_end = max(label_ends) if label_ends else None
    if max_label_end is None or max_label_end > raw_data_end:
        issues.append("training label interval exceeds raw_data_end")
    if pd.to_datetime(raw[DATE_COL]).dt.normalize().max() > raw_data_end:
        issues.append("raw frame contains rows after raw_data_end")
    if decision_date > raw_data_end:
        issues.append("decision_date exceeds raw_data_end")
    expected_train_sample_end = "2026-06-26"
    if split_metadata.get("train_sample_end") != expected_train_sample_end:
        issues.append(f"train_sample_end is {split_metadata.get('train_sample_end')}, expected {expected_train_sample_end}")
    try:
        assert_market_truncation_equivalence(raw, [decision_date])
        feature_truncation = {"status": "pass", "sampled_dates": [decision_date.strftime("%Y-%m-%d")]}
    except Exception as exc:  # noqa: BLE001
        feature_truncation = {"status": "fail", "error": str(exc)}
        issues.append("feature truncation equivalence failed")
    scaler_lengths_ok = (
        len(scalers.stock_mean) > 0
        and len(scalers.factor_mean) > 0
        and len(scalers.market_mean) > 0
    )
    if not scaler_lengths_ok:
        issues.append("scalers are empty")
    checks = {
        "raw_max_date_lte_2026_07_03": pd.to_datetime(raw[DATE_COL]).dt.normalize().max().strftime("%Y-%m-%d"),
        "train_label_end_lte_raw_data_end": max_label_end.strftime("%Y-%m-%d") if max_label_end is not None else None,
        "train_sample_end": split_metadata.get("train_sample_end"),
        "feature_end_for_inference": split_metadata.get("feature_end_for_inference"),
        "decision_date": split_metadata.get("decision_date"),
        "scaler_fit_scope": split_metadata.get("scaler_fit_scope"),
        "scaler_fit_sample_count": len(train_samples),
        "no_validation_set": bool(split_metadata.get("no_validation_set")),
        "prediction_uses_future_label": False,
        "feature_truncation_equivalence": feature_truncation,
    }
    return {
        "leakage_audit": "pass" if not issues else "fail",
        "checks": checks,
        "issues": issues,
    }


def train_fixed_final(
    *,
    config: dict[str, Any],
    train_samples: list[Any],
    columns: Any,
    scalers: DFeatureScalers,
    output_dir: Path,
    data_path: Path,
) -> dict[str, Any]:
    target_device = device()
    set_seed(int(config["seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "environment.json", environment_summary(data_path))
    scalers.to_json(output_dir / "scalers.json")
    train_samples = scalers.transform_samples(train_samples)
    train_loader = DataLoader(
        DStockRankingDataset(train_samples),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_d_samples,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    train_eval_loader = DataLoader(
        DStockRankingDataset(train_samples),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_d_samples,
        num_workers=0,
    )
    model = make_model(config, train_samples[0], columns).to(target_device)
    criterion = PortfolioAwareRankingLoss(**config["loss"]).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(int(config["training"]["num_epochs"])):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            moved = _move_tensor_batch(batch, target_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(moved)
            breakdown = criterion(outputs, moved["labels"], moved["relevance"], moved["mask"])
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            optimizer.step()
            losses.append(float(breakdown.total.detach().cpu()))
        train_pred, train_daily, train_summary = evaluate_loader(model, train_eval_loader, target_device, config)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "elapsed_minutes": (time.time() - started) / 60.0,
            **{f"train_{key}": value for key, value in train_summary.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        train_daily.to_csv(output_dir / "latest_train_daily_metrics.csv", index=False)
        train_pred.to_csv(output_dir / "latest_train_predictions.csv", index=False)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": row["train_loss"],
                    "train_mean_portfolio_return": row.get("train_mean_portfolio_return"),
                    "elapsed_minutes": row["elapsed_minutes"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    final_model = output_dir / "final_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": int(config["training"]["num_epochs"]),
            "columns": {
                "stock": columns.stock,
                "factor": columns.factor,
                "market": columns.market,
            },
            "fixed_final_epoch_no_validation": True,
        },
        final_model,
    )
    return {
        "model": model,
        "target_device": target_device,
        "train_samples_transformed": train_samples,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "final_model": str(final_model),
        "history": history,
    }


def attach_final_portfolio_weights(predictions: pd.DataFrame, portfolio_config: dict[str, Any]) -> pd.DataFrame:
    frame = predictions.copy()
    frame["portfolio_weight"] = 0.0
    frame["portfolio_selected"] = False
    top_k = int(portfolio_config.get("top_k", 5))
    mode = str(portfolio_config.get("mode", "topk_equal"))
    for _, day in frame.groupby("sample_end_date", sort=True):
        scores = day["score"].to_numpy(dtype=np.float64)
        weights = weights_from_scores(
            scores,
            top_k=top_k,
            mode=mode,
            temperature=float(portfolio_config.get("softmax_temperature", 1.0)),
            min_weight=float(portfolio_config.get("min_weight", 0.0)),
            max_weight=float(portfolio_config.get("max_weight", 1.0)),
        )
        frame.loc[day.index, "portfolio_weight"] = weights
        frame.loc[day.index, "portfolio_selected"] = weights > 1e-12
    return frame


def predict_final(
    *,
    config: dict[str, Any],
    raw: pd.DataFrame,
    scalers: DFeatureScalers,
    model: torch.nn.Module,
    target_device: torch.device,
    output_dir: Path,
    decision_date: pd.Timestamp,
    buy_date: pd.Timestamp,
    sell_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_for_prediction = raw[pd.to_datetime(raw[DATE_COL]).dt.normalize() <= decision_date].copy()
    sample, _ = build_inference_sample(
        raw_for_prediction,
        sequence_length=int(config["features"]["sequence_length"]),
        min_stocks_per_day=1,
    )
    if pd.Timestamp(sample.sample_date).normalize() != decision_date:
        raise ValueError(f"inference sample date {sample.sample_date} is not decision date {decision_date.date()}")
    transformed_sample = scalers.transform_sample(sample)
    loader = DataLoader(DStockRankingDataset([transformed_sample]), batch_size=1, shuffle=False, collate_fn=collate_d_samples)
    model.eval()
    with torch.no_grad():
        batch = _move_tensor_batch(next(iter(loader)), target_device)
        outputs = model(batch)
        pred = predictions_from_batch(outputs, batch)
    pred = attach_final_portfolio_weights(pred, config["portfolio"])
    if "portfolio_logit" not in pred.columns:
        pred["portfolio_logit"] = pred["score"]
    if "regime_risk" not in pred.columns:
        pred["regime_risk"] = 0.0
    ranked = pred.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["decision_date"] = decision_date.strftime("%Y-%m-%d")
    ranked["buy_date"] = buy_date.strftime("%Y-%m-%d")
    ranked["planned_sell_date"] = sell_date.strftime("%Y-%m-%d")
    ranked["final_weight"] = ranked["portfolio_weight"].clip(lower=0.0)
    top_k = int(config["portfolio"]["top_k"])
    ranked["selection_reason"] = np.where(
        ranked["final_weight"] > 1e-12,
        f"selected_top_{top_k}_equal_weight",
        "not_selected_rank_below_top_k",
    )
    ranked["filter_reason"] = np.where(ranked["final_weight"] > 1e-12, "", "rank_below_selected_cutoff")
    ranked = ranked[
        [
            "decision_date",
            "buy_date",
            "planned_sell_date",
            "rank",
            "stock_id",
            "score",
            "portfolio_logit",
            "regime_risk",
            "final_weight",
            "portfolio_selected",
            "selection_reason",
            "filter_reason",
        ]
    ]
    final_dir = output_dir / "final_prediction"
    final_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(final_dir / "ranked_scores.csv", index=False, encoding="utf-8")
    result = ranked[ranked["final_weight"] > 1e-12][["stock_id", "final_weight"]].copy()
    result = result.head(5).rename(columns={"final_weight": "weight"})
    result["weight"] = result["weight"].astype(float)
    if len(result) > 5:
        raise ValueError("result.csv would contain more than 5 stocks")
    if (result["weight"] < -1e-12).any() or float(result["weight"].sum()) > 1.0 + 1e-12:
        raise ValueError("result.csv violates nonnegative/weight-sum constraints")
    result.to_csv(final_dir / "result.csv", index=False, encoding="utf-8")
    manifest = {
        "status": "completed",
        "decision_date": decision_date.strftime("%Y-%m-%d"),
        "feature_end_for_inference": decision_date.strftime("%Y-%m-%d"),
        "buy_date": buy_date.strftime("%Y-%m-%d"),
        "planned_sell_date": sell_date.strftime("%Y-%m-%d"),
        "rule": "D02 production refit model, top-k equal-weight portfolio from Champion config",
        "max_stocks": 5,
        "output_stock_count": int(len(result)),
        "weight_sum": float(result["weight"].sum()),
        "cash_weight": float(max(0.0, 1.0 - result["weight"].sum())),
        "uses_future_market_data": False,
        "uses_future_return_labels": False,
        "ranked_scores": "final_prediction/ranked_scores.csv",
        "result_csv": "final_prediction/result.csv",
    }
    write_json(final_dir / "prediction_manifest.json", manifest)
    return ranked, result, manifest


def write_report(output_dir: Path, manifest: dict[str, Any], result: pd.DataFrame, ranked: pd.DataFrame, stderr_path: Path) -> None:
    top20 = ranked.head(20)
    result_md = result.to_markdown(index=False)
    top20_md = top20[
        ["rank", "stock_id", "score", "final_weight", "selection_reason"]
    ].to_markdown(index=False)
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip() if stderr_path.exists() else ""
    report = f"""# D02 Production Refit Report

## Status

- Status: `{manifest['status']}`
- Source Champion Program: `{manifest['source_champion_program']}`
- Fixed epochs: `{manifest['fixed_epochs']}`
- Final checkpoint: `{manifest['final_model']}`
- Elapsed minutes: `{manifest['elapsed_minutes']:.4f}`

## Data Window

- raw_data_start: `{manifest['raw_data_start']}`
- raw_data_end: `{manifest['raw_data_end']}`
- train_sample_start: `{manifest['train_sample_start']}`
- train_sample_end: `{manifest['train_sample_end']}`
- feature_end_for_inference: `{manifest['feature_end_for_inference']}`
- decision_date: `{manifest['decision_date']}`
- buy_date: `{manifest['buy_date']}`
- planned_sell_date: `{manifest['planned_sell_date']}`

## Leakage Audit

- leakage_audit: `{manifest['leakage_audit']}`
- scaler_fit_scope: `production_train_samples_only`
- validation_set_used: `false`
- future_return_used_for_prediction: `false`

## result.csv

{result_md}

## ranked_scores Top 20

{top20_md}

## stderr

```text
{stderr_text if stderr_text else '(empty)'}
```
"""
    (output_dir / "PRODUCTION_REFIT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="D02 production refit and final 2026-07-03 prediction.")
    parser.add_argument("--d02-dir", default=str(D02_DIR))
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--raw-data-end", default="2026-07-03")
    parser.add_argument("--decision-date", default="2026-07-03")
    args = parser.parse_args()

    d02_dir = Path(args.d02_dir).resolve()
    output_dir = Path(__file__).resolve().parent
    data_path = Path(args.data_path).resolve() if args.data_path else Path(read_json(d02_dir / "generated_config.json")["data"]["train_csv"]).resolve()
    raw_data_end = pd.Timestamp(args.raw_data_end).normalize()
    decision_date = pd.Timestamp(args.decision_date).normalize()
    forward_days = next_weekday_trading_days(decision_date, 5)
    buy_date = forward_days[0]
    sell_date = forward_days[4]

    raw_all = pd.read_csv(data_path)
    raw_all[DATE_COL] = pd.to_datetime(raw_all[DATE_COL]).dt.normalize()
    raw = raw_all[raw_all[DATE_COL] <= raw_data_end].copy()
    if raw.empty:
        raise ValueError("no raw rows available through raw_data_end")
    train_start_candidates = sorted(value for value in raw[DATE_COL].unique() if pd.Timestamp(value).year >= 2022)
    train_start = pd.Timestamp(train_start_candidates[0]).normalize()
    data_md5 = md5_file(data_path)
    config = prepare_production_config(d02_dir, output_dir, data_path)
    write_json(output_dir / "generated_config_production_refit.json", config)

    train_samples, columns = build_production_samples(config, raw, train_start, raw_data_end)
    scalers = DFeatureScalers.fit(train_samples)
    split_metadata = build_split_metadata(
        raw=raw,
        data_path=data_path,
        data_md5=data_md5,
        train_samples=train_samples,
        train_start=train_start,
        raw_data_end=raw_data_end,
        decision_date=decision_date,
        buy_date=buy_date,
        sell_date=sell_date,
        config=config,
    )
    write_json(output_dir / "split_metadata.json", split_metadata)
    audit = run_leakage_audit(
        raw=raw,
        train_samples=train_samples,
        split_metadata=split_metadata,
        raw_data_end=raw_data_end,
        decision_date=decision_date,
        scalers=scalers,
    )
    write_json(output_dir / "leakage_audit.json", audit)
    if audit["leakage_audit"] != "pass":
        raise RuntimeError(f"leakage audit failed: {audit['issues']}")

    training_result = train_fixed_final(
        config=config,
        train_samples=train_samples,
        columns=columns,
        scalers=scalers,
        output_dir=output_dir,
        data_path=data_path,
    )
    ranked, result, prediction_manifest = predict_final(
        config=config,
        raw=raw,
        scalers=scalers,
        model=training_result["model"],
        target_device=training_result["target_device"],
        output_dir=output_dir,
        decision_date=decision_date,
        buy_date=buy_date,
        sell_date=sell_date,
    )
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_champion_program": "D_AUTO_CHAMPION_REF_20260709_110423",
        "d02_snapshot": str(d02_dir),
        "output_dir": str(output_dir),
        "data_path": str(data_path),
        "data_md5": data_md5,
        "raw_data_start": split_metadata["raw_data_start"],
        "raw_data_end": split_metadata["raw_data_end"],
        "train_sample_start": split_metadata["train_sample_start"],
        "train_sample_end": split_metadata["train_sample_end"],
        "train_label_end_end": split_metadata["train_label_end_end"],
        "feature_end_for_inference": split_metadata["feature_end_for_inference"],
        "decision_date": split_metadata["decision_date"],
        "buy_date": split_metadata["buy_date"],
        "planned_sell_date": split_metadata["planned_sell_date"],
        "fixed_epochs": int(config["training"]["num_epochs"]),
        "seed": int(config["seed"]),
        "no_validation_set": True,
        "no_hyperparameter_search": True,
        "no_future_model_selection": True,
        "leakage_audit": audit["leakage_audit"],
        "leakage_audit_path": "leakage_audit.json",
        "final_model": "final_model.pth",
        "elapsed_minutes": float(training_result["elapsed_minutes"]),
        "prediction_manifest": prediction_manifest,
        "artifacts": {
            "generated_config": "generated_config_production_refit.json",
            "final_model": "final_model.pth",
            "scalers": "scalers.json",
            "split_metadata": "split_metadata.json",
            "leakage_audit": "leakage_audit.json",
            "training_history": "training_history.csv",
            "ranked_scores": "final_prediction/ranked_scores.csv",
            "result_csv": "final_prediction/result.csv",
            "prediction_manifest": "final_prediction/prediction_manifest.json",
            "report": "PRODUCTION_REFIT_REPORT.md",
        },
    }
    write_json(output_dir / "production_refit_manifest.json", manifest)
    write_report(output_dir, manifest, result, ranked, output_dir / "train_stderr.log")
    print(json.dumps({"status": "completed", "manifest": str(output_dir / "production_refit_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

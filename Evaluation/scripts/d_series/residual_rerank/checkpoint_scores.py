from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decision_dates_from_range(
    raw: pd.DataFrame,
    *,
    date_column: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> list[str]:
    dates = pd.to_datetime(raw[date_column]).dt.normalize()
    lower = pd.Timestamp(start).normalize()
    upper = pd.Timestamp(end).normalize()
    return [
        pd.Timestamp(value).strftime("%Y-%m-%d")
        for value in sorted(dates.loc[(dates >= lower) & (dates <= upper)].unique())
    ]


def canonical_score_frame(
    frame: pd.DataFrame,
    *,
    marker: str,
    experiment_id: str,
) -> pd.DataFrame:
    date_column = (
        "decision_date" if "decision_date" in frame.columns else "sample_end_date"
    )
    required = {date_column, "stock_id", "score"}
    if missing := required.difference(frame.columns):
        raise KeyError(f"missing score columns: {sorted(missing)}")
    result = frame[[date_column, "stock_id", "score"]].copy()
    result = result.rename(columns={date_column: "decision_date"})
    result["decision_date"] = pd.to_datetime(result["decision_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    result["stock_id"] = (
        result["stock_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    result["score"] = pd.to_numeric(result["score"], errors="raise").astype(float)
    if result.duplicated(["decision_date", "stock_id"]).any():
        raise ValueError("score export contains duplicate decision/stock rows")
    result = result.sort_values(["decision_date", "stock_id"]).reset_index(drop=True)
    result.insert(0, "experiment_id", experiment_id)
    result.insert(0, "marker", marker)
    return result


def write_canonical_scores(frame: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )


def assert_byte_identical(left: str | Path, right: str | Path) -> dict[str, object]:
    left_path = Path(left)
    right_path = Path(right)
    left_bytes = left_path.read_bytes()
    right_bytes = right_path.read_bytes()
    if left_bytes != right_bytes:
        raise ValueError(
            f"score passes are not byte-identical: {left_path} != {right_path}"
        )
    digest = hashlib.sha256(left_bytes).hexdigest()
    return {
        "identical": True,
        "bytes": len(left_bytes),
        "sha256": digest,
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
    }


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _import_h_modules(code_root: str | Path) -> tuple[Any, Any, Any]:
    source = str(Path(code_root).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    dataset = importlib.import_module("dataset")
    featurework = importlib.import_module("featurework")
    model = importlib.import_module("model")
    return dataset, featurework, model


def export_checkpoint_scores(
    *,
    h_code_root: str | Path,
    config_path: str | Path,
    data_path: str | Path,
    scalers_path: str | Path,
    checkpoint_path: str | Path,
    start: str | None,
    end: str | None,
    output_path: str | Path,
    experiment_id: str,
    seed: int,
    force_cpu: bool = False,
) -> dict[str, object]:
    set_determinism(seed)
    dataset_module, featurework, model_module = _import_h_modules(h_code_root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw = pd.read_csv(data_path)
    raw[featurework.DATE_COL] = pd.to_datetime(
        raw[featurework.DATE_COL]
    ).dt.normalize()
    if end is None:
        upper = pd.Timestamp(raw[featurework.DATE_COL].max()).normalize()
    else:
        upper = pd.Timestamp(end).normalize()
    raw = raw.loc[raw[featurework.DATE_COL] <= upper].copy()
    if start is None:
        lower = upper
    else:
        lower = pd.Timestamp(start).normalize()
    sample_dates = decision_dates_from_range(
        raw,
        date_column=featurework.DATE_COL,
        start=lower,
        end=upper,
    )
    if not sample_dates:
        raise ValueError(f"no inference dates in {lower.date()}..{upper.date()}")

    features = config.get("features", {})
    samples, columns = dataset_module.build_d_samples(
        raw,
        sequence_length=int(features["sequence_length"]),
        horizon=int(config.get("label", {}).get("horizon", 5)),
        buy_offset=int(config.get("label", {}).get("buy_offset", 1)),
        sell_offset=int(config.get("label", {}).get("sell_offset", 5)),
        min_stocks_per_day=1,
        sample_dates=sample_dates,
        include_labels=False,
        industry_map=None,
        futures_frame=None,
        enable_industry=False,
        enable_futures=False,
        industry_missing_label="UNKNOWN",
        futures_lag_days=0,
    )
    if len(samples) != len(sample_dates):
        raise ValueError(
            f"inference sample count differs: expected {len(sample_dates)}, got {len(samples)}"
        )
    scalers = dataset_module.DFeatureScalers.from_json(scalers_path)
    samples = scalers.transform_samples(samples)
    loader = DataLoader(
        dataset_module.DStockRankingDataset(samples),
        batch_size=max(1, int(config.get("training", {}).get("batch_size", 4))),
        shuffle=False,
        collate_fn=dataset_module.collate_d_samples,
        num_workers=0,
    )
    target_device = torch.device(
        "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    )
    ranker = model_module.MarketIntelligenceRanker(
        stock_input_dim=len(columns.stock),
        market_input_dim=len(columns.market),
        factor_input_dim=len(columns.factor),
        sequence_length=int(features["sequence_length"]),
        config=config["model"],
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=target_device,
        weights_only=False,
    )
    ranker.load_state_dict(checkpoint["model_state_dict"])
    ranker.to(target_device)
    ranker.eval()

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            moved = {
                key: value.to(target_device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            outputs = ranker(moved)
            scores = outputs["ranking_scores"].detach().cpu().numpy()
            mask = moved["mask"].detach().cpu().numpy().astype(bool)
            for batch_index, decision_date in enumerate(batch["sample_dates"]):
                count = int(mask[batch_index].sum())
                for stock_index in range(count):
                    rows.append(
                        {
                            "decision_date": decision_date,
                            "stock_id": batch["stock_codes"][batch_index][stock_index],
                            "score": float(scores[batch_index, stock_index]),
                        }
                    )
    canonical = canonical_score_frame(
        pd.DataFrame(rows),
        marker="TEST",
        experiment_id=experiment_id,
    )
    write_canonical_scores(canonical, output_path)
    counts = canonical.groupby("decision_date")["stock_id"].size()
    report = {
        "marker": "TEST",
        "experiment_id": experiment_id,
        "device": str(target_device),
        "seed": int(seed),
        "date_start": str(canonical["decision_date"].min()),
        "date_end": str(canonical["decision_date"].max()),
        "decision_date_count": int(canonical["decision_date"].nunique()),
        "row_count": int(len(canonical)),
        "min_stock_count": int(counts.min()),
        "max_stock_count": int(counts.max()),
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(Path(output_path).resolve()),
        "output_sha256": sha256_file(output_path),
        "sources": {
            "config_sha256": sha256_file(config_path),
            "data_sha256": sha256_file(data_path),
            "scalers_sha256": sha256_file(scalers_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        },
        "outcome_columns_emitted": False,
        "industry_enabled": False,
        "futures_enabled": False,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deterministic score-only tables from frozen D/H0 checkpoints"
    )
    parser.add_argument("--h-code-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scalers", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_checkpoint_scores(
        h_code_root=args.h_code_root,
        config_path=args.config,
        data_path=args.data,
        scalers_path=args.scalers,
        checkpoint_path=args.checkpoint,
        start=args.start,
        end=args.end,
        output_path=args.output,
        experiment_id=args.experiment_id,
        seed=args.seed,
        force_cpu=args.cpu,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

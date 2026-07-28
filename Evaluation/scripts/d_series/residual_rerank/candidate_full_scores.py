from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from checkpoint_scores import sha256_file, write_canonical_scores


def canonical_full_score_frame(
    *,
    decision_date: pd.Timestamp | str,
    stock_codes: Iterable[object],
    scores: Iterable[float],
    experiment_id: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "stock_id": [
                str(value).replace(".0", "").zfill(6) for value in stock_codes
            ],
            "score": [float(value) for value in scores],
        }
    )
    if frame["stock_id"].duplicated().any():
        raise ValueError("full score export contains duplicate stock codes")
    frame["decision_date"] = pd.Timestamp(decision_date).strftime("%Y-%m-%d")
    frame["experiment_id"] = str(experiment_id)
    frame["marker"] = "TEST"
    return (
        frame[
            ["marker", "experiment_id", "decision_date", "stock_id", "score"]
        ]
        .sort_values(["decision_date", "stock_id"])
        .reset_index(drop=True)
    )


def _set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)


def _import_candidate_modules(code_root: str | Path) -> tuple[Any, Any, Any]:
    source = str(Path(code_root).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    return (
        importlib.import_module("dataset"),
        importlib.import_module("featurework"),
        importlib.import_module("model"),
    )


def export_candidate_full_scores(
    *,
    code_root: str | Path,
    config_path: str | Path,
    history_path: str | Path,
    test_path: str | Path,
    checkpoint_path: str | Path,
    scalers_path: str | Path,
    output_path: str | Path,
    experiment_id: str,
    seed: int,
    force_cpu: bool = False,
) -> dict[str, object]:
    _set_determinism(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    dataset, featurework, model_module = _import_candidate_modules(code_root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    frames = [pd.read_csv(history_path), pd.read_csv(test_path)]
    raw = pd.concat(frames, ignore_index=True)
    raw[featurework.DATE_COL] = pd.to_datetime(
        raw[featurework.DATE_COL]
    ).dt.normalize()
    raw = (
        raw.drop_duplicates(
            [featurework.STOCK_COL, featurework.DATE_COL],
            keep="last",
        )
        .sort_values([featurework.DATE_COL, featurework.STOCK_COL])
        .reset_index(drop=True)
    )
    sample, columns = dataset.build_inference_sample(
        raw,
        sequence_length=int(config["features"]["sequence_length"]),
        min_stocks_per_day=1,
    )
    scalers = dataset.DFeatureScalers.from_json(scalers_path)
    sample = scalers.transform_sample(sample)
    batch = dataset.collate_d_samples([sample])
    device = torch.device(
        "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    )
    moved = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    ranker = model_module.MarketIntelligenceRanker(
        stock_input_dim=len(columns.stock),
        market_input_dim=len(columns.market),
        factor_input_dim=len(columns.factor),
        sequence_length=int(config["features"]["sequence_length"]),
        config=config["model"],
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    ranker.load_state_dict(checkpoint["model_state_dict"])
    ranker.to(device)
    ranker.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = ranker(moved)
    scores = (
        outputs["ranking_scores"][0, : len(sample.stock_codes)]
        .detach()
        .cpu()
        .numpy()
    )
    canonical = canonical_full_score_frame(
        decision_date=sample.sample_date,
        stock_codes=sample.stock_codes,
        scores=scores,
        experiment_id=experiment_id,
    )
    write_canonical_scores(canonical, output_path)
    return {
        "marker": "TEST",
        "experiment_id": experiment_id,
        "device": str(device),
        "decision_date": canonical["decision_date"].iloc[0],
        "row_count": int(len(canonical)),
        "unique_stock_count": int(canonical["stock_id"].nunique()),
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(Path(output_path).resolve()),
        "output_sha256": sha256_file(output_path),
        "outcome_columns_emitted": False,
        "sources": {
            "config_sha256": sha256_file(config_path),
            "history_sha256": sha256_file(history_path),
            "test_sha256": sha256_file(test_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "scalers_sha256": sha256_file(scalers_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all Candidate-D scores from an isolated package copy"
    )
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scalers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_candidate_full_scores(
        code_root=args.code_root,
        config_path=args.config,
        history_path=args.history,
        test_path=args.test_data,
        checkpoint_path=args.checkpoint,
        scalers_path=args.scalers,
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

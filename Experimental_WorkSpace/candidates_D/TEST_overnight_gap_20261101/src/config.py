from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_id": "D001",
    "seed": 20260707,
    "data": {
        "train_csv": "data/train.csv",
        "date_col": "日期",
        "stock_col": "股票代码",
        "min_stocks_per_day": 10,
    },
    "label": {
        "horizon": 5,
        "buy_offset": 1,
        "sell_offset": 5,
    },
    "features": {
        "version": "d_market_overnight_v2",
        "sequence_length": 60,
    },
    "model": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "use_factor_encoder": False,
    },
    "loss": {
        "temperature": 1.0,
        "pairwise_weight": 0.25,
        "pairwise_margin": 0.05,
        "portfolio_return_weight": 0.0,
        "portfolio_volatility_weight": 0.0,
    },
    "portfolio": {
        "top_k": 5,
        "mode": "topk_equal",
        "softmax_temperature": 1.0,
        "max_weight": 1.0,
        "min_weight": 0.0,
    },
    "training": {
        "batch_size": 2,
        "num_epochs": 20,
        "learning_rate": 5e-5,
        "weight_decay": 1e-5,
        "grad_clip": 5.0,
        "num_workers": 0,
        "early_stop_patience": 5,
    },
    "validation": {
        "mode": "purged_walk_forward",
        "folds": [
            ["fold1", "2024-01-02", "2024-12-31"],
            ["fold2", "2025-01-02", "2025-12-31"],
            ["fold3", "2026-01-19", "2026-03-12"],
        ],
    },
    "output": {
        "run_dir": "Experimental_WorkSpace/candidates_D/TEST_overnight_gap_20261101/runs/manual",
    },
}


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            deep_update(config, json.load(handle))
    if overrides:
        deep_update(config, overrides)
    return config


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

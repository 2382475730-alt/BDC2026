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
        "version": "d_market_v1",
        "sequence_length": 60,
    },
    "model": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "use_factor_encoder": False,
        "use_regime_conditioned_head": False,
        "regime_concentration_relaxation": 0.0,
    },
    "loss": {
        "temperature": 1.0,
        "pairwise_weight": 0.25,
        "pairwise_margin": 0.05,
        "portfolio_return_weight": 0.0,
        "portfolio_volatility_weight": 0.0,
        "downside_weight": 0.0,
        "downside_margin": 0.0,
        "cvar_weight": 0.0,
        "cvar_alpha": 0.05,
        "negative_weight_weight": 0.0,
        "win_rate_weight": 0.0,
        "win_rate_margin": 0.0,
        "win_rate_temperature": 0.01,
        "positive_tail_penalty_weight": 0.0,
        "positive_tail_cap": 0.08,
        "return_softcap": 0.0,
    },
    "portfolio": {
        "top_k": 5,
        "mode": "topk_equal",
        "softmax_temperature": 1.0,
        "max_weight": 1.0,
        "min_weight": 0.0,
        "regime_risk_threshold": 0.55,
        "low_risk_top_k": 2,
        "low_risk_cash_ratio": 0.0,
        "high_risk_top_k": 4,
        "high_risk_cash_ratio": 0.25,
        "signal_gap_cash_threshold": 0.0,
        "weak_signal_cash_ratio": 0.0,
        "stable_candidate_multiplier": 3,
        "stable_score_margin": 0.0,
        "regime_down_return_threshold": -0.02,
        "regime_high_vol_threshold": 0.035,
        "regime_low_breadth_threshold": 0.48,
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
        "run_dir": "experiments/D001_manual_run",
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

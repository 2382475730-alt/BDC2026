# D001 Market Encoder Baseline

Updated: 2026-07-08

D001 is the first D-series code baseline. It is intentionally not a C-series extension. It introduces an explicit market-state branch and keeps date-grouped stock ranking samples as the training unit.

## Scope

Implemented:

- causal market-state feature extraction from history available at prediction date T;
- date-grouped dataset output with `market_features`, `factor_features`, `stock_sequence`, and labels;
- market-conditioned fusion model with ranking and portfolio logits;
- listwise/pairwise ranking loss plus optional portfolio-aware terms;
- evaluation helpers for return, ranking, stability, and risk metrics;
- dry-run training entrypoint and checkpoint-aware inference entrypoint;
- leakage and smoke tests.

Current status:

- first training round completed under the C-equivalent train/validation split;
- best exploratory run: `experiments/round1_D001_market_only_top2_20260708/`;
- best checkpoint: `experiments/round1_D001_market_only_top2_20260708/best_model.pth`;
- no C-series modification;
- no candidate submission packaging.

## Layout

```text
D001_MarketEncoder_Baseline/
  code/src/
    config.py
    dataset.py
    evaluator.py
    featurework.py
    loss.py
    model.py
    test.py
    train.py
  configs/
    d001_market_encoder_baseline.json
  experiments/
    README.md
  model_cards/
    D001_MARKET_ENCODER_BASELINE.md
  tests/
```

## Quick Checks

From this directory:

```powershell
python code/src/train.py --config configs/d001_market_encoder_baseline.json --dry-run --data-path ..\..\candidates_C\C0_BaseModel\data\train.csv --max-dates 20
```

If `pytest` is available, run `python -m pytest tests`. In the current workspace `.venv`, `pytest` is not installed, so the tests can also be called directly from Python:

```powershell
@'
from tests.test_d001_market_features import test_market_features_are_truncation_equivalent, test_purged_split_removes_overlapping_label_windows
from tests.test_d001_smoke import test_d001_forward_and_loss_smoke

test_market_features_are_truncation_equivalent()
test_purged_split_removes_overlapping_label_windows()
test_d001_forward_and_loss_smoke()
print("manual tests passed")
'@ | python -
```

The dry run builds samples, fits scalers on the training-date side only, performs one forward/loss pass, and exits without training.

## Round 1 Training

Report:

```text
experiments/ROUND1_REPORT_20260708.md
```

Resume command for the best exploratory run:

```powershell
python code/src/train.py --config configs/d001_round1_market_only_top2_20260708.json --resume experiments\round1_D001_market_only_top2_20260708\last_model.pth --time-budget-minutes 60
```

## Round 2 Training

Report:

```text
experiments/R2_REPORT_20260708.md
```

Current R2 validation champion:

```text
experiments/r2_D001_three_set_portfolio_return_20260708/best_model.pth
```

This checkpoint is selected by validation only. It improves three-set validation robust final_score to `0.030488`, but local-test metrics remain negative, so it is not promotion-ready.

## T+5 Protocol

D-series training now treats the competition target as a single T+5 portfolio problem:

```text
label(T) = open(T+5) / open(T+1) - 1
```

All D001 train, validation, holdout, and one-shot test samples use this same label. The split is chronological and purged:

- `train`: used for model weights, scaler fitting, and feature statistics only.
- `validation`: used for checkpoint selection, top-k choice, loss/model iteration, and stability checks.
- `holdout_backtest`: multiple frozen T+5 anchors after validation, used only as a stability diagnostic.
- `test`: exactly one T+5 anchor, matching the competition rule that the final test is one week. Do not select checkpoints or tune hyperparameters from test scores.

Current R2 split over `..\..\candidates_C\C0_BaseModel\data\train.csv`:

```text
train             sample_end before validation, with label_end < 2026-01-05
validation        2026-01-05..2026-02-20 sample_end dates
holdout_backtest  2026-02-23..2026-03-12 sample_end dates
test              2026-03-12 one-week T+5 anchor only
```

The active R2 config is:

```text
configs/d001_r2_three_set_top2_20260708.json
```

Checkpoint selection remains `val_robust_final_score`; holdout and one-shot test metrics are recorded for reporting only.

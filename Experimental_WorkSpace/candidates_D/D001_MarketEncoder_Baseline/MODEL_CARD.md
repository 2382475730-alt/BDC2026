# MODEL_CARD: D001 Market Encoder Baseline

Updated: 2026-07-08

## Hypothesis

C-series assumes future stock ranking is mostly learnable from individual historical sequences plus implicit cross-sectional attention. D001 tests a different hypothesis: the same stock sequence can imply different opportunity depending on the current market regime.

## Architecture

```text
market features at T -> MarketEncoder --------\
                                               -> MarketConditionedFusion -> ranking score
stock history up to T -> StockTemporalEncoder -/                           -> portfolio logits
```

D001 keeps the stock temporal branch because a market encoder alone cannot rank stocks. The difference from C-series is that market state is explicit and is injected into every stock representation before cross-stock fusion.

## Data Boundary

Features are computed using data available at or before sample date T. Labels use the official open-to-open target:

```text
label = open(T+5) / open(T+1) - 1
```

Training and validation must be split by sample date, not random rows. Purged validation removes training samples whose label interval overlaps the validation period.

## Train / Validation / Test Protocol

Starting with R2, D001 separates model selection, holdout diagnostics, and the competition-style one-week test:

- Train fits model weights and scalers.
- Validation selects checkpoints and iteration choices.
- Holdout backtest is a frozen multi-anchor diagnostic after checkpoint selection and must not drive hyperparameter search.
- Test is exactly one T+5 anchor, matching the competition's one-week evaluation.

The current R2 split uses `Experimental_WorkSpace/candidates_C/C0_BaseModel/data/train.csv`:

```text
validation: 2026-01-05..2026-02-20 sample_end dates
holdout_backtest: 2026-02-23..2026-03-12 sample_end dates
test: 2026-03-12 one-week T+5 anchor
```

Training samples are required to have `label_end_date < 2026-01-05`, so the model does not train on label windows overlapping validation or local test.

## Difference From C-Series

- D001 uses separate market-state features instead of treating all context as stock-level sequence features.
- D001 has an independent dataset builder and does not reuse the C-series `RankingDataset` class.
- D001 exposes portfolio logits and portfolio-aware loss hooks, though D001 config keeps the portfolio loss coefficient at zero.
- D001 records experiment metadata before training.

## Expected Advantage

The model can learn that momentum, volatility, and liquidity patterns should be interpreted differently across market states such as broad risk-on, narrow dispersion, high volatility, or liquidity contraction.

## Risks

- Market proxy features are built from the available stock universe, not a separately reported CSI300 index file.
- The initial market branch may underperform C-series until D002/D003 factor and fusion variants are trained.
- Stronger validation may lower local one-shot self-score while improving generalization discipline.

## Current Status

First training round completed on 2026-07-08 under the C-equivalent train/validation split. Best exploratory run is `round1_D001_market_only_top2_20260708`, with validation robust final_score `0.0025435417511850827` and mean validation portfolio return `0.044969475614683084`.

This is not promotion-ready. It remains below the retained C1 legacy robust validation score and has not been tested with formal walk-forward or local official-style self-score.

R2 retrains D001 under the three-set T+5 protocol with config `configs/d001_r2_three_set_top2_20260708.json`.

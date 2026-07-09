# D-Series Round 1 Training Report

Date: 2026-07-08

## Scope

This round trained and iterated D-series models only under:

```text
Experimental_WorkSpace/candidates_D/D001_MarketEncoder_Baseline/
```

No C-series files or `candidate/` submission packages were modified.

## Data And Split

- Data file: `Experimental_WorkSpace/candidates_C/C0_BaseModel/data/train.csv`
- Data md5: `5f389f4817dc1d33ebf53e7fed6425fd`
- Rows: `367152`
- Stocks: `300`
- Raw date range: `2021-01-04` to `2026-03-19`
- Split mode: `c_legacy_last_two_months`, matching the C-series legacy validation boundary.
- Validation target range: `2026-01-19` to `2026-03-19`
- D validation sample dates with complete T+5 labels: `2026-01-19` to `2026-03-12`
- D train samples: `1158`
- D validation samples: `33`
- Leakage check: max train label end is before validation start.

Environment:

- Python: `.venv/Scripts/python.exe`
- Python version: `3.11.9`
- Torch: `2.10.0+cu128`
- CUDA: available
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`

## Runs

| Run | Best epoch | Best robust final_score | Mean final_score | Min final_score | Mean portfolio return | Hit rate | Worst20 return | Volatility | Max drawdown | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `round1_D001_market_only_top2_20260708` | 20 | 0.002544 | 0.173750 | -0.396939 | 0.044969 | 0.636364 | -0.041251 | 0.088342 | -0.107469 | -0.095372 |
| `round1_D001_factor_fusion_top3_20260708` | 1 | -0.009681 | 0.137522 | -0.353154 | 0.032522 | 0.575758 | -0.052352 | 0.069875 | -0.172068 | -0.066546 |
| `round1_D001_market_only_20260708` | 23 | -0.025721 | 0.099624 | -0.318191 | 0.022489 | 0.757576 | -0.042420 | 0.053041 | -0.151238 | -0.086084 |
| `round1_D001_factor_fusion_20260708` | 1 | -0.089863 | 0.044822 | -0.404128 | 0.012726 | 0.545455 | -0.061481 | 0.056021 | -0.302762 | -0.066546 |
| `round1_D001_market_only_lowlr_20260708` | 1 | -0.166061 | 0.025797 | -0.613730 | 0.009805 | 0.545455 | -0.069978 | 0.057223 | -0.411009 | -0.103822 |

Best checkpoint:

```text
experiments/round1_D001_market_only_top2_20260708/best_model.pth
```

Resume checkpoint:

```text
experiments/round1_D001_market_only_top2_20260708/last_model.pth
```

Resume command:

```powershell
python code/src/train.py --config configs/d001_round1_market_only_top2_20260708.json --resume experiments\round1_D001_market_only_top2_20260708\last_model.pth --time-budget-minutes 60
```

## Quality Assessment

The best D run is usable as an exploratory D-series baseline, but it is not promotion-ready.

Positive signals:

- Validation mean portfolio return reached `4.50%` on the C-equivalent validation window.
- Robust final_score became slightly positive at `0.00254`, unlike the initial top5 D001 run.
- Top-2 concentration materially improved stability versus top5 selection for this D model.
- Checkpoint/resume works through `last_model.pth`.

Weak signals:

- Rank IC is still negative (`-0.095372`), so the model is not reliably ordering the full cross-section.
- The robust score is only barely above zero and is still far below C1 legacy validation.
- The best run depends on a validation-selected portfolio size (`top_k=2`), which creates overfitting risk.
- Worst validation days remain poor, especially around late January and mid-March 2026.

## Comparison To C-Series

C1 retained legacy training record:

```text
Best robust_val_score: 0.118085
Best mean_final_score: 0.147696
Best min_final_score: 0.048992
```

D round 1 best:

```text
Best robust_val_score: 0.002544
Best mean_final_score: 0.173750
Best min_final_score: -0.396939
```

D has a higher mean final_score in this validation window, but much worse minimum-day behavior. C1 is still the stronger baseline because its robust score is materially higher.

## Overfitting And Stability

Observed overfitting pattern:

- Training loss steadily decreases across all runs.
- Validation score peaks at specific epochs and then degrades.
- FactorFusion overfits immediately: best checkpoint is epoch 1.
- MarketOnly-Top2 peaks at epoch 20 and then deteriorates through epoch 35.

This is not classic "train score high, validation score low" only. It is a stability problem: the model can find profitable validation days but fails on a few adverse regime days, which dominates robust scoring.

## Upside Potential

There is further upside, but the next gains are more likely from objective/portfolio alignment and regime filtering than from simply training longer.

Evidence:

- Changing only portfolio size from top5 to top2 turned robust final_score from negative to slightly positive.
- Factor features improved some return statistics under top3 but were too unstable without stronger regularization.
- The D architecture is learning some useful signal, but not yet a reliable cross-sectional ordering.

## Recommended Next Steps

1. Add a risk-aware/cash gate using only pre-T market state and score dispersion, then validate it with walk-forward. Do not tune the gate only on the current 33-day validation window.
2. Add portfolio-aware training where evaluation can use `portfolio_logits`, not only `ranking_scores`.
3. Run at least 3 seeds for MarketOnly-Top2 to measure seed variance before trusting the score.
4. Run frozen walk-forward against post-2026-03-19 BaoStock future data before any promotion discussion.
5. Keep C1 as champion. Treat current D best as an auxiliary expert candidate, not as a replacement.

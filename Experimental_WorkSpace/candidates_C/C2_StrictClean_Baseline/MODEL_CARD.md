# C2 Model Card

Former ID: `C302_formal_purge_clean_c4_cutoff_20260313`.

## Role

C2 is the retained strict-clean baseline. It is a formal-purge clean retrain from C0/C4 logic and exists to measure leakage-controlled generalization.

## Training Data

Training data: `2021-01-04..2026-03-13`.

Formal evaluation window: `2026-03-23..2026-06-26`.

Leakage control: train sample dates end at `2026-03-13`. With the 5-trading-day label horizon, the latest label endpoint is `2026-03-20`, before formal starts on `2026-03-23`. The fixed formal holdout evaluation loads the trained model and does not refit on formal dates.

## Training Result

Best epoch: 1.

Best robust validation score: 0.075640.

Early stopped at epoch 6.

## Fixed-Model Formal Holdout

13 folds.

| Cost | Mean net | Median net | Hit rate | Worst 20 pct mean | Max drawdown proxy |
|---|---:|---:|---:|---:|---:|
| 0bps | 0.010025 | -0.002965 | 0.4615 | -0.056273 | -0.116063 |
| 10bps | 0.008025 | -0.004965 | 0.4615 | -0.058273 | -0.121694 |

## Current Portfolio

| Stock | Weight |
|---|---:|
| 600549 | 0.500000 |
| 002353 | 0.349978 |
| 000657 | 0.050022 |
| 600118 | 0.050000 |
| 002028 | 0.050000 |

## Notes

C2 is clean enough to use as the formal leakage-control baseline, but it is not a production champion. Its main value is measuring future D-series bottom-level training-program changes under strict sample separation.

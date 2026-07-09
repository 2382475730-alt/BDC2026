# C0 Model Card

Former ID: `C4_recent_weighted_train`.

## Role

C0 is the retained base Transformer model. It is the historical C4 recent-weighted training line and serves as the lowest retained C-series model baseline.

## Intended Use

- Baseline model checkpoint for comparison.
- Reference point for future D-series bottom-level training-program experiments.
- Not the current production champion.

## Known Formal Metrics

Fixed/frozen formal window: `2026-03-23..2026-06-26`, 13 folds, 10bps primary cost.

| metric | value |
|---|---:|
| mean_net_score | 0.011875 |
| median_net_score | 0.010792 |
| hit_rate | 0.692308 |
| worst_20pct_mean | -0.056864 |
| max_drawdown_proxy | -0.081745 |

## Notes

C0 is kept because C1's champion logic builds on this trained Transformer family. Future D-series models should compare against C0 when changing the bottom-level training program.

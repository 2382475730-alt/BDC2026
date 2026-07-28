# C1 Model Card

## Role

C1 is the retained return-seeking champion candidate from the 2026-07-03 night search.

Former ID: `C77`.

It keeps the C0 trained Transformer checkpoints and changes only inference-time portfolio selection.

## Regime Logic

- Compute `_gate_volume_ratio_5 = volume / volume.shift(5)` before scaling.
- Use the cross-sectional mean on the prediction date.
- If `mean(_gate_volume_ratio_5) <= 1.111401`, use fixed manual ranks `[1, 2, 12, 15, 17]` with weights `[0.05, 0.15, 0.50, 0.15, 0.15]`.
- Otherwise use the C0 floor-capped softmax fallback with `temperature=0.03`, `min_weight=0.05`, and `max_weight=0.50`.
- `market_state` is not used because it is derived from the evaluation interval and would leak future returns.

## Validated Metrics

| window | mean_net_score | hit_rate | worst_20pct_mean | max_drawdown_proxy |
|---|---:|---:|---:|---:|
| formal_2026q2 | 0.030653 | 0.769231 | -0.027266 | -0.046147 |
| shadow_2026q1 | 0.005751 | 0.714286 | -0.015433 | -0.030866 |
| shadow_2025h2 | 0.014801 | 0.631579 | -0.037340 | -0.070232 |
| holdout_2025h1 | 0.006821 | 0.470588 | -0.040890 | -0.131219 |

Four-window mean: `0.014507`.

## Selection Note

C1 beats C0 on all four mean_net_score windows and remains the main retained C-series champion baseline.

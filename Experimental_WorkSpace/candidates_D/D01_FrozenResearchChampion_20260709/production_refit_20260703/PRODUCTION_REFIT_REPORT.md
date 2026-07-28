# D02 Production Refit Report

## Status

- Status: `completed`
- Source Champion Program: `D_AUTO_CHAMPION_REF_20260709_110423`
- Fixed epochs: `12`
- Final checkpoint: `final_model.pth`
- Elapsed minutes: `4.9965`

## Data Window

- raw_data_start: `2021-01-04`
- raw_data_end: `2026-07-03`
- train_sample_start: `2022-01-04`
- train_sample_end: `2026-06-26`
- feature_end_for_inference: `2026-07-03`
- decision_date: `2026-07-03`
- buy_date: `2026-07-06`
- planned_sell_date: `2026-07-10`

## Leakage Audit

- leakage_audit: `pass`
- scaler_fit_scope: `production_train_samples_only`
- validation_set_used: `false`
- future_return_used_for_prediction: `false`

## result.csv

|   stock_id |   weight |
|-----------:|---------:|
|     301165 |      0.5 |
|     688126 |      0.5 |

## ranked_scores Top 20

|   rank |   stock_id |    score |   final_weight | selection_reason              |
|-------:|-----------:|---------:|---------------:|:------------------------------|
|      1 |     301165 | 1.86931  |            0.5 | selected_top_2_equal_weight   |
|      2 |     688126 | 1.45303  |            0.5 | selected_top_2_equal_weight   |
|      3 |     688521 | 0.701293 |            0   | not_selected_rank_below_top_k |
|      4 |     300476 | 0.601121 |            0   | not_selected_rank_below_top_k |
|      5 |     688183 | 0.55246  |            0   | not_selected_rank_below_top_k |
|      6 |     300442 | 0.507302 |            0   | not_selected_rank_below_top_k |
|      7 |     000975 | 0.459725 |            0   | not_selected_rank_below_top_k |
|      8 |     301308 | 0.436199 |            0   | not_selected_rank_below_top_k |
|      9 |     002463 | 0.429359 |            0   | not_selected_rank_below_top_k |
|     10 |     002600 | 0.417013 |            0   | not_selected_rank_below_top_k |
|     11 |     000938 | 0.406743 |            0   | not_selected_rank_below_top_k |
|     12 |     300502 | 0.382696 |            0   | not_selected_rank_below_top_k |
|     13 |     688082 | 0.370854 |            0   | not_selected_rank_below_top_k |
|     14 |     688223 | 0.370161 |            0   | not_selected_rank_below_top_k |
|     15 |     300433 | 0.36731  |            0   | not_selected_rank_below_top_k |
|     16 |     688396 | 0.332636 |            0   | not_selected_rank_below_top_k |
|     17 |     300661 | 0.318211 |            0   | not_selected_rank_below_top_k |
|     18 |     688008 | 0.285096 |            0   | not_selected_rank_below_top_k |
|     19 |     000657 | 0.27765  |            0   | not_selected_rank_below_top_k |
|     20 |     300274 | 0.2561   |            0   | not_selected_rank_below_top_k |

## stderr

```text
(empty)
```

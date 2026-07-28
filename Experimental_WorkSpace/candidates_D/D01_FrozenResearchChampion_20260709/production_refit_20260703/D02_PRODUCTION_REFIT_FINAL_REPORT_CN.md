# D02 Production Refit ??????

## 1. D02 ?????????

- D02 ???`Experimental_WorkSpace\candidates_D\D02_FrozenResearchChampion_20260709`
- ?? Champion Program?`D_AUTO_CHAMPION_REF_20260709_110423`
- D02 ????? D ?? research champion ???????? challenger???????
- ????`code_snapshot/`?`generated_config.json`?`best_model.pth`?`last_model.pth`?`scalers.json`?`split_metadata.json`?`leakage_audit.json`?`metrics_v2.json`?`weekly_backtest.csv`?`official_proxy_result.json`?`champion_gate_decision.json`?`RUN_REPORT.md`?`MODEL_CARD.md`?`D02_SNAPSHOT_MANIFEST.json`?

## 2. Production Refit ????

| ?? | ? |
| --- | --- |
| raw_data_start | `2021-01-04` |
| raw_data_end | `2026-07-03` |
| train_sample_start | `2022-01-04` |
| train_sample_end | `2026-06-26` |
| train_label_end_end | `2026-07-03` |
| feature_end_for_inference | `2026-07-03` |
| decision_date | `2026-07-03` |
| buy_date | `2026-07-06` |
| planned_sell_date | `2026-07-10` |

`train_sample_end` ?????? `2026-06-26`????? T+5 label ????? T???????? `2026-07-03`?

## 3. ???????

- ?? D02/Champion ?????????`code_snapshot/src/`?
- ?? Champion ???seed `20260708`?Market Intelligence ranker?portfolio-return loss?`top_k=2` equal-weight portfolio?
- ? research ???? validation best checkpoint?Production Refit ???? `12` epoch??????? early stopping?????????
- ???????`production_refit_20260703/generated_config_production_refit.json`?

## 4. Leakage Audit ??

- ???`pass`?
- ??????? 2026-07-03?`2026-07-03`?
- ???????`2026-07-03`?
- scaler fit ???`production_train_samples_only`???? `1083`?
- ?????????????`False`?
- feature truncation equivalence?`pass`?

## 5. ????

- ???`completed`?
- ???`4.9965` ???
- ?? checkpoint?`production_refit_20260703/final_model.pth`?
- stderr???

## 6. ??????? result.csv

|   stock_id |   weight |
|-----------:|---------:|
|     301165 |      0.5 |
|     688126 |      0.5 |

## 7. ranked_scores ? 20 ?

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

## 8. candidate/ ????

??? `candidate/` ???????????????????????

## 9. ?????????

?????????? D ????????????????????????????????????? `candidate/`??? docker/app ????????`train.sh/test.sh/init.sh`??? readme?5 ??????????????????

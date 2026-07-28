# D001 vs Official Baseline Same-Validation Comparison

- Validation source: `D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU\Experimental_WorkSpace\candidates_D\automation_runs\D_AUTO_CHAMPION_REF_20260709_110423\best_val_predictions.csv`
- Date range: 2025-01-03 to 2026-01-05
- Validation samples: 243
- Aligned rows: 72415 / D rows 72415
- Official missing sequences: 0
- Official model: `D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU\candidate\THU-BDC2026-main\model\60_158+39`

## Metrics

|mode|top_k|mean_final_score|robust_final_score|min_final_score|mean_portfolio_return|hit_rate|rank_ic|ndcg_at_5|finite_score_ratio|
|---|---|---|---|---|---|---|---|---|---|
|D_current_top2|2|0.100610|-0.249709|-1.067123|0.037120|0.654321|0.005407|0.224595|1.000000|
|D_top5_diagnostic|5|0.082422|-0.223482|-0.937260|0.026377|0.637860|0.005407|0.224595|1.000000|
|official_default_top5|5|0.028692|-0.124276|-0.481199|0.012528|0.633745|0.002017|0.145255|0.168724|
|official_top2_diagnostic|2|0.040534|-0.197040|-0.751377|0.016525|0.563786|0.002017|0.145255|0.168724|

## Deltas

Primary current behavior: `D_current_top2 - official_default_top5`.

```json
{
  "mean_final_score": 0.07191868556614678,
  "robust_final_score": -0.12543392005191092,
  "min_final_score": -0.5859233331607121,
  "mean_portfolio_return": 0.024591336465848386,
  "hit_rate": 0.020576131687242816,
  "rank_ic": 0.0033907386980020393,
  "ndcg_at_5": 0.07933992564788228
}
```

Same top_k=5 diagnostic: `D_top5_diagnostic - official_default_top5`.

```json
{
  "mean_final_score": 0.053730659419069834,
  "robust_final_score": -0.09920664767530507,
  "min_final_score": -0.4560603642288466,
  "mean_portfolio_return": 0.013848589675718007,
  "hit_rate": 0.004115226337448541,
  "rank_ic": 0.0033907386980020393,
  "ndcg_at_5": 0.07933992564788228
}
```

Same top_k=2 diagnostic: `D_current_top2 - official_top2_diagnostic`.

```json
{
  "mean_final_score": 0.060076637463241905,
  "robust_final_score": -0.052669984485402166,
  "min_final_score": -0.3157454356989049,
  "mean_portfolio_return": 0.020594439931885044,
  "hit_rate": 0.09053497942386834,
  "rank_ic": 0.0033907386980020393,
  "ndcg_at_5": 0.07933992564788228
}
```

## Notes

- The official checkpoint is frozen from `candidate/THU-BDC2026-main`; it is scored on D001's validation dates and labels.
- The official package default portfolio emits top5 equal weights; the D-series source predictions are evaluated with top2 equal weights by default.
- Top2/top5 diagnostic rows reuse each model's ranking scores while changing only the portfolio selection size.
- Official baseline scores are all NaN on this validation input; official portfolio metrics therefore use the same `np.argsort(scores)[::-1]` fallback behavior used by its `predict.py`.
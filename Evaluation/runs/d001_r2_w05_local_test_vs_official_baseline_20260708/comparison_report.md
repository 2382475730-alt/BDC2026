# D001 vs Official Baseline Same-Validation Comparison

- Validation source: `D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU\Experimental_WorkSpace\candidates_D\D001_MarketEncoder_Baseline\experiments\r2_D001_three_set_portfolio_return_20260708\best_test_predictions.csv`
- Date range: 2026-02-24 to 2026-03-12
- Validation samples: 13
- Aligned rows: 3892 / D rows 3892
- Official missing sequences: 0
- Official model: `D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU\candidate\THU-BDC2026-main\model\60_158+39`

## Metrics

|mode|top_k|mean_final_score|robust_final_score|min_final_score|mean_portfolio_return|hit_rate|rank_ic|ndcg_at_5|finite_score_ratio|
|---|---|---|---|---|---|---|---|---|---|
|D_current_top2|2|-0.015086|-0.069170|-0.195366|-0.007109|0.461538|-0.147376|0.108270|1.000000|
|D_top5_diagnostic|5|-0.078458|-0.199514|-0.481976|-0.014442|0.384615|-0.147376|0.108270|1.000000|
|official_default_top5|5|-0.099116|-0.154516|-0.283782|-0.024494|0.307692|0.000000|0.062975|0.000000|
|official_top2_diagnostic|2|-0.181015|-0.289580|-0.542898|-0.043015|0.153846|0.000000|0.062975|0.000000|

## Deltas

Primary current behavior: `D_current_top2 - official_default_top5`.

```json
{
  "mean_final_score": 0.08403012768473388,
  "robust_final_score": 0.08534578339021329,
  "min_final_score": 0.08841564670299856,
  "mean_portfolio_return": 0.017384352726982716,
  "hit_rate": 0.15384615384615385,
  "rank_ic": -0.14737627584397123,
  "ndcg_at_5": 0.04529538857844567
}
```

Same top_k=5 diagnostic: `D_top5_diagnostic - official_default_top5`.

```json
{
  "mean_final_score": 0.02065787384862673,
  "robust_final_score": -0.04499766972293168,
  "min_final_score": -0.198193938056568,
  "mean_portfolio_return": 0.01005193852459061,
  "hit_rate": 0.07692307692307693,
  "rank_ic": -0.14737627584397123,
  "ndcg_at_5": 0.04529538857844567
}
```

Same top_k=2 diagnostic: `D_current_top2 - official_top2_diagnostic`.

```json
{
  "mean_final_score": 0.16592889759812976,
  "robust_final_score": 0.22040971178159455,
  "min_final_score": 0.34753161154301243,
  "mean_portfolio_return": 0.03590608601846794,
  "hit_rate": 0.3076923076923077,
  "rank_ic": -0.14737627584397123,
  "ndcg_at_5": 0.04529538857844567
}
```

## Notes

- The official checkpoint is frozen from `candidate/THU-BDC2026-main`; it is scored on D001's validation dates and labels.
- The official package default portfolio emits top5 equal weights; the D-series source predictions are evaluated with top2 equal weights by default.
- Top2/top5 diagnostic rows reuse each model's ranking scores while changing only the portfolio selection size.
- Official baseline scores are all NaN on this validation input; official portfolio metrics therefore use the same `np.argsort(scores)[::-1]` fallback behavior used by its `predict.py`.
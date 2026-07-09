# D001 vs Official Baseline Same-Validation Comparison

- Validation source: `D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU\Experimental_WorkSpace\candidates_D\D001_MarketEncoder_Baseline\experiments\r2_D001_three_set_portfolio_return_20260708\best_val_predictions.csv`
- Date range: 2026-01-05 to 2026-02-13
- Validation samples: 30
- Aligned rows: 8970 / D rows 8970
- Official missing sequences: 0
- Official model: `D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU\candidate\THU-BDC2026-main\model\60_158+39`

## Metrics

|mode|top_k|mean_final_score|robust_final_score|min_final_score|mean_portfolio_return|hit_rate|rank_ic|ndcg_at_5|finite_score_ratio|
|---|---|---|---|---|---|---|---|---|---|
|D_current_top2|2|0.134955|0.030488|-0.213270|0.038837|0.700000|0.023587|0.195268|1.000000|
|D_top5_diagnostic|5|0.070100|-0.021421|-0.234971|0.017127|0.666667|0.023587|0.195268|1.000000|
|official_default_top5|5|0.006120|-0.120228|-0.415039|0.007131|0.666667|0.000000|0.126863|0.000000|
|official_top2_diagnostic|2|0.100745|-0.060071|-0.435311|0.031293|0.666667|0.000000|0.126863|0.000000|

## Deltas

Primary current behavior: `D_current_top2 - official_default_top5`.

```json
{
  "mean_final_score": 0.1288354342007869,
  "robust_final_score": 0.15071575190062736,
  "min_final_score": 0.20176982653358846,
  "mean_portfolio_return": 0.031705148267404476,
  "hit_rate": 0.033333333333333326,
  "rank_ic": 0.02358726804896178,
  "ndcg_at_5": 0.06840542422904683
}
```

Same top_k=5 diagnostic: `D_top5_diagnostic - official_default_top5`.

```json
{
  "mean_final_score": 0.063980649406393,
  "robust_final_score": 0.09880701446721822,
  "min_final_score": 0.18006853294247713,
  "mean_portfolio_return": 0.009995344968046988,
  "hit_rate": 0.0,
  "rank_ic": 0.02358726804896178,
  "ndcg_at_5": 0.06840542422904683
}
```

Same top_k=2 diagnostic: `D_current_top2 - official_top2_diagnostic`.

```json
{
  "mean_final_score": 0.034209493554034784,
  "robust_final_score": 0.09055892677104442,
  "min_final_score": 0.22204093761073354,
  "mean_portfolio_return": 0.007543384796008461,
  "hit_rate": 0.033333333333333326,
  "rank_ic": 0.02358726804896178,
  "ndcg_at_5": 0.06840542422904683
}
```

## Notes

- The official checkpoint is frozen from `candidate/THU-BDC2026-main`; it is scored on D001's validation dates and labels.
- The official package default portfolio emits top5 equal weights; the D-series source predictions are evaluated with top2 equal weights by default.
- Top2/top5 diagnostic rows reuse each model's ranking scores while changing only the portfolio selection size.
- Official baseline scores are all NaN on this validation input; official portfolio metrics therefore use the same `np.argsort(scores)[::-1]` fallback behavior used by its `predict.py`.
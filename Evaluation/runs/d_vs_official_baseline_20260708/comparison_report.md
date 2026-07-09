# D001 vs Official Baseline Same-Validation Comparison

- Validation source: `D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU\Experimental_WorkSpace\candidates_D\D001_MarketEncoder_Baseline\experiments\round1_D001_market_only_top2_20260708\best_val_predictions.csv`
- Date range: 2026-01-19 to 2026-03-12
- Validation samples: 33
- Aligned rows: 9872 / D rows 9872
- Official missing sequences: 0
- Official model: `D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU\candidate\THU-BDC2026-main\model\60_158+39`

## Metrics

|mode|top_k|mean_final_score|robust_final_score|min_final_score|mean_portfolio_return|hit_rate|rank_ic|ndcg_at_5|finite_score_ratio|
|---|---|---|---|---|---|---|---|---|---|
|D_current_top2|2|0.173750|0.002544|-0.396939|0.044969|0.636364|-0.095372|0.217906|1.000000|
|D_top5_diagnostic|5|0.095312|-0.005376|-0.240316|0.021391|0.666667|-0.095372|0.217906|1.000000|
|official_default_top5|5|-0.057769|-0.164950|-0.415039|-0.010628|0.454545|0.000000|0.096103|0.000000|
|official_top2_diagnostic|2|-0.012767|-0.171806|-0.542898|-0.000994|0.393939|0.000000|0.096103|0.000000|

## Deltas

Primary current behavior: `D_current_top2 - official_default_top5`.

```json
{
  "mean_final_score": 0.231519332518651,
  "robust_final_score": 0.16749377158525044,
  "min_final_score": 0.018100796073982572,
  "mean_portfolio_return": 0.05559769659422189,
  "hit_rate": 0.18181818181818182,
  "rank_ic": -0.0953715538089294,
  "ndcg_at_5": 0.12180327060945341
}
```

Same top_k=5 diagnostic: `D_top5_diagnostic - official_default_top5`.

```json
{
  "mean_final_score": 0.15308137814641923,
  "robust_final_score": 0.15957401310013683,
  "min_final_score": 0.17472349465881135,
  "mean_portfolio_return": 0.03201966203240303,
  "hit_rate": 0.2121212121212121,
  "rank_ic": -0.0953715538089294,
  "ndcg_at_5": 0.12180327060945341
}
```

Same top_k=2 diagnostic: `D_current_top2 - official_top2_diagnostic`.

```json
{
  "mean_final_score": 0.18651730313309606,
  "robust_final_score": 0.1743498544529604,
  "min_final_score": 0.14595914086597717,
  "mean_portfolio_return": 0.045963741999774786,
  "hit_rate": 0.24242424242424243,
  "rank_ic": -0.0953715538089294,
  "ndcg_at_5": 0.12180327060945341
}
```

## Notes

- The official checkpoint is frozen from `candidate/THU-BDC2026-main`; it is scored on D001's validation dates and labels.
- The official package default portfolio emits top5 equal weights; D001 current round-1 config uses top2 equal weights.
- Top2/top5 diagnostic rows reuse each model's ranking scores while changing only the portfolio selection size.
- Official baseline scores are all NaN on this validation input; official portfolio metrics therefore use the same `np.argsort(scores)[::-1]` fallback behavior used by its `predict.py`.
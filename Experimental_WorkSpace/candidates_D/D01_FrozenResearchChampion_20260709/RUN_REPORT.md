# D-Series Auto Iteration Report: D_AUTO_CHAMPION_REF_20260709_110423

- Decision: `REJECT`
- Reason: mean_weekly_excess; positive_diff_rate; diff_without_best_2_weeks; best_2_weeks_contribution_ratio; serious_challenger_3_seeds; seed_mean_excess_positive; seed_min_excess_not_materially_bad; cost_drag_le_half_gross_improvement
- Official baseline score: `0.012528499297343995`
- Candidate official-proxy score: `0.03711983576319238`
- Weekly cumulative net return: `2.4696850615960635`
- Mean weekly net return: `0.02812305849117497`
- Median weekly net return: `0.011561056351378442`
- Max drawdown abs: `0.30052236067111693`
- Worst week: `-0.1384535469931582`
- CVaR 95 loss abs: `0.11100414457583074`
- Diff without best 2 weeks: `None`
- Best 2 weeks contribution ratio: `None`

## Gate Checks

|check|passed|value|threshold|
|---|---:|---:|---:|
|leakage_audit_pass|True|pass|None|
|reproducibility_check_pass|True|pass|None|
|same_protocol_declared|True|fixed by orchestrator|None|
|exceeds_official_baseline|True|0.03711983576319238|0.012528499297343995|
|net_cumulative_return_gt_champion|True|2.4696850615960635|-inf|
|mean_weekly_excess|False|None|0.001|
|median_weekly_excess|True|None|0.0|
|positive_diff_rate|False|None|0.52|
|max_drawdown_abs|True|0.30052236067111693|inf|
|CVaR_95_loss_abs|True|0.11100414457583074|inf|
|diff_without_best_2_weeks|False|None|0.0|
|best_2_weeks_contribution_ratio|False|None|0.5|
|serious_challenger_3_seeds|False|1|3|
|seed_mean_excess_positive|False|None|0.0|
|seed_min_excess_not_materially_bad|False|None|-0.005|
|cost_drag_le_half_gross_improvement|False|0.07949999999999999|0.012295668232924193|

# Improvement-THU Workspace Layout

Updated: 2026-07-08

## Root

The workspace root is:

```text
D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU
```

The root is not a final submission package.

## Main Directories

```text
BaoStock_Datasets/
Experimental_WorkSpace/
Evaluation/
candidate/
docs/
skills/
_cleanup_reports/
.venv/
```

## Data

`BaoStock_Datasets/` contains the local BaoStock datasets. It replaces the old directory:

```text
03_data_baostock_expansion_20260630/
```

## Experiments

`Experimental_WorkSpace/` contains experiment candidates and the shared training environment. It replaces the old directory:

```text
08_experiment_generalization_selfscore_20260701/
```

Current retained C-series model baselines:

```text
Experimental_WorkSpace/candidates_C/C0_BaseModel/
Experimental_WorkSpace/candidates_C/C1_Champion_RegimeGate/
Experimental_WorkSpace/candidates_C/C2_StrictClean_Baseline/
```

Current D-series experiment package:

```text
Experimental_WorkSpace/candidates_D/D001_MarketEncoder_Baseline/
Experimental_WorkSpace/candidates_D/D_experiment_registry.json
Experimental_WorkSpace/candidates_D/automation_runs/
Experimental_WorkSpace/candidates_D/data_snapshots/
```

Model definitions:

- `C0`: former `C4_recent_weighted_train`, the base Transformer model.
- `C1`: former `C77_c4_regime_raw_volume5_gate_c69_else_c4`, the current champion regime-gated inference baseline.
- `C2`: former `C302_formal_purge_clean_c4_cutoff_20260313`, the strict clean baseline.
- `D001`: Market Encoder baseline for the D-series Market Intelligence lineage. Round1 and R2 have been trained; no D-series model has been promoted.

`candidates_C/` represents one candidate family that shares the same bottom-level training program lineage.

`candidates_D/` represents a second candidate family for explicit market-state, factor, stock-temporal, and portfolio-aware modeling. The current D001 package contains code, configs, and experiment outputs.

Within a candidate family, every candidate can keep its own package-local code under `code/src/` for reproducibility. The family boundary is determined by the bottom-level training program lineage, not by whether every inference script is byte-identical.

Future bottom-level training-program experiments should use a separate candidate-family folder. The current D-series folder is:

```text
Experimental_WorkSpace/candidates_D/
```

Examples:

- Inference-only changes such as rerank rules, regime gates, or portfolio weight modes can stay inside the same candidate family.
- Changes to `train.py`, `model.py`, loss, labels, feature construction, dataset construction, optimizer schedule, or checkpoint-selection semantics should start a new candidate family.
- A new candidate-family folder must include its own `README.md`.

## Evaluation

`Evaluation/` contains shared evaluation scripts, configurations, generated runs, and reports.

Current categories:

```text
Evaluation/scripts/d_series/
Evaluation/scripts/walk_forward/
Evaluation/scripts/reporting/
Evaluation/scripts/self_score/
Evaluation/scripts/leakage_audit/
Evaluation/scripts/scoreboard/
```

It replaces the old directory:

```text
04_framework_walkforward_eval_20260701/
```

## Submission Candidates

`candidate/` remains reserved for submission candidates and official/reference packages.

Current submission/reference packages:

```text
candidate/C1_Submission_App_20260703/
candidate/Improvement_THU_original_candidate_20260705/
candidate/THU-BDC2026-main/
```

## Cleanup Record

The C0/C1/C2 rename and pruning manifest is:

```text
_cleanup_reports/model_workspace_rename_manifest_20260707.json
```

## Rules

- Keep submission packages in `candidate/` clean.
- Run exploratory training and D-series search in `Experimental_WorkSpace/`, not in `candidate/`.
- Keep shared evaluation scripts in `Evaluation/`.
- Use `BaoStock_Datasets/` as the local BaoStock data source.
- Treat fixed `output/result.csv` self-score as a one-shot diagnostic; use walk-forward reports to discuss generalization risk.
- When changing workspace structure or folder semantics, update `AGENTS.md`, this layout file, and the relevant local `README.md` in the same task.

## D-Series Automation Additions

`Evaluation/scripts/d_series/` contains reusable D-series research-validation utilities:

```text
split_and_purge.py
leakage_audit.py
official_proxy_eval.py
weekly_backtest.py
metrics_v2.py
paired_champion_gate.py
experiment_registry.py
production_refit.py
```

`Experimental_WorkSpace/candidates_D/run_d_auto_iteration.py` orchestrates smoke checks and automated D-series challenger runs. Its generated artifacts stay under:

```text
Experimental_WorkSpace/candidates_D/automation_runs/
```

`Experimental_WorkSpace/candidates_D/data_snapshots/` stores derived D-series experiment input snapshots and manifests. Canonical market source records remain in `BaoStock_Datasets/`.

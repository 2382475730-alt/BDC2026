# F-Series Futures-Industry Fusion

Updated: 2026-07-26

`candidates_F/` is an isolated research lineage for causal industry-relative
and exchange-traded futures features. It is not a submission package.

No retained numbered F-series model exists. Smoke, Quick, fold, and seed runs
use run IDs. `F01_...` may be allocated only after the preregistered data,
leakage, robustness, same-protocol baseline, runtime, and reproducibility gates
all pass.

The 2026-07-26 Quick run stopped after only 2/4 folds were positive and the
recent 2026Q1 fold was negative. Its checkpoints are retained only as `TEST`
evidence; see `TEST_STATUS.json` and `F_experiment_registry.json`.

The external-data reporting status is currently unverified. All outputs in this
family are therefore `RESEARCH_ONLY_NOT_SUBMISSION_ELIGIBLE` unless the exact
source files, URLs, and MD5 values are confirmed against the team's
2026-07-18 reporting email.

## Layout

```text
candidates_F/
  code/src/          # isolated training implementation derived from D01
  configs/           # frozen control and fusion configurations
  automation_runs/   # compact run evidence
  tests/             # feature and leakage tests
  TEST_STATUS.json
  F_experiment_registry.json
  F_PREREGISTRATION.md
  F_STAGE_PLAN.json
```

Canonical market data remains in `BaoStock_Datasets/`. Candidate packages are
not modified or promoted by this lineage.

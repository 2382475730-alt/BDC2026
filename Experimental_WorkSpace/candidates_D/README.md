# candidates_D Directory

Updated: 2026-07-08

`candidates_D/` is the D-series candidate-family directory for a new bottom-level training lineage.

This family is separate from `Experimental_WorkSpace/candidates_C/`. C-series is a sequence ranking expert built around stock history, temporal Transformer encoding, cross-stock attention, and ranking loss. D-series starts a new hypothesis:

```text
future ranking = market regime state + factor exposure + stock temporal behavior + portfolio constraints
```

## Current Contents

```text
candidates_D/
  D001_MarketEncoder_Baseline/
  automation_runs/
  data_snapshots/
  D_experiment_registry.json
  run_d_auto_iteration.py
```

`D001_MarketEncoder_Baseline/` is the active D001 experiment package. Round 1 has been trained and scored under the former C-equivalent validation split. R2 uses the three-set T+5 protocol:

```text
train       purged samples before validation
validation  2026-01-05..2026-02-20 sample_end dates
holdout     2026-02-23..2026-03-12 multi-anchor diagnostic
test        2026-03-12 one-week T+5 anchor
```

It has not been promoted into `candidate/`.

## Family Boundary

Models belong here when they change bottom-level training semantics relative to C-series, including:

- model structure, especially explicit market/factor/portfolio branches;
- dataset construction and date-grouped sample semantics;
- label construction or purge rules;
- loss design or portfolio-aware objective terms;
- checkpoint selection or validation protocol.

Inference-only rerank changes to C-series do not belong here.

## Data And Evaluation Boundary

- Training data should be documented by path, source, date range, and md5 before any long run.
- Validation should use walk-forward, purged splits, or the documented D001 three-set T+5 protocol. Local one-shot self-score is only diagnostic.
- Future labels may be used for scoring and supervised training labels, but not for feature fitting, scaler fitting, inference rules, or parameter search outside the training fold.
- Do not place model artifacts in `BaoStock_Datasets/`.
- Do not promote anything from this directory into `candidate/` unless explicitly requested.

## Required Recordkeeping

Every D-series experiment should have an entry in `D_experiment_registry.json` before a training run starts. Entries must record architecture, feature version, label version, training dates, validation dates, seed, hyperparameters, metrics, checkpoint path, and status.

## Automated Iteration Framework

`run_d_auto_iteration.py` runs the D-series smoke and challenger workflow. It writes all run-local artifacts under `automation_runs/<experiment_id>/`, including:

- generated config and code snapshot;
- environment and data hash metadata;
- split metadata and leakage audit;
- training logs, checkpoints, predictions, and resume command;
- official-proxy result, weekly validation/backtest result, `metrics_v2.json`, Champion Gate decision, and production-refit manifest;
- `RUN_REPORT.md` for the run.

`data_snapshots/` stores derived experiment input snapshots with manifests. The current snapshot `c0_train_plus_baostock_future_20260703.csv` combines retained C0 training history with the BaoStock future file through 2026-07-03 so production-refit metadata can use `feature_end_for_inference = 2026-07-03`. Do not treat this folder as the canonical market-data source; keep source provenance in the snapshot manifest and `D_experiment_registry.json`.

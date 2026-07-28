# Evaluation Workspace

Updated: 2026-07-26

`Evaluation/` contains shared validation, walk-forward, leakage-audit,
scoreboard, portfolio-policy, and reporting code. It is not a submission package
and does not own retained model weights.

## Layout

```text
Evaluation/
├─ scripts/
│  ├─ d_series/          # D-series protocols, gates, policies, and tests
│  ├─ walk_forward/      # rolling evaluation
│  ├─ reporting/         # report generation
│  ├─ self_score/        # shared self-score utilities
│  ├─ leakage_audit/     # leakage checks
│  └─ scoreboard/        # same-protocol comparisons
├─ configs/
├─ runs/                 # small, valid, interpretable outputs only
└─ reports/
```

## D-Series Utilities

The `scripts/d_series/` package includes:

- split/purge, leakage, official-proxy, weekly-backtest, metrics, registry, and
  production-refit helpers;
- paired Champion Gate logic;
- R4 nested-validation and frozen-checkpoint inference helpers;
- R5 expanding-window, top-k objective, and sealed-holdout helpers;
- Position Control R1 OOF training and policy evaluation;
- Defensive Position R3 policy evaluation;
- unit tests for R4, R5, position control, and defensive-position semantics.

These utilities may be reused by future experiments, but completed controllers
and historical outputs can contain legacy IDs. New work must explicitly point
to:

```text
Experimental_WorkSpace/candidates_D/D01_FrozenResearchChampion_20260709/
```

and use new run IDs. Deleted run-level checkpoints or prediction matrices are
not current inputs.

## Candidate Validation

Submission packages validate themselves from their package roots:

```powershell
$python = ".\Experimental_WorkSpace\shared_envs\pytorch-cu128-py311\Scripts\python.exe"

Push-Location ".\candidate\Candidate-C"
& $python -m compileall -q code
& $python code\src\validate_submission.py --app-dir .
Pop-Location

Push-Location ".\candidate\Candidate-D"
& $python -m compileall -q code
& $python code\src\validate_submission.py --app-dir .
Pop-Location

Push-Location ".\candidate\Candidate-I"
& $python -m compileall -q code
& $python code\src\validate_submission.py --app-dir .
Pop-Location
```

Package inference and training commands are documented in each package's
`readme.md`. Do not run exploratory training inside `candidate/`.

`Candidate-I` is a user-authorized submission-format TEST package. Its source
residual-rerank run was `REJECTED_AT_DATA_GATE`; package validation must not be
reported as research promotion or evidence of improvement over Candidate-D or
the official baseline.

## Scoreboard Usage

`scripts/scoreboard/compare_d_vs_official_baseline.py` compares a supplied
D-series prediction table with the protected official checkpoint on the same
dates and labels. Pass the prediction source explicitly; do not rely on retired
experiment paths.

## Three-candidate recent rolling validation

`scripts/walk_forward/tri_candidate_recent_rolling.py` implements the frozen
two-round protocol for `THU-BDC2026-main`, `Candidate-C`, and `Candidate-D`.
It enforces same-byte training inputs per round, from-scratch training,
double-inference prediction locks, date-gated W2 materialization, transaction
cost sensitivity, benchmark comparison, and leakage checks.

The completed 2026-07-22/23 run is stored under
`runs/tri_candidate_recent_rolling_20260722/`; its reader-facing Chinese report
is `reports/tri_candidate_recent_rolling_20260722.md`. The two observed weeks
support provisional labels only. Do not use this run for promotion until the
same frozen protocol has accumulated at least eight independent weeks.

## External-data TEST assessment

The 2026-07-26 staged industry, CFFEX futures, and point-in-time financial
feature assessment is documented in
`reports/TEST_external_data_futures_financial_assessment_20260726.md`.
Both F and G Quick gates failed; the report and referenced checkpoints are
research-only `TEST` evidence, not promotion inputs.

The H-series follow-up causal and data-quality audit is stored under
`runs/TEST_H_single_week_stability_20260726/`. Its executed notebook,
machine-readable audit, and reader-facing report verify source-manifest hashes
and block H2 (industry/futures residual), H3 (financial-event residual), and H4
(corporate-event residual) from training. Only the D01 control H0 and
loss-only H1 branch are eligible for the preregistered Quick stage.

That Quick is now terminal: H1 was `REJECTED_AT_QUICK_GATE` after 16 nested
outer runs and 98 paired five-trading-day TOP5_EQUAL decisions. Serious and
sealed evaluation did not run. See `TEST_H_FINAL_REPORT.md` in the run
directory for the decision, diagnostics, invariants, and protection warning.

The 2026-07-27 H5 follow-up is stored under
`runs/TEST_H5_predicted_boundary_20260727/`. It changed only H1's boundary
negative selection, passed CUDA smoke and cache-equivalence checks, then was
`REJECTED_AT_SENTINEL` on seed 20260726 with the exposed 2025Q2 and 2026Q1
outer windows. Full Quick did not run because three sentinel gates failed and
the exposure registry could not prove four unconsumed outer windows. The run
contains the pre-registration, H0 reuse audit, validation-exposure registry,
tests/coverage, cache manifest, per-file protection comparison, leakage audit,
and final reader report.

## H5 two-week retrospective exposed diagnostic

`TEST_H5_TWO_WEEK_RETROSPECTIVE_EXPOSED_DIAGNOSTIC` is stored under
`runs/TEST_H5_two_week_retrospective_20260727/`; its reader report is
`reports/TEST_H5_two_week_retrospective_20260727.md`. The controller and
one-shot scorer are:

```text
Experimental_WorkSpace/candidates_H/code/run_h5_two_week_retrospective.py
Evaluation/scripts/walk_forward/score_test_h5_two_week_retrospective.py
```

The run used the exact W1/W2 training CSV bytes from the completed
three-candidate rolling test, fixed seed 20260726 and final epoch 1, and locked
two byte-identical inference passes per round before one hash-verified
validation parse. Its 10 bps/side two-week compound result was -12.0785%.
This is a `RETROSPECTIVE_EXPOSED_DIAGNOSTIC` with status `DIAGNOSTIC_ONLY`,
not independent holdout or promotion evidence.

## Candidate-D prospective 8+4-window stability test

`TEST_D_PROSPECTIVE_8W_20260727` is the preregistered prospective follow-up to
the exposed two-week diagnostic. Its machine-readable protocol is
`configs/walk_forward/TEST_D_PROSPECTIVE_8W_20260727.json`, and its controller
is `scripts/walk_forward/prospective_d_stability.py`.

The completed July windows and the already elapsed 2026-07-20 through
2026-07-24 period are excluded. P01 starts with the 2026-07-31 decision cutoff.
The protocol requires at least eight new valid windows and has four
preregistered contingency windows. Training and inference cannot receive
outcome paths, prediction locks must precede the buy open, later-window rows
are rejected by both training and scoring date gates, and pre-finalization
public receipts contain hashes but no performance values.

Decision-date causal inputs are built by
`BaoStock_Datasets/scripts/build_prospective_causal_prefix.py`. Sell-date
outcomes and the same-period SSE Composite benchmark are independently built
by `BaoStock_Datasets/scripts/build_prospective_outcome_bundle.py` after the
09:31 scoring gate and are written only to the external sealed vault. The
machine-readable data chain and both frozen program hashes are recorded in
`configs/walk_forward/TEST_D_PROSPECTIVE_DATA_PLAN_20260727.json`.

Run evidence and prediction locks belong under
`Experimental_WorkSpace/candidates_D/automation_runs/TEST_D_PROSPECTIVE_8W_20260727/`.
This directory is `TEST` evidence only and never authorizes a candidate
promotion.

## Boundaries

- Future returns may be used for evaluation only, never for feature/scaler
  fitting, model training, parameter selection, or inference rules.
- A one-shot self-score is diagnostic, not generalization evidence.
- Report official and candidate scores, data path, validation range, and metric
  definition together.
- Do not modify `candidate/` or save checkpoints under `Evaluation/`.
- Remove caches and ephemeral logs after verification.

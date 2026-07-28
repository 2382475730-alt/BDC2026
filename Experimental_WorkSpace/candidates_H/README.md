# H-Series Single-Week Stability TEST

Updated: 2026-07-27

`candidates_H/` is the isolated TEST lineage for aligning training and
checkpoint-selection semantics with the competition's final five-trading-day,
at-most-five-stock output.

No retained numbered H-series model exists. Smoke, Quick, fold, and seed runs
use TEST run IDs. `H01_...` may be allocated only after every frozen data,
leakage, Quick, Serious, same-protocol baseline, reproducibility, runtime, and
submission-eligibility gate passes and the user explicitly authorizes
promotion.

## Primary Change

Relative to D01, this lineage may change the loss and nested checkpoint
selection:

- the portfolio-return surrogate has a hard top-5 forward pass;
- downside/CVaR and positive-tail soft-cap terms are explicit;
- inner development windows select epochs;
- outer folds are evaluated once;
- seed agreement is reported and never uses future labels at inference.

External data is not assumed valid. The industry/futures, financial-event, and
corporate-event branches remain blocked until their branch-specific data gates
pass.

## Terminal TEST Result

The 2026-07-26 H1 loss-only branch was `REJECTED_AT_QUICK_GATE` after the full
2-seed, 4-fold nested Quick. It failed fold consistency, cross-seed direction,
strict paired win rate, and leave-best-two robustness. Serious, policy lock,
sealed evaluation, and promotion were not run. No `H01` exists.

The reader report and machine-readable evidence are under
`Evaluation/runs/TEST_H_single_week_stability_20260726/`.

The 2026-07-27 H5 follow-up
`TEST_H5_PREDICTED_BOUNDARY_OBJECTIVE_V2` changed only H1's boundary-negative
selection to predicted-boundary hard negatives. Its CUDA smoke passed, but it
was `REJECTED_AT_SENTINEL` on seed 20260726 and the exposed 2025Q2/2026Q1
rejection screens. Decision change was non-degenerate (22 of 23 five-day
decisions), while changed-decision strict win rate was only 31.82%, CVaR20 was
worse than H0, and the normalized rank-5/rank-6 margin was worse than H1.
Full Quick, Serious, sealed evaluation, and promotion were not run.

The H5 report, validation-exposure registry, cache manifest, tests, and
task-relative per-file protection audit are under
`Evaluation/runs/TEST_H5_predicted_boundary_20260727/`.

## Two-week retrospective exposed diagnostic

`TEST_H5_TWO_WEEK_RETROSPECTIVE_EXPOSED_DIAGNOSTIC` ran the frozen H5
predicted-boundary objective on the same-byte W1/W2 training inputs previously
used by the three candidates. Both rounds used seed 20260726, a from-scratch
fixed final epoch 1, no validation/checkpoint selection/early stopping, no
external features, and strict `TOP5_EQUAL`. Each round's full 300-stock score
table and five-stock result were independently inferred twice and locked
before the isolated validation CSV was hash-verified and parsed once.

The observed gross returns were +5.8793% for W1 and -16.6035% for W2. The
two-week compounded return was -11.7004% gross and -12.0785% at 10 bps per
side, versus -6.4334% for `THU-BDC2026-main`, -5.6872% for Candidate-C,
+9.1129% for Candidate-D, and -2.2837% for HS300 equal weight under the same
cost convention. The result is permanently `RETROSPECTIVE_EXPOSED_DIAGNOSTIC`
and `DIAGNOSTIC_ONLY`; `promotion_evidence=false`.

The run, score, leakage audit, and report are under
`Evaluation/runs/TEST_H5_two_week_retrospective_20260727/` and
`Evaluation/reports/TEST_H5_two_week_retrospective_20260727.md`.

## Layout

```text
candidates_H/
  code/                 # isolated TEST implementation derived from D01/F code
  configs/              # frozen TEST configurations
  tests/                # objective, leakage, nesting, and policy tests
  automation_runs/      # TEST checkpoints, predictions, logs, and summaries
  H_PREREGISTRATION.md
  H_STAGE_PLAN.json
  H_experiment_registry.json
  TEST_STATUS.json
```

## Boundaries

- Use D01's self-contained market snapshot as the control data.
- Keep every new model, prediction, and report marked `TEST`.
- Do not write to `candidate/`, D01, F, G, or `BaoStock_Datasets/`.
- Keep 2025Q4 sealed. The range 2026-04-03..2026-07-03 has been permanently
  reclassified as exposed H-series training input and may never again be used
  as a holdout or promotion evidence.
- Treat 2026-07-06..2026-07-17 as an already exposed scoring window. W2 used
  raw rows only through 2026-07-10; all conclusions remain diagnostic.
- Do not train H2 with the 2026 static industry snapshot.
- Do not train H3 from latest-revision-only financial history.
- Do not open H4 without compliant point-in-time corporate-event data.
- Treat the existing 2025Q1/Q2/Q3 and 2026Q1 outer windows as exposed
  rejection screens. Do not use them as the sole promotion evidence.
- A future full Quick requires both a passing sentinel and at least four outer
  windows proven unconsumed by an authoritative pre-task exposure ledger.

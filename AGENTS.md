# Project Agents

This workspace is a BDC2026 stock-ranking competition workspace. The root is
not a submission package. Submission candidates live under `candidate/`,
experiments under `Experimental_WorkSpace/`, and shared evaluation code under
`Evaluation/`.

## Available Project Skills

- `$bdc2026-submission-steward`: submission cleanup, package inspection,
  manifests, entrypoints, and final validation.
- `$bdc2026-experiment-runner`: model iteration, rerank search, GPU training,
  self-score, walk-forward validation, and champion reports.

## Pre-Operation Orientation

Before any file operation, code edit, training/evaluation run, cleanup, or
packaging task:

- Read this `AGENTS.md`.
- Read `WORKSPACE_LAYOUT_20260705.md` when touching workspace layout, models,
  data, evaluation scripts, or candidate packages.
- For a large change, consult the official specifications under `docs/`,
  especially the code rules and task description PDFs.
- Read the local `README.md` before operating under `BaoStock_Datasets/`,
  `Experimental_WorkSpace/`, `Evaluation/`, or a candidate-family directory.
- Inspect the immediate children of a target before moving, deleting, copying,
  or generating files.
- Confirm whether the target belongs to data, experiment, evaluation, or
  submission scope.

Do not start destructive cleanup, model promotion, data replacement, or
submission packaging until the relevant purpose and boundaries are clear.

Large changes include training-program changes, model-family creation, data
replacement, evaluation-framework changes, submission-package changes,
dependency/environment changes, and workspace restructuring.

## Workspace Rules

- Keep `candidate/` packages clean; never use them for exploratory work.
- Run training and search in `Experimental_WorkSpace/`.
- Keep shared evaluation and reporting code in `Evaluation/`.
- Do not promote artifacts into `candidate/` unless the user explicitly asks.
- Treat package-local `data/test.csv` self-score as a one-shot diagnostic.
  Discuss generalization using walk-forward and robustness evidence.
- Prefer relative paths inside candidate packages and absolute paths in
  user-facing reports.
- Use `BaoStock_Datasets/` for local market data and data-source records; never
  write model artifacts there.
- Keep models with the same bottom-level training program in one family
  directory. Inference-only reranking or weighting changes may remain in that
  family; changes to model structure, loss, labels, features, datasets,
  optimizer schedule, or checkpoint selection require a new lineage/family.
- A retained research model uses
  `<series><two-digit-id>_<role-or-method>_<date>`. Smoke, Quick, seed, and fold
  runs use run IDs and do not consume model numbers.
- A submission identity uses `candidate/Candidate-<series>/`, with at most one
  current candidate per series. Dates and experiment IDs belong in manifests,
  not directory names.
- `candidate/THU-BDC2026-main/` is the protected official reference; its name
  and contents must not change.
- When a model uses the same data and validation set as the official reference,
  it must beat that reference under the same scoring protocol before being
  called an improvement. Report both scores, data path, validation range, and
  metric/script together. Secondary diagnostics cannot override a failed
  official-baseline comparison or robustness gate.

## Documentation Maintenance

When structure or operating boundaries change:

- Update `AGENTS.md`, `WORKSPACE_LAYOUT_20260705.md`, and the affected local
  `README.md` in the same task.
- Add a local `README.md` for every new candidate-family directory.
- Keep historical cleanup reports in `_cleanup_reports/` unchanged as
  provenance.
- After documentation edits, scan active docs for stale canonical paths and
  verify every current path exists.

## Current Identities

- Official reference: `candidate/THU-BDC2026-main/` (protected, unchanged).
- C submission fallback: `candidate/Candidate-C/`. The duplicate C0/C1/C2
  research directories were retired; their decisions remain in cleanup
  reports.
- D Alpha Champion: `D01_FrozenResearchChampion_20260709` at
  `Experimental_WorkSpace/candidates_D/D01_FrozenResearchChampion_20260709/`.
  Its legacy ID is `D02`; `RENAMING_ALIAS.json` preserves the mapping.
- D submission identity: `candidate/Candidate-D/`, built from D01's fixed
  12-epoch production refit. It is a packaging identity, not a new training
  program.
- D Defensive submission-format TEST identity:
  `candidate/Candidate-D-Defensive-TEST/`. It reuses Candidate-D's frozen
  model and applies the forced complete-link policy
  `EQ_V0.050_C0.150_T0.000_R0.100` with five equal weights. Its source V2
  experiment was `REJECTED_AT_COMBINATION_GATE`; the package exists only
  because the user explicitly requested a packaging exception on 2026-07-28.
  It is not a Defensive Champion and does not replace Candidate-D.
- I submission-format TEST identity: `candidate/Candidate-I/`. It reuses the
  Candidate-D production model and applies the fixed lag-1 residual rerank
  tuple `(0.05, 0.05, 0.05)` with top-5 equal weights. The source TEST was
  `REJECTED_AT_DATA_GATE`; Candidate-I exists only because the user explicitly
  requested a package-level exception on 2026-07-28. It does not replace
  Candidate-D and must not be described as a validated improvement.
- D Defensive Champion: `NONE`. A defensive result cannot replace the Alpha
  Champion through a defensive-position gate.
- E retained numbered model: `NONE`. E R1 and E Position R2 ended in rejection;
  only lightweight code, preregistration, audits, registry, and final reports
  remain.
- F retained numbered model: `NONE`. The industry + equity-index futures Quick
  test failed its 3/4-fold and recent-fold gates; its checkpoints are retained
  only as explicitly marked `TEST` evidence under
  `Experimental_WorkSpace/candidates_F/`.
- G retained numbered model: `NONE`. Point-in-time financial + industry +
  futures runs under `Experimental_WorkSpace/candidates_G/` are explicitly
  marked `TEST` and are not submission packages.
- H retained numbered model: `NONE`. H1 was rejected at its 2-seed, 4-fold
  Quick gate; H2/H3/H4 were blocked by their data gates. Serious and sealed
  evaluation were not run. All evidence under
  `Experimental_WorkSpace/candidates_H/` remains TEST-only, and `H01` does not
  exist.
- The next retained D-series research snapshot is `D02_...`; the first future
  E/F/G/H snapshot that passes its frozen promotion gates is respectively
  `E01_...`, `F01_...`, `G01_...`, or `H01_...`.

Detailed decisions for retired C/D/E models are in
`_cleanup_reports/retired_model_conclusions_20260720.md`.

## Data and Environment

- `BaoStock_Datasets/`: canonical local BaoStock datasets.
- `BaoStock_Datasets/data/fusion_research_20260726/`: research-only frozen
  industry, CFFEX equity-index futures, and point-in-time financial data with
  source manifests. Reporting-email eligibility is unverified.
- `Experimental_WorkSpace/candidates_D/D01_FrozenResearchChampion_20260709/data/`:
  self-contained D01 source snapshot and its provenance records.
- `Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/`: the single shared
  GPU environment. Do not recreate a duplicate root `.venv`.
- `Evaluation/`: centralized evaluation scripts, configs, small interpretable
  runs, and reports.

## Directory README Files

- `BaoStock_Datasets/README.md`
- `Experimental_WorkSpace/README.md`
- `Experimental_WorkSpace/candidates_C/README.md`
- `Experimental_WorkSpace/candidates_D/README.md`
- `Experimental_WorkSpace/candidates_E/README.md`
- `Experimental_WorkSpace/candidates_F/README.md`
- `Experimental_WorkSpace/candidates_G/README.md`
- `Experimental_WorkSpace/candidates_H/README.md`
- `Evaluation/README.md`

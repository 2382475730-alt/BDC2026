# Project Agents

This workspace is a BDC2026 stock-ranking competition workspace. The root is not itself the final submission package; submission candidates live under `candidate/`, experiments live under `Experimental_WorkSpace/`, and shared evaluation scripts live under `Evaluation/`.

## Available Project Skills

- `$bdc2026-submission-steward`: use for submission candidate cleanup, package inspection, manifest updates, entrypoint checks, and final validation.
- `$bdc2026-experiment-runner`: use for model iteration, rerank search, GPU training, self-score scoring, walk-forward validation, and champion reports.

## Pre-Operation Orientation

Before any file operation, code edit, training run, evaluation run, cleanup, or packaging task, first understand the relevant directory structure.

Required checks:

- Read this `AGENTS.md`.
- Read `WORKSPACE_LAYOUT_20260705.md` when the task touches workspace layout, model directories, data directories, evaluation scripts, or candidate packages.
- Before any large change, consult the official competition specifications under `docs/`, especially code rules, task description, and competition notice PDFs.
- Read the target directory's local `README.md` when operating under:
  - `BaoStock_Datasets/`
  - `Experimental_WorkSpace/`
  - `Evaluation/`
- Inspect the immediate child directories of the target path before moving, deleting, copying, or generating files.
- Confirm whether the target belongs to data, experiment, evaluation, or submission scope before acting.

Do not start destructive cleanup, model promotion, data replacement, or submission packaging until the relevant folder purpose and boundaries are clear.

Large changes include training-program changes, model-family creation, data replacement, evaluation framework changes, submission package changes, dependency/environment changes, and workspace restructuring.

## Workspace Rules

- Keep candidate submission directories clean.
- Run exploratory training and search in experiment directories, not in final candidates.
- Keep shared walk-forward, self-score, leakage-audit, and scoreboard scripts under `Evaluation/`.
- Do not promote model artifacts into `candidate/` unless the user explicitly asks.
- Treat local `data/test.csv` self-score as a one-shot diagnostic; use walk-forward reports to discuss generalization risk.
- Prefer relative paths inside candidate packages and absolute paths in user-facing reports.
- Use `BaoStock_Datasets/` for local market data and data-source records; do not write model artifacts there.
- Use `Experimental_WorkSpace/` for model experiments, retained C-series baselines, future D-series training-program variants, and shared experiment runtimes.
- Keep models with the same bottom-level training program in the same candidate-family folder, such as `Experimental_WorkSpace/candidates_C/`.
- When changing the bottom-level training program, create a separate family folder, such as `Experimental_WorkSpace/candidates_D/`.
- Within a candidate family, each candidate may keep its own reproducible package-local code copy under `code/src/`; the shared lineage is established by matching bottom-level training program files such as `train.py`, `model.py`, feature engineering, loss, and validation code.
- Inference-only changes such as `predict.py`, rerank config, weight mode, or regime gate do not by themselves require a new candidate-family folder.
- Changes to the bottom-level training program, model structure, loss, label construction, feature construction, dataset construction, optimizer schedule, or checkpoint-selection semantics require a new candidate-family folder.
- Use `Evaluation/` for shared evaluation scripts, configs, generated runs, and reports.
- Use `candidate/` only for clean submission candidates and official/reference packages.
- Do not modify `candidate/` packages during exploratory training or evaluation.
- For every model iteration, when using the same training data and validation set as the official baseline package at `candidate/THU-BDC2026-main/`, the candidate must beat that official baseline under the same scoring protocol before it can be treated as a valid improvement. Report the official-baseline score, candidate score, data path, validation range, and scoring script or metric definition together; do not claim progress from a model that only improves secondary diagnostics while failing this baseline comparison.

## Documentation Maintenance

When workspace structure, candidate-family rules, data-source rules, evaluation rules, or submission boundaries change, update documentation in the same task.

Required documentation updates:

- Update `AGENTS.md` when operating rules or required workflow behavior changes.
- Update `WORKSPACE_LAYOUT_20260705.md` when a top-level directory, candidate-family directory, or canonical path changes.
- Update the target directory `README.md` when changing the purpose, contents, or usage boundary of `BaoStock_Datasets/`, `Experimental_WorkSpace/`, `Evaluation/`, or a candidate-family folder such as `candidates_C/`.
- Add a local `README.md` for any new candidate-family folder, for example `Experimental_WorkSpace/candidates_D/README.md`.
- Keep historical cleanup reports in `_cleanup_reports/` as provenance; do not rewrite them only to remove old path names.
- After documentation changes, scan active docs for stale canonical paths and verify the new paths exist.

## Current Baselines

- `C0`: former `C4`, base Transformer model at `Experimental_WorkSpace/candidates_C/C0_BaseModel/`.
- `C1`: former `C77`, current champion regime-gated baseline at `Experimental_WorkSpace/candidates_C/C1_Champion_RegimeGate/`.
- `C2`: former `C302`, strict clean baseline at `Experimental_WorkSpace/candidates_C/C2_StrictClean_Baseline/`.
- `D001`: Market Encoder baseline at `Experimental_WorkSpace/candidates_D/D001_MarketEncoder_Baseline/`; round1 and R2 have been trained, but no D-series model has been promoted into `candidate/`.
- Future bottom-level training-program variants after D should continue in the D family when they share the D-series Market Intelligence lineage, or use a later family letter for a different lineage.

## Data Directories

- `BaoStock_Datasets/`: local BaoStock datasets.
- `Experimental_WorkSpace/`: retained C-series baselines, future D-series experiments, and shared experiment runtime.
- `Evaluation/`: centralized evaluation scripts, configs, generated runs, and reports.

## Directory README Files

- `BaoStock_Datasets/README.md`: data directory purpose, data-source boundaries, and current dataset notes.
- `Experimental_WorkSpace/README.md`: experiment workspace purpose, retained C-series baselines, future D-series placement, and shared environment notes.
- `Experimental_WorkSpace/candidates_C/README.md`: C-series candidate-family rules and bottom-level training-program lineage notes.
- `Experimental_WorkSpace/candidates_D/README.md`: D-series candidate-family rules, registry requirements, and D001 status.
- `Evaluation/README.md`: evaluation workspace purpose, script categories, output directories, and evaluation boundaries.

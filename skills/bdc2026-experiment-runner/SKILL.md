---
name: bdc2026-experiment-runner
description: Run BDC2026 model iteration experiments outside submission candidates, including GPU training, rerank searches, self-score scoring, BaoStock extended-data evaluation, frozen walk-forward validation, rolling retrain checks, and champion decision reports. Use when the user asks to improve model self-score, test generalization, analyze overfitting, compare branches, or continue quantitative stock-ranking experiments.
---

# BDC2026 Experiment Runner

## Core Rule

Run experiments in the experiment workspace, not in final candidate packages. Promote only the minimal selected artifacts after the user asks for promotion.

## Workflow

1. Read `WORKSPACE_LAYOUT_20260705.md` and `references/experiment-protocol.md`.
2. Create or reuse a timestamped folder under `Experimental_WorkSpace/`.
3. Copy candidate packages into isolated run directories before modifying config, data, model, or rerank files.
4. Record environment details before long runs: Python path, torch version, CUDA availability, GPU model, and key package versions.
5. For each candidate, save:
   - configuration and source candidate
   - training logs
   - `output/result.csv`
   - `temp/tmp.csv`
   - `candidate_summary.json`
   - `fold_predictions.csv` and `fold_scores.csv` when walk-forward is run
6. Rank candidates by the user-requested priority, but always disclose overfitting risk when local `data/test.csv` self-score conflicts with walk-forward evidence.

## Evaluation Defaults

- Local self-score is a one-shot weighted open-to-open score over the provided official-style `data/test.csv`.
- Walk-forward validation is a robustness diagnostic, not the official final score.
- Use future labels only for scoring. Do not use them to fit features, scalers, labels, rerank parameters, or stock selection rules.
- Prefer lightweight rerank and portfolio controls before changing the Transformer architecture.

## Boundaries

- Do not overwrite `candidate/` packages during search.
- Do not use hidden or future returns to construct inference rules.
- Do not leave failed candidates, caches, logs, or alternate models inside a submission candidate.
- Do not claim a candidate is robust based only on one official self-score window.

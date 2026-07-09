---
name: bdc2026-submission-steward
description: Maintain, inspect, validate, and package the BDC2026 stock-ranking submission candidates in this workspace. Use when the user asks to clean a submission folder, explain candidate package contents, update manifests, run init/train/test/score/validate entrypoints, compare candidate submission directories, or prepare a final package without polluting it with experiment artifacts.
---

# BDC2026 Submission Steward

## Core Rule

Treat submission candidates as reviewable package artifacts. Do not run exploratory training, broad rerank search, or temporary experiments inside a candidate submission directory unless the user explicitly asks for a candidate-local change.

## Workflow

1. Identify the target candidate directory. If the user does not specify one, inspect `candidate/` and ask or report the options before modifying anything.
2. Read `WORKSPACE_LAYOUT_20260705.md` and `references/submission-checklist.md` before making package decisions.
3. Keep edits scoped to required submission assets: `code/`, `data/`, `model/`, `output/`, `temp/`, shell entrypoints, dependency files, Docker files, and documentation.
4. Move or ignore development clutter; never silently delete user-created assets outside the requested target.
5. Validate with the candidate's own commands whenever possible:
   - `python -m compileall -q code`
   - `python code/src/test.py --work-dir . --output output/result.csv`
   - `python code/src/score_self.py --work-dir . --test-data data/test.csv --output output/result.csv`
   - `python code/src/validate_submission.py --app-dir . --write-manifest`
   - `python code/src/validate_submission.py --app-dir .`
6. Report exact file paths, key scores, and any validation gaps.

## Boundaries

- Do not treat the workspace root as a final submission package.
- Do not mix experiment outputs from `Experimental_WorkSpace/` into a candidate unless promoting a selected champion.
- Do not add network download behavior to official `train.sh` or `test.sh`.
- Preserve the official reference candidate separately from the current best candidate.

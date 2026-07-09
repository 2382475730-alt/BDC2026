# Submission Checklist

## Workspace Map

- `candidate/`: submission candidates and official reference copies.
- `candidate/C1_Submission_App_20260703/`: current cleaned candidate package, unless the user selects another target.
- `candidate/Improvement_THU_original_candidate_20260705/`: archived original candidate reference.
- `candidate/THU-BDC2026-main/`: official reference code copy.
- `Experimental_WorkSpace/`: experiment outputs and candidate training runs.
- `BaoStock_Datasets/`: BaoStock expanded data.
- `Evaluation/`: centralized walk-forward, self-score, leakage-audit, scoreboard, and reporting framework.
- `docs/`: competition PDFs.

## Clean Candidate Shape

A final candidate should contain only package-relevant assets:

- `code/`
- `data/train.csv`
- `data/test.csv`
- `model/`
- `output/result.csv`
- `temp/`
- `init.sh`, `train.sh`, `test.sh`
- `pyproject.toml`, `uv.lock`, Docker files when present
- `readme.md`, model notes, and manifest files required by the project

## Result Constraints

`output/result.csv` must have:

- columns exactly `stock_id,weight`
- 1 to 5 rows
- no duplicate stock IDs
- non-negative weights
- total weight less than or equal to 1

The current risk-controlled target is five stocks with max single-stock weight near 0.5 unless the user explicitly changes the rule.

## Validation Notes

Prefer running validation from the target candidate directory. If a command cannot run because dependencies are missing, report the exact dependency or environment gap instead of guessing.

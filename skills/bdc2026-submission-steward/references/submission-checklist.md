# Submission Checklist

## Workspace Map

- `candidate/THU-BDC2026-main/`: protected official reference; name and contents
  must remain unchanged.
- `candidate/Candidate-C/`: current C-series fallback submission package.
- `candidate/Candidate-D/`: current D-series submission candidate derived from
  D01's fixed production refit.
- `Experimental_WorkSpace/candidates_D/D01_FrozenResearchChampion_20260709/`:
  retained D Alpha Champion source; legacy ID D02.
- `Experimental_WorkSpace/`: research artifacts and shared runtime.
- `BaoStock_Datasets/`: canonical local market data.
- `Evaluation/`: shared evaluation code and small interpretable results.
- `docs/`: official competition specifications.

At most one current `Candidate-<series>` directory may exist per series. Do not
create a parallel `candidates/` submission root.

## Clean Candidate Shape

A current package should contain only package-relevant assets:

- `code/src/`
- `data/train.csv`
- `data/test.csv`
- `model/`
- `output/result.csv`
- empty `temp/`
- `init.sh`, `train.sh`, `test.sh`
- Docker and dependency files required by the package
- `readme.md`, `MODEL_CARD.md`, and `code/submission_manifest.json`

Exclude validation prediction matrices, ranked-score diagnostics, copied
automation runs, research reports, old checkpoints, stdout/stderr logs,
`__pycache__`, `.pyc`, absolute workspace paths, and network downloads.

## Entrypoints

- Paths must resolve relative to the package root.
- Training starts from `data/train.csv`, uses a fixed seed, and writes the
  package model artifacts.
- Inference reads package-local data/model artifacts and writes
  `output/result.csv`.
- The code must run offline.
- Official limits: training no more than 8 hours, inference no more than
  5 minutes, and uncompressed package/Docker content below 10 GB.

## Result Constraints

`output/result.csv` must have:

- columns exactly `stock_id,weight`
- 1 to 5 rows
- unique six-digit stock IDs
- non-negative weights
- total weight less than or equal to 1

## Validation

From the candidate root, run:

```powershell
python -m compileall -q code
python code/src/test.py --work-dir . --output output/result.csv
python code/src/score_self.py --work-dir . --test-data data/test.csv --output output/result.csv
python code/src/validate_submission.py --app-dir . --write-manifest
python code/src/validate_submission.py --app-dir .
```

For a newly constructed family package, also reproduce the fixed-seed training
when feasible and compare model state, scalers, and result hashes with the source
refit. Record timing, size, dependency/CUDA status, hashes, and any robustness
caveat.

Do not mark a package `SUBMISSION_CANDIDATE_READY` until all applicable checks
pass. Submission readiness does not by itself prove research improvement.

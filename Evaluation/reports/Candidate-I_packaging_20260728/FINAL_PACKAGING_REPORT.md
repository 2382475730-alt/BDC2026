# Candidate-I packaging report

## Outcome

`candidate/Candidate-I/` is a clean, independently validated BDC2026
submission-format package. It does not replace Candidate-D.

Package status: `SUBMISSION_FORMAT_READY`  
Research status: `REJECTED_AT_DATA_GATE`

The package uses Candidate-D's byte-identical production model and scaler, then
applies the fixed, non-outcome-selected residual-rerank tuple
`(momentum=0.05, volume=0.05, reversal=0.05)`. The final policy is top-5 equal
weight, fully invested, with HHI 0.20.

## Complete inference run

- Device: CUDA
- Decision date: 2026-07-03
- Factor as-of date: 2026-07-02
- Full model scores: 300
- Complete raw-factor rows: 299/300
- Selected incomplete-factor rows: 0/5
- Pass 1: 13.7769 seconds
- Pass 2: 13.4446 seconds
- Outputs byte-identical: yes
- `result.csv` SHA-256:
  `55f9b92fb8a4e4b0439aea4527db616dfb8d9ee3dd1c1605f780d9f42fc82251`

```csv
stock_id,weight
688126,0.2
002463,0.2
300442,0.2
301308,0.2
300433,0.2
```

## Official package checks

All package-local checks passed:

- required directories and entrypoints;
- required README sections;
- offline review scripts;
- official `output/result.csv` path;
- exact `stock_id,weight` schema;
- five unique six-digit stocks;
- non-negative weights summing to 1;
- source manifest;
- package size: 56,176,307 bytes, below 10GB;
- inference time below 5 minutes;
- no `__pycache__`, `.pyc`, or absolute workspace paths.

The full residual-rerank test suite passed 32/32 tests. The new package rerank
module has 85% line coverage.

Candidate-D, D01, and `THU-BDC2026-main` remained byte-for-byte unchanged.

## Important limitation

The source research TEST stopped before calibration or outer outcome access
because its data gate failed. The symmetric lambda is an explicit
non-outcome packaging choice made to satisfy the user's request, not a
performance-selected parameter.

Therefore submission-format readiness must not be described as:

- evidence that Candidate-I improves Candidate-D;
- evidence that Candidate-I exceeds the official baseline;
- research promotion or a retained champion;
- a guarantee of future or leaderboard returns.


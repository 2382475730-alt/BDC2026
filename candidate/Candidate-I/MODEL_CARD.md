# Candidate-I Model Card

## Identity

- Submission directory: `candidate/Candidate-I`
- Base model: Candidate-D fixed 12-epoch production refit
- Rerank source: `TEST_D_ALPHA_RESIDUAL_RERANK_20260728`
- Research status: `REJECTED_AT_DATA_GATE`
- Package status: `SUBMISSION_FORMAT_READY / RESEARCH_REJECTED`
- Replacement status: does not replace Candidate-D

## Frozen Inference Contract

- Base-model seed: `20260708`
- Base-model epochs: `12`
- Rerank lambdas: momentum `0.05`, volume `0.05`, reversal `0.05`
- Lambda rule: smallest symmetric nonzero tuple in the preregistered grid
- Lambda selected from outcomes: no
- Factor availability: strictly earlier than the decision date
- Portfolio: top-5 equal, fully invested
- HHI: `0.20`

## Source Hashes

- Training data SHA-256:
  `8e9cf37e5c53e4edafddc3f6f6b5843d4037376334e6f53cba068699095df1ad`
- Base production model SHA-256:
  `bd85f504599d1995bd19f254c78136be1cf42e74a349f32ed6bdec5ccb0ef86d`
- Base production scalers SHA-256:
  `3d647f554d5e09c50e56488931deabd54ff08750edb645f265ff91d00bc62b38`

## Intended Use

Offline BDC2026 submission-format reproduction of a deterministic Candidate-D
ranking plus lag-1 residual rerank. It writes a constrained
`stock_id,weight` portfolio.

## Research Limitation

The source TEST stopped before calibration because eight Candidate-D top-10
seed/fold/date/stock rows lacked complete volume and amount confirmation.
Calibration and outer outcomes were not read. The present fixed lambda is a
non-outcome packaging choice authorized by the user, not a validated
hyperparameter selection.

Missing raw confirmation components receive the frozen implementation's
cross-sectional neutral rank. This makes inference total and deterministic but
does not cure the failed research gate.

## Claims

- Submission structure and deterministic output may be validated.
- No claim is made that Candidate-I improves Candidate-D or the official
  baseline.
- No claim is made about future returns or leaderboard performance.

## Package Validation

- CUDA inference pass 1: 13.7769 seconds
- CUDA inference pass 2: 13.4446 seconds
- Full base-model score count: 300
- Factor as-of date for the packaged output: 2026-07-02
- Complete raw-factor rows: 299/300
- Selected stocks with incomplete raw factors: 0/5
- Output rows: 5
- Weight sum: 1.0
- HHI: 0.20
- Double-run output: byte-identical
- `output/result.csv` SHA-256:
  `55f9b92fb8a4e4b0439aea4527db616dfb8d9ee3dd1c1605f780d9f42fc82251`
- Submission layout validator: PASS
- Unit tests: 32/32 PASS
- Residual-rerank line coverage: 85%

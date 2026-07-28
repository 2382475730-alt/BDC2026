# Candidate-D Defensive TEST Model Card

## Identity

- Package: `candidate/Candidate-D-Defensive-TEST`
- Base research snapshot: `D01_FrozenResearchChampion_20260709`
- Base submission identity: `candidate/Candidate-D` (unchanged)
- Defensive policy: `EQ_V0.050_C0.150_T0.000_R0.100`
- Status: `FORCED_TEST_PACKAGE_NOT_RESEARCH_GATE_APPROVED`

## Frozen Model

- Seed: `20260708`
- Epochs: `12`
- Training: fixed final epoch, no validation selection or new training
- Train sample window: `2022-01-04..2026-06-26`
- Production model SHA-256:
  `bd85f504599d1995bd19f254c78136be1cf42e74a349f32ed6bdec5ccb0ef86d`
- Scalers SHA-256:
  `3d647f554d5e09c50e56488931deabd54ff08750edb645f265ff91d00bc62b38`

## Portfolio Layer

The frozen Candidate-D rank scores are reranked with lag-1 volatility, reversal
risk, and complete-link return-correlation clusters. The first feasible pool in
the frozen sequence top-20, top-50, top-100, full ranking is used. The result is
five equal-weight stocks, fully invested, with at most two stocks per cluster.

One-shot submission inference has no prior locked portfolio, so turnover is zero
and the frozen policy's turnover lambda is also zero. The market-regime
multiplier uses the predeclared neutral fallback of 1.0; top-k and the weight
template never change.

## Research Evidence

The V2 feasibility gate passed 192/192 exposed contexts, but no policy passed
both-seed noninferiority. The source experiment therefore stopped at
`REJECTED_AT_COMBINATION_GATE` without an outer score. This package was created
only after an explicit request to force a submission-shaped artifact and must
not be represented as a promoted Defensive Champion.

## Intended Use

Offline reproduction, structural submission validation, and explicit
user-directed experimental submission. It is not approved to replace the Alpha
Champion or the existing Candidate-D package.

## Limitations

The selected policy is post-gate and failed the preregistered two-seed
noninferiority requirement. Financial returns are non-stationary; neither local
format validity nor deterministic output implies future or leaderboard
performance.

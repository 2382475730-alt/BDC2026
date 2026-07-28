# D01 Frozen Research Champion

Updated: 2026-07-20

## Identity

- Canonical snapshot: `D01_FrozenResearchChampion_20260709`
- Legacy snapshot: `D02_FrozenResearchChampion_20260709`
- Source program: `D_AUTO_CHAMPION_REF_20260709_110423`
- Role: retained D-series Alpha Research Champion
- Defensive Champion: `NONE`
- Submission package: no
- Submission identity derived from this snapshot: `candidate/Candidate-D/`

The renumbering is an identity cleanup, not a training-program change.
`RENAMING_ALIAS.json` is the authoritative mapping.

## Program

- Feature version: `d_market_v1`
- Label: `open(T+5) / open(T+1) - 1`
- Research train: `2022-01-04` through `2024-12-25`, purged before validation
- Research validation: `2025-01-03` through `2026-01-05`
- Seed: `20260708`
- Best/fixed production epoch: `12`
- Portfolio: Top-2 equal weight

The architecture combines stock-temporal encoding, an explicit market encoder,
market-conditioned cross-stock fusion, ranking output, and a portfolio head.

## Recorded Evidence

| Metric | D01 | Official proxy |
| --- | ---: | ---: |
| Mean portfolio return, same recorded protocol | 0.0371198358 | 0.0125284993 |
| Validation samples | 243 | 243 |

The mean-return check passed, but the broader research Champion Gate was
`REJECT`: robustness, multi-seed, and cost checks did not pass. D01 is the
retained incumbent after cleanup; this evidence must not be presented as a new
fully accepted research improvement.

## Reproduction and Packaging

The production refit uses a fixed seed, 12 epochs, no validation set, no early
stopping, and no hyperparameter search. Candidate-D reproduction matched all 59
model-state tensors exactly and reproduced the scaler and result hashes.

Canonical movable inputs:

- `D01_ACTIVE_CONFIG.json`
- `data/train.csv`
- `code_snapshot/src/`
- `production_refit_20260703/`

`D02_SNAPSHOT_MANIFEST.json` and completed reports remain unmodified historical
provenance and therefore retain the legacy ID and original paths. Use
`D01_SNAPSHOT_MANIFEST.json` for the current canonical view.

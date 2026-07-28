# Candidate-D Model Card

## Identity

- Submission directory: `candidate/Candidate-D`
- Research snapshot: `D01_FrozenResearchChampion_20260709`
- Legacy research ID: `D02_FrozenResearchChampion_20260709`
- Source program: `D_AUTO_CHAMPION_REF_20260709_110423`
- Status: `SUBMISSION_CANDIDATE_READY`

## Frozen Training Contract

- Seed: `20260708`
- Epochs: `12`
- Training mode: fixed final epoch, no validation set, no hyperparameter search
- Train sample window: `2022-01-04..2026-06-26`
- Label end: no later than `2026-07-03`
- Portfolio: Top-2 equal weight

## Source Hashes

- Training data SHA-256: `8e9cf37e5c53e4edafddc3f6f6b5843d4037376334e6f53cba068699095df1ad`
- Production model SHA-256: `bd85f504599d1995bd19f254c78136be1cf42e74a349f32ed6bdec5ccb0ef86d`
- Production scalers SHA-256: `3d647f554d5e09c50e56488931deabd54ff08750edb645f265ff91d00bc62b38`

## Intended Use

BDC2026 offline reproduction and submission review. The model ranks the latest available CSI 300 cross-section and writes a constrained `stock_id,weight` portfolio.

## Limitations

Financial returns are non-stationary. Research-gate evidence and one-shot local diagnostics do not guarantee future or official leaderboard performance. Candidate readiness requires package-level reproducibility and format validation.

## Package Validation

- Full fixed-seed reproduction: 12 epochs in 3.782 minutes on CUDA
- Source/reproduced model tensors: 59/59 exactly equal
- Maximum absolute parameter difference: 0
- Model-state SHA-256: `c22f866f4753d9deabdc900ad598a326497350b5b19c33972e25145dc3e87830`
- Source/reproduced `result.csv`: byte-identical
- Reproduced-checkpoint inference: 23.4 seconds
- Validation record: workspace `_cleanup_reports/candidate_validation_20260720.json`

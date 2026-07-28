# E-Series Family

Updated: 2026-07-20

The E-series workspace is a lightweight record of completed, rejected research.
There is no retained E checkpoint, numbered E model, production refit, or
submission package.

## Terminal Decisions

- E R1 Recent Regime Mixture:
  `REJECT_E_ALPHA_CHAMPION`. All six preregistered Quick variants failed at
  least one frozen gate; no Serious run, E4, finalist, Production Refit, or E01
  was created. The 2026-04-07 through 2026-07-03 Recent Holdout remained
  unopened.
- E Position R2:
  `NO_FIVE_STOCK_POLICY_EXCEEDS_E2`. The best mean-return reference failed the
  PDR, 2026Q1, without-best-two-weeks, T+5-retention, and official same-protocol
  evidence gates. No E02 was created.

D01 remains the D Alpha Champion; Defensive Champion remains `NONE`.

## Retained Contents

- lightweight model/protocol code and configs under
  `E001_RecentRegimeMixture_Ranker/`
- preregistration files and hashes
- manifests
- final Chinese reports
- concise `E_experiment_registry.json`
- root-level audits, summaries, and reference tables under `automation_runs/`

All failed weights, completed run subdirectories, large prediction matrices,
logs, caches, copied source trees, and the derived E data snapshot were removed.
The retained controllers are historical protocol implementations, not turnkey
entrypoints after artifact cleanup.

## Future Numbering

Smoke, Quick, seed, and fold runs use run IDs and do not consume formal model
numbers. A future E challenger receives `E01_...` only after its code, config,
seed, policy, hashes, and frozen gates are locked and passed. No E model may be
promoted into `candidate/` without a separate explicit packaging decision.

Detailed retirement evidence is in
`_cleanup_reports/retired_model_conclusions_20260720.md`.

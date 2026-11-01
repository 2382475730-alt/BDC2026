# D-Series Family

Updated: 2026-07-28

`candidates_D/` is the Market Intelligence research lineage. It contains one
retained research snapshot, lightweight protocol history, and reusable
evaluation controllers.

## Current Identities

- Alpha Champion:
  `D01_FrozenResearchChampion_20260709/`
- Legacy identity:
  `D02_FrozenResearchChampion_20260709`
- Submission identity:
  `candidate/Candidate-D/`
- Submission-format TEST identity:
  `candidate/Candidate-I/` (user-authorized packaging exception;
  `RESEARCH_REJECTED`, does not replace Candidate-D)
- Defensive Champion:
  `NONE`
- Next retained research ID:
  `D02_...`

`RENAMING_ALIAS.json` records the D02-to-D01 mapping. Candidate-D is built from
D01's fixed 12-epoch production refit; it is not a new training program or a
second research champion.

## D01 Contents

```text
D01_FrozenResearchChampion_20260709/
├─ code_snapshot/                 # frozen research implementation
├─ data/                          # source train snapshot + provenance
├─ production_refit_20260703/     # fixed-epoch refit source
├─ D01_SNAPSHOT_MANIFEST.json
├─ RENAMING_ALIAS.json
├─ MODEL_CARD.md
└─ compact research evidence
```

The canonical training snapshot is
`D01_FrozenResearchChampion_20260709/data/train.csv`. Active D01 configs use
relative paths so the snapshot can be moved as a unit.

## Automation and Controllers

`automation_runs/` contains only root-level summaries, state, split manifests,
audits, gate decisions, and final reports. Completed run subdirectories,
checkpoints, prediction matrices, logs, and copied source trees were retired.

The retained `run_d_*.py` files are protocol-history/controllers from completed
rounds. Some locked protocols intentionally refer to deleted run-level evidence
or legacy IDs and are not turnkey current entrypoints. For a new iteration:

1. use D01's `code_snapshot/src/` and `data/train.csv` as the explicit source;
2. assign run IDs to smoke/quick/fold/seed work;
3. write new run artifacts under `automation_runs/`;
4. allocate `D02_...` only if a challenger passes all frozen gates and is worth
   long-term retention.

Shared validation helpers remain in `Evaluation/scripts/d_series/`.

## Evidence and Promotion Rules

D01's same-protocol mean-return comparison exceeded the official proxy on the
recorded 2025-01-03 through 2026-01-05 validation range. However, the broader
research Champion Gate was `REJECT` because robustness, seed, and cost checks
failed. This caveat must accompany any baseline claim.

Candidate-D separately passed packaging, fixed-seed reproduction, inference,
format, offline-review, and size validation. Submission readiness does not erase
the research robustness caveat.

Candidate-I packages the 2026-07-28 inference-only residual-rerank TEST using
the non-outcome-selected symmetric tuple `(0.05, 0.05, 0.05)` and top-5 equal
weights. The source TEST stopped at its data gate before calibration or outer
outcome access. Candidate-I is structurally reviewable because the user
explicitly requested the package, but it is not a retained D champion, D02, or
validated improvement.

The prospective stability protocol
`automation_runs/TEST_D_PROSPECTIVE_8W_20260727/` was preregistered on
2026-07-27. It evaluates Candidate-D against the protected official reference
over eight new independent weekly windows, with four frozen contingency
windows. The run uses isolated ordinary copies, from-scratch training,
pre-buy prediction locks, external sealed scoring, and a hash-chained exposure
audit. It is diagnostic `TEST` evidence and does not modify Candidate-D, D01,
or either champion identity.

The D Alpha Champion and Defensive Champion are separate identities. A
defensive-position result cannot replace the Alpha Champion. Defensive Champion
remains `NONE`.

## Retired Models

`D001`, former `D03`, and former `D04` were removed after their conclusions and
hashes were preserved in:

- `D_experiment_registry.json`
- `_cleanup_reports/retired_model_conclusions_20260720.md`
- `_cleanup_reports/workspace_cleanup_renumber_20260717.json`

Historical reports keep their original IDs and are not bulk-renamed.

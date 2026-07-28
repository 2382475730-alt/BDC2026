# Experimental Workspace

Updated: 2026-07-26

`Experimental_WorkSpace/` contains research lineages, retained protocol
evidence, controllers, and the single shared GPU environment. It is not a
submission directory.

## Current Layout

```text
Experimental_WorkSpace/
├─ candidates_C/
│  └─ README.md
├─ candidates_D/
│  ├─ D01_FrozenResearchChampion_20260709/
│  ├─ automation_runs/
│  ├─ D_experiment_registry.json
│  ├─ README.md
│  └─ run_d_*.py
├─ candidates_E/
│  ├─ E001_RecentRegimeMixture_Ranker/
│  ├─ automation_runs/
│  ├─ manifests/
│  ├─ E_experiment_registry.json
│  ├─ preregistration and final reports
│  └─ README.md
├─ candidates_F/
│  ├─ code/
│  ├─ automation_runs/
│  ├─ F_PREREGISTRATION.md
│  ├─ TEST_STATUS.json
│  └─ README.md
├─ candidates_G/
│  ├─ code/
│  ├─ automation_runs/
│  ├─ G_PREREGISTRATION.md
│  └─ README.md
├─ candidates_H/
│  ├─ configs/
│  ├─ H_PREREGISTRATION.md
│  ├─ H_STAGE_PLAN.json
│  ├─ TEST_STATUS.json
│  └─ README.md
├─ shared_envs/
│  └─ pytorch-cu128-py311/
└─ PYTORCH_ENVIRONMENT.md
```

## Current Identities

- C-family experiment copies were retired. The surviving submission fallback is
  `candidate/Candidate-C/`.
- D Alpha Champion is `D01_FrozenResearchChampion_20260709` (legacy ID `D02`).
  Its fixed production refit is the source of `candidate/Candidate-D/`.
- D Defensive Champion is `NONE`.
- E has no retained numbered model. E R1 and E Position R2 were rejected; only
  lightweight code, preregistration, audits, registry, and final reports remain.
- F has no retained numbered model. Its industry + futures Quick experiment
  failed 2/4 and recent-fold gates; outputs are marked `TEST`.
- G has no retained numbered model. Its point-in-time financial + industry +
  futures work is isolated and every output is marked `TEST`.
- H has no retained numbered model. Its single-week-stability objective and
  nested-selection work is isolated and every output is marked `TEST`.
- The next retained D snapshot is `D02_...`; future E/F/G models earn
  `E01_...`, `F01_...`, or `G01_...` only after all frozen gates pass. H work
  may earn `H01_...` only after all frozen gates pass and the user approves.
  The 2026-07-26 H1 Quick failed, so no H model was retained.

## Boundaries

- Train, search, and generate experiment artifacts here, never in `candidate/`.
- Keep shared validation logic under `Evaluation/`.
- Keep canonical market data under `BaoStock_Datasets/`.
- D01's required source snapshot is self-contained under
  `candidates_D/D01_FrozenResearchChampion_20260709/data/`.
- Completed run directories, repeated checkpoints, prediction matrices, logs,
  caches, and copied source trees are not long-term artifacts.
- `automation_runs/` keeps only root-level summaries, state, audits, split
  manifests, decisions, and final reports.
- Historical controllers and locked reports may mention legacy IDs. Consult the
  family README before reusing a controller; do not treat deleted run paths as
  current inputs.

## Environment

Use:

```text
Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/
```

The duplicate root `.venv` was removed after dependency and CUDA verification.
See `PYTORCH_ENVIRONMENT.md`.

## Promotion

Research promotion is explicit. A candidate package must use package-relative
paths, run offline, satisfy official time/size/result constraints, and pass its
validator before receiving `SUBMISSION_CANDIDATE_READY`. A submission package
is a packaging identity; it does not create a new training-program lineage.

Cleanup provenance is recorded under `_cleanup_reports/`.

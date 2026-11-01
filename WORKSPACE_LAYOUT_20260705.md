# Improvement-THU Workspace Layout

Updated: 2026-07-28

The workspace root is:

```text
D:\Study\Competitions\2026年中国高校计算机大赛—大数据挑战赛-提交版本\Improvement-THU
```

The root is a research workspace, not a final submission package.

## Canonical Layout

```text
Improvement-THU/
├─ BaoStock_Datasets/                  # canonical local market data
├─ Experimental_WorkSpace/
│  ├─ candidates_C/
│  │  └─ README.md                    # C-family history only
│  ├─ candidates_D/
│  │  ├─ D01_FrozenResearchChampion_20260709/
│  │  │  ├─ code_snapshot/
│  │  │  ├─ data/
│  │  │  └─ production_refit_20260703/
│  │  ├─ automation_runs/             # root-level reports/state/audits only
│  │  ├─ D_experiment_registry.json
│  │  ├─ README.md
│  │  └─ run_d_*.py                   # retained protocol controllers
│  ├─ candidates_E/
│  │  ├─ E001_RecentRegimeMixture_Ranker/  # lightweight code/config only
│  │  ├─ automation_runs/             # root-level audits/summaries only
│  │  ├─ manifests/
│  │  ├─ E_experiment_registry.json
│  │  ├─ preregistration and final reports
│  │  └─ README.md
│  ├─ candidates_F/                    # TEST industry + index-futures evidence
│  │  ├─ code/
│  │  ├─ automation_runs/
│  │  ├─ preregistration/stage plan
│  │  ├─ TEST_STATUS.json
│  │  └─ README.md
│  ├─ candidates_G/                    # TEST point-in-time financial fusion
│  │  ├─ code/
│  │  ├─ automation_runs/
│  │  ├─ preregistration/stage plan
│  │  └─ README.md
│  ├─ candidates_H/                    # TEST single-week stability lineage
│  │  ├─ configs/
│  │  ├─ preregistration/stage plan
│  │  ├─ TEST_STATUS.json
│  │  └─ README.md
│  └─ shared_envs/
│     └─ pytorch-cu128-py311/         # only retained GPU environment
├─ Evaluation/                        # shared evaluation code and reports
├─ candidate/
│  ├─ THU-BDC2026-main/               # protected official reference
│  ├─ Candidate-C/                    # C-family fallback submission
│  ├─ Candidate-D/                    # D-family current submission candidate
│  └─ Candidate-I/                    # submission-format TEST; research rejected
├─ docs/                              # official competition specifications
├─ skills/
└─ _cleanup_reports/                  # immutable history + current cleanup audit
```

There is no canonical root `.venv`; use the shared environment under
`Experimental_WorkSpace/shared_envs/`.

The candidate tree also contains `candidate/Candidate-D-Defensive-TEST/`, a
forced submission-format TEST package whose V2 source was rejected at the
combination gate. It does not replace `candidate/Candidate-D/`.

## Model Identities

Research snapshots and submission packages have separate names:

- Research: `<series><two-digit-id>_<role-or-method>_<date>`.
- Submission: `candidate/Candidate-<series>/`.

Current identities:

- `THU-BDC2026-main`: official reference; name and contents unchanged.
- `Candidate-C`: C-series fallback submission. Former experiment-local C0/C1/C2
  copies were retired.
- `D01_FrozenResearchChampion_20260709`: current D Alpha Champion. Legacy ID:
  `D02_FrozenResearchChampion_20260709`.
- `Candidate-D`: submission package derived from D01's fixed production refit;
  it does not define a new training program.
- `Candidate-D-Defensive-TEST`: user-authorized submission-format TEST package
  derived from Candidate-D plus the forced complete-link defensive policy
  `EQ_V0.050_C0.150_T0.000_R0.100`. Its source V2 experiment was rejected at
  the combination gate. It is not a Defensive Champion, does not replace
  Candidate-D, and carries no improvement or promotion claim.
- `Candidate-I`: user-authorized submission-format TEST package derived from
  Candidate-D plus the fixed lag-1 residual rerank tuple
  `(0.05, 0.05, 0.05)`. Its source TEST was rejected at the data gate. It does
  not replace Candidate-D and carries no improvement or promotion claim.
- Defensive Champion: `NONE`.
- E retained numbered model: `NONE`.
- F retained numbered model: `NONE`; its four-fold industry + futures run is
  rejected and retained only as `TEST` evidence.
- G retained numbered model: `NONE`; all financial-fusion outputs are marked
  `TEST` while reporting eligibility and robustness remain unresolved.
- H retained numbered model: `NONE`; the loss-only H1 branch was rejected at
  Quick, H2/H3/H4 were blocked by data gates, and Serious/sealed evaluation did
  not run. All H artifacts remain TEST-only.

The next retained D research model starts at `D02_...`. Smoke/Quick/fold/seed
runs use run IDs and do not consume formal numbers. Future E/F/G/H snapshots
start at `E01_...`, `F01_...`, `G01_...`, or `H01_...` only after passing their
frozen promotion gates and receiving explicit promotion approval.

## Scope Boundaries

- `candidate/` contains clean submission/reference packages only.
- `Experimental_WorkSpace/` contains retained research lineage, controllers,
  registries, and the shared runtime.
- `Evaluation/` contains shared code; it does not own model weights.
- `BaoStock_Datasets/` contains data and provenance, never model artifacts.
- Research-only auxiliary data is frozen under
  `BaoStock_Datasets/data/fusion_research_20260726/`; it is not automatically
  submission-eligible.
- D01 is self-contained at
  `Experimental_WorkSpace/candidates_D/D01_FrozenResearchChampion_20260709/data/train.csv`.
- Historical reports may retain legacy IDs. Active docs, registries, configs,
  and canonical paths use the new identities.

## Evaluation

Reusable categories live under:

```text
Evaluation/scripts/d_series/
Evaluation/scripts/walk_forward/
Evaluation/scripts/reporting/
Evaluation/scripts/self_score/
Evaluation/scripts/leakage_audit/
Evaluation/scripts/scoreboard/
```

Keep only small, valid, interpretable outputs under `Evaluation/runs/` and
`Evaluation/reports/`.

## Cleanup Provenance

- Pre-cleanup manifest:
  `_cleanup_reports/workspace_cleanup_renumber_20260717.json`
- Execution log:
  `_cleanup_reports/workspace_cleanup_execution_20260720.json`
- Retired-model conclusions:
  `_cleanup_reports/retired_model_conclusions_20260720.md`
- Candidate validation:
  `_cleanup_reports/candidate_validation_20260720.json`

Do not rewrite older reports merely to replace historical path names.

## Operating Rules

- Train and search only in `Experimental_WorkSpace/`.
- Do not modify `candidate/THU-BDC2026-main/`.
- Do not promote a failed model into `candidate/`.
  `Candidate-D-Defensive-TEST` and `Candidate-I` are documented 2026-07-28
  user-authorized packaging exceptions and remain research-rejected TEST
  identities, not promotion precedents.
- Use the same-protocol official-baseline comparison plus robustness evidence
  before claiming an improvement.
- Update this file and the affected local README whenever canonical structure
  changes.

## Overnight feature research TEST (2026-11-01)

`Experimental_WorkSpace/candidates_D/TEST_overnight_gap_20261101/` is an
untrained research-only lineage replacing D01's duplicate stock feature with
an overnight return. It requires new scalers and weights; no candidate or
champion has been promoted or modified.

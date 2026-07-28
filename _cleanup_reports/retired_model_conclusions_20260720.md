# Retired Model Conclusions - 2026-07-20

This lightweight record preserves the decision-level evidence for model artifacts removed by `workspace_cleanup_renumber_20260717.json`. It does not replace the immutable pre-cleanup manifest.

## Retained identities

- Official reference: `candidate/THU-BDC2026-main` (name and content protected).
- C-family submission: legacy `C1_Submission_App_20260703` -> `candidate/Candidate-C`.
- D-family research champion: legacy `D02_FrozenResearchChampion_20260709` -> `D01_FrozenResearchChampion_20260709`.
- D-family submission: `candidate/Candidate-D`, derived from the D01 fixed 12-epoch production refit.
- D-family defensive champion: `NONE`.

## Retired D-series snapshots

### D001_MarketEncoder_Baseline

- Disposition: superseded baseline package; remove after Candidate-D completes package validation and D01 is self-contained.
- Pre-cleanup size: 121,303,702 bytes.
- Reason: D01 preserves the selected Market Intelligence lineage, code, production checkpoint, scalers, configuration, evidence, and training data needed for future iteration. Keeping D001 would duplicate the same lineage without being the retained champion.

### D03_R5_L1_F0_ResearchCandidate_20260711

- Disposition: remove the complete numbered snapshot; future retained D-series work resumes at D02 after renumbering.
- Pre-cleanup size: 37,121,722 bytes.
- Original designation: `RESEARCH_CANDIDATE_ONLY`.
- Gate result: `STOP_AFTER_QUICK_NO_QUALIFIED_CHALLENGER`; `quick_to_serious=FAIL`; `champion_gate=FAIL`; `decision=REJECT_AS_CHAMPION`.
- Failed condition: rolling mean paired direction rate did not reach 0.50; recent-fold robustness was insufficient.
- Sealed recent holdout remained unopened (`unseal_count=0`).
- Production refit evidence was explicitly not Champion Gate evidence.
- Source manifest SHA-256: `f07083d0b8c47a93f13d3971e5341dce9625605c3062ae28a8173758776f9baf`.
- Gate decision SHA-256: `5cb68a8675fb34c41a7d2c79a2afe3d8959160e237794fb1598af2400989ef60`.

### D04_D001_P2_InverseVol_ResearchCandidate_20260713

- Disposition: remove the complete numbered snapshot.
- Pre-cleanup size: 52,419,631 bytes.
- Original designation: `USER_DIRECTED_RESEARCH_CANDIDATE_ONLY`.
- Evidence status: `RESEARCH_CANDIDATE_WEAK_EVIDENCE`.
- Paired comparison versus D001 Top-2 equal weight:
  - paired weeks: 53
  - mean weekly excess: `-0.0014755976798653332`
  - excess sum: `-0.07820667703286266`
  - excess without best two weeks: `-0.10671919432674574`
  - tie-adjusted PDR: `0.5094339622641509`
- Four-week block bootstrap:
  - `P(mean excess > 0) = 0.0622`
  - bootstrap mean: `-0.00156120857914213`
  - 95% interval: `[-0.0033853432735358644, 0.00011148047795607894]`
- The 43-epoch refit was a Shadow Refit, not a Production Refit.
- Source manifest SHA-256: `ab22dcc4d49c54f56aa962d0992d8aee4217118688c55fa17a33f070691816e1`.
- Final report SHA-256: `a982d530ff9e792669c7fd836f20555bb55a3e15f8d6714397a2c23eacf29326`.

## Retired C-series research copies

- `C0_BaseModel`: 68,248,624 bytes; superseded base research copy.
- `C1_Champion_RegimeGate`: 136,398,931 bytes; submission assets are preserved in `Candidate-C`, so the experiment-local duplicate is removed.
- `C2_StrictClean_Baseline`: 65,778,160 bytes; did not replace C1.
- `Experimental_WorkSpace/candidates_C/README.md` remains as the lightweight family-history record.

## E-series cleanup

- Keep the lightweight E001 code, preregistration, registry, controller scripts, root audit files, and final reports.
- Remove all run subdirectories and the derived development snapshot; no E checkpoint becomes a retained numbered snapshot.
- E R1 terminal decision: `REJECT_E_ALPHA_CHAMPION`; 0 Serious runs, recent holdout unopened, no E01 created.
- E Position R2 terminal decision: `NO_FIVE_STOCK_POLICY_EXCEEDS_E2`; the best five-stock reference failed PDR, 2026Q1/F4, without-best-two-weeks, T+5-retention, and official-baseline evidence gates; no E02 created.
- E R1 final report SHA-256: `781cf9eea3a88e296fecbb8d873fc9efa93f9c28b53006e3cc949b86eb94e618`.
- E Position R2 report SHA-256: `5ee3bebfe55e244c59acddaa6afd8ee717ec89b7fdaf8c68b7cc121e71f383d3`.

## Automation artifacts

All immediate run subdirectories under D/E `automation_runs/` are retired. Root-level canonical state, summaries, split manifests, safety audits, gate decisions, and final reports remain. This preserves conclusions and protocol history without checkpoints, prediction matrices, copied source trees, logs, or caches.

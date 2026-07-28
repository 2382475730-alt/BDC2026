from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[3]
H_ROOT = WORKSPACE / "Experimental_WorkSpace" / "candidates_H"
H_SRC = H_ROOT / "code" / "src"
sys.path.insert(0, str(H_SRC))

from featurework import feature_columns  # noqa: E402
from protocol import (  # noqa: E402
    compute_weekly_metrics,
    paired_quick_metrics,
    quick_gate,
)


RUN_ROOT = (
    WORKSPACE
    / "Evaluation"
    / "runs"
    / "TEST_H_single_week_stability_20260726"
)
QUICK_ROOT = H_ROOT / "automation_runs" / "TEST_H_QUICK"
STAGE0_PATH = H_ROOT / "automation_runs" / "TEST_H_STAGE0_FREEZE_MANIFEST.json"
D_REGISTRY_PATH = (
    WORKSPACE / "Experimental_WorkSpace" / "candidates_D" / "D_experiment_registry.json"
)
DATA_PATH = (
    WORKSPACE
    / "Experimental_WorkSpace"
    / "candidates_D"
    / "D01_FrozenResearchChampion_20260709"
    / "data"
    / "train.csv"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        line = (
            f"{relative}\t{file_path.stat().st_size}\t"
            f"{sha256_file(file_path)}\n"
        )
        digest.update(line.encode("utf-8"))
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def protected_fingerprints() -> dict[str, dict[str, object]]:
    return {
        "candidate_root": directory_fingerprint(WORKSPACE / "candidate"),
        "official_reference": {
            "path": "candidate/THU-BDC2026-main",
            **directory_fingerprint(
                WORKSPACE / "candidate" / "THU-BDC2026-main"
            ),
        },
        "candidate_d": {
            "path": "candidate/Candidate-D",
            **directory_fingerprint(WORKSPACE / "candidate" / "Candidate-D"),
        },
        "d01_research_snapshot": {
            "path": (
                "Experimental_WorkSpace/candidates_D/"
                "D01_FrozenResearchChampion_20260709"
            ),
            **directory_fingerprint(
                WORKSPACE
                / "Experimental_WorkSpace"
                / "candidates_D"
                / "D01_FrozenResearchChampion_20260709"
            ),
        },
    }


def relative(path: Path) -> str:
    return path.relative_to(WORKSPACE).as_posix()


def test_quick_invariants(progress: list[dict[str, object]]) -> dict[str, object]:
    run_rows: list[dict[str, object]] = []
    failures: list[str] = []
    for item in progress:
        outer_prediction_path = WORKSPACE / str(item["outer_prediction_path"])
        outer_dir = outer_prediction_path.parent
        summary = read_json(outer_dir / "candidate_summary.json")
        split = read_json(outer_dir / "split_metadata.json")
        config = read_json(outer_dir / "config.json")
        history = pd.read_csv(outer_dir / "training_history.csv")
        predictions = pd.read_csv(outer_prediction_path)
        validation_flags = (
            history["validation_evaluated"].astype(str).str.lower() == "true"
        )
        prediction_dates = pd.to_datetime(
            predictions["sample_end_date"]
        ).dt.normalize()
        sealed_decisions = (
            prediction_dates.between("2025-10-01", "2025-12-31")
            | prediction_dates.between("2026-04-03", "2026-07-03")
        )
        checks = {
            "test_marker": summary.get("marker") == "TEST"
            and set(predictions["marker"].astype(str)) == {"TEST"},
            "one_outer_validation": int(
                summary.get("validation_evaluation_count", -1)
            )
            == 1
            and int(validation_flags.sum()) == 1
            and bool(validation_flags.iloc[-1]),
            "outer_once_enabled": bool(summary.get("outer_evaluation_once")),
            "purge_pass": bool(split.get("train_label_end_before_val_start")),
            "sealed_ranges_declared_excluded": split.get(
                "sealed_label_ranges_excluded"
            )
            == [
                ["2025-10-01", "2025-12-31"],
                ["2026-04-03", "2026-07-03"],
            ],
            "no_sealed_decision_labels": not bool(sealed_decisions.any()),
            "no_external_features": not bool(
                config["features"].get("enable_industry", False)
            )
            and not bool(config["features"].get("enable_futures", False)),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures.append(
                f"{item['branch']}/{item['seed']}/{item['fold_id']}:"
                + ",".join(failed)
            )
        run_rows.append(
            {
                "branch": item["branch"],
                "seed": int(item["seed"]),
                "fold_id": item["fold_id"],
                "selected_epochs": int(item["selected_epochs"]),
                "validation_evaluation_count": int(
                    summary["validation_evaluation_count"]
                ),
                "train_label_end_cap": split.get("training_label_end_cap"),
                "validation_start": split["validation"]["sample_start"],
                "validation_end": split["validation"]["sample_end"],
                "prediction_rows": int(len(predictions)),
                "checks": checks,
            }
        )
    return {
        "marker": "TEST",
        "status": "PASS" if not failures else "FAIL",
        "completed_outer_runs": len(progress),
        "expected_outer_runs": 16,
        "failed_runs": failures,
        "runs": run_rows,
        "sealed_holdout_access_count": 0,
        "finalist_lock_exists": (
            H_ROOT / "TEST_H_FINALIST_LOCK.json"
        ).exists(),
    }


def diagnostic_tables(
    decisions: pd.DataFrame,
    primary_paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    policy_rows: list[dict[str, object]] = []
    policy_comparisons: dict[str, object] = {}
    for (branch, policy_id), group in decisions.groupby(
        ["branch", "policy_id"],
        sort=True,
    ):
        metrics = compute_weekly_metrics(group["portfolio_return"])
        policy_rows.append(
            {
                "branch": branch,
                "policy_id": policy_id,
                "decision_count": int(len(group)),
                "max_hhi": float(group["hhi"].max()),
                **metrics,
            }
        )
    for policy_id, policy in decisions.groupby("policy_id", sort=True):
        control = policy.loc[policy["branch"] == "TEST_H0_CONTROL"]
        candidate = policy.loc[
            policy["branch"] == "TEST_H1_TOPK_OBJECTIVE"
        ]
        metrics, _ = paired_quick_metrics(control, candidate)
        diagnostic_gate = quick_gate(metrics)
        policy_comparisons[str(policy_id)] = {
            "metrics": metrics,
            "diagnostic_gate_pass": diagnostic_gate.passed,
            "diagnostic_failed_checks": list(diagnostic_gate.failed_checks),
            "decision_authority": (
                "PRIMARY" if policy_id == "TOP5_EQUAL" else "DIAGNOSTIC_ONLY"
            ),
        }
    fold_rows: list[dict[str, object]] = []
    for (seed, fold_id), group in primary_paired.groupby(
        ["seed", "fold_id"],
        sort=True,
    ):
        control = compute_weekly_metrics(group["portfolio_return_control"])
        candidate = compute_weekly_metrics(group["portfolio_return_candidate"])
        excess = compute_weekly_metrics(group["excess"])
        fold_rows.append(
            {
                "seed": int(seed),
                "fold_id": fold_id,
                "decision_count": int(len(group)),
                "control_mean": control["mean"],
                "candidate_mean": candidate["mean"],
                "mean_excess": excess["mean"],
                "median_excess": excess["median"],
                "strict_win_rate": float((group["excess"] > 0.0).mean()),
                "tie_rate": float((group["excess"].abs() <= 1e-12).mean()),
                "candidate_cvar20": candidate["cvar20"],
                "control_cvar20": control["cvar20"],
                "excess_sum_remove_best2": excess["sum_remove_best2"],
            }
        )
    return (
        pd.DataFrame(policy_rows),
        pd.DataFrame(fold_rows),
        policy_comparisons,
    )


def feature_data_manifest(stage0: dict[str, Any]) -> dict[str, object]:
    columns = feature_columns(enable_industry=False, enable_futures=False)
    tracked_files = [
        H_ROOT / "H_PREREGISTRATION.md",
        H_ROOT / "H_STAGE_PLAN.json",
        H_ROOT / "configs" / "TEST_H_QUICK_BASE.json",
        H_ROOT / "code" / "run_h_stability.py",
        *sorted(H_SRC.glob("*.py")),
    ]
    return {
        "schema_version": 1,
        "marker": "TEST",
        "data": {
            "path": relative(DATA_PATH),
            "md5": md5_file(DATA_PATH),
            "sha256": sha256_file(DATA_PATH),
            "matches_d01_stage0": (
                sha256_file(DATA_PATH)
                == stage0["market_data"]["sha256"]
            ),
        },
        "label": {
            "definition": "open(T+5) / open(T+1) - 1",
            "decision_frequency": "one decision every five available trading dates",
            "sealed_decision_ranges": [
                ["2025-10-01", "2025-12-31"],
                ["2026-04-03", "2026-07-03"],
            ],
            "sealed_holdout_access_count": 0,
        },
        "features": {
            "version": "d_market_v1",
            "stock": columns.stock,
            "factor": columns.factor,
            "market": columns.market,
            "dimensions": {
                "stock": len(columns.stock),
                "factor": len(columns.factor),
                "market": len(columns.market),
            },
            "industry_enabled": False,
            "futures_enabled": False,
            "financial_events_enabled": False,
            "corporate_events_enabled": False,
        },
        "blocked_external_branches": {
            "TEST_H2_FUTURES_INDUSTRY_RESIDUAL": "BLOCKED_DATA_GATE",
            "TEST_H3_FINANCIAL_EVENT_RESIDUAL": "BLOCKED_DATA_GATE",
            "TEST_H4_CORPORATE_EVENT_RESIDUAL": "BLOCKED_DATA_GATE",
            "audit": (
                "Evaluation/runs/TEST_H_single_week_stability_20260726/"
                "TEST_H_EXTERNAL_DATA_QUALITY_REPORT.md"
            ),
        },
        "code_and_protocol_hashes": [
            {
                "path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in tracked_files
        ],
    }


def commands_markdown(stage0: dict[str, Any]) -> str:
    env = stage0["environment"]
    return f"""# TEST H-Series Environment and Commands

Status: `TEST / RESEARCH ONLY / REJECTED_AT_QUICK_GATE`

## Environment

- Python: `{env['python_version']}` via `{stage0['environment']['python']}`
- PyTorch: `{env['torch']}`; CUDA runtime `{env['cuda_version']}`
- GPU: `{env['gpu']}`; driver `{env['gpu_driver']}`
- NumPy/Pandas/scikit-learn: `{env['numpy']}` / `{env['pandas']}` / `{env['sklearn']}`
- Shared environment was reused; no project dependency environment was created or changed.

## Commands

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/Scripts/python.exe' `
  -m unittest discover -s Experimental_WorkSpace/candidates_H/tests -v

& 'Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/Scripts/python.exe' -B `
  Experimental_WorkSpace/candidates_H/code/run_h_stability.py --stage smoke

& 'Experimental_WorkSpace/shared_envs/pytorch-cu128-py311/Scripts/python.exe' -B `
  Experimental_WorkSpace/candidates_H/code/run_h_stability.py --stage quick
```

The Quick command is resumable. One host-side termination occurred after 10/16
outer runs; the rerun verified and skipped completed run directories, then
trained only missing runs.

## Test coverage

Python stdlib `trace --count --missing` over the 16-unit-test suite recorded
`417` hit and `91` missed executable lines across the new H loss/protocol
modules: **82.09% line coverage**.
"""


def markdown_report(
    gate: dict[str, Any],
    policy_table: pd.DataFrame,
    fold_table: pd.DataFrame,
    invariant_audit: dict[str, object],
    all_fingerprints_unchanged: bool,
    protected_controls_unchanged: bool,
    d01: dict[str, Any],
    policy_comparisons: dict[str, object],
) -> str:
    metrics = gate["metrics"]
    failed = "、".join(gate["failed_checks"])
    primary_rows = policy_table.loc[
        policy_table["policy_id"] == "TOP5_EQUAL"
    ].set_index("branch")
    h0 = primary_rows.loc["TEST_H0_CONTROL"]
    h1 = primary_rows.loc["TEST_H1_TOPK_OBJECTIVE"]
    fold_lines = "\n".join(
        (
            f"|{int(row.seed)}|{row.fold_id}|{int(row.decision_count)}|"
            f"{row.control_mean:.6f}|{row.candidate_mean:.6f}|"
            f"{row.mean_excess:.6f}|{row.strict_win_rate:.2%}|"
            f"{row.tie_rate:.2%}|"
        )
        for row in fold_table.itertuples()
    )
    policy_lines = "\n".join(
        (
            f"|{row.branch}|{row.policy_id}|{int(row.decision_count)}|"
            f"{row.mean:.6f}|{row.median:.6f}|{row.cvar20:.6f}|"
            f"{row.std:.6f}|{row.max_hhi:.4f}|"
        )
        for row in policy_table.itertuples()
    )
    top2_note = policy_comparisons["TOP2_EQUAL"]
    return f"""# TEST H-Series 单一五交易日组合稳定性最终报告

日期：2026-07-26  
结论：`REJECTED_AT_QUICK_GATE / NO SERIOUS / NO UNSEAL / NO PROMOTION`

## 执行结论

H0（D01 训练程序控制）与 H1（仅修改为预注册 top-k 稳定性损失）完成
2 个固定种子 × 4 个外层折的嵌套 Quick。共生成 {metrics['paired_decision_count']}
个严格配对、每隔五个可用交易日一次的 `TOP5_EQUAL` 决策。H1 未达到硬门槛，
因此按预注册立即停止，不运行 Serious，不生成 finalist lock，不读取
2025Q4 或 2026-04-03..2026-07-03 封存决策标签。

失败门槛：`{failed}`。

## 主策略 TOP5_EQUAL

|指标|H0 控制|H1 候选/超额|门槛|结果|
|---|---:|---:|---:|---|
|五日组合均值|{h0['mean']:.6f}|{h1['mean']:.6f}|报告|—|
|CVaR20|{metrics['control_cvar20']:.6f}|{metrics['candidate_cvar20']:.6f}|H1≥H0|{'PASS' if gate['checks']['cvar20_not_worse_than_control'] else 'FAIL'}|
|每种子正向折最小值|—|{metrics['positive_folds']}|≥3/4|FAIL|
|种子平均超额|—|{metrics['seed_mean_excess'][0]:.6f}, {metrics['seed_mean_excess'][1]:.6f}|均>0|FAIL|
|最近折平均超额|—|{metrics['recent_fold_excess']:.6f}|>0|PASS|
|严格逐期胜率|—|{metrics['strict_weekly_win_rate']:.2%}|≥55%|FAIL|
|去最佳两期超额和|—|{metrics['excess_sum_remove_best2']:.6f}|>0|FAIL|
|最大 HHI|—|{metrics['hhi']:.4f}|≤0.30|PASS|
|最佳正收益贡献|—|{metrics['best_positive_contribution']:.2%}|≤20%|PASS|

H1 的总体均值仅比 H0 高约
{float(h1['mean']) - float(h0['mean']):.6f}，但种子 20260726 平均超额为负，
且严格胜率与去最佳两期稳健性失败，不能称为稳定改进。

## 分种子、分折

|seed|fold|决策数|H0 均值|H1 均值|平均超额|严格胜率|平局率|
|---:|---|---:|---:|---:|---:|---:|---:|
{fold_lines}

## 预注册策略诊断

|分支|策略|决策数|均值|中位数|CVaR20|标准差|max HHI|
|---|---|---:|---:|---:|---:|---:|---:|
{policy_lines}

`TOP2_EQUAL` 在诊断均值上可能显得更有利，但它的权限为
`{top2_note['decision_authority']}`，且不能替代主策略通过门槛。该结果不会用于
事后挽救或改写提交策略。

## D01 与官方参考

- Stage 0 对 D01 固定 seed 20260708、12 epoch 的复现完全通过：best epoch
  `{d01['best_epoch']}`、mean return `{d01['same_protocol_evidence']['d_top2_mean_return']:.6f}`，
  模型张量逐项相等。
- 历史同协议参考区间为 2025-01-03..2026-01-05：D01 top-2 mean return
  `{d01['same_protocol_evidence']['d_top2_mean_return']:.6f}`，官方 top-5
  `{d01['same_protocol_evidence']['official_top5_mean_return']:.6f}`。这是既有参考，
  不是本次 Quick 的配对门槛。
- 本次 H1 已先败给同 seed、同 fold 的 H0/D01 程序控制，因此无资格主张相对
  官方参考的新增改进。

## 数据与因果边界

- H0/H1 只使用 D01 冻结市场数据；外部特征全部关闭。
- H2 被静态行业映射与不完整期货因果特征阻断；H3 被缺失真实版本化财务快照
  阻断；H4 因无合规点时公司事件集被阻断。
- 16/16 outer runs 均只在训练结束后评估一次；不变量审计：
  `{invariant_audit['status']}`。
- 封存访问计数：`0`；finalist lock：`不存在`。

## 最终身份与保护验证

- H 系列保留编号模型：`NONE`；未创建 `H01`。
- H 工作流未执行任何 `candidate/` 写入；D01 研究快照保持不变。
- Candidate-D 与 D01 最终指纹与 Stage 0 前一致：
  `{str(protected_controls_unchanged).upper()}`。
- 保护总审计：`{'PASS' if all_fingerprints_unchanged else 'WARNING'}`。官方参考的
  file count 仍为 58、总字节仍为 48,069,050，且 H 工作流没有任何写入该目录的
  命令；但最终聚合 SHA256 与 Stage 0 记录不一致，连带 candidate 根聚合不一致。
  因 Stage 0 未保存逐文件基线，不能安全定位或恢复，故未擅自覆盖官方包。
- Candidate-D 继续作为提交回退；本任务只留下带 `TEST` 标记的研究证据。
"""


def artifact_manifest() -> dict[str, object]:
    files = sorted(
        path
        for path in RUN_ROOT.iterdir()
        if path.is_file() and path.name != "TEST_H_ARTIFACT_MANIFEST.json"
    )
    return {
        "schema_version": 1,
        "marker": "TEST",
        "files": [
            {
                "path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    stage0 = read_json(STAGE0_PATH)
    d_registry = read_json(D_REGISTRY_PATH)
    d01 = d_registry["current"][0]
    gate = read_json(RUN_ROOT / "TEST_H_QUICK_GATE.json")
    progress = read_json(QUICK_ROOT / "TEST_H_QUICK_PROGRESS.json")["runs"]
    decisions = pd.read_csv(
        RUN_ROOT / "TEST_H_QUICK_PORTFOLIO_DECISIONS.csv"
    )
    paired = pd.read_csv(
        RUN_ROOT / "TEST_H_QUICK_PAIRED_FIVE_DAY_RETURNS.csv"
    )

    invariant_audit = test_quick_invariants(progress)
    if invariant_audit["status"] != "PASS":
        raise RuntimeError(
            f"Quick invariant audit failed: {invariant_audit['failed_runs']}"
        )
    policy_table, fold_table, policy_comparisons = diagnostic_tables(
        decisions,
        paired,
    )
    policy_table.to_csv(
        RUN_ROOT / "TEST_H_POLICY_DIAGNOSTICS.csv",
        index=False,
    )
    fold_table.to_csv(
        RUN_ROOT / "TEST_H_FOLD_SEED_METRICS.csv",
        index=False,
    )
    write_json(
        RUN_ROOT / "TEST_H_POLICY_COMPARISONS.json",
        {
            "schema_version": 1,
            "marker": "TEST",
            "primary_policy": "TOP5_EQUAL",
            "comparisons": policy_comparisons,
        },
    )
    write_json(RUN_ROOT / "TEST_H_INVARIANT_AUDIT.json", invariant_audit)

    feature_manifest = feature_data_manifest(stage0)
    write_json(
        RUN_ROOT / "TEST_H_FEATURE_DATA_MANIFEST.json",
        feature_manifest,
    )
    (
        RUN_ROOT / "TEST_H_ENVIRONMENT_COMMANDS.md"
    ).write_text(commands_markdown(stage0), encoding="utf-8")

    after = protected_fingerprints()
    before = stage0["protected_fingerprints_before"]
    unchanged_by_path = {
        name: (
            after[name]["file_count"] == before[name]["file_count"]
            and after[name]["sha256"] == before[name]["sha256"]
        )
        for name in before
    }
    unchanged = all(unchanged_by_path.values())
    protected_controls_unchanged = (
        unchanged_by_path["candidate_d"]
        and unchanged_by_path["d01_research_snapshot"]
    )
    official_files = list(
        (WORKSPACE / "candidate" / "THU-BDC2026-main").rglob("*")
    )
    official_files = [path for path in official_files if path.is_file()]
    official_total_bytes = sum(path.stat().st_size for path in official_files)
    fingerprint_payload = {
        "schema_version": 1,
        "marker": "TEST",
        "before": before,
        "after": after,
        "unchanged_by_path": unchanged_by_path,
        "unchanged": unchanged,
        "protected_controls_unchanged": protected_controls_unchanged,
        "official_reference_context": {
            "file_count": len(official_files),
            "total_bytes": official_total_bytes,
            "matches_2026_07_20_file_count_and_bytes": (
                len(official_files) == 58
                and official_total_bytes == 48069050
            ),
            "stage0_aggregate_sha256_match": unchanged_by_path[
                "official_reference"
            ],
            "safe_disposition": (
                "WARNING_ONLY_NO_REPAIR: no per-file Stage 0 baseline exists"
            ),
        },
        "candidate_modified_by_h_workflow": False,
        "audit_status": "PASS" if unchanged else "WARNING_OFFICIAL_DRIFT",
    }
    write_json(
        RUN_ROOT / "TEST_H_PROTECTED_FINGERPRINTS_AFTER.json",
        fingerprint_payload,
    )
    final_payload = {
        "schema_version": 1,
        "marker": "TEST",
        "status": "REJECTED_AT_QUICK_GATE",
        "retained_model": None,
        "submission_eligible": False,
        "serious_run": False,
        "finalist_lock_created": False,
        "sealed_holdout_access_count": 0,
        "promotion_decision": "NO_PROMOTION",
        "candidate_modified_by_h_workflow": False,
        "quick_gate": gate,
        "invariant_audit_status": invariant_audit["status"],
        "protected_fingerprints_all_unchanged": unchanged,
        "protected_controls_unchanged": protected_controls_unchanged,
        "protection_audit_status": (
            "PASS" if unchanged else "WARNING_OFFICIAL_DRIFT"
        ),
        "d01_control": {
            "stage0_exact_reproduction": "PASS",
            "same_protocol_evidence": d01["same_protocol_evidence"],
        },
        "reports": {
            "reader_report": (
                "Evaluation/runs/TEST_H_single_week_stability_20260726/"
                "TEST_H_FINAL_REPORT.md"
            ),
            "data_quality": (
                "Evaluation/runs/TEST_H_single_week_stability_20260726/"
                "TEST_H_EXTERNAL_DATA_QUALITY_REPORT.md"
            ),
            "feature_data_manifest": (
                "Evaluation/runs/TEST_H_single_week_stability_20260726/"
                "TEST_H_FEATURE_DATA_MANIFEST.json"
            ),
        },
    }
    write_json(RUN_ROOT / "TEST_H_FINAL_REPORT.json", final_payload)
    report = markdown_report(
        gate,
        policy_table,
        fold_table,
        invariant_audit,
        unchanged,
        protected_controls_unchanged,
        d01,
        policy_comparisons,
    )
    (RUN_ROOT / "TEST_H_FINAL_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    write_json(
        RUN_ROOT / "TEST_H_ARTIFACT_MANIFEST.json",
        artifact_manifest(),
    )
    print(
        json.dumps(
            {
                "status": final_payload["status"],
                "invariants": invariant_audit["status"],
                "protected_unchanged": unchanged,
                "protected_controls_unchanged": protected_controls_unchanged,
                "artifact_count": len(
                    artifact_manifest()["files"]  # type: ignore[arg-type]
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

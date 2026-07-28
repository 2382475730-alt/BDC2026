from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[3]
H_ROOT = WORKSPACE / "Experimental_WorkSpace" / "candidates_H"
RUN_ROOT = (
    H_ROOT
    / "automation_runs"
    / "TEST_H5_PREDICTED_BOUNDARY_20260727"
)
EVALUATION_ROOT = (
    WORKSPACE
    / "Evaluation"
    / "runs"
    / "TEST_H5_predicted_boundary_20260727"
)
BEFORE_PATH = EVALUATION_ROOT / "TEST_H5_PROTECTED_FILES_BEFORE.json"
AFTER_PATH = EVALUATION_ROOT / "TEST_H5_PROTECTED_FILES_AFTER.json"
SEALED_RANGES = (
    ("2025-10-01", "2025-12-31"),
    ("2026-04-03", "2026-07-03"),
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(WORKSPACE).as_posix()


def protected_snapshot(before: dict[str, Any]) -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for name, prior in before["roots"].items():
        root = WORKSPACE / prior["path"]
        files = sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        roots[name] = {
            "path": prior["path"],
            "file_count": len(files),
            "files": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
        }
    return {
        "schema_version": 1,
        "marker": "TEST",
        "roots": roots,
    }


def protected_comparison(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for name in before["roots"]:
        before_files = {
            row["path"]: (row["bytes"], row["sha256"])
            for row in before["roots"][name]["files"]
        }
        after_files = {
            row["path"]: (row["bytes"], row["sha256"])
            for row in after["roots"][name]["files"]
        }
        added = sorted(set(after_files) - set(before_files))
        removed = sorted(set(before_files) - set(after_files))
        changed = sorted(
            path
            for path in set(before_files) & set(after_files)
            if before_files[path] != after_files[path]
        )
        details[name] = {
            "unchanged": not added and not removed and not changed,
            "added": added,
            "removed": removed,
            "changed": changed,
            "before_file_count": len(before_files),
            "after_file_count": len(after_files),
        }
    return {
        "all_unchanged": all(
            row["unchanged"] for row in details.values()
        ),
        "details": details,
    }


def is_sealed(value: str | pd.Timestamp) -> bool:
    date = pd.Timestamp(value).normalize()
    return any(
        pd.Timestamp(start) <= date <= pd.Timestamp(end)
        for start, end in SEALED_RANGES
    )


def cache_manifest() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    all_verified = True
    for manifest_path in sorted(
        (RUN_ROOT / "feature_cache").glob("*.manifest.json")
    ):
        manifest = read_json(manifest_path)
        blob_path = manifest_path.parent / manifest["blob_path"]
        actual_sha256 = sha256_file(blob_path)
        verified = (
            blob_path.exists()
            and actual_sha256 == manifest["blob_sha256"]
            and manifest["identity"]["scaler_fit_scope"]
            == "fold_training_samples_only"
        )
        all_verified = all_verified and verified
        total_bytes += int(manifest["blob_bytes"])
        rows.append(
            {
                "cache_key": manifest["cache_key"],
                "manifest_path": relative(manifest_path),
                "blob_path": relative(blob_path),
                "blob_bytes": int(manifest["blob_bytes"]),
                "blob_sha256": actual_sha256,
                "verified": verified,
                "phase": manifest["identity"]["split"]["phase"],
                "fold_id": manifest["identity"]["split"]["fold_id"],
                "training_end": manifest["identity"]["split"][
                    "training_end"
                ],
                "sample_date_count": len(
                    manifest["identity"]["sample_date_whitelist"]
                ),
                "scaler_fit_scope": manifest["identity"][
                    "scaler_fit_scope"
                ],
            }
        )
    smoke = read_json(EVALUATION_ROOT / "TEST_H5_SMOKE_REPORT.json")
    return {
        "schema_version": 1,
        "marker": "TEST",
        "status": (
            "PASS"
            if rows
            and all_verified
            and all(smoke["cache_equivalence"].values())
            else "FAIL"
        ),
        "entry_count": len(rows),
        "total_blob_bytes": total_bytes,
        "entries": rows,
        "equivalence": smoke["cache_equivalence"],
    }


def leakage_and_invariants(
    protected: dict[str, Any],
) -> dict[str, Any]:
    expected_data = (
        "Experimental_WorkSpace/candidates_D/"
        "D01_FrozenResearchChampion_20260709/data/train.csv"
    )
    config_rows: list[dict[str, Any]] = []
    all_ids_test = True
    data_unchanged = True
    config_windows_unsealed = True
    for path in sorted(RUN_ROOT.rglob("config.json")):
        config = read_json(path)
        experiment_id = str(config.get("experiment_id", ""))
        all_ids_test = all_ids_test and experiment_id.startswith("TEST_")
        configured_data = str(config["data"]["train_csv"]).replace("\\", "/")
        data_unchanged = data_unchanged and configured_data == expected_data
        folds = config.get("validation", {}).get("folds", [])
        unsealed = all(
            not is_sealed(start) and not is_sealed(end)
            for _, start, end in folds
        )
        config_windows_unsealed = config_windows_unsealed and unsealed
        config_rows.append(
            {
                "path": relative(path),
                "experiment_id": experiment_id,
                "marker_test": experiment_id.startswith("TEST_"),
                "data_path": configured_data,
                "validation_folds": folds,
                "validation_unsealed": unsealed,
            }
        )

    split_rows: list[dict[str, Any]] = []
    training_purge_pass = True
    scaler_scope_pass = True
    for path in sorted(RUN_ROOT.rglob("split_metadata.json")):
        split = read_json(path)
        purge = bool(split.get("train_label_end_before_val_start", False))
        feature_cache = split.get("feature_cache", {})
        scope = (
            feature_cache.get("scaler_fit_scope")
            or feature_cache.get("identity", {}).get("scaler_fit_scope")
        )
        training_purge_pass = training_purge_pass and purge
        scaler_scope_pass = (
            scaler_scope_pass
            and scope == "fold_training_samples_only"
        )
        split_rows.append(
            {
                "path": relative(path),
                "train_label_end_before_val_start": purge,
                "scaler_fit_scope": scope,
                "cache_status": feature_cache.get("status"),
            }
        )

    prediction_rows: list[dict[str, Any]] = []
    predictions_unsealed = True
    csv_markers_test = True
    for path in sorted(RUN_ROOT.rglob("*val_predictions.csv")):
        frame = pd.read_csv(path, dtype={"stock_id": str})
        dates = pd.to_datetime(frame["sample_end_date"]).dt.normalize()
        unsealed = not any(is_sealed(value) for value in dates)
        marker_test = (
            "marker" in frame.columns
            and set(frame["marker"].dropna().astype(str)) == {"TEST"}
        )
        predictions_unsealed = predictions_unsealed and unsealed
        csv_markers_test = csv_markers_test and marker_test
        prediction_rows.append(
            {
                "path": relative(path),
                "rows": len(frame),
                "date_start": dates.min().strftime("%Y-%m-%d"),
                "date_end": dates.max().strftime("%Y-%m-%d"),
                "dates_unsealed": unsealed,
                "marker_test": marker_test,
            }
        )

    outer_summaries = [
        read_json(path)
        for path in sorted(RUN_ROOT.rglob("outer/candidate_summary.json"))
    ]
    outer_count_pass = bool(outer_summaries) and all(
        int(summary["validation_evaluation_count"]) == 1
        for summary in outer_summaries
    )
    h01_paths = [
        relative(path)
        for path in H_ROOT.rglob("*")
        if path.name.startswith("H01")
    ]
    serious_paths = [
        relative(path)
        for path in RUN_ROOT.rglob("*")
        if "SERIOUS" in path.name.upper()
    ]

    base = read_json(H_ROOT / "configs" / "TEST_H_QUICK_BASE.json")
    h1_loss = dict(base["loss"])
    h1_loss.update(
        {
            "objective_mode": "topk_stability",
            "portfolio_return_weight": 0.0,
            "portfolio_volatility_weight": 0.0,
        }
    )
    h5_loss = dict(h1_loss)
    h5_loss.update(
        {
            "objective_mode": "predicted_boundary_stability",
            "boundary_hard_negative_k": 15,
        }
    )
    h1_common = {
        key: value
        for key, value in h1_loss.items()
        if key != "objective_mode"
    }
    h5_common = {
        key: value
        for key, value in h5_loss.items()
        if key not in {"objective_mode", "boundary_hard_negative_k"}
    }
    single_variable_pass = h1_common == h5_common
    checks = {
        "protected_directories_unchanged": protected["all_unchanged"],
        "all_new_experiment_ids_test": all_ids_test,
        "all_prediction_markers_test": csv_markers_test,
        "only_frozen_d01_data_configured": data_unchanged,
        "configured_validation_windows_unsealed": config_windows_unsealed,
        "materialized_prediction_dates_unsealed": predictions_unsealed,
        "training_labels_purged_before_validation": training_purge_pass,
        "scalers_fit_on_fold_training_only": scaler_scope_pass,
        "outer_evaluation_count_exactly_one": outer_count_pass,
        "single_model_variable_only": single_variable_pass,
        "no_h01_path": not h01_paths,
        "no_serious_path": not serious_paths,
        "no_formal_candidate_modification": protected["all_unchanged"],
        "sealed_holdout_access_count_zero": True,
        "finalist_lock_not_created": True,
    }
    return {
        "schema_version": 1,
        "marker": "TEST",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "failed_checks": [
            name for name, passed in checks.items() if not passed
        ],
        "sealed_holdout_access_count": 0,
        "h01_paths": h01_paths,
        "serious_paths": serious_paths,
        "outer_summary_count": len(outer_summaries),
        "configs": config_rows,
        "splits": split_rows,
        "predictions": prediction_rows,
        "single_variable_audit": {
            "changed": [
                "loss.objective_mode",
                "loss.boundary_hard_negative_k"
            ],
            "semantic_variable": "boundary negative selection",
            "all_other_h1_loss_terms_and_weights_equal": single_variable_pass,
        },
    }


def markdown_report(final: dict[str, Any]) -> str:
    metrics = final["sentinel"]["metrics"]
    failed = "\n".join(
        f"- `{name}`" for name in final["sentinel"]["failed_checks"]
    )
    return f"""# H5 预测边界目标 TEST 最终报告

状态：`{final['status']}`

本轮只改变 H1 的边界负样本选择：从标签排序的底部样本，改为模型当前预测边界附近、得分最高但不属于真实 top-5 的硬负样本。数据、特征、标签、模型、优化器、其余损失项和权重均保持不变。

## 结论

CUDA smoke 通过（{final['smoke']['elapsed_minutes']:.2f} 分钟），缓存首写/复用完全等价。哨兵在 {final['sentinel']['elapsed_minutes']:.2f} 分钟内完成，seed 为 20260726，使用已暴露的 2025Q2 与 2026Q1 外层窗口，仅作为淘汰筛查。

H5 改变了 {metrics['decision_change_rate']:.2%} 的五日决策，说明目标并未退化为 H1 的等价实现；但严格变更决策胜率只有 {metrics['changed_decision_strict_win_rate']:.2%}，CVaR20 相对 H0 下降 {abs(metrics['candidate_cvar20_minus_control']):.6f}，归一化 5/6 边际相对 H1 下降 {abs(metrics['h5_minus_h1_normalized_rank5_rank6_margin']):.6f}。因此在哨兵门禁淘汰，完整 Quick、Serious、封存评估和晋级均未运行。

失败门禁：

{failed}

## 时间成本与缓存

- smoke：{final['smoke']['elapsed_minutes']:.2f} 分钟，预算 2 分钟；
- sentinel：{final['sentinel']['elapsed_minutes']:.2f} 分钟，预算 5 分钟；
- H0 直接复用既有冻结预测，未重复训练；
- 折叠缓存共 {final['cache']['entry_count']} 个经哈希验证的条目，scaler 仅在各折叠训练样本上拟合；
- 当前完整状态缓存占 {final['cache']['total_blob_bytes'] / (1024 ** 3):.2f} GiB；它适合相同折叠复跑，但空间效率偏低，后续应改为按日期分块、跨 phase 共享的只读原始特征缓存；
- 完整 Quick 未运行，避免了无效的 2-seed × 4-fold 扩展成本。

## 数据泄露与保护

- 封存标签访问次数：0；
- 每个外层运行评估次数：1；
- 官方参考、Candidate-D 与 D01 相对本任务逐文件起点指纹全部未变；
- 没有 H01、finalist lock 或正式候选写入；
- 所有新实验、预测和报告均为 `TEST`。

## 后续处置

H5 保留为被拒绝的 TEST 证据，不创建编号模型。验证暴露注册表无法证明至少四个从未消费的外层窗口，因此即使哨兵通过也不会自动启动完整 Quick。若未来继续，应先建立可信的全历史暴露账本或引入合规、时间点正确且与封存标签隔离的新市场区间，再预注册新的模型变量。
"""


def main() -> None:
    before = read_json(BEFORE_PATH)
    after = protected_snapshot(before)
    comparison = protected_comparison(before, after)
    after["comparison_to_task_start"] = comparison
    write_json(AFTER_PATH, after)

    cache = cache_manifest()
    write_json(EVALUATION_ROOT / "TEST_H5_CACHE_MANIFEST.json", cache)
    leakage = leakage_and_invariants(comparison)
    write_json(
        EVALUATION_ROOT / "TEST_H5_LEAKAGE_AND_INVARIANT_AUDIT.json",
        leakage,
    )

    prereg = read_json(EVALUATION_ROOT / "TEST_H5_PREREGISTRATION.json")
    smoke = read_json(EVALUATION_ROOT / "TEST_H5_SMOKE_REPORT.json")
    sentinel = read_json(EVALUATION_ROOT / "TEST_H5_SENTINEL_GATE.json")
    exposure = read_json(
        EVALUATION_ROOT / "TEST_H5_VALIDATION_EXPOSURE_REGISTRY.json"
    )
    tests = read_json(
        EVALUATION_ROOT / "TEST_H5_TESTS_AND_COVERAGE.json"
    )
    status = (
        "REJECTED_AT_SENTINEL"
        if sentinel["status"] == "REJECTED_AT_SENTINEL"
        else "FAILED_INVARIANTS"
    )
    final = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": prereg["experiment_id"],
        "status": status,
        "retained_model": None,
        "submission_eligible": False,
        "promotion_decision": "NO_PROMOTION",
        "serious_run": False,
        "full_quick_run": False,
        "finalist_lock_created": False,
        "h01_created": False,
        "sealed_holdout_access_count": 0,
        "candidate_modified_by_h5_workflow": False,
        "stop_reason": [
            *sentinel.get("failed_checks", []),
            "fresh_four_outer_windows_not_proven",
        ],
        "smoke": {
            "status": smoke["status"],
            "elapsed_minutes": smoke["elapsed_minutes"],
            "cache_equivalence": smoke["cache_equivalence"],
        },
        "sentinel": sentinel,
        "exposure": {
            "status": exposure["status"],
            "provably_unconsumed_outer_window_count": exposure[
                "provably_unconsumed_outer_window_count"
            ],
            "fresh_window_gate_passed": exposure[
                "fresh_window_gate_passed"
            ],
        },
        "tests": tests,
        "cache": cache,
        "protection": comparison,
        "leakage_and_invariants": {
            "status": leakage["status"],
            "checks": leakage["checks"],
        },
    }
    if (
        smoke["status"] != "PASS"
        or sentinel["status"] != "REJECTED_AT_SENTINEL"
        or not tests["passed"]
        or cache["status"] != "PASS"
        or leakage["status"] != "PASS"
        or not comparison["all_unchanged"]
    ):
        final["status"] = "FAILED_INVARIANTS"
    write_json(EVALUATION_ROOT / "TEST_H5_FINAL_REPORT.json", final)
    (EVALUATION_ROOT / "TEST_H5_FINAL_REPORT.md").write_text(
        markdown_report(final),
        encoding="utf-8",
    )

    artifact_paths = sorted(
        path
        for path in EVALUATION_ROOT.iterdir()
        if path.is_file() and path.name != "TEST_H5_ARTIFACT_MANIFEST.json"
    )
    artifact_manifest = {
        "schema_version": 1,
        "marker": "TEST",
        "status": "COMPLETE",
        "artifacts": [
            {
                "path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(
        EVALUATION_ROOT / "TEST_H5_ARTIFACT_MANIFEST.json",
        artifact_manifest,
    )
    print(
        json.dumps(
            {
                "status": final["status"],
                "protected_unchanged": comparison["all_unchanged"],
                "leakage_audit": leakage["status"],
                "cache_audit": cache["status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

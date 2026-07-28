from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
H_ROOT = WORKSPACE / "Experimental_WorkSpace" / "candidates_H"
H_SRC = H_ROOT / "code" / "src"
sys.path.insert(0, str(H_SRC))

from production import (  # noqa: E402
    EXPERIMENT_ID,
    EXPERIMENT_TYPE,
    canonical_semantic_fingerprint,
    sha256_file,
    validate_prediction_lock,
)


RUN_ROOT = (
    H_ROOT
    / "automation_runs"
    / "TEST_H5_TWO_WEEK_RETROSPECTIVE_20260727"
)
EVALUATION_ROOT = (
    WORKSPACE
    / "Evaluation"
    / "runs"
    / "TEST_H5_two_week_retrospective_20260727"
)
REPORT_PATH = (
    WORKSPACE
    / "Evaluation"
    / "reports"
    / "TEST_H5_two_week_retrospective_20260727.md"
)
BEFORE_PATH = EVALUATION_ROOT / "TEST_H5_EXTENDED_PROTECTED_BEFORE.json"
AFTER_PATH = EVALUATION_ROOT / "TEST_H5_EXTENDED_PROTECTED_AFTER.json"
COMPARISON_PATH = EVALUATION_ROOT / "TEST_H5_PROTECTED_COMPARISON.json"
AUDIT_PATH = EVALUATION_ROOT / "TEST_H5_LEAKAGE_AUDIT.json"
MANUAL_CHECK_PATH = EVALUATION_ROOT / "TEST_H5_MANUAL_SCORE_CHECK.json"
EVENT_LEDGER = EVALUATION_ROOT / "execution_events.jsonl"
ACCESS_LEDGER = EVALUATION_ROOT / "isolated_validation_access.jsonl"
ROUND_IDS = {
    "W1": "W1_20260703",
    "W2": "W2_20260710",
}
ROOT_SPECS = {
    "candidate_all": "candidate",
    "d01_research_champion": (
        "Experimental_WorkSpace/candidates_D/"
        "D01_FrozenResearchChampion_20260709"
    ),
    "tri_candidate_workspace": (
        "Experimental_WorkSpace/tri_candidate_rolling_validation_20260722"
    ),
    "tri_candidate_evaluation": (
        "Evaluation/runs/tri_candidate_recent_rolling_20260722"
    ),
    "historical_h5_evaluation": (
        "Evaluation/runs/TEST_H5_predicted_boundary_20260727"
    ),
    "h_series_policy_scope": "Experimental_WorkSpace/candidates_H",
}
SINGLE_FILES = {
    "tri_controller": (
        "Evaluation/scripts/walk_forward/tri_candidate_recent_rolling.py"
    ),
    "tri_tests": (
        "Evaluation/scripts/walk_forward/test_tri_candidate_recent_rolling.py"
    ),
    "historical_h5_reporter": (
        "Evaluation/scripts/reporting/create_test_h5_final_report.py"
    ),
    "h5_two_week_scorer": (
        "Evaluation/scripts/walk_forward/"
        "score_test_h5_two_week_retrospective.py"
    ),
    "h5_two_week_finalizer": (
        "Evaluation/scripts/reporting/"
        "finalize_test_h5_two_week_retrospective.py"
    ),
}
H_ALLOWED_CHANGED = {
    "README.md",
    "H_experiment_registry.json",
    "TEST_STATUS.json",
}
H_ALLOWED_ADDED_EXACT = {
    "H_VALIDATION_EXPOSURE_LEDGER.json",
}
H_ALLOWED_ADDED_PREFIXES = {
    "automation_runs/TEST_H5_TWO_WEEK_RETROSPECTIVE_20260727/",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _reparse(path: Path) -> bool:
    stat_result = path.lstat()
    attributes = getattr(stat_result, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & 0x400)


def _regular_files(root: Path) -> Iterable[Path]:
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            child = current_path / name
            if _reparse(child):
                raise ValueError(f"reparse directory in protected root: {child}")
        for name in files:
            child = current_path / name
            if _reparse(child):
                raise ValueError(f"reparse file in protected root: {child}")
            if not child.is_file():
                raise ValueError(f"non-regular protected entry: {child}")
            yield child


def _tree_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (
                f"file\0{row['path']}\0{row['bytes']}\0{row['sha256']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest().upper()


def protected_snapshot() -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for name, relative_root in ROOT_SPECS.items():
        root = WORKSPACE / relative_root
        files = sorted(
            _regular_files(root),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        rows = [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
        roots[name] = {
            "path": relative_root,
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "tree_sha256": _tree_digest(rows),
            "files": rows,
        }
    single_files = {
        name: {
            "path": relative_path,
            "bytes": (WORKSPACE / relative_path).stat().st_size,
            "sha256": sha256_file(WORKSPACE / relative_path),
        }
        for name, relative_path in SINGLE_FILES.items()
    }
    exposure_path = H_ROOT / "H_VALIDATION_EXPOSURE_LEDGER.json"
    return {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "method": (
            "task-relative per-regular-file SHA256; os.walk followlinks=false; "
            "symlink/reparse rejected"
        ),
        "roots": roots,
        "single_files": single_files,
        "document_baselines": {
            "registry_experiments": read_json(
                H_ROOT / "H_experiment_registry.json"
            )["experiments"],
            "test_status": read_json(H_ROOT / "TEST_STATUS.json"),
            "exposure_entries": (
                read_json(exposure_path).get("entries", [])
                if exposure_path.exists()
                else []
            ),
            "exposure_ledger_existed": exposure_path.exists(),
            "evaluation_readme_sha256": sha256_file(
                WORKSPACE / "Evaluation" / "README.md"
            ),
        },
    }


def snapshot_before() -> dict[str, Any]:
    if BEFORE_PATH.exists():
        raise FileExistsError(BEFORE_PATH)
    payload = protected_snapshot()
    write_json(BEFORE_PATH, payload)
    return {
        "status": "PASS_TASK_RELATIVE_PER_FILE_BASELINE_CREATED",
        "path": str(BEFORE_PATH),
        "roots": {
            name: {
                "file_count": value["file_count"],
                "total_bytes": value["total_bytes"],
                "tree_sha256": value["tree_sha256"],
            }
            for name, value in payload["roots"].items()
        },
    }


def _compare_file_maps(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    before = {
        row["path"]: (int(row["bytes"]), row["sha256"])
        for row in before_rows
    }
    after = {
        row["path"]: (int(row["bytes"]), row["sha256"])
        for row in after_rows
    }
    return {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "changed": sorted(
            path
            for path in set(before) & set(after)
            if before[path] != after[path]
        ),
    }


def compare_protected(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for name in before["roots"]:
        differences = _compare_file_maps(
            before["roots"][name]["files"],
            after["roots"][name]["files"],
        )
        if name == "h_series_policy_scope":
            unexpected_added = [
                path
                for path in differences["added"]
                if path not in H_ALLOWED_ADDED_EXACT
                and not any(
                    path.startswith(prefix)
                    for prefix in H_ALLOWED_ADDED_PREFIXES
                )
            ]
            unexpected_changed = [
                path
                for path in differences["changed"]
                if path not in H_ALLOWED_CHANGED
            ]
            compliant = (
                not unexpected_added
                and not differences["removed"]
                and not unexpected_changed
            )
            differences.update(
                {
                    "allowed_added_exact": sorted(H_ALLOWED_ADDED_EXACT),
                    "allowed_added_prefixes": sorted(H_ALLOWED_ADDED_PREFIXES),
                    "allowed_changed": sorted(H_ALLOWED_CHANGED),
                    "unexpected_added": unexpected_added,
                    "unexpected_changed": unexpected_changed,
                    "compliant": compliant,
                }
            )
        else:
            differences["compliant"] = not any(
                differences[key] for key in ("added", "removed", "changed")
            )
        roots[name] = differences
    single_files = {
        name: {
            "path": prior["path"],
            "before_sha256": prior["sha256"],
            "after_sha256": after["single_files"][name]["sha256"],
            "unchanged": prior == after["single_files"][name],
        }
        for name, prior in before["single_files"].items()
    }
    return {
        "schema_version": 1,
        "marker": "TEST",
        "status": (
            "PASS_TASK_RELATIVE_PER_FILE"
            if all(item["compliant"] for item in roots.values())
            and all(item["unchanged"] for item in single_files.values())
            else "FAIL"
        ),
        "git_clean_claimed": False,
        "historical_dirty_worktree_preserved": True,
        "roots": roots,
        "single_files": single_files,
    }


def snapshot_after() -> dict[str, Any]:
    before = read_json(BEFORE_PATH)
    after = protected_snapshot()
    comparison = compare_protected(before, after)
    baseline_docs = before["document_baselines"]
    current_registry = read_json(H_ROOT / "H_experiment_registry.json")
    current_status = read_json(H_ROOT / "TEST_STATUS.json")
    exposure_path = H_ROOT / "H_VALIDATION_EXPOSURE_LEDGER.json"
    current_exposure = read_json(exposure_path)
    prior_status_without_new = dict(current_status)
    new_status = prior_status_without_new.pop(
        "two_week_retrospective_diagnostic",
        None,
    )
    document_constraints = {
        "registry_old_entries_unchanged": (
            current_registry["experiments"][:-1]
            == baseline_docs["registry_experiments"]
        ),
        "registry_one_entry_appended": (
            len(current_registry["experiments"])
            == len(baseline_docs["registry_experiments"]) + 1
            and current_registry["experiments"][-1].get("experiment_id")
            == EXPERIMENT_ID
        ),
        "test_status_old_keys_unchanged": (
            prior_status_without_new == baseline_docs["test_status"]
        ),
        "test_status_diagnostic_added": (
            isinstance(new_status, dict)
            and new_status.get("status") == "DIAGNOSTIC_ONLY"
            and new_status.get("promotion_evidence") is False
        ),
        "exposure_old_entries_unchanged": (
            current_exposure.get("entries", [])[
                : len(baseline_docs["exposure_entries"])
            ]
            == baseline_docs["exposure_entries"]
        ),
        "exposure_entries_appended": (
            len(current_exposure.get("entries", []))
            >= len(baseline_docs["exposure_entries"]) + 3
        ),
        "evaluation_readme_updated": (
            sha256_file(WORKSPACE / "Evaluation" / "README.md")
            != baseline_docs["evaluation_readme_sha256"]
            and EXPERIMENT_ID
            in (WORKSPACE / "Evaluation" / "README.md").read_text(
                encoding="utf-8"
            )
        ),
        "h_readme_required_markers": all(
            value
            in (H_ROOT / "README.md").read_text(encoding="utf-8")
            for value in (
                EXPERIMENT_ID,
                EXPERIMENT_TYPE,
                "DIAGNOSTIC_ONLY",
                "promotion_evidence=false",
                "2026-04-03..2026-07-03",
                "2026-07-06..2026-07-17",
            )
        ),
    }
    comparison["document_constraints"] = document_constraints
    if not all(document_constraints.values()):
        comparison["status"] = "FAIL"
    write_json(AFTER_PATH, after)
    write_json(COMPARISON_PATH, comparison)
    if comparison["status"] != "PASS_TASK_RELATIVE_PER_FILE":
        raise ValueError("protected task-relative comparison failed")
    return comparison


def check(
    name: str,
    expected: Any,
    observed: Any,
    evidence: Any,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "expected": expected,
        "observed": observed,
        "evidence": evidence,
        "passed": bool(passed),
    }


def manual_score_check() -> dict[str, Any]:
    weekly = pd.read_csv(EVALUATION_ROOT / "TEST_H5_weekly_scores.csv")
    holdings = pd.read_csv(EVALUATION_ROOT / "TEST_H5_holding_details.csv")
    recomputed_weekly = (
        holdings.groupby("round_id", sort=False)["weighted_return"].sum().to_dict()
    )
    weekly_match = all(
        abs(
            float(row.weekly_return_gross)
            - float(recomputed_weekly[row.round_id])
        )
        <= 1e-12
        for row in weekly.itertuples()
    )
    gross = 1.0
    net10 = 1.0
    for row in weekly.sort_values("round_id").itertuples():
        gross *= 1.0 + float(row.weekly_return_gross)
        net10 *= 1.0 + float(row.weekly_return_net_10bps)
    score = read_json(EVALUATION_ROOT / "TEST_H5_TWO_WEEK_SCORE.json")
    expected = score["two_week"]["TEST-H5"]
    payload = {
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "weekly_holding_sum_match": weekly_match,
        "recomputed_two_week_gross_compound": gross - 1.0,
        "reported_two_week_gross_compound": expected[
            "two_week_gross_compound"
        ],
        "recomputed_two_week_net_10bps_compound": net10 - 1.0,
        "reported_two_week_net_10bps_compound": expected[
            "two_week_net_10bps_compound"
        ],
    }
    payload["passed"] = (
        weekly_match
        and abs(
            payload["recomputed_two_week_gross_compound"]
            - payload["reported_two_week_gross_compound"]
        )
        <= 1e-12
        and abs(
            payload["recomputed_two_week_net_10bps_compound"]
            - payload["reported_two_week_net_10bps_compound"]
        )
        <= 1e-12
    )
    write_json(MANUAL_CHECK_PATH, payload)
    return payload


def build_audit() -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in EVENT_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stages = [event["stage"] for event in events]
    required_order = [
        "code_config_frozen",
        "round_inputs_prepared",
        "cuda_smoke_passed",
        "round_training_started",
        "round_prediction_locked",
        "round_training_started",
        "round_prediction_locked",
        "dual_prediction_locks_verified",
        "scoring_started_after_dual_lock",
        "scoring_completed",
    ]
    access_events = [
        json.loads(line)
        for line in ACCESS_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    freeze = read_json(EVALUATION_ROOT / "TEST_H5_CODE_CONFIG_FREEZE.json")
    freeze_matches = all(
        (WORKSPACE / name).stat().st_size == int(metadata["bytes"])
        and sha256_file(WORKSPACE / name) == metadata["sha256"]
        for name, metadata in freeze["files"].items()
    )
    locks: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    configs: dict[str, Any] = {}
    lock_validity = True
    for round_key, round_id in ROUND_IDS.items():
        lock_path = RUN_ROOT / round_id / "prediction_lock.json"
        lock = read_json(lock_path)
        try:
            validate_prediction_lock(lock)
        except (ValueError, OSError, KeyError):
            lock_validity = False
        locks[round_key] = lock
        manifests[round_key] = read_json(
            RUN_ROOT / round_id / "training" / "training_manifest.json"
        )
        configs[round_key] = read_json(
            RUN_ROOT / round_id / "frozen_config.json"
        )
    expected_samples = {
        "W1": (337852, 300, 1083, "2026-06-26", "2026-07-03"),
        "W2": (339352, 300, 1088, "2026-07-03", "2026-07-10"),
    }
    boundaries_pass = all(
        (
            int(locks[key]["input"]["rows"]),
            int(locks[key]["input"]["stocks"]),
            int(locks[key]["input"]["sample_count"]),
            locks[key]["input"]["sample_end"],
            locks[key]["input"]["max_label_end"],
        )
        == expected_samples[key]
        for key in ("W1", "W2")
    )
    ordinary_copies = all(
        locks[key]["input"]["samefile"] is False
        and locks[key]["input"]["readonly"] is True
        and locks[key]["input"]["source_sha256"]
        == locks[key]["input"]["copy_sha256"]
        for key in ("W1", "W2")
    )
    production_pass = all(
        manifest["fixed_epoch"] == 1
        and manifest["epochs_completed"] == 1
        and manifest["validation_evaluation_count"] == 0
        and manifest["early_stopping"] is False
        and manifest["checkpoint_selection"] is False
        and manifest["from_scratch"] is True
        and manifest["parent_checkpoint"] is None
        and manifest["optimizer_loaded"] is False
        and manifest["scaler_loaded"] is False
        and manifest["cache_enabled"] is False
        and manifest["cuda_available"] is True
        and manifest["loss"]["finite"] is True
        and manifest["gradient"]["finite"] is True
        and manifest["within_360_seconds"] is True
        for manifest in manifests.values()
    )
    semantic_equal = (
        canonical_semantic_fingerprint(configs["W1"])
        == canonical_semantic_fingerprint(configs["W2"])
    )
    validation_path_leaked = any(
        "validation_20260706_20260717.csv"
        in path.read_text(encoding="utf-8", errors="ignore")
        for path in [
            RUN_ROOT / ROUND_IDS[key] / "frozen_config.json"
            for key in ("W1", "W2")
        ]
        + [
            RUN_ROOT
            / ROUND_IDS[key]
            / "training"
            / "training_manifest.json"
            for key in ("W1", "W2")
        ]
    )
    post_score_model_events = [
        event
        for event in events
        if event["event_index"]
        > next(
            item["event_index"]
            for item in events
            if item["stage"] == "scoring_completed"
        )
        and event["stage"]
        in {
            "round_training_started",
            "round_prediction_locked",
        }
    ]
    protected = read_json(COMPARISON_PATH)
    manual = manual_score_check()
    score = read_json(EVALUATION_ROOT / "TEST_H5_TWO_WEEK_SCORE.json")
    status = read_json(H_ROOT / "TEST_STATUS.json")
    added_h_entries = protected["roots"]["h_series_policy_scope"]["added"]
    forbidden_named_entries = [
        path
        for path in added_h_entries
        if (
            "H01" in Path(path).name
            or "SERIOUS" in Path(path).name.upper()
            or "FINALIST" in Path(path).name.upper()
        )
    ]
    checks = [
        check(
            "stage_order",
            required_order,
            stages,
            str(EVENT_LEDGER.relative_to(WORKSPACE)),
            stages == required_order,
        ),
        check(
            "code_and_config_freeze",
            "all frozen bytes and SHA256 unchanged",
            freeze_matches,
            str(
                (
                    EVALUATION_ROOT / "TEST_H5_CODE_CONFIG_FREEZE.json"
                ).relative_to(WORKSPACE)
            ),
            freeze_matches,
        ),
        check(
            "dual_prediction_locks",
            "W1 and W2 strict locks validate",
            lock_validity,
            [locks[key]["artifacts"] for key in ("W1", "W2")],
            lock_validity,
        ),
        check(
            "isolated_validation_access",
            "one hash-before-parse event after both locks",
            access_events,
            str(ACCESS_LEDGER.relative_to(WORKSPACE)),
            len(access_events) == 1
            and access_events[0].get("event") == "parsed"
            and access_events[0].get("sha256_verified_before_parse") is True
            and access_events[0].get("parse_count") == 1,
        ),
        check(
            "validation_path_absent_from_train_and_infer",
            False,
            validation_path_leaked,
            "two frozen configs and two training manifests",
            not validation_path_leaked,
        ),
        check(
            "ordinary_readonly_training_copies",
            True,
            ordinary_copies,
            [locks[key]["input"] for key in ("W1", "W2")],
            ordinary_copies,
        ),
        check(
            "training_windows_and_samples",
            expected_samples,
            {
                key: (
                    locks[key]["input"]["rows"],
                    locks[key]["input"]["stocks"],
                    locks[key]["input"]["sample_count"],
                    locks[key]["input"]["sample_end"],
                    locks[key]["input"]["max_label_end"],
                )
                for key in ("W1", "W2")
            },
            "prediction lock input sections",
            boundaries_pass,
        ),
        check(
            "production_refit_policy",
            (
                "CUDA; epoch1; no validation/selection/early-stop/resume/cache; "
                "finite loss/grad; <=360s"
            ),
            {
                key: {
                    name: manifests[key].get(name)
                    for name in (
                        "device",
                        "fixed_epoch",
                        "validation_evaluation_count",
                        "from_scratch",
                        "cache_enabled",
                        "within_360_seconds",
                    )
                }
                for key in ("W1", "W2")
            },
            "training manifests",
            production_pass,
        ),
        check(
            "round_semantic_identity",
            True,
            semantic_equal,
            {
                key: canonical_semantic_fingerprint(configs[key])
                for key in ("W1", "W2")
            },
            semantic_equal,
        ),
        check(
            "protected_scopes",
            "PASS_TASK_RELATIVE_PER_FILE",
            protected["status"],
            str(COMPARISON_PATH.relative_to(WORKSPACE)),
            protected["status"] == "PASS_TASK_RELATIVE_PER_FILE",
        ),
        check(
            "manual_score_recalculation",
            True,
            manual["passed"],
            str(MANUAL_CHECK_PATH.relative_to(WORKSPACE)),
            manual["passed"],
        ),
        check(
            "post_score_model_events",
            [],
            post_score_model_events,
            str(EVENT_LEDGER.relative_to(WORKSPACE)),
            not post_score_model_events,
        ),
        check(
            "diagnostic_only_no_formal_gate",
            {
                "status": "DIAGNOSTIC_ONLY",
                "promotion_evidence": False,
                "formal_stability_gate_active": False,
                "cvar_reported": False,
            },
            {
                "status": score["status"],
                "promotion_evidence": score["promotion_evidence"],
                "formal_stability_gate_active": score[
                    "interpretation_constraints"
                ]["formal_stability_gate_active"],
                "cvar_reported": score["interpretation_constraints"][
                    "cvar_reported"
                ],
            },
            "TEST_H5_TWO_WEEK_SCORE.json",
            score["status"] == "DIAGNOSTIC_ONLY"
            and score["promotion_evidence"] is False
            and score["interpretation_constraints"][
                "formal_stability_gate_active"
            ]
            is False
            and score["interpretation_constraints"]["cvar_reported"] is False,
        ),
        check(
            "no_H01_Serious_Finalist_creation",
            [],
            forbidden_named_entries,
            str(H_ROOT.relative_to(WORKSPACE)),
            not forbidden_named_entries
            and status["status"] == "FINAL_REJECTED_AT_H5_SENTINEL",
        ),
    ]
    payload = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
    }
    write_json(AUDIT_PATH, payload)
    if payload["status"] != "PASS":
        failed = [item["name"] for item in checks if not item["passed"]]
        raise ValueError(f"H5 leakage audit failed: {failed}")
    return payload


def _percent(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def write_report(score: dict[str, Any], audit: dict[str, Any]) -> None:
    weekly = pd.read_csv(EVALUATION_ROOT / "TEST_H5_weekly_scores.csv")
    comparison = pd.read_csv(
        EVALUATION_ROOT / "TEST_H5_two_week_comparison.csv"
    )
    h5 = score["two_week"]["TEST-H5"]
    lines = [
        "# TEST H5 两周回溯收益诊断",
        "",
        "> 结论身份：`RETROSPECTIVE_EXPOSED_DIAGNOSTIC` / "
        "`DIAGNOSTIC_ONLY`。这不是盲测、独立留出或晋级证据。",
        "",
        "## 结论",
        "",
        (
            f"H5 在 10 bps/边成本下的两周复合净收益为 "
            f"**{_percent(h5['two_week_net_10bps_compound'])}**；"
            f"相对官方参考差值为 "
            f"**{_percent(score['differences']['vs_official_reference'])}**，"
            f"相对 Candidate-D 为 "
            f"**{_percent(score['differences']['vs_candidate_d'])}**。"
        ),
        "",
        "无论数值高低，本结果固定为诊断用途：训练期和评分期在本任务前均已暴露，"
        "不得用于 H01、Serious、Finalist 或候选晋级。",
        "",
        "## 两周逐周收益",
        "",
        "| 周次 | 买入—卖出 | 选股 | 毛收益 | 10 bps/边净收益 | 等权基准 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in weekly.itertuples():
        lines.append(
            f"| {row.round_id} | {row.buy_date}—{row.sell_date} | "
            f"{row.selected_stocks} | {_percent(row.weekly_return_gross)} | "
            f"{_percent(row.weekly_return_net_10bps)} | "
            f"{_percent(row.benchmark_equal_weight_return)} |"
        )
    lines.extend(
        [
            "",
            "## 两周复合比较",
            "",
            "| 模型 | 毛收益 | 0 bps/边 | 10 bps/边 | 20 bps/边 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples():
        lines.append(
            f"| {row.candidate} | "
            f"{_percent(row.two_week_gross_compound)} | "
            f"{_percent(row.two_week_net_0bps_compound)} | "
            f"{_percent(row.two_week_net_10bps_compound)} | "
            f"{_percent(row.two_week_net_20bps_compound)} |"
        )
    lines.extend(
        [
            "",
            "## 防泄露与执行约束",
            "",
            f"- 审计：`{audit['status']}`，"
            f"{sum(item['passed'] for item in audit['checks'])}/"
            f"{len(audit['checks'])} 项通过。",
            "- W1/W2 使用与其他 candidates 完全相同的冻结训练 CSV 字节；"
            "两轮均 seed=20260726、固定 final epoch=1、从零训练。",
            "- 无验证集选择、早停、恢复训练、W1→W2 热启动或历史缓存命中。",
            "- 两轮各独立推理两次，完整 300 股分数表和严格 Top5 等权结果逐字节一致后锁定。",
            "- 双锁后才校验隔离验证 SHA256，并从同一内存字节仅解析一次；"
            "原始验证文件没有复制到工作区。",
            "- 不启用正式稳定性门禁，不报告 CVaR，不修改 candidate 包。",
            "",
            "## 暴露边界",
            "",
            "- `2026-04-03..2026-07-03` 已永久重分类为 H 系列训练暴露区间，"
            "不得再作 holdout 或 promotion evidence。",
            "- `2026-07-06..2026-07-17` 在本任务前已是评分暴露窗口；"
            "其中 W2 的原始行只纳入到 `2026-07-10`。",
            "",
            "完整机器可读结果见 "
            "`Evaluation/runs/TEST_H5_two_week_retrospective_20260727/`。",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def report() -> dict[str, Any]:
    score = read_json(EVALUATION_ROOT / "TEST_H5_TWO_WEEK_SCORE.json")
    audit = build_audit()
    write_report(score, audit)
    summary = {
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "status": "DIAGNOSTIC_ONLY",
        "audit_status": audit["status"],
        "score_path": str(
            (EVALUATION_ROOT / "TEST_H5_TWO_WEEK_SCORE.json").relative_to(
                WORKSPACE
            )
        ),
        "audit_path": str(AUDIT_PATH.relative_to(WORKSPACE)),
        "report_path": str(REPORT_PATH.relative_to(WORKSPACE)),
    }
    write_json(EVALUATION_ROOT / "TEST_H5_COMPLETION_SUMMARY.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protection, audit, and reporting for TEST H5 two-week diagnostic"
    )
    parser.add_argument(
        "action",
        choices=("snapshot-before", "snapshot-after", "report"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "snapshot-before":
        result = snapshot_before()
    elif args.action == "snapshot-after":
        result = snapshot_after()
    else:
        result = report()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

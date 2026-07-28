from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

import Evaluation.scripts.walk_forward.prospective_d_stability as controller
from Evaluation.scripts.walk_forward.prospective_d_stability import (
    append_hash_chained_event,
    build_candidate_d_runtime_config,
    build_prediction_lock,
    evaluate_final_gates,
    sha256_file,
    store_sealed_score,
    strip_candidate_d_runtime_fields,
    validate_audit_chain,
    validate_scoring_prefix_dates,
    validate_preregistration,
    validate_training_prefix,
    window_phase,
)


CN = ZoneInfo("Asia/Shanghai")


def valid_config() -> dict:
    windows = [
        ("P01", "2026-07-31", "2026-08-03", "2026-08-07"),
        ("P02", "2026-08-07", "2026-08-10", "2026-08-14"),
        ("P03", "2026-08-14", "2026-08-17", "2026-08-21"),
        ("P04", "2026-08-21", "2026-08-24", "2026-08-28"),
        ("P05", "2026-08-28", "2026-08-31", "2026-09-04"),
        ("P06", "2026-09-04", "2026-09-07", "2026-09-11"),
        ("P07", "2026-09-11", "2026-09-14", "2026-09-18"),
        ("P08", "2026-09-18", "2026-09-21", "2026-09-28"),
    ]
    return {
        "schema_version": "test_d_prospective_8w_v1",
        "experiment_id": "TEST_D_PROSPECTIVE_8W_20260727",
        "classification": "PROSPECTIVE_SEQUENTIAL_DIAGNOSTIC",
        "timezone": "Asia/Shanghai",
        "minimum_valid_windows": 8,
        "extend_to_windows_if_inconclusive": 12,
        "performance_blinding": "UNTIL_MINIMUM_WINDOW_COUNT",
        "candidates": ["THU-BDC2026-main", "Candidate-D"],
        "excluded_exposed_periods": [
            {
                "decision_date": "2026-07-03",
                "buy_date": "2026-07-06",
                "sell_date": "2026-07-10",
            },
            {
                "decision_date": "2026-07-10",
                "buy_date": "2026-07-13",
                "sell_date": "2026-07-17",
            },
            {
                "decision_date": "2026-07-17",
                "buy_date": "2026-07-20",
                "sell_date": "2026-07-24",
            },
        ],
        "windows": [
            {
                "window_id": window_id,
                "decision_date": decision,
                "decision_lock_time": f"{decision}T18:00:00+08:00",
                "buy_date": buy,
                "buy_deadline": f"{buy}T09:30:00+08:00",
                "sell_date": sell,
                "score_not_before": f"{sell}T09:31:00+08:00",
            }
            for window_id, decision, buy, sell in windows
        ],
        "frozen_protocol": {
            "candidate_d": {
                "seed": 20260708,
                "epochs": 12,
                "sequence_length": 60,
                "top_k": 2,
                "weights": [0.5, 0.5],
            },
            "cost_bps_per_side": [0, 10, 20],
        },
        "decision_gates": {
            "minimum_valid_windows": 8,
            "minimum_official_excess_wins": 6,
            "mean_official_excess_positive": True,
            "median_official_excess_positive": True,
            "compound_net10_beats_official": True,
            "compound_net10_beats_sse": True,
            "leave_best_two_out_official_excess_positive": True,
            "minimum_single_week_net10": -0.12,
            "cost_direction_must_hold": True,
        },
    }


def candidate_d_config() -> dict:
    return {
        "experiment_id": "Candidate-D_ProductionRefit_20260703",
        "seed": 20260708,
        "features": {"sequence_length": 60},
        "portfolio": {"top_k": 2, "mode": "topk_equal"},
        "training": {"num_epochs": 12, "learning_rate": 0.00005},
        "submission": {
            "raw_data_end": "2026-07-03",
            "train_sample_start": "2022-01-04",
            "train_sample_end": "2026-06-26",
            "expected_train_sample_count": 1083,
        },
    }


def write_fake_candidate_sources(root: Path) -> None:
    official = root / "candidate" / "THU-BDC2026-main"
    candidate_d = root / "candidate" / "Candidate-D"
    for candidate in (official, candidate_d):
        for relative in ("code/src", "data", "output", "temp"):
            (candidate / relative).mkdir(parents=True, exist_ok=True)
        (candidate / "data" / "train.csv").write_text(
            "股票代码,日期,开盘\n000001,2026-07-01,10\n",
            encoding="utf-8",
        )
        (candidate / "data" / "test.csv").write_text(
            "股票代码,日期,开盘\n000001,2099-01-01,99\n",
            encoding="utf-8",
        )
        (candidate / "code" / "src" / "train.py").write_text(
            "print('fake train')\n",
            encoding="utf-8",
        )
    (official / "code" / "src" / "predict.py").write_text(
        "print('fake predict')\n",
        encoding="utf-8",
    )
    (official / "code" / "src" / "config.py").write_text(
        "config = {'seed': 42}\n",
        encoding="utf-8",
    )
    official_model = official / "model" / "60_158+39"
    official_model.mkdir(parents=True)
    (official_model / "config.json").write_text("{}\n", encoding="utf-8")
    (official_model / "best_model.pth").write_bytes(b"old official model")
    (official_model / "scaler.pkl").write_bytes(b"old official scaler")

    (candidate_d / "code" / "src" / "test.py").write_text(
        "print('fake test')\n",
        encoding="utf-8",
    )
    d_model = candidate_d / "model"
    d_model.mkdir(parents=True)
    base = candidate_d_config()
    base["data"] = {"min_stocks_per_day": 10}
    base["label"] = {"sell_offset": 5}
    base["submission"]["train_sample_start"] = "2026-07-01"
    (d_model / "config.json").write_text(
        json.dumps(base, ensure_ascii=False),
        encoding="utf-8",
    )
    (d_model / "final_model.pth").write_bytes(b"old d model")
    (d_model / "scalers.json").write_text("{}\n", encoding="utf-8")


def write_synthetic_market_prefix(path: Path) -> None:
    dates = pd.bdate_range("2026-07-01", "2026-07-31")
    rows = []
    for date in dates:
        for stock in range(1, 11):
            rows.append(
                {
                    "股票代码": f"{stock:06d}",
                    "日期": date.strftime("%Y-%m-%d"),
                    "开盘": 10.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_execution_freeze_fixture(
    root: Path,
) -> tuple[Path, Path, Path, dict[str, str]]:
    program_paths = {
        "causal_prefix": root / "scripts" / "causal.py",
        "outcome_bundle": root / "scripts" / "outcome.py",
        "tencent_fallback": root / "scripts" / "tencent.py",
        "controller": root / "scripts" / "controller.py",
    }
    for key, path in program_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen {key}\n", encoding="utf-8")
    hashes = {
        key: sha256_file(path)
        for key, path in program_paths.items()
    }
    plan = {
        "schema_version": "test_d_prospective_data_plan_v2",
        "finalization_status": "FROZEN_BEFORE_P01_WITH_PROVIDER_FAILOVER",
        "acquisition_program": {
            "path": "scripts/causal.py",
            "sha256": hashes["causal_prefix"],
            "fallback_helper": {
                "path": "scripts/tencent.py",
                "sha256": hashes["tencent_fallback"],
            },
            "source_policy": "baostock_then_tencent_qfq",
            "provider_mix_within_window_allowed": False,
            "future_query_allowed": False,
            "historical_overlap_replacement_allowed": False,
        },
        "outcome_acquisition_program": {
            "path": "scripts/outcome.py",
            "sha256": hashes["outcome_bundle"],
            "fallback_helper": {
                "path": "scripts/tencent.py",
                "sha256": hashes["tencent_fallback"],
            },
            "source_policy": "baostock_then_tencent_qfq",
            "provider_mix_within_window_allowed": False,
            "stocks_and_benchmark_must_switch_together": True,
            "later_window_query_allowed": False,
            "training_workspace_write_allowed": False,
            "candidate_package_write_allowed": False,
        },
        "evaluation_program": {
            "path": "scripts/controller.py",
            "sha256": hashes["controller"],
        },
    }
    plan_path = root / "data-plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    experiment_root = root / "experiment"
    audit_path = experiment_root / "audit" / "events.jsonl"
    append_hash_chained_event(audit_path, {"event": "PROTOCOL_INITIALIZED"})
    append_hash_chained_event(
        audit_path,
        {
            "event": "EXECUTION_FREEZE_ACTIVATED_BEFORE_P01",
            "data_plan_path": str(plan_path.resolve()),
            "data_plan_sha256": sha256_file(plan_path),
            "program_sha256": hashes,
            "source_policy": "baostock_then_tencent_qfq",
            "provider_mix_within_window_allowed": False,
            "performance_values_present": False,
        },
    )
    return (
        plan_path,
        experiment_root,
        program_paths["controller"],
        hashes,
    )


def complete_fake_candidate_process(
    args: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> dict:
    del timeout_seconds
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("fake execution\n", encoding="utf-8")
    if args[1].endswith("train.py"):
        if cwd.name == "THU-BDC2026-main":
            model = cwd / "model" / "60_158+39"
            model.mkdir(parents=True, exist_ok=True)
            (model / "best_model.pth").write_bytes(b"official checkpoint")
            (model / "scaler.pkl").write_bytes(b"official scaler")
            (model / "config.json").write_text("{}\n", encoding="utf-8")
        else:
            (cwd / "model" / "final_model.pth").write_bytes(b"d checkpoint")
            (cwd / "model" / "scalers.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
    else:
        output = cwd / "output" / "result.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        if cwd.name == "Candidate-D":
            output.write_text(
                "stock_id,weight\n000001,0.5\n000002,0.5\n",
                encoding="utf-8",
            )
        else:
            output.write_text(
                "stock_id,weight\n000009,1.0\n",
                encoding="utf-8",
            )
    return {
        "command": args,
        "cwd": str(cwd),
        "started_at": "2026-08-03T09:00:00+08:00",
        "completed_at": "2026-08-03T09:10:00+08:00",
        "elapsed_seconds": 1.0,
        "return_code": 0,
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "environment_keys": [],
        "outcome_paths_in_environment": False,
    }


class ProspectiveDStabilityTests(unittest.TestCase):
    def test_preregistration_accepts_eight_future_nonoverlapping_windows(self) -> None:
        validated = validate_preregistration(valid_config())
        self.assertEqual(len(validated["windows"]), 8)

    def test_preregistration_rejects_fewer_than_eight_windows(self) -> None:
        config = valid_config()
        config["windows"] = config["windows"][:7]
        with self.assertRaisesRegex(ValueError, "at least 8"):
            validate_preregistration(config)

    def test_preregistration_rejects_exposed_or_overlapping_window(self) -> None:
        config = valid_config()
        config["windows"][0]["decision_date"] = "2026-07-10"
        config["windows"][0]["decision_lock_time"] = "2026-07-10T18:00:00+08:00"
        config["windows"][0]["buy_date"] = "2026-07-13"
        config["windows"][0]["buy_deadline"] = "2026-07-13T09:30:00+08:00"
        config["windows"][0]["sell_date"] = "2026-07-17"
        config["windows"][0]["score_not_before"] = "2026-07-17T09:31:00+08:00"
        with self.assertRaisesRegex(ValueError, "exposed"):
            validate_preregistration(config)

    def test_window_phase_enforces_decision_buy_and_score_time_gates(self) -> None:
        window = valid_config()["windows"][0]
        self.assertEqual(
            window_phase(window, datetime(2026, 7, 30, 12, tzinfo=CN), False, False),
            "WAIT_DECISION",
        )
        self.assertEqual(
            window_phase(window, datetime(2026, 7, 31, 18, 1, tzinfo=CN), False, False),
            "PREPARE_AND_LOCK",
        )
        self.assertEqual(
            window_phase(window, datetime(2026, 8, 3, 9, 31, tzinfo=CN), False, False),
            "MISSED",
        )
        self.assertEqual(
            window_phase(window, datetime(2026, 8, 7, 9, 32, tzinfo=CN), True, False),
            "READY_TO_SCORE",
        )
        self.assertEqual(
            window_phase(window, datetime(2026, 8, 7, 9, 32, tzinfo=CN), True, True),
            "SCORED",
        )

    def test_training_prefix_rejects_future_rows_and_requires_decision_date(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.csv"
            frame = pd.DataFrame(
                {
                    "股票代码": ["000001", "000002", "000001", "000002"],
                    "日期": ["2026-07-31", "2026-07-31", "2026-08-03", "2026-08-03"],
                    "开盘": [10.0, 20.0, 11.0, 21.0],
                }
            )
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "future"):
                validate_training_prefix(path, "2026-07-31")
            frame.iloc[:2].to_csv(path, index=False, encoding="utf-8-sig")
            report = validate_training_prefix(path, "2026-07-31")
            self.assertEqual(report["date_end"], "2026-07-31")
            self.assertEqual(report["rows"], 2)

    def test_scoring_prefix_rejects_rows_after_current_sell_date(self) -> None:
        frame = pd.DataFrame(
            {
                "股票代码": ["000001", "000001", "000001"],
                "日期": pd.to_datetime(
                    ["2026-08-03", "2026-08-07", "2026-08-10"]
                ),
                "开盘": [10.0, 11.0, 12.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "later-window"):
            validate_scoring_prefix_dates(frame, "2026-08-07")
        report = validate_scoring_prefix_dates(frame.iloc[:2].copy(), "2026-08-07")
        self.assertEqual(report["date_end"], "2026-08-07")

    def test_candidate_d_runtime_patch_changes_only_causal_metadata(self) -> None:
        base = candidate_d_config()
        runtime = build_candidate_d_runtime_config(
            base,
            experiment_id="TEST_D_PROSPECTIVE_8W_20260727_P01",
            decision_date="2026-07-31",
            train_sample_end="2026-07-24",
            train_sample_count=1098,
        )
        self.assertEqual(strip_candidate_d_runtime_fields(runtime), strip_candidate_d_runtime_fields(base))
        self.assertEqual(runtime["submission"]["raw_data_end"], "2026-07-31")
        self.assertEqual(runtime["submission"]["train_sample_end"], "2026-07-24")
        self.assertEqual(runtime["submission"]["expected_train_sample_count"], 1098)

    def test_prediction_lock_requires_two_identical_results_before_buy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "run1.csv"
            second = root / "run2.csv"
            content = "stock_id,weight\n000001,0.5\n000002,0.5\n"
            first.write_text(content, encoding="utf-8")
            second.write_text(content, encoding="utf-8")
            lock = build_prediction_lock(
                window=valid_config()["windows"][0],
                candidate="Candidate-D",
                locked_at="2026-08-03T09:20:00+08:00",
                training_sha256="A" * 64,
                checkpoint_hashes={"model/final_model.pth": "B" * 64},
                config_hashes={"model/config.json": "C" * 64},
                result_paths=[first, second],
            )
            self.assertTrue(lock["repeated_inference_byte_identical"])
            self.assertEqual(lock["result_sha256"], sha256_file(first))

            second.write_text("stock_id,weight\n000003,0.5\n000004,0.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                build_prediction_lock(
                    window=valid_config()["windows"][0],
                    candidate="Candidate-D",
                    locked_at="2026-08-03T09:20:00+08:00",
                    training_sha256="A" * 64,
                    checkpoint_hashes={"model/final_model.pth": "B" * 64},
                    config_hashes={"model/config.json": "C" * 64},
                    result_paths=[first, second],
                )

    def test_prediction_lock_rejects_post_deadline_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "run1.csv"
            second = root / "run2.csv"
            content = "stock_id,weight\n000001,0.5\n000002,0.5\n"
            first.write_text(content, encoding="utf-8")
            second.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deadline"):
                build_prediction_lock(
                    window=valid_config()["windows"][0],
                    candidate="Candidate-D",
                    locked_at="2026-08-03T09:31:00+08:00",
                    training_sha256="A" * 64,
                    checkpoint_hashes={"model/final_model.pth": "B" * 64},
                    config_hashes={"model/config.json": "C" * 64},
                    result_paths=[first, second],
                )

    def test_append_only_audit_chain_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            append_hash_chained_event(path, {"event": "INIT"})
            append_hash_chained_event(path, {"event": "LOCK", "window_id": "P01"})
            self.assertEqual(validate_audit_chain(path)["event_count"], 2)
            rows = path.read_text(encoding="utf-8").splitlines()
            changed = json.loads(rows[0])
            changed["event"]["event"] = "TAMPERED"
            rows[0] = json.dumps(changed, ensure_ascii=False, sort_keys=True)
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audit chain"):
                validate_audit_chain(path)

    def test_execution_freeze_verifies_plan_programs_and_leakage_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, experiment, controller_path, hashes = (
                write_execution_freeze_fixture(root)
            )
            report = controller.verify_execution_freeze(
                data_plan_path=plan,
                experiment_root=experiment,
                repository_root=root,
                controller_path=controller_path,
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["program_sha256"], hashes)
            self.assertFalse(report["provider_mix_within_window_allowed"])

    def test_execution_freeze_rejects_plan_or_program_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, experiment, controller_path, _ = (
                write_execution_freeze_fixture(root)
            )
            causal = root / "scripts" / "causal.py"
            causal.write_text("# drift\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "program hash mismatch"):
                controller.verify_execution_freeze(
                    data_plan_path=plan,
                    experiment_root=experiment,
                    repository_root=root,
                    controller_path=controller_path,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, experiment, controller_path, _ = (
                write_execution_freeze_fixture(root)
            )
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload["acquisition_program"]["future_query_allowed"] = True
            plan.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "data plan hash mismatch"):
                controller.verify_execution_freeze(
                    data_plan_path=plan,
                    experiment_root=experiment,
                    repository_root=root,
                    controller_path=controller_path,
                )

    def test_execution_freeze_requires_activation_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, experiment, controller_path, _ = (
                write_execution_freeze_fixture(root)
            )
            audit = experiment / "audit" / "events.jsonl"
            first = audit.read_text(encoding="utf-8").splitlines()[0]
            audit.write_text(first + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "activation event"):
                controller.verify_execution_freeze(
                    data_plan_path=plan,
                    experiment_root=experiment,
                    repository_root=root,
                    controller_path=controller_path,
                )

    def test_synthetic_dry_run_and_sanitized_process_runner(self) -> None:
        dry_run = controller.synthetic_dry_run(valid_config())
        self.assertEqual(dry_run["status"], "PASS")
        self.assertFalse(dry_run["uses_real_future_data"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution = controller.run_process(
                [controller.sys.executable, "-c", "print('SAFE_PROCESS_OK')"],
                cwd=root,
                log_path=root / "process.log",
                timeout_seconds=30,
            )
            self.assertEqual(execution["return_code"], 0)
            self.assertFalse(execution["outcome_paths_in_environment"])
            self.assertNotIn(
                "SAFE_PROCESS_OK",
                " ".join(execution["environment_keys"]),
            )
            with self.assertRaisesRegex(ValueError, "forbidden token"):
                controller.run_process(
                    [controller.sys.executable, "-c", "print('outcome')"],
                    cwd=root,
                    log_path=root / "forbidden.log",
                    timeout_seconds=30,
                )

    def test_sealed_score_keeps_performance_out_of_public_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vault = root / "external-vault"
            receipt = root / "public" / "P01.json"
            secret = {
                "window_id": "P01",
                "candidate": "Candidate-D",
                "weekly_return_gross": 0.03,
                "weekly_return_net_10bps": 0.026,
            }
            public = store_sealed_score(
                vault_root=vault,
                public_receipt_path=receipt,
                window_id="P01",
                scored_at="2026-08-07T09:35:00+08:00",
                score_not_before="2026-08-07T09:31:00+08:00",
                outcome_sha256="D" * 64,
                secret_payload=secret,
            )
            public_text = receipt.read_text(encoding="utf-8")
            self.assertNotIn("weekly_return", public_text)
            self.assertNotIn("0.03", public_text)
            self.assertEqual(public["status"], "SEALED")
            self.assertTrue((vault / "sealed_scores" / "P01.json").is_file())

    def test_sealed_score_recovers_after_secret_only_partial_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vault = root / "external-vault"
            receipt = root / "public" / "P01.json"
            secret = {
                "window_id": "P01",
                "candidate_scores": {"Candidate-D": {"net_10bps": 0.01}},
            }
            first = store_sealed_score(
                vault_root=vault,
                public_receipt_path=receipt,
                window_id="P01",
                scored_at="2026-08-07T09:35:00+08:00",
                score_not_before="2026-08-07T09:31:00+08:00",
                outcome_sha256="D" * 64,
                secret_payload=secret,
            )
            receipt.unlink()
            recovered = store_sealed_score(
                vault_root=vault,
                public_receipt_path=receipt,
                window_id="P01",
                scored_at="2026-08-07T09:40:00+08:00",
                score_not_before="2026-08-07T09:31:00+08:00",
                outcome_sha256="D" * 64,
                secret_payload=secret,
            )
            repeated = store_sealed_score(
                vault_root=vault,
                public_receipt_path=receipt,
                window_id="P01",
                scored_at="2026-08-07T09:45:00+08:00",
                score_not_before="2026-08-07T09:31:00+08:00",
                outcome_sha256="D" * 64,
                secret_payload=secret,
            )
            self.assertEqual(
                first["sealed_payload_sha256"],
                recovered["sealed_payload_sha256"],
            )
            self.assertEqual(recovered, repeated)
            with self.assertRaisesRegex(ValueError, "conflict"):
                store_sealed_score(
                    vault_root=vault,
                    public_receipt_path=receipt,
                    window_id="P01",
                    scored_at="2026-08-07T09:50:00+08:00",
                    score_not_before="2026-08-07T09:31:00+08:00",
                    outcome_sha256="E" * 64,
                    secret_payload=secret,
                )

    def test_train_lock_resumes_second_candidate_without_retraining_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fake_candidate_sources(root)
            config_path = root / "protocol.json"
            config_path.write_text(
                json.dumps(valid_config(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            experiment = root / "experiment"
            public = root / "public"
            prefix = root / "market-prefix.csv"
            write_synthetic_market_prefix(prefix)
            calls: list[tuple[str, str]] = []
            fail_candidate_d_once = True

            def flaky_process(
                args: list[str],
                *,
                cwd: Path,
                log_path: Path,
                timeout_seconds: int,
            ) -> dict:
                nonlocal fail_candidate_d_once
                action = "train" if args[1].endswith("train.py") else "predict"
                calls.append((cwd.name, action))
                if (
                    cwd.name == "Candidate-D"
                    and action == "train"
                    and fail_candidate_d_once
                ):
                    fail_candidate_d_once = False
                    raise RuntimeError("injected Candidate-D interruption")
                return complete_fake_candidate_process(
                    args,
                    cwd=cwd,
                    log_path=log_path,
                    timeout_seconds=timeout_seconds,
                )

            with mock.patch.object(controller, "ROOT", root):
                controller.initialize_protocol(config_path, experiment, public)
                round_dir = controller.prepare_window(
                    config_path=config_path,
                    experiment_root=experiment,
                    window_id="P01",
                    market_prefix=prefix,
                    now=datetime(2026, 7, 31, 18, 30, tzinfo=CN),
                )
                with (
                    mock.patch.object(
                        controller,
                        "run_process",
                        side_effect=flaky_process,
                    ),
                    mock.patch.object(
                        controller,
                        "local_now",
                        return_value=datetime(2026, 8, 3, 9, 20, tzinfo=CN),
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "interruption"):
                        controller.train_and_lock_window(
                            config_path=config_path,
                            experiment_root=experiment,
                            window_id="P01",
                            now=datetime(2026, 8, 3, 9, 0, tzinfo=CN),
                            python_executable=Path("python"),
                        )
                    locks = controller.train_and_lock_window(
                        config_path=config_path,
                        experiment_root=experiment,
                        window_id="P01",
                        now=datetime(2026, 8, 3, 9, 10, tzinfo=CN),
                        python_executable=Path("python"),
                    )
            self.assertEqual(len(locks), 2)
            self.assertEqual(
                calls.count(("THU-BDC2026-main", "train")),
                1,
            )
            self.assertTrue(
                (
                    round_dir
                    / "candidates"
                    / "THU-BDC2026-main"
                    / "prediction_lock.json"
                ).is_file()
            )
            events = [
                json.loads(line)["event"]
                for line in (
                    experiment / "audit" / "events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                sum(
                    event.get("event") == "PREDICTIONS_LOCKED"
                    for event in events
                ),
                1,
            )

    def test_prepare_cleans_failed_staging_and_retries_without_partial_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fake_candidate_sources(root)
            config_path = root / "protocol.json"
            config_path.write_text(
                json.dumps(valid_config(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            experiment = root / "experiment"
            public = root / "public"
            prefix = root / "market-prefix.csv"
            write_synthetic_market_prefix(prefix)
            original_copytree = controller.shutil.copytree
            copy_calls = 0

            def fail_second_candidate_copy(*args, **kwargs):
                nonlocal copy_calls
                copy_calls += 1
                if copy_calls == 2:
                    raise RuntimeError("injected copy interruption")
                return original_copytree(*args, **kwargs)

            with mock.patch.object(controller, "ROOT", root):
                controller.initialize_protocol(config_path, experiment, public)
                with mock.patch.object(
                    controller.shutil,
                    "copytree",
                    side_effect=fail_second_candidate_copy,
                ):
                    with self.assertRaisesRegex(RuntimeError, "copy interruption"):
                        controller.prepare_window(
                            config_path=config_path,
                            experiment_root=experiment,
                            window_id="P01",
                            market_prefix=prefix,
                            now=datetime(2026, 7, 31, 18, 30, tzinfo=CN),
                        )
                self.assertFalse((experiment / "rounds" / "P01").exists())
                self.assertFalse(
                    (experiment / "rounds" / ".P01.preparing").exists()
                )
                round_dir = controller.prepare_window(
                    config_path=config_path,
                    experiment_root=experiment,
                    window_id="P01",
                    market_prefix=prefix,
                    now=datetime(2026, 7, 31, 18, 35, tzinfo=CN),
                )
            self.assertTrue((round_dir / "round_manifest.json").is_file())

    def test_prepare_reconciles_missing_audit_event_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fake_candidate_sources(root)
            config_path = root / "protocol.json"
            config_path.write_text(
                json.dumps(valid_config(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            experiment = root / "experiment"
            public = root / "public"
            prefix = root / "market-prefix.csv"
            write_synthetic_market_prefix(prefix)
            real_append = controller.append_hash_chained_event
            fail_once = True

            def interrupted_append(path: Path, event: dict) -> dict:
                nonlocal fail_once
                if event.get("event") == "WINDOW_PREPARED" and fail_once:
                    fail_once = False
                    raise RuntimeError("injected audit interruption")
                return real_append(path, event)

            with mock.patch.object(controller, "ROOT", root):
                controller.initialize_protocol(config_path, experiment, public)
                with mock.patch.object(
                    controller,
                    "append_hash_chained_event",
                    side_effect=interrupted_append,
                ):
                    with self.assertRaisesRegex(RuntimeError, "audit interruption"):
                        controller.prepare_window(
                            config_path=config_path,
                            experiment_root=experiment,
                            window_id="P01",
                            market_prefix=prefix,
                            now=datetime(2026, 7, 31, 18, 30, tzinfo=CN),
                        )
                round_dir = controller.prepare_window(
                    config_path=config_path,
                    experiment_root=experiment,
                    window_id="P01",
                    market_prefix=prefix,
                    now=datetime(2026, 7, 31, 18, 35, tzinfo=CN),
                )
                repeated = controller.prepare_window(
                    config_path=config_path,
                    experiment_root=experiment,
                    window_id="P01",
                    market_prefix=prefix,
                    now=datetime(2026, 7, 31, 18, 40, tzinfo=CN),
                )
            self.assertEqual(round_dir, repeated)
            prepared_events = controller.audit_events_for_window(
                experiment / "audit" / "events.jsonl",
                event_name="WINDOW_PREPARED",
                window_id="P01",
            )
            self.assertEqual(len(prepared_events), 1)

    def test_final_gate_requires_robust_excess_not_one_lucky_week(self) -> None:
        rows = []
        for index in range(8):
            d_net10 = 0.02 if index < 6 else -0.01
            official_net10 = 0.005
            rows.append(
                {
                    "window_id": f"P{index + 1:02d}",
                    "candidate_d_net_0bps": d_net10 + 0.002,
                    "candidate_d_net_10bps": d_net10,
                    "candidate_d_net_20bps": d_net10 - 0.002,
                    "official_net_10bps": official_net10,
                    "sse_return": 0.0,
                }
            )
        decision = evaluate_final_gates(rows, valid_config()["decision_gates"])
        self.assertEqual(decision["decision"], "PASS")
        self.assertTrue(decision["gates"]["leave_best_two_out_official_excess_positive"])

        unstable = deepcopy(rows)
        for row in unstable:
            row["candidate_d_net_10bps"] = -0.01
        unstable[0]["candidate_d_net_10bps"] = 0.50
        decision = evaluate_final_gates(unstable, valid_config()["decision_gates"])
        self.assertEqual(decision["decision"], "FAIL")
        self.assertFalse(decision["gates"]["median_official_excess_positive"])

    def test_borderline_eight_window_result_extends_without_changing_rules(self) -> None:
        rows = []
        for index in range(8):
            rows.append(
                {
                    "window_id": f"P{index + 1:02d}",
                    "candidate_d_net_0bps": 0.012,
                    "candidate_d_net_10bps": 0.010,
                    "candidate_d_net_20bps": 0.008,
                    "official_net_10bps": 0.005,
                    "sse_return": 0.03,
                }
            )
        gates = deepcopy(valid_config()["decision_gates"])
        gates["minimum_official_excess_win_rate_for_extension"] = 0.75
        gates["extension_at_eight"] = {
            "eligible_if_failed_non_safety_gate_count_is_at_most": 2,
            "hard_fail_gates": [
                "minimum_single_week_net10",
                "cost_direction_must_hold",
            ],
        }
        decision = evaluate_final_gates(rows, gates)
        self.assertEqual(decision["decision"], "INCONCLUSIVE_EXTEND")
        self.assertEqual(decision["next_required_window_count"], 12)

    def test_hard_tail_failure_does_not_receive_extension(self) -> None:
        rows = []
        for index in range(8):
            rows.append(
                {
                    "window_id": f"P{index + 1:02d}",
                    "candidate_d_net_0bps": 0.022,
                    "candidate_d_net_10bps": -0.13 if index == 7 else 0.020,
                    "candidate_d_net_20bps": -0.132 if index == 7 else 0.018,
                    "official_net_10bps": 0.005,
                    "sse_return": 0.0,
                }
            )
        gates = deepcopy(valid_config()["decision_gates"])
        gates["extension_at_eight"] = {
            "eligible_if_failed_non_safety_gate_count_is_at_most": 2,
            "hard_fail_gates": [
                "minimum_single_week_net10",
                "cost_direction_must_hold",
            ],
        }
        decision = evaluate_final_gates(rows, gates)
        self.assertEqual(decision["decision"], "FAIL")
        self.assertFalse(decision["gates"]["minimum_single_week_net10"])

    def test_synthetic_protocol_lifecycle_keeps_scores_sealed_until_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fake_candidate_sources(root)
            config_path = root / "protocol.json"
            config_path.write_text(
                json.dumps(valid_config(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            experiment = root / "experiment"
            public = root / "public"
            vault = root / "external-vault"
            prefix = root / "market-prefix.csv"
            write_synthetic_market_prefix(prefix)

            with mock.patch.object(controller, "ROOT", root):
                state = controller.initialize_protocol(config_path, experiment, public)
                self.assertFalse(state["performance_values_present"])
                repeated_state = controller.initialize_protocol(
                    config_path,
                    experiment,
                    public,
                )
                self.assertEqual(
                    repeated_state["preregistration_sha256"],
                    state["preregistration_sha256"],
                )
                before_status = controller.protocol_status(
                    config_path=config_path,
                    experiment_root=experiment,
                    public_run=public,
                    now=datetime(2026, 7, 30, 12, tzinfo=CN),
                )
                self.assertEqual(before_status["windows"][0]["phase"], "WAIT_DECISION")
                round_dir = controller.prepare_window(
                    config_path=config_path,
                    experiment_root=experiment,
                    window_id="P01",
                    market_prefix=prefix,
                    now=datetime(2026, 7, 31, 18, 30, tzinfo=CN),
                )
                self.assertFalse(
                    (
                        round_dir
                        / "candidates"
                        / "Candidate-D"
                        / "data"
                        / "test.csv"
                    ).exists()
                )

                def fake_run_process(
                    args: list[str],
                    *,
                    cwd: Path,
                    log_path: Path,
                    timeout_seconds: int,
                ) -> dict:
                    del timeout_seconds
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text("fake execution\n", encoding="utf-8")
                    if args[1].endswith("train.py"):
                        if cwd.name == "THU-BDC2026-main":
                            model = cwd / "model" / "60_158+39"
                            model.mkdir(parents=True, exist_ok=True)
                            (model / "best_model.pth").write_bytes(b"official checkpoint")
                            (model / "scaler.pkl").write_bytes(b"official scaler")
                            (model / "config.json").write_text(
                                "{}\n",
                                encoding="utf-8",
                            )
                        else:
                            (cwd / "model" / "final_model.pth").write_bytes(
                                b"d checkpoint"
                            )
                            (cwd / "model" / "scalers.json").write_text(
                                "{}\n",
                                encoding="utf-8",
                            )
                    else:
                        output = cwd / "output" / "result.csv"
                        output.parent.mkdir(parents=True, exist_ok=True)
                        if cwd.name == "Candidate-D":
                            output.write_text(
                                "stock_id,weight\n000001,0.5\n000002,0.5\n",
                                encoding="utf-8",
                            )
                        else:
                            output.write_text(
                                "stock_id,weight\n000009,1.0\n",
                                encoding="utf-8",
                            )
                    return {
                        "command": args,
                        "cwd": str(cwd),
                        "started_at": "2026-08-03T09:00:00+08:00",
                        "completed_at": "2026-08-03T09:10:00+08:00",
                        "elapsed_seconds": 1.0,
                        "return_code": 0,
                        "log_path": str(log_path),
                        "log_sha256": sha256_file(log_path),
                        "environment_keys": [],
                        "outcome_paths_in_environment": False,
                    }

                with (
                    mock.patch.object(
                        controller,
                        "run_process",
                        side_effect=fake_run_process,
                    ),
                    mock.patch.object(
                        controller,
                        "local_now",
                        return_value=datetime(2026, 8, 3, 9, 20, tzinfo=CN),
                    ),
                ):
                    locks = controller.train_and_lock_window(
                        config_path=config_path,
                        experiment_root=experiment,
                        window_id="P01",
                        now=datetime(2026, 8, 3, 9, 0, tzinfo=CN),
                        python_executable=Path("python"),
                    )
                self.assertEqual(len(locks), 2)

                outcome_rows = []
                for date, multiplier in (
                    ("2026-08-03", 1.0),
                    ("2026-08-07", 1.1),
                ):
                    for stock in range(1, 11):
                        outcome_rows.append(
                            {
                                "股票代码": f"{stock:06d}",
                                "日期": date,
                                "开盘": (
                                    10.0
                                    if stock == 9
                                    else 10.0 * multiplier
                                ),
                            }
                        )
                outcome = root / "outcome.csv"
                pd.DataFrame(outcome_rows).to_csv(
                    outcome,
                    index=False,
                    encoding="utf-8-sig",
                )
                sse = root / "sse.csv"
                pd.DataFrame(
                    {
                        "日期": ["2026-08-03", "2026-08-07"],
                        "开盘": [100.0, 100.0],
                    }
                ).to_csv(sse, index=False, encoding="utf-8-sig")
                receipt = controller.score_window(
                    config_path=config_path,
                    experiment_root=experiment,
                    public_run=public,
                    vault_root=vault,
                    window_id="P01",
                    outcome_market=outcome,
                    sse_open_csv=sse,
                    now=datetime(2026, 8, 7, 9, 35, tzinfo=CN),
                )
                repeated_receipt = controller.score_window(
                    config_path=config_path,
                    experiment_root=experiment,
                    public_run=public,
                    vault_root=vault,
                    window_id="P01",
                    outcome_market=outcome,
                    sse_open_csv=sse,
                    now=datetime(2026, 8, 7, 9, 40, tzinfo=CN),
                )
                self.assertEqual(receipt, repeated_receipt)
                score_events = [
                    json.loads(line)["event"]
                    for line in (
                        experiment / "audit" / "events.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                    if json.loads(line)["event"].get("event")
                    == "SCORE_SEALED"
                ]
                self.assertEqual(len(score_events), 1)
                self.assertEqual(
                    controller.read_json(experiment / "state.json")[
                        "sealed_score_count"
                    ],
                    1,
                )
                self.assertFalse(receipt["performance_fields_disclosed"])
                self.assertNotIn(
                    "weekly_return",
                    (public / "sealed_receipts" / "P01.json").read_text(
                        encoding="utf-8"
                    ),
                )
                after_status = controller.protocol_status(
                    config_path=config_path,
                    experiment_root=experiment,
                    public_run=public,
                    now=datetime(2026, 8, 7, 9, 40, tzinfo=CN),
                )
                self.assertEqual(after_status["windows"][0]["phase"], "SCORED")

                for index in range(2, 9):
                    window_id = f"P{index:02d}"
                    secret = {
                        "window_id": window_id,
                        "candidate_scores": {
                            "Candidate-D": {
                                "gross": 0.022,
                                "net_0bps": 0.022,
                                "net_10bps": 0.020,
                                "net_20bps": 0.018,
                            },
                            "THU-BDC2026-main": {
                                "gross": 0.007,
                                "net_0bps": 0.007,
                                "net_10bps": 0.005,
                                "net_20bps": 0.003,
                            },
                        },
                        "sse_return": 0.0,
                        "hs300_equal_weight_return": 0.0,
                    }
                    store_sealed_score(
                        vault_root=vault,
                        public_receipt_path=public
                        / "sealed_receipts"
                        / f"{window_id}.json",
                        window_id=window_id,
                        scored_at="2026-12-01T10:00:00+08:00",
                        score_not_before="2026-11-01T09:31:00+08:00",
                        outcome_sha256=f"{index}" * 64,
                        secret_payload=secret,
                    )
                with self.assertRaisesRegex(ValueError, "audit"):
                    controller.finalize_scores(
                        config_path=config_path,
                        experiment_root=experiment,
                        public_run=public,
                        vault_root=vault,
                    )
                for index in range(2, 9):
                    window_id = f"P{index:02d}"
                    public_receipt = controller.read_json(
                        public / "sealed_receipts" / f"{window_id}.json"
                    )
                    append_hash_chained_event(
                        experiment / "audit" / "events.jsonl",
                        {
                            "event": "SCORE_SEALED",
                            "at": "2026-12-01T10:00:00+08:00",
                            "window_id": window_id,
                            "outcome_sha256": public_receipt["outcome_sha256"],
                            "sealed_payload_sha256": public_receipt[
                                "sealed_payload_sha256"
                            ],
                            "performance_values_present": False,
                        },
                    )
                persisted_state = controller.read_json(experiment / "state.json")
                persisted_state["sealed_score_count"] = 8
                controller.write_json(experiment / "state.json", persisted_state)
                report = controller.finalize_scores(
                    config_path=config_path,
                    experiment_root=experiment,
                    public_run=public,
                    vault_root=vault,
                )
                self.assertEqual(report["decision"]["decision"], "PASS")
                audit = controller.integrity_audit(
                    config_path=config_path,
                    experiment_root=experiment,
                    public_run=public,
                )
                self.assertEqual(audit["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

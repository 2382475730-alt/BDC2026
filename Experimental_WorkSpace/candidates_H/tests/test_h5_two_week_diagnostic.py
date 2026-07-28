from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "code" / "src"
sys.path.insert(0, str(SRC))

from dataset import DSample  # noqa: E402
from featurework import (  # noqa: E402
    AMOUNT_COL,
    CLOSE_COL,
    DATE_COL,
    HIGH_COL,
    LOW_COL,
    OPEN_COL,
    STOCK_COL,
    TURNOVER_COL,
    VOLUME_COL,
)
from production import (  # noqa: E402
    OneShotValidationReader,
    assert_byte_identical,
    assert_lock_pair_ready,
    build_prediction_lock,
    build_training_samples,
    canonical_semantic_fingerprint,
    infer_unlabeled,
    rank_top5_equal,
    train_fixed_final_epoch,
    validate_fixed_production_config,
    validate_prediction_lock,
)


def sample(
    sample_date: str,
    label_end: str | None,
    *,
    stocks: int = 5,
) -> DSample:
    return DSample(
        sample_date=pd.Timestamp(sample_date),
        label_end_date=pd.Timestamp(label_end) if label_end else None,
        market_features=np.asarray([1.0, 2.0], dtype=np.float32),
        factor_features=np.ones((stocks, 2), dtype=np.float32),
        stock_sequence=np.ones((stocks, 3, 2), dtype=np.float32),
        labels=np.linspace(-0.02, 0.03, stocks, dtype=np.float32),
        relevance=np.linspace(-1.0, 1.0, stocks, dtype=np.float32),
        stock_codes=[f"{index:06d}" for index in range(stocks)],
    )


def frozen_config() -> dict:
    return {
        "experiment_id": "TEST_H5_TWO_WEEK_RETROSPECTIVE_EXPOSED_DIAGNOSTIC",
        "seed": 20260726,
        "data": {
            "train_csv": "",
            "industry_csv": "",
            "futures_csv": "",
            "date_col": DATE_COL,
            "stock_col": STOCK_COL,
            "min_stocks_per_day": 10,
        },
        "label": {"horizon": 5, "buy_offset": 1, "sell_offset": 5},
        "features": {
            "version": "d_market_v1",
            "sequence_length": 60,
            "enable_industry": False,
            "enable_futures": False,
            "industry_missing_label": "UNKNOWN",
            "futures_lag_days": 0,
        },
        "model": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1,
            "use_factor_encoder": False,
        },
        "loss": {
            "objective_mode": "predicted_boundary_stability",
            "ranking_weight": 1.0,
            "temperature": 1.0,
            "pairwise_weight": 0.25,
            "pairwise_margin": 0.05,
            "portfolio_return_weight": 0.0,
            "portfolio_volatility_weight": 0.0,
            "topk_return_weight": 0.05,
            "downside_weight": 0.08,
            "downside_temperature": 0.02,
            "cvar_weight": 0.04,
            "cvar_alpha": 0.20,
            "positive_tail_weight": 0.02,
            "positive_tail_cap": 0.08,
            "topk": 5,
            "topk_temperature": 0.10,
            "boundary_hard_negative_k": 15,
        },
        "portfolio": {"top_k": 5, "mode": "topk_equal"},
        "training": {
            "batch_size": 4,
            "num_epochs": 1,
            "learning_rate": 0.00005,
            "weight_decay": 0.00001,
            "grad_clip": 5.0,
            "num_workers": 0,
            "final_checkpoint_policy": "fixed_epoch_1",
            "validation_enabled": False,
            "early_stopping_enabled": False,
            "checkpoint_selection_enabled": False,
        },
        "cache": {"enabled": False},
        "production": {
            "from_scratch": True,
            "resume_path": None,
            "parent_checkpoint": None,
            "load_optimizer": False,
            "load_scaler": False,
        },
    }


class TrainingBoundaryTests(unittest.TestCase):
    def test_training_samples_keep_only_labels_available_by_cutoff(self) -> None:
        samples = [
            sample("2026-06-25", "2026-07-02"),
            sample("2026-06-26", "2026-07-03"),
            sample("2026-06-29", "2026-07-06"),
            sample("2026-07-03", None),
        ]
        selected, metadata = build_training_samples(samples, "2026-07-03")
        self.assertEqual(
            [item.sample_date.strftime("%Y-%m-%d") for item in selected],
            ["2026-06-25", "2026-06-26"],
        )
        self.assertEqual(metadata["sample_end"], "2026-06-26")
        self.assertEqual(metadata["max_label_end"], "2026-07-03")
        self.assertTrue(metadata["all_labels_available_by_cutoff"])

    def test_fixed_production_config_rejects_selection_resume_and_cache(self) -> None:
        config = frozen_config()
        validate_fixed_production_config(config)
        mutations = [
            ("training", "num_epochs", 2),
            ("training", "validation_enabled", True),
            ("training", "early_stopping_enabled", True),
            ("training", "checkpoint_selection_enabled", True),
            ("cache", "enabled", True),
            ("production", "from_scratch", False),
            ("production", "resume_path", "old.pth"),
            ("production", "parent_checkpoint", "w1.pth"),
            ("production", "load_optimizer", True),
            ("production", "load_scaler", True),
        ]
        for section, key, bad_value in mutations:
            with self.subTest(section=section, key=key):
                bad = json.loads(json.dumps(config))
                bad[section][key] = bad_value
                with self.assertRaises(ValueError):
                    validate_fixed_production_config(bad)

    def test_semantic_fingerprint_rejects_model_loss_seed_feature_drift(self) -> None:
        config = frozen_config()
        baseline = canonical_semantic_fingerprint(config)
        mutations = [
            ("seed", None, 7),
            ("features", "sequence_length", 30),
            ("model", "d_model", 64),
            ("loss", "objective_mode", "topk_stability"),
            ("loss", "boundary_hard_negative_k", 10),
            ("training", "learning_rate", 0.001),
            ("portfolio", "top_k", 3),
        ]
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                drift = json.loads(json.dumps(config))
                if key is None:
                    drift[section] = value
                else:
                    drift[section][key] = value
                self.assertNotEqual(canonical_semantic_fingerprint(drift), baseline)


class PredictionContractTests(unittest.TestCase):
    def test_top5_equal_is_deterministic_and_strict(self) -> None:
        scores = pd.DataFrame(
            {
                "stock_id": ["000006", "000005", "000004", "000003", "000002", "000001"],
                "score": [0.1, 0.2, 0.2, 0.3, 0.4, 0.5],
            }
        )
        full, result, diagnostics = rank_top5_equal(scores)
        self.assertEqual(
            result["stock_id"].tolist(),
            ["000001", "000002", "000003", "000004", "000005"],
        )
        self.assertEqual(result["weight"].tolist(), [0.2] * 5)
        self.assertAlmostEqual(float(result["weight"].sum()), 1.0)
        self.assertAlmostEqual(diagnostics["hhi"], 0.2)
        self.assertEqual(full.loc[full["rank"] == 5, "stock_id"].item(), "000005")
        self.assertEqual(full.loc[full["rank"] == 6, "stock_id"].item(), "000006")

    def test_top5_rejects_nonfinite_and_too_few_stocks(self) -> None:
        with self.assertRaises(ValueError):
            rank_top5_equal(
                pd.DataFrame({"stock_id": ["a"] * 5, "score": [1, 2, 3, 4, 5]})
            )
        with self.assertRaises(ValueError):
            rank_top5_equal(
                pd.DataFrame(
                    {
                        "stock_id": ["1", "2", "3", "4", "5"],
                        "score": [1, 2, 3, 4, np.nan],
                    }
                )
            )

    def test_two_inference_passes_must_be_byte_identical(self) -> None:
        self.assertEqual(assert_byte_identical(b"same", b"same"), True)
        with self.assertRaises(ValueError):
            assert_byte_identical(b"same", b"same\n")

    def test_lock_validation_detects_tamper_and_requires_strict_top5(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score_frame = pd.DataFrame(
                {
                    "stock_id": [f"{index:06d}" for index in range(300)],
                    "score": np.arange(300, dtype=float),
                }
            )
            full, result, diagnostics = rank_top5_equal(score_frame)
            scores_payload = full.to_csv(
                index=False, float_format="%.12g", lineterminator="\n"
            ).encode()
            result_payload = result.to_csv(
                index=False, float_format="%.12g", lineterminator="\n"
            ).encode()
            artifacts = {}
            for name, payload in {
                "input_copy": b"input",
                "checkpoint": b"checkpoint",
                "scaler": b"scaler",
                "scores_pass1": scores_payload,
                "scores_pass2": scores_payload,
                "result_pass1": result_payload,
                "result_pass2": result_payload,
                "training_manifest": b"manifest",
            }.items():
                path = root / name
                path.write_bytes(payload)
                artifacts[name] = path
            lock = build_prediction_lock(
                round_id="W1",
                decision_date="2026-07-03",
                config=frozen_config(),
                artifacts=artifacts,
                input_metadata={
                    "source_sha256": "source",
                    "copy_sha256": "source",
                    "samefile": False,
                    "readonly": True,
                    "rows": 337852,
                    "stocks": 300,
                    "sample_count": 1083,
                    "sample_end": "2026-06-26",
                    "max_label_end": "2026-07-03",
                },
                prediction_metadata=diagnostics,
            )
            validate_prediction_lock(lock)
            artifacts["checkpoint"].write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                validate_prediction_lock(lock)


class ProductionIntegrationTests(unittest.TestCase):
    def test_cpu_synthetic_fixed_epoch_and_unlabeled_double_inference(self) -> None:
        dates = pd.bdate_range("2025-01-02", periods=72)
        rows = []
        for stock in range(10):
            for index, date in enumerate(dates):
                open_price = 10.0 + stock * 0.2 + index * (0.01 + stock * 0.0002)
                close_price = open_price * (1.0 + (stock - 4.5) * 0.0001)
                volume = 100000.0 + stock * 1000.0 + index * 10.0
                rows.append(
                    {
                        STOCK_COL: stock + 1,
                        DATE_COL: date.strftime("%Y-%m-%d"),
                        OPEN_COL: open_price,
                        CLOSE_COL: close_price,
                        HIGH_COL: max(open_price, close_price) * 1.01,
                        LOW_COL: min(open_price, close_price) * 0.99,
                        VOLUME_COL: volume,
                        AMOUNT_COL: volume * close_price,
                        TURNOVER_COL: 1.0 + stock * 0.01,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "train.csv"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            config = frozen_config()
            config["data"]["train_csv"] = str(data_path)
            run_dir = root / "run"
            manifest = train_fixed_final_epoch(
                config,
                data_path,
                run_dir,
                dates[-1].strftime("%Y-%m-%d"),
                require_cuda=False,
                max_seconds=120.0,
            )
            self.assertEqual(manifest["fixed_epoch"], 1)
            self.assertEqual(manifest["validation_evaluation_count"], 0)
            self.assertFalse((run_dir / "best_model.pth").exists())
            first_scores, first_result, first_meta = infer_unlabeled(
                config,
                data_path,
                manifest["artifacts"]["checkpoint"],
                manifest["artifacts"]["scaler"],
                require_cuda=False,
            )
            second_scores, second_result, second_meta = infer_unlabeled(
                config,
                data_path,
                manifest["artifacts"]["checkpoint"],
                manifest["artifacts"]["scaler"],
                require_cuda=False,
            )
            self.assertTrue(assert_byte_identical(first_scores, second_scores))
            self.assertTrue(assert_byte_identical(first_result, second_result))
            self.assertEqual(first_meta, second_meta)
            self.assertIsNone(first_meta["label_end_date"])
            self.assertTrue(first_meta["labels_zero"])


class ValidationAccessTests(unittest.TestCase):
    def test_reader_blocks_before_both_locks_and_allows_one_verified_parse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation.csv"
            payload = b"stock_id,date,open\n000001,2026-07-06,10.0\n"
            validation.write_bytes(payload)
            expected_sha = __import__("hashlib").sha256(payload).hexdigest()
            ledger = root / "access.jsonl"
            reader = OneShotValidationReader(validation, expected_sha, ledger)
            with self.assertRaises(PermissionError):
                reader.read_csv({"W1": None, "W2": None})
            locks = {"W1": {"verified": True}, "W2": {"verified": True}}
            frame = reader.read_csv(locks)
            self.assertEqual(len(frame), 1)
            with self.assertRaises(PermissionError):
                reader.read_csv(locks)
            events = [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([event["event"] for event in events], ["parsed"])
            second_process_reader = OneShotValidationReader(
                validation, expected_sha, ledger
            )
            with self.assertRaises(PermissionError):
                second_process_reader.read_csv(locks)

    def test_wrong_hash_blocks_parser_and_leaves_no_parsed_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation.csv"
            validation.write_bytes(b"a,b\n1,2\n")
            ledger = root / "access.jsonl"
            reader = OneShotValidationReader(validation, "0" * 64, ledger)
            with self.assertRaises(ValueError):
                reader.read_csv(
                    {"W1": {"verified": True}, "W2": {"verified": True}}
                )
            self.assertFalse(ledger.exists())

    def test_lock_pair_requires_both_rounds_verified(self) -> None:
        with self.assertRaises(PermissionError):
            assert_lock_pair_ready({"W1": {"verified": True}})
        with self.assertRaises(PermissionError):
            assert_lock_pair_ready(
                {"W1": {"verified": True}, "W2": {"verified": False}}
            )
        self.assertTrue(
            assert_lock_pair_ready(
                {"W1": {"verified": True}, "W2": {"verified": True}}
            )
        )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    unittest.main()

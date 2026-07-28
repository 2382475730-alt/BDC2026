from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
G_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = G_ROOT / "code" / "src" / "train.py"
DATA_ROOT = ROOT / "BaoStock_Datasets" / "data" / "fusion_research_20260726"
AUTOMATION_ROOT = G_ROOT / "automation_runs"
FOLDS = [
    ("G_DEV_01_2025_Q1", "2025-01-03", "2025-03-31"),
    ("G_DEV_02_2025_Q2", "2025-04-01", "2025-06-30"),
    ("G_DEV_03_2025_Q3", "2025-07-01", "2025-09-30"),
    ("G_DEV_04_2026_Q1", "2026-01-05", "2026-03-27"),
]
VARIANTS = {
    "TEST_G_CONTROL": G_ROOT / "configs" / "g_control_template.json",
    "TEST_G_FINANCIAL": G_ROOT / "configs" / "g_financial_fusion_template.json",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_data_gates() -> dict:
    industry = read_json(DATA_ROOT / "industry" / "validation.json")
    futures = read_json(DATA_ROOT / "cffex" / "validation.json")
    financial = read_json(DATA_ROOT / "financial" / "validation.json")
    if industry.get("status") != "pass":
        raise RuntimeError("industry data validation gate failed")
    if futures.get("status") != "pass":
        raise RuntimeError("futures data validation gate failed")
    if financial.get("status") != "pass":
        raise RuntimeError("financial data validation gate failed")
    return {"industry": industry, "futures": futures, "financial": financial}


def run_command(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; see {stdout_path} and {stderr_path}"
        )


def smoke(seed: int) -> dict:
    gates = require_data_gates()
    config = read_json(VARIANTS["TEST_G_FINANCIAL"])
    config["seed"] = seed
    config["experiment_id"] = f"TEST_G_SMOKE_S{seed}"
    config["output"]["run_dir"] = str(
        (AUTOMATION_ROOT / f"TEST_G_SMOKE_S{seed}").relative_to(ROOT)
    ).replace("\\", "/")
    config_path = AUTOMATION_ROOT / f"TEST_G_SMOKE_S{seed}_config.json"
    write_json(config_path, config)
    stdout_path = AUTOMATION_ROOT / f"TEST_G_SMOKE_S{seed}_stdout.log"
    stderr_path = AUTOMATION_ROOT / f"TEST_G_SMOKE_S{seed}_stderr.log"
    run_command(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(config_path),
            "--dry-run",
            "--max-dates",
            "180",
        ],
        stdout_path,
        stderr_path,
    )
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = json.loads("\n".join(lines))
    finite = np.isfinite(float(result["loss"]))
    passed = bool(
        finite
        and int(result["stock_feature_dim"]) == 43
        and int(result["market_feature_dim"]) == 25
        and int(result["factor_feature_dim"]) == 8
    )
    summary = {
        "status": "pass" if passed else "fail",
        "seed": seed,
        "data_gates": gates,
        "dry_run": result,
        "test_marker": "TEST",
        "expected_dimensions": {"stock": 43, "factor": 8, "market": 25},
    }
    write_json(AUTOMATION_ROOT / "TEST_G_SMOKE_SUMMARY.json", summary)
    if not passed:
        raise RuntimeError("TEST G-series smoke gate failed")
    return summary


def weekly_metrics(path: Path, variant: str, fold: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["sample_end_date"]).dt.normalize()
    frame["week"] = frame["date"].dt.to_period("W-FRI").astype(str)
    weekly = frame.sort_values("date").groupby("week", as_index=False).tail(1)
    return weekly[["date", "portfolio_return"]].rename(
        columns={"portfolio_return": f"{variant}_return"}
    ).assign(fold=fold)


def paired_summary(paired: pd.DataFrame) -> dict:
    excess = (
        paired["TEST_G_FINANCIAL_return"].astype(float)
        - paired["TEST_G_CONTROL_return"].astype(float)
    )
    ordered = np.sort(excess.to_numpy())
    without_best_two = ordered[:-2] if len(ordered) > 2 else np.asarray([], dtype=float)
    epsilon = 1e-12
    pdr = float(((excess > epsilon).astype(float) + 0.5 * (excess.abs() <= epsilon)).mean())
    fold_means = paired.assign(excess=excess).groupby("fold")["excess"].mean().to_dict()
    recent = float(fold_means.get("G_DEV_04_2026_Q1", float("nan")))
    positive_folds = sum(float(value) > 0.0 for value in fold_means.values())
    result = {
        "paired_week_count": int(len(paired)),
        "mean_weekly_excess": float(excess.mean()),
        "median_weekly_excess": float(excess.median()),
        "paired_direction_rate": pdr,
        "excess_sum": float(excess.sum()),
        "excess_sum_without_best_two_weeks": (
            float(without_best_two.sum()) if len(without_best_two) else None
        ),
        "positive_fold_count": int(positive_folds),
        "fold_mean_excess": {key: float(value) for key, value in fold_means.items()},
        "recent_fold_mean_excess": recent,
    }
    result["quick_gate_passed"] = bool(
        positive_folds >= 3
        and np.isfinite(recent)
        and recent > 0.0
        and pdr >= 0.50
        and result["excess_sum_without_best_two_weeks"] is not None
        and float(result["excess_sum_without_best_two_weeks"]) > 0.0
    )
    return result


def quick(seed: int, epochs: int) -> dict:
    gates = require_data_gates()
    if not (AUTOMATION_ROOT / "TEST_G_SMOKE_SUMMARY.json").exists():
        smoke(seed)
    run_records: list[dict] = []
    weekly_frames: list[pd.DataFrame] = []
    for fold, start, end in FOLDS:
        for variant, template_path in VARIANTS.items():
            run_id = f"TEST_G_QUICK_{variant}_{fold}_S{seed}"
            run_dir = AUTOMATION_ROOT / run_id
            config = read_json(template_path)
            config["experiment_id"] = run_id
            config["seed"] = seed
            config["training"]["num_epochs"] = epochs
            config["training"]["early_stop_patience"] = epochs
            config["validation"]["folds"] = [[fold, start, end]]
            config["output"]["run_dir"] = str(run_dir.relative_to(ROOT)).replace("\\", "/")
            config_path = AUTOMATION_ROOT / f"{run_id}_config.json"
            write_json(config_path, config)
            run_command(
                [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)],
                AUTOMATION_ROOT / f"{run_id}_stdout.log",
                AUTOMATION_ROOT / f"{run_id}_stderr.log",
            )
            summary = read_json(run_dir / "candidate_summary.json")
            run_records.append(
                {
                    "run_id": run_id,
                    "variant": variant,
                    "fold": fold,
                    "fold_start": start,
                    "fold_end": end,
                    "seed": seed,
                    "epochs": int(summary["epochs"]),
                    "best_epoch": int(summary["best_epoch"]),
                    "best_score": float(summary["best_score"]),
                    "elapsed_minutes": float(summary["elapsed_minutes"]),
                }
            )
            weekly_frames.append(
                weekly_metrics(run_dir / "best_val_daily_metrics.csv", variant, fold)
            )

    all_weekly = pd.concat(weekly_frames, ignore_index=True)
    control = all_weekly[all_weekly["TEST_G_CONTROL_return"].notna()][
        ["date", "fold", "TEST_G_CONTROL_return"]
    ]
    fusion = all_weekly[all_weekly["TEST_G_FINANCIAL_return"].notna()][
        ["date", "fold", "TEST_G_FINANCIAL_return"]
    ]
    paired = control.merge(fusion, on=["date", "fold"], how="inner", validate="one_to_one")
    paired["excess"] = (
        paired["TEST_G_FINANCIAL_return"] - paired["TEST_G_CONTROL_return"]
    )
    paired.to_csv(AUTOMATION_ROOT / "TEST_G_QUICK_PAIRED_WEEKLY.csv", index=False)
    comparison = paired_summary(paired)
    result = {
        "schema_version": 1,
        "test_marker": "TEST",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_QUICK" if comparison["quick_gate_passed"] else "STOP_AFTER_QUICK",
        "submission_eligibility": "RESEARCH_ONLY_REPORTING_UNVERIFIED",
        "data_gates": gates,
        "run_records": run_records,
        "comparison": comparison,
        "next_action": (
            "eligible_for_preregistered_multi_seed serious stage"
            if comparison["quick_gate_passed"]
            else "do not expand features, products, seeds, or open sealed holdout"
        ),
    }
    write_json(AUTOMATION_ROOT / "TEST_G_QUICK_SUMMARY.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the preregistered TEST G-series financial experiment.")
    parser.add_argument("--mode", choices=["smoke", "quick"], required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = smoke(args.seed) if args.mode == "smoke" else quick(args.seed, args.epochs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

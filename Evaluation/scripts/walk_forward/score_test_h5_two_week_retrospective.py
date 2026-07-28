from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
H_CODE = (
    WORKSPACE
    / "Experimental_WorkSpace"
    / "candidates_H"
    / "code"
)
H_SRC = H_CODE / "src"
sys.path.insert(0, str(H_SRC))

from production import (  # noqa: E402
    EXPERIMENT_ID,
    EXPERIMENT_TYPE,
    OneShotValidationReader,
    assert_lock_pair_ready,
    sha256_file,
    validate_prediction_lock,
)


H_ROOT = WORKSPACE / "Experimental_WorkSpace" / "candidates_H"
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
BASE_CONFIG_PATH = H_ROOT / "configs" / "TEST_H_QUICK_BASE.json"
DUAL_LOCK_PATH = EVALUATION_ROOT / "TEST_H5_DUAL_LOCK_VERIFICATION.json"
ACCESS_LEDGER = EVALUATION_ROOT / "isolated_validation_access.jsonl"
EVENT_LEDGER = EVALUATION_ROOT / "execution_events.jsonl"
COMPARATOR_WEEKLY_PATH = (
    WORKSPACE
    / "Evaluation"
    / "runs"
    / "tri_candidate_recent_rolling_20260722"
    / "weekly_scores.csv"
)
VALIDATION_SHA256 = (
    "B21FB1BEEFAE407729C305E01E274695FE083C718B695F06A2146D7512B6AFC9"
)
ROUND_SPECS = {
    "W1": {
        "round_id": "W1_20260703",
        "decision_date": "2026-07-03",
        "buy_date": "2026-07-06",
        "sell_date": "2026-07-10",
    },
    "W2": {
        "round_id": "W2_20260710",
        "decision_date": "2026-07-10",
        "buy_date": "2026-07-13",
        "sell_date": "2026-07-17",
    },
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_event(stage: str, **details: Any) -> None:
    lines = (
        EVENT_LEDGER.read_text(encoding="utf-8").splitlines()
        if EVENT_LEDGER.exists()
        else []
    )
    payload = {
        "event_index": len(lines) + 1,
        "at_utc": utc_now(),
        "stage": stage,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        **details,
    }
    with EVENT_LEDGER.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        )


def verified_locks() -> tuple[dict[str, Any], dict[str, Any]]:
    verification = read_json(DUAL_LOCK_PATH)
    if verification.get("both_locks_verified") is not True:
        raise PermissionError("dual lock verification is not complete")
    locks: dict[str, Any] = {}
    readiness: dict[str, Any] = {}
    for round_key, spec in ROUND_SPECS.items():
        path = RUN_ROOT / spec["round_id"] / "prediction_lock.json"
        lock = read_json(path)
        validate_prediction_lock(lock)
        expected_sidecar = path.with_suffix(path.suffix + ".sha256").read_text(
            encoding="ascii"
        ).split()[0]
        if sha256_file(path) != expected_sidecar:
            raise ValueError(f"{round_key} prediction lock sidecar mismatch")
        if (
            verification["locks"][round_key]["lock_sha256"]
            != expected_sidecar
        ):
            raise ValueError(f"{round_key} dual verification hash mismatch")
        locks[round_key] = lock
        readiness[round_key] = {"verified": True}
    assert_lock_pair_ready(readiness)
    return locks, readiness


def normalize_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def weekly_h5_score(
    validation: pd.DataFrame,
    result: pd.DataFrame,
    spec: dict[str, str],
    *,
    code_col: str,
    date_col: str,
    open_col: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    buy_date = pd.Timestamp(spec["buy_date"])
    sell_date = pd.Timestamp(spec["sell_date"])
    buy = validation.loc[
        validation[date_col] == buy_date,
        [code_col, open_col],
    ].rename(columns={open_col: "open_buy"})
    sell = validation.loc[
        validation[date_col] == sell_date,
        [code_col, open_col],
    ].rename(columns={open_col: "open_sell"})
    prices = buy.merge(sell, on=code_col, how="inner", validate="one_to_one")
    if len(prices) != 300:
        raise ValueError(
            f"{spec['round_id']} requires 300 matched validation stocks, "
            f"observed {len(prices)}"
        )
    day = result.merge(
        prices,
        left_on="stock_id",
        right_on=code_col,
        how="left",
        validate="one_to_one",
    )
    if day[["open_buy", "open_sell"]].isna().any().any():
        raise ValueError("H5 selected stock is missing a validation open price")
    day["stock_return"] = day["open_sell"] / day["open_buy"] - 1.0
    day["weighted_return"] = day["weight"] * day["stock_return"]
    gross = float(day["weighted_return"].sum())
    benchmark = float(
        (prices["open_sell"] / prices["open_buy"] - 1.0).mean()
    )
    row: dict[str, Any] = {
        "round_id": spec["round_id"],
        "candidate": "TEST-H5",
        "decision_date": spec["decision_date"],
        "buy_date": spec["buy_date"],
        "sell_date": spec["sell_date"],
        "weekly_return_gross": gross,
        "weekly_return_net_0bps": gross,
        "weekly_return_net_10bps": gross - 0.002,
        "weekly_return_net_20bps": gross - 0.004,
        "benchmark_equal_weight_return": benchmark,
        "gross_excess_vs_benchmark": gross - benchmark,
        "net_10bps_excess_vs_benchmark": gross - 0.002 - benchmark,
        "weight_sum": float(result["weight"].sum()),
        "selected_count": int(len(result)),
        "selected_stocks": ",".join(result["stock_id"].tolist()),
        "max_weight": float(result["weight"].max()),
        "hhi": float(np.square(result["weight"].to_numpy(dtype=float)).sum()),
        "effective_holdings": float(
            1.0 / np.square(result["weight"].to_numpy(dtype=float)).sum()
        ),
        "worst_holding_return": float(day["stock_return"].min()),
        "holding_return_dispersion": float(
            day["stock_return"].std(ddof=0)
        ),
        "result_sha256": None,
        "validation_ok": True,
    }
    day.insert(0, "round_id", spec["round_id"])
    day.insert(1, "candidate", "TEST-H5")
    return row, day


def compound(values: list[float]) -> float:
    result = 1.0
    for value in values:
        result *= 1.0 + float(value)
    return result - 1.0


def comparator_summary(comparator_weekly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate, group in comparator_weekly.groupby("candidate", sort=False):
        ordered = group.sort_values("round_id")
        rows.append(
            {
                "candidate": candidate,
                "two_week_gross_compound": compound(
                    ordered["weekly_return_gross"].astype(float).tolist()
                ),
                "two_week_net_0bps_compound": compound(
                    ordered["weekly_return_net_0bps"].astype(float).tolist()
                ),
                "two_week_net_10bps_compound": compound(
                    ordered["weekly_return_net_10bps"].astype(float).tolist()
                ),
                "two_week_net_20bps_compound": compound(
                    ordered["weekly_return_net_20bps"].astype(float).tolist()
                ),
            }
        )
    benchmark = (
        comparator_weekly.loc[
            comparator_weekly["candidate"] == "THU-BDC2026-main"
        ]
        .sort_values("round_id")["benchmark_equal_weight_return"]
        .astype(float)
        .tolist()
    )
    rows.append(
        {
            "candidate": "HS300_EQUAL_WEIGHT",
            "two_week_gross_compound": compound(benchmark),
            "two_week_net_0bps_compound": compound(benchmark),
            "two_week_net_10bps_compound": compound(
                [value - 0.002 for value in benchmark]
            ),
            "two_week_net_20bps_compound": compound(
                [value - 0.004 for value in benchmark]
            ),
        }
    )
    return pd.DataFrame(rows)


def score(validation_path: Path) -> dict[str, Any]:
    if ACCESS_LEDGER.exists():
        raise PermissionError("isolated validation access ledger already exists")
    locks, readiness = verified_locks()
    record_event(
        "scoring_started_after_dual_lock",
        both_locks_verified=True,
        validation_parse_count_before=0,
    )
    reader = OneShotValidationReader(
        validation_path,
        VALIDATION_SHA256,
        ACCESS_LEDGER,
    )
    validation = reader.read_csv(readiness)
    config = read_json(BASE_CONFIG_PATH)
    code_col = str(config["data"]["stock_col"])
    date_col = str(config["data"]["date_col"])
    if code_col not in validation or date_col not in validation:
        raise ValueError("isolated validation schema differs from frozen market schema")
    open_col = str(validation.columns[2])
    validation[code_col] = validation[code_col].map(normalize_code)
    validation[date_col] = pd.to_datetime(validation[date_col]).dt.normalize()
    if (
        len(validation) != 3000
        or validation[code_col].nunique() != 300
        or validation[date_col].min() != pd.Timestamp("2026-07-06")
        or validation[date_col].max() != pd.Timestamp("2026-07-17")
    ):
        raise ValueError("isolated validation rows/stocks/date range mismatch")

    weekly_rows: list[dict[str, Any]] = []
    holding_frames: list[pd.DataFrame] = []
    previous_weights: dict[str, float] | None = None
    for round_key, spec in ROUND_SPECS.items():
        lock = locks[round_key]
        result_entry = lock["artifacts"]["result_pass1"]
        result_path = Path(result_entry["path"])
        result = pd.read_csv(result_path, dtype={"stock_id": str})
        result["stock_id"] = result["stock_id"].map(normalize_code)
        result["weight"] = result["weight"].astype(float)
        row, holdings = weekly_h5_score(
            validation,
            result,
            spec,
            code_col=code_col,
            date_col=date_col,
            open_col=open_col,
        )
        row["result_sha256"] = result_entry["sha256"]
        current_weights = dict(
            zip(result["stock_id"], result["weight"], strict=True)
        )
        if previous_weights is None:
            row["jaccard_prev_week"] = None
            row["turnover_prev_week"] = None
        else:
            previous = set(previous_weights)
            current = set(current_weights)
            row["jaccard_prev_week"] = len(previous & current) / len(
                previous | current
            )
            universe = previous | current
            row["turnover_prev_week"] = 0.5 * sum(
                abs(current_weights.get(code, 0.0) - previous_weights.get(code, 0.0))
                for code in universe
            )
        previous_weights = current_weights
        weekly_rows.append(row)
        holding_frames.append(holdings)

    h5_weekly = pd.DataFrame(weekly_rows)
    holdings = pd.concat(holding_frames, ignore_index=True)
    comparator_weekly = pd.read_csv(COMPARATOR_WEEKLY_PATH)
    comparison_weekly = pd.concat(
        [comparator_weekly, h5_weekly],
        ignore_index=True,
        sort=False,
    )
    two_week = comparator_summary(comparison_weekly)
    h5_row = two_week.loc[two_week["candidate"] == "TEST-H5"].iloc[0]
    official_row = two_week.loc[
        two_week["candidate"] == "THU-BDC2026-main"
    ].iloc[0]
    c_row = two_week.loc[two_week["candidate"] == "Candidate-C"].iloc[0]
    d_row = two_week.loc[two_week["candidate"] == "Candidate-D"].iloc[0]
    benchmark_row = two_week.loc[
        two_week["candidate"] == "HS300_EQUAL_WEIGHT"
    ].iloc[0]
    primary = "two_week_net_10bps_compound"
    summary = {
        "schema_version": 1,
        "marker": "TEST",
        "experiment_id": EXPERIMENT_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "status": "DIAGNOSTIC_ONLY",
        "blind_test": False,
        "independent_holdout": False,
        "promotion_evidence": False,
        "validation": {
            "sha256": VALIDATION_SHA256,
            "sha256_verified_before_parse": True,
            "parse_count": 1,
            "rows": int(len(validation)),
            "stocks": int(validation[code_col].nunique()),
            "date_start": validation[date_col].min().strftime("%Y-%m-%d"),
            "date_end": validation[date_col].max().strftime("%Y-%m-%d"),
            "raw_copy_created_in_workspace": False,
        },
        "weekly": weekly_rows,
        "two_week": {
            row["candidate"]: {
                key: float(row[key])
                for key in (
                    "two_week_gross_compound",
                    "two_week_net_0bps_compound",
                    "two_week_net_10bps_compound",
                    "two_week_net_20bps_compound",
                )
            }
            for row in two_week.to_dict(orient="records")
        },
        "primary_metric": primary,
        "h5_primary": float(h5_row[primary]),
        "differences": {
            "vs_official_reference": float(h5_row[primary] - official_row[primary]),
            "vs_candidate_c": float(h5_row[primary] - c_row[primary]),
            "vs_candidate_d": float(h5_row[primary] - d_row[primary]),
            "vs_hs300_equal_weight": float(
                h5_row[primary] - benchmark_row[primary]
            ),
        },
        "interpretation_constraints": {
            "formal_stability_gate_active": False,
            "cvar_reported": False,
            "promotion_allowed": False,
            "serious_allowed": False,
            "h01_creation_allowed": False,
        },
    }
    h5_weekly.to_csv(
        EVALUATION_ROOT / "TEST_H5_weekly_scores.csv",
        index=False,
        lineterminator="\n",
    )
    comparison_weekly.to_csv(
        EVALUATION_ROOT / "TEST_H5_weekly_comparison.csv",
        index=False,
        lineterminator="\n",
    )
    two_week.to_csv(
        EVALUATION_ROOT / "TEST_H5_two_week_comparison.csv",
        index=False,
        lineterminator="\n",
    )
    holdings.to_csv(
        EVALUATION_ROOT / "TEST_H5_holding_details.csv",
        index=False,
        lineterminator="\n",
    )
    output_path = EVALUATION_ROOT / "TEST_H5_TWO_WEEK_SCORE.json"
    write_json(output_path, summary)
    record_event(
        "scoring_completed",
        score_path=str(output_path.relative_to(WORKSPACE).as_posix()),
        validation_parse_count_after=1,
        status="DIAGNOSTIC_ONLY",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot scorer for the locked TEST H5 retrospective predictions"
    )
    parser.add_argument(
        "--validation-path",
        required=True,
        type=Path,
        help="External isolated validation CSV; accepted only after both locks verify",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score(args.validation_path)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

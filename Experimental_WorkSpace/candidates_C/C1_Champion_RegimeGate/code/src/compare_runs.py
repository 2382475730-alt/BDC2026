from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


RUN_FIELDS = (
    "run_name",
    "score_mode",
    "mean_fold_score",
    "median_fold_score",
    "min_fold_score",
    "std_fold_score",
    "cv_robust_score",
    "base_cv_robust_score",
    "rerank_cv_robust_score",
    "rerank_gain",
)


def read_run_summary(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_run_summary(summary: dict[str, object]) -> dict[str, object]:
    cv_summary = summary.get("cv_summary", {})
    if not isinstance(cv_summary, dict):
        cv_summary = {}
    base = summary.get("base_cv_summary", {})
    if not isinstance(base, dict):
        base = {}
    rerank = summary.get("rerank_cv_summary", {})
    if not isinstance(rerank, dict):
        rerank = {}
    row = {
        "run_name": summary.get("run_name", ""),
        "score_mode": summary.get("score_mode", ""),
        "mean_fold_score": cv_summary.get("mean_fold_score", 0.0),
        "median_fold_score": cv_summary.get("median_fold_score", 0.0),
        "min_fold_score": cv_summary.get("min_fold_score", 0.0),
        "std_fold_score": cv_summary.get("std_fold_score", 0.0),
        "cv_robust_score": cv_summary.get("cv_robust_score", 0.0),
        "base_cv_robust_score": base.get("cv_robust_score", 0.0),
        "rerank_cv_robust_score": rerank.get("cv_robust_score", 0.0),
    }
    row["rerank_gain"] = float(row["rerank_cv_robust_score"]) - float(row["base_cv_robust_score"])
    return row


def compare_run_summaries(paths: Sequence[Path]) -> pd.DataFrame:
    rows = [flatten_run_summary(read_run_summary(path)) for path in paths]
    return pd.DataFrame(rows, columns=RUN_FIELDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare walk-forward validation run summaries.")
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    table = compare_run_summaries(args.summaries)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
    else:
        print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

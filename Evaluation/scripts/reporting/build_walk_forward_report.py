from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path
from typing import Any


CURRENT_SELF_SCORE = 0.0670806033288513
MIN_OFFICIAL_SELF_SCORE = 0.060


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined report from walk-forward candidate summaries.")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-name", default="baseline_current")
    parser.add_argument("--primary-metric", default="mean_net_score")
    return parser.parse_args()


def load_summaries(runs_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(runs_dir.glob("*/candidate_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path)
        fold_scores = path.parent / "fold_scores.csv"
        if fold_scores.exists():
            with fold_scores.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            data["validation_bad_count"] = sum(1 for row in rows if row.get("validation_ok") != "True")
        else:
            data["validation_bad_count"] = None
        summaries.append(data)
    if not summaries:
        raise FileNotFoundError(f"no candidate_summary.json files found under {runs_dir}")
    return summaries


def metric(summary: dict[str, Any], name: str) -> float:
    value = summary.get("metrics", {}).get(name)
    return float(value) if value is not None else float("-inf")


def passes_rules(summary: dict[str, Any], baseline: dict[str, Any] | None) -> tuple[bool, list[str]]:
    reasons = []
    official = summary.get("official_self_score")
    if official is not None and float(official) < MIN_OFFICIAL_SELF_SCORE:
        reasons.append(f"official self-score below {MIN_OFFICIAL_SELF_SCORE}")
    metrics = summary.get("metrics", {})
    if not metrics:
        reasons.append("missing walk-forward metrics")
        return False, reasons
    validation_bad_count = summary.get("validation_bad_count")
    if validation_bad_count:
        reasons.append(f"portfolio validation failed in {validation_bad_count} folds")

    if baseline and baseline.get("metrics"):
        base = baseline["metrics"]
        if metrics.get("mean_net_score", float("-inf")) <= base.get("mean_net_score", float("-inf")):
            reasons.append("mean net score not above baseline")
        if metrics.get("median_net_score", float("-inf")) < base.get("median_net_score", float("-inf")):
            reasons.append("median net score below baseline")
        if metrics.get("hit_rate", float("-inf")) < base.get("hit_rate", float("-inf")):
            reasons.append("hit rate below baseline")
        base_worst = base.get("worst_20pct_mean")
        cand_worst = metrics.get("worst_20pct_mean")
        if base_worst is not None and cand_worst is not None:
            floor = base_worst * 1.1 if base_worst < 0 else base_worst * 0.9
            if cand_worst < floor:
                reasons.append("worst 20% mean fails tolerance")
    return not reasons, reasons


def main() -> int:
    args = parse_args()
    summaries = load_summaries(Path(args.runs_dir).resolve())
    baseline = next((item for item in summaries if item.get("candidate_name") == args.baseline_name), None)
    ranked = sorted(summaries, key=lambda item: metric(item, args.primary_metric), reverse=True)

    lines = [
        "# Walk-forward Candidate Ranking",
        "",
        f"- Current formal self-score: `{CURRENT_SELF_SCORE}`",
        f"- Minimum official self-score for promotion: `{MIN_OFFICIAL_SELF_SCORE}`",
        f"- Baseline candidate: `{args.baseline_name}`",
        "",
        "| Rank | Candidate | Policy | Folds | Bad Folds | Mean | Median | Hit Rate | Worst 20% | Official Self | Pass | Notes |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    for idx, item in enumerate(ranked, start=1):
        ok, reasons = passes_rules(item, baseline)
        metrics = item.get("metrics", {})
        lines.append(
            "| {rank} | {name} | {policy} | {folds} | {bad} | {mean:.8f} | {median:.8f} | {hit:.3f} | {worst:.8f} | {official} | {passed} | {notes} |".format(
                rank=idx,
                name=item.get("candidate_name", ""),
                policy=item.get("policy", ""),
                folds=metrics.get("fold_count", 0),
                bad=item.get("validation_bad_count"),
                mean=float(metrics.get("mean_net_score", 0.0)),
                median=float(metrics.get("median_net_score", 0.0)),
                hit=float(metrics.get("hit_rate", 0.0)),
                worst=float(metrics.get("worst_20pct_mean", 0.0)),
                official=item.get("official_self_score"),
                passed="yes" if ok else "no",
                notes="; ".join(reasons) if reasons else "promotion eligible",
            )
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "Evaluation" / "runs" / "TEST_H_single_week_stability_20260726"
NOTEBOOK = OUTPUT / "TEST_H_external_data_quality_audit.ipynb"


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "H Audit (bundled Python)",
        "language": "python",
        "name": "h-audit",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# TEST H External-Data Quality Audit

## tl;dr

- H0/H1 may proceed because they use only the frozen D01 market snapshot.
- H2 is blocked: the industry table is a 2026-07-26 static snapshot, most
  groups are too small for stable group ranks, and the futures table lacks the
  required spot basis/annualized term structure/risk-state fields while the
  prior consumer used lag 0.
- H3 is blocked: the financial table contains one latest revised value per
  stock/report period rather than true versioned announcement snapshots.
- H4 is blocked because no compliant point-in-time corporate-event dataset is
  present.

All source-manifest files and processed files are checked against their frozen
hash records. These data remain TEST-only because reporting-email eligibility
is unverified."""
        ),
        nbf.v4.new_markdown_cell(
            """## Context & Methods

This notebook profiles raw-ingestion evidence and processed ML feature tables
at their intended grain. It checks completeness, uniqueness, source hashes,
time availability, robust numeric outliers, and branch-specific causal rules.

### Key Assumptions

- The H-series goal's causal rules override the older aggregate `status=pass`
  flags.
- A 2026 static industry taxonomy cannot be backfilled into 2022-2026 history.
- `available_date=max(notice_date, update_date)` is conservative but does not
  manufacture the missing historical versions of revised filings.
- Same-day futures closes are unavailable unless inference timing proves
  otherwise; the default safe consumer lag is therefore at least one trading
  day."""
        ),
        nbf.v4.new_code_cell(
            """from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
assert (ROOT / "AGENTS.md").exists(), ROOT
RUN_DIR = ROOT / "Evaluation" / "runs" / "TEST_H_single_week_stability_20260726"
DATA_ROOT = ROOT / "BaoStock_Datasets" / "data" / "fusion_research_20260726"
RUN_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "industry": DATA_ROOT / "industry",
    "cffex": DATA_ROOT / "cffex",
    "financial": DATA_ROOT / "financial",
}
PROCESSED = {
    "industry": DATASETS["industry"] / "processed" / "csi300_industry_static.csv",
    "cffex": DATASETS["cffex"] / "processed" / "cffex_market_features.csv",
    "financial": DATASETS["financial"] / "processed" / "financial_point_in_time.csv",
}

def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_manifest(dataset_dir: Path) -> dict[str, object]:
    manifest_path = dataset_dir / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    for raw_relative, expected in manifest.get("files", {}).items():
        relative = raw_relative.replace("\\\\", "/")
        path = dataset_dir / relative
        exists = path.exists()
        checks.append(
            {
                "relative_path": relative,
                "exists": exists,
                "md5_matches": exists and digest(path, "md5") == expected.get("md5"),
                "sha256_matches": exists and digest(path, "sha256") == expected.get("sha256"),
            }
        )
    return {
        "manifest": manifest,
        "file_count": len(checks),
        "missing_count": sum(not row["exists"] for row in checks),
        "md5_mismatch_count": sum(row["exists"] and not row["md5_matches"] for row in checks),
        "sha256_mismatch_count": sum(row["exists"] and not row["sha256_matches"] for row in checks),
        "all_files_match": all(
            row["exists"] and row["md5_matches"] and row["sha256_matches"] for row in checks
        ),
    }

def numeric_profile(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    output = []
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        finite = numeric[np.isfinite(numeric)]
        if finite.empty:
            output.append(
                {
                    "field": column,
                    "coverage": 0.0,
                    "nonfinite_count": int((numeric.notna() & ~np.isfinite(numeric)).sum()),
                }
            )
            continue
        q01, q25, q50, q75, q99 = finite.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        iqr = float(q75 - q25)
        lower = float(q25 - 3.0 * iqr)
        upper = float(q75 + 3.0 * iqr)
        output.append(
            {
                "field": column,
                "coverage": float(numeric.notna().mean()),
                "zero_rate": float((finite == 0.0).mean()),
                "q01": float(q01),
                "median": float(q50),
                "q99": float(q99),
                "nonfinite_count": int((numeric.notna() & ~np.isfinite(numeric)).sum()),
                "robust_3iqr_outlier_count": int(((finite < lower) | (finite > upper)).sum()),
                "robust_3iqr_outlier_rate": float(((finite < lower) | (finite > upper)).mean()),
            }
        )
    return output

manifest_checks = {name: verify_manifest(path) for name, path in DATASETS.items()}
{name: {k: value[k] for k in ("file_count", "missing_count", "md5_mismatch_count", "sha256_mismatch_count", "all_files_match")} for name, value in manifest_checks.items()}"""
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            """industry = pd.read_csv(PROCESSED["industry"], dtype={"stock_code": str})
futures = pd.read_csv(PROCESSED["cffex"])
financial = pd.read_csv(PROCESSED["financial"], dtype={"stock_code": str})

industry["stock_code"] = industry["stock_code"].str.zfill(6)
financial["stock_code"] = financial["stock_code"].str.zfill(6)
for column in ("date",):
    futures[column] = pd.to_datetime(futures[column], errors="raise")
for column in ("report_date", "notice_date", "update_date", "available_date"):
    financial[column] = pd.to_datetime(financial[column], errors="raise")

pd.DataFrame(
    [
        {
            "dataset": "industry",
            "grain": "stock_code",
            "rows": len(industry),
            "columns": len(industry.columns),
            "duplicate_keys": int(industry.duplicated(["stock_code"]).sum()),
            "date_min": industry["classification_asof_date"].min(),
            "date_max": industry["classification_asof_date"].max(),
        },
        {
            "dataset": "cffex",
            "grain": "trading date",
            "rows": len(futures),
            "columns": len(futures.columns),
            "duplicate_keys": int(futures.duplicated(["date"]).sum()),
            "date_min": futures["date"].min().date().isoformat(),
            "date_max": futures["date"].max().date().isoformat(),
        },
        {
            "dataset": "financial",
            "grain": "stock_code + report_date (latest revision only)",
            "rows": len(financial),
            "columns": len(financial.columns),
            "duplicate_keys": int(financial.duplicated(["stock_code", "report_date"]).sum()),
            "date_min": financial["report_date"].min().date().isoformat(),
            "date_max": financial["report_date"].max().date().isoformat(),
        },
    ]
)"""
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            """industry_sizes = industry["industry"].fillna("UNKNOWN").str.strip().value_counts()
industry_summary = {
    "row_count": int(len(industry)),
    "stock_count": int(industry["stock_code"].nunique()),
    "duplicate_stock_count": int(industry.duplicated(["stock_code"]).sum()),
    "missing_industry_count": int(industry["industry"].isna().sum() + industry["industry"].fillna("").str.strip().eq("").sum()),
    "industry_count": int(len(industry_sizes)),
    "small_group_count_lt_5": int((industry_sizes < 5).sum()),
    "small_group_stock_count_lt_5": int(industry_sizes[industry_sizes < 5].sum()),
    "small_group_stock_share_lt_5": float(industry_sizes[industry_sizes < 5].sum() / len(industry)),
    "min_group_size": int(industry_sizes.min()),
    "median_group_size": float(industry_sizes.median()),
    "max_group_size": int(industry_sizes.max()),
    "classification_asof_date": str(industry["classification_asof_date"].iloc[0]),
    "historical_point_in_time_all": bool(industry["classification_is_historical_point_in_time"].fillna(False).all()),
}
pd.DataFrame(
    [
        {"metric": key, "value": value}
        for key, value in industry_summary.items()
    ]
)"""
        ),
        nbf.v4.new_code_cell(
            """futures_value_columns = [
    column for column in futures.columns if column != "date" and not column.endswith("_available")
]
futures_field_profile = numeric_profile(futures, futures_value_columns)
futures_manifest = manifest_checks["cffex"]["manifest"]
f_template = json.loads(
    (ROOT / "Experimental_WorkSpace" / "candidates_F" / "configs" / "f_fusion_template.json").read_text(encoding="utf-8")
)
required_corrected_fields = {
    "spot_basis": "fut_if_spot_basis",
    "annualized_term_structure": "fut_if_annualized_term_structure",
    "spot_basis_change": "fut_if_spot_basis_change",
    "risk_state": "fut_risk_state",
}
futures_summary = {
    "row_count": int(len(futures)),
    "duplicate_date_count": int(futures.duplicated(["date"]).sum()),
    "date_start": futures["date"].min().date().isoformat(),
    "date_end": futures["date"].max().date().isoformat(),
    "all_values_finite": bool(np.isfinite(futures[futures_value_columns].to_numpy(dtype=float)).all()),
    "prior_consumer_lag_days": int(f_template["features"].get("futures_lag_days", 0)),
    "main_contract_selection": futures_manifest["validation"]["main_contract_selection"],
    "same_contract_return_rule": futures_manifest["validation"]["return_rule"],
    "required_corrected_field_presence": {
        name: column in futures.columns for name, column in required_corrected_fields.items()
    },
}
pd.DataFrame(futures_field_profile)"""
        ),
        nbf.v4.new_code_cell(
            """financial_metrics = [
    "epsjb",
    "bps",
    "roejq",
    "zzcjll",
    "xsjll",
    "zcfzl",
    "totaloperaterevetz",
    "parentnetprofittz",
    "jyxjlyysr",
    "ld",
    "sd",
]
financial_field_profile = numeric_profile(financial, financial_metrics)
market_cutoff = pd.Timestamp("2026-07-03")
availability_delay = (financial["available_date"] - financial["report_date"]).dt.days
cutoff_age = (market_cutoff - financial["report_date"]).dt.days
for row in financial_field_profile:
    mask = financial[row["field"]].notna()
    row["availability_delay_days_median"] = float(availability_delay[mask].median())
    row["availability_delay_days_p95"] = float(availability_delay[mask].quantile(0.95))
    row["age_at_cutoff_days_median"] = float(cutoff_age[mask].median())
    row["age_at_cutoff_days_p95"] = float(cutoff_age[mask].quantile(0.95))

revision_delay = (financial["update_date"] - financial["notice_date"]).dt.days
financial_summary = {
    "row_count": int(len(financial)),
    "stock_count": int(financial["stock_code"].nunique()),
    "duplicate_stock_report_count": int(financial.duplicated(["stock_code", "report_date"]).sum()),
    "future_available_count": int((financial["available_date"] > market_cutoff).sum()),
    "revision_delayed_count": int((financial["update_date"] > financial["notice_date"]).sum()),
    "revision_delayed_share": float((financial["update_date"] > financial["notice_date"]).mean()),
    "revision_delay_days_median": float(revision_delay.median()),
    "revision_delay_days_p95": float(revision_delay.quantile(0.95)),
    "has_version_identifier": any(
        token in column.lower() for column in financial.columns for token in ("version", "revision_id", "snapshot_id")
    ),
    "latest_revision_only_grain": int(financial.duplicated(["stock_code", "report_date"]).sum()) == 0,
}
pd.DataFrame(financial_field_profile)"""
        ),
        nbf.v4.new_code_cell(
            """source_records = {}
for name, check in manifest_checks.items():
    manifest = check["manifest"]
    processed_relative = str(PROCESSED[name].relative_to(DATASETS[name])).replace("/", "\\\\")
    expected_processed = manifest.get("files", {}).get(processed_relative, {})
    source_records[name] = {
        "dataset_id": manifest.get("dataset_id"),
        "provider": manifest.get("source_provider"),
        "url": manifest.get("source_url") or manifest.get("source_url_template"),
        "retrieved_at_utc": manifest.get("retrieved_at_utc"),
        "reporting_status": manifest.get("competition_reporting_status"),
        "processed_file": str(PROCESSED[name].relative_to(ROOT)).replace("\\\\", "/"),
        "processed_md5": digest(PROCESSED[name], "md5"),
        "processed_sha256": digest(PROCESSED[name], "sha256"),
        "processed_md5_matches_manifest": digest(PROCESSED[name], "md5") == expected_processed.get("md5"),
        "processed_sha256_matches_manifest": digest(PROCESSED[name], "sha256") == expected_processed.get("sha256"),
        "all_manifest_files_match": check["all_files_match"],
    }
pd.DataFrame(source_records).T"""
        ),
        nbf.v4.new_code_cell(
            """branch_gates = {
    "TEST_H0_CONTROL": {
        "status": "ENABLED",
        "uses_external_data": False,
        "reason": "Frozen D01 market snapshot reproduced exactly.",
    },
    "TEST_H1_TOPK_OBJECTIVE": {
        "status": "ENABLED",
        "uses_external_data": False,
        "reason": "Only the preregistered objective/checkpoint semantics change.",
    },
    "TEST_H2_FUTURES_INDUSTRY_RESIDUAL": {
        "status": "BLOCKED_DATA_GATE",
        "uses_external_data": True,
        "reason": (
            "Industry is not historical point-in-time; small groups violate the >=5 stability rule; "
            "futures lacks spot-basis/annualized-term/risk-state fields and the prior consumer lag is 0."
        ),
    },
    "TEST_H3_FINANCIAL_EVENT_RESIDUAL": {
        "status": "BLOCKED_DATA_GATE",
        "uses_external_data": True,
        "reason": (
            "The source exposes latest revised values at one stock/report grain, not true versioned "
            "announcement snapshots. Conservative available_date prevents direct look-ahead but cannot recreate versions."
        ),
    },
    "TEST_H4_CORPORATE_EVENT_RESIDUAL": {
        "status": "BLOCKED_DATA_GATE",
        "uses_external_data": True,
        "reason": "No compliant point-in-time increase/decrease/repurchase event dataset is present.",
    },
}

findings = [
    {
        "id": "DQ-H-001",
        "dataset": "industry",
        "severity": "HIGH",
        "confidence": "HIGH",
        "failed_rule": "historical point-in-time classification and stable group size",
        "evidence": industry_summary,
        "impact": "Static 2026 labels can time-travel across historical folds; tiny groups create unstable pseudo-industry ranks.",
        "likely_cause": "The public component endpoint returned only a current snapshot with a fine-grained taxonomy.",
        "remediation": "Acquire historical effective-dated classifications or keep H2 blocked; shrink only genuine point-in-time small groups upward.",
    },
    {
        "id": "DQ-H-002",
        "dataset": "cffex",
        "severity": "HIGH",
        "confidence": "HIGH",
        "failed_rule": "corrected futures basis/term/risk fields and lag-safe consumption",
        "evidence": futures_summary,
        "impact": "A market-wide same-day signal is not a causal, cross-sectional residual and cannot satisfy H2.",
        "likely_cause": "The F-series table was designed for a preliminary fusion smoke test.",
        "remediation": "Add CSI300 spot, annualized expiry-aware basis/term structure, OI/risk regimes, lag at least one day, and stock beta/size interactions.",
    },
    {
        "id": "DQ-H-003",
        "dataset": "financial",
        "severity": "HIGH",
        "confidence": "HIGH",
        "failed_rule": "true versioned announcement snapshots",
        "evidence": financial_summary,
        "impact": "Event surprises and revision histories cannot be reconstructed; H3 would model stale/latest levels instead of point-in-time events.",
        "likely_cause": "The endpoint returns the latest revision for each report period.",
        "remediation": "Acquire versioned announcement snapshots with event timestamps; retain notice/update versions and derive event/decay/missingness features.",
    },
    {
        "id": "DQ-H-004",
        "dataset": "corporate_events",
        "severity": "HIGH",
        "confidence": "HIGH",
        "failed_rule": "compliant point-in-time source availability",
        "evidence": {"dataset_present": False},
        "impact": "H4 cannot be trained without inventing or backfilling event states.",
        "likely_cause": "No approved source was frozen in the research dataset.",
        "remediation": "Keep H4 closed until a reported, hashed, effective-dated source distinguishes plan/progress/completion.",
    },
]

audit = {
    "schema_version": 1,
    "marker": "TEST",
    "created_date": "2026-07-26",
    "intended_use": "H-series branch eligibility for single-week stability experiments",
    "source_records": source_records,
    "manifest_integrity": {
        name: {
            key: check[key]
            for key in (
                "file_count",
                "missing_count",
                "md5_mismatch_count",
                "sha256_mismatch_count",
                "all_files_match",
            )
        }
        for name, check in manifest_checks.items()
    },
    "industry": {
        "summary": industry_summary,
        "gate_passed": False,
    },
    "cffex": {
        "summary": futures_summary,
        "field_profile": futures_field_profile,
        "gate_passed": False,
    },
    "financial": {
        "summary": financial_summary,
        "field_profile": financial_field_profile,
        "gate_passed": False,
    },
    "corporate_events": {
        "dataset_present": False,
        "gate_passed": False,
    },
    "branch_gates": branch_gates,
    "findings": findings,
}

audit_path = RUN_DIR / "TEST_H_EXTERNAL_DATA_QUALITY_AUDIT.json"
audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

report_lines = [
    "# TEST H External-Data Quality Gate",
    "",
    "Status: `H0/H1 ENABLED; H2/H3/H4 BLOCKED_DATA_GATE`",
    "",
    "## Dataset and grain summary",
    "",
    "|Dataset|Grain|Rows|Key duplicates|Manifest hashes|",
    "|---|---|---:|---:|---|",
    f"|Industry|stock|{len(industry)}|{industry_summary['duplicate_stock_count']}|{'PASS' if manifest_checks['industry']['all_files_match'] else 'FAIL'}|",
    f"|CFFEX|date|{len(futures)}|{futures_summary['duplicate_date_count']}|{'PASS' if manifest_checks['cffex']['all_files_match'] else 'FAIL'}|",
    f"|Financial|stock + report period, latest revision only|{len(financial)}|{financial_summary['duplicate_stock_report_count']}|{'PASS' if manifest_checks['financial']['all_files_match'] else 'FAIL'}|",
    "",
    "## Branch decisions",
    "",
    "|Branch|Decision|Reason|",
    "|---|---|---|",
]
for branch, decision in branch_gates.items():
    report_lines.append(f"|{branch}|{decision['status']}|{decision['reason']}|")
report_lines.extend(
    [
        "",
        "## High-severity findings",
        "",
        f"- Industry: classification date {industry_summary['classification_asof_date']}; "
        f"{industry_summary['small_group_count_lt_5']}/{industry_summary['industry_count']} groups have fewer than five stocks, "
        f"covering {industry_summary['small_group_stock_share_lt_5']:.1%} of stocks.",
        f"- Futures: prior consumer lag={futures_summary['prior_consumer_lag_days']}; "
        f"required corrected fields present={futures_summary['required_corrected_field_presence']}.",
        f"- Financial: {financial_summary['revision_delayed_share']:.1%} of rows have update date after notice date; "
        f"there is no version identifier and the grain is latest-revision-only.",
        "- Corporate events: no compliant frozen dataset is present.",
        "",
        "## Safe next action",
        "",
        "Proceed only with H0/H1. Do not train H2-H4, do not access sealed labels, "
        "and do not reinterpret aggregate source validation passes as branch eligibility.",
        "",
        "The full field coverage, age, robust-outlier, source URL/date, and hash evidence is in "
        "`TEST_H_EXTERNAL_DATA_QUALITY_AUDIT.json` and this executed notebook.",
    ]
)
report_path = RUN_DIR / "TEST_H_EXTERNAL_DATA_QUALITY_REPORT.md"
report_path.write_text("\\n".join(report_lines) + "\\n", encoding="utf-8")

pd.DataFrame(
    [
        {"branch": branch, "status": decision["status"], "reason": decision["reason"]}
        for branch, decision in branch_gates.items()
    ]
)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Takeaways

The aggregate ingestion checks are useful but insufficient for H-series
eligibility. All three frozen source trees are internally hash-consistent, yet
the intended causal use fails: industry is not effective-dated, futures is not
the required lagged cross-sectional residual, and financial history is not
versioned. The smallest safe remediation is to keep H2-H4 closed and continue
only with the market-data control/objective ablation."""
        ),
    ]
    return notebook


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()

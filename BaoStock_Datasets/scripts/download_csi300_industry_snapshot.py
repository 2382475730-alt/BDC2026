from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


SOURCE_URL = (
    "https://datacenter-web.eastmoney.com/api/data/v1/get"
    "?reportName=RPT_INDEX_TS_COMPONENT"
    "&columns=SECUCODE%2CSECURITY_CODE%2CTYPE%2CSECURITY_NAME_ABBR%2CINDUSTRY"
    "&quoteColumns=&quoteType=0&source=WEB&client=WEB"
    "&filter=%28TYPE%3D%221%22%29"
    "&pageNumber=1&pageSize=500&sortColumns=SECURITY_CODE&sortTypes=1"
)


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_code(value: object) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    if not digits:
        raise ValueError(f"invalid stock code: {value!r}")
    return digits[-6:].zfill(6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a public CSI 300 industry snapshot.")
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.out_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        SOURCE_URL,
        timeout=float(args.timeout_seconds),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    payload = response.json()
    rows = ((payload.get("result") or {}).get("data") or [])
    if not payload.get("success") or not rows:
        raise RuntimeError(f"industry source returned no usable rows: {payload.get('message')}")

    raw_path = raw_dir / f"eastmoney_csi300_component_industry_{args.asof_date}.json"
    raw_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    universe_raw = pd.read_csv(args.universe_csv, dtype=str)
    stock_column = universe_raw.columns[0]
    universe = sorted({normalize_code(value) for value in universe_raw[stock_column]})
    source = pd.DataFrame(rows)
    source["stock_code"] = source["SECURITY_CODE"].map(normalize_code)
    source["industry"] = source["INDUSTRY"].fillna("UNKNOWN").astype(str).str.strip()
    source.loc[source["industry"] == "", "industry"] = "UNKNOWN"
    source = source.rename(columns={"SECURITY_NAME_ABBR": "stock_name"})
    processed = source[["stock_code", "stock_name", "industry"]].drop_duplicates("stock_code")
    processed = processed[processed["stock_code"].isin(universe)].sort_values("stock_code")
    processed["classification_asof_date"] = args.asof_date
    processed["classification_is_historical_point_in_time"] = False
    processed["competition_use_status"] = "RESEARCH_ONLY_REPORTING_UNVERIFIED"

    processed_path = processed_dir / "csi300_industry_static.csv"
    processed.to_csv(processed_path, index=False, encoding="utf-8")

    matched = set(processed["stock_code"])
    missing = sorted(set(universe) - matched)
    coverage = len(matched) / max(1, len(universe))
    validation = {
        "status": "pass" if coverage >= 0.99 and not processed["stock_code"].duplicated().any() else "fail",
        "universe_count": len(universe),
        "matched_count": len(matched),
        "coverage": coverage,
        "industry_count": int(processed["industry"].nunique()),
        "unknown_count": int((processed["industry"] == "UNKNOWN").sum()),
        "duplicate_stock_count": int(processed["stock_code"].duplicated().sum()),
        "missing_stock_codes": missing,
        "static_taxonomy_warning": (
            "This snapshot is not historical point-in-time classification. "
            "It may be used only in the explicitly labelled research branch."
        ),
    }
    validation_path = output_dir / "validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "dataset_id": f"csi300_industry_static_{args.asof_date}",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "source_provider": "Eastmoney public CSI 300 component endpoint",
        "classification_asof_date": args.asof_date,
        "training_market_cutoff": "2026-07-03",
        "competition_reporting_status": "UNVERIFIED_AFTER_2026-07-18_RESEARCH_ONLY",
        "files": {
            str(raw_path.relative_to(output_dir)): {
                "md5": file_hash(raw_path, "md5"),
                "sha256": file_hash(raw_path, "sha256"),
            },
            str(processed_path.relative_to(output_dir)): {
                "md5": file_hash(processed_path, "md5"),
                "sha256": file_hash(processed_path, "sha256"),
            },
            str(validation_path.relative_to(output_dir)): {
                "md5": file_hash(validation_path, "md5"),
                "sha256": file_hash(validation_path, "sha256"),
            },
        },
        "validation": validation,
    }
    manifest_path = output_dir / "SOURCE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

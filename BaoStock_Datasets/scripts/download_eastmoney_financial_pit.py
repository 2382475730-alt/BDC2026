from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


SOURCE_URL = "https://datacenter.eastmoney.com/securities/api/data/v1/get"
REPORT_NAME = "RPT_F10_FINANCE_MAINFINADATA"
RAW_METRICS = [
    "EPSJB",
    "BPS",
    "ROEJQ",
    "ZZCJLL",
    "XSJLL",
    "ZCFZL",
    "TOTALOPERATEREVETZ",
    "PARENTNETPROFITTZ",
    "JYXJLYYSR",
    "LD",
    "SD",
]
OUTPUT_COLUMNS = [
    "stock_code",
    "stock_name",
    "org_type",
    "report_date",
    "report_type",
    "notice_date",
    "update_date",
    "available_date",
    *[column.lower() for column in RAW_METRICS],
]


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_stock_code(value: object) -> str:
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits[-6:].zfill(6)


def valid_raw_file(path: Path, stock_code: str) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("success") is True
        and payload.get("code") == 0
        and payload.get("request", {}).get("stock_code") == stock_code
        and isinstance(payload.get("result", {}).get("data"), list)
    )


def fetch_stock(
    sessions: tuple[requests.Session, ...],
    stock_code: str,
    raw_dir: Path,
    timeout: float,
    retries: int,
) -> Path:
    output = raw_dir / f"{stock_code}.json"
    if valid_raw_file(output, stock_code):
        return output
    if output.exists():
        output.unlink()
    params = {
        "reportName": REPORT_NAME,
        "columns": "ALL",
        "filter": f'(SECURITY_CODE="{stock_code}")',
        "pageNumber": 1,
        "pageSize": 500,
        "sortColumns": "REPORT_DATE",
        "sortTypes": -1,
        "source": "HSF10",
        "client": "PC",
    }
    partial = raw_dir / f"{stock_code}.json.part"
    last_error: Exception | None = None
    for attempt in range(retries):
        session = sessions[attempt % len(sessions)]
        try:
            response = session.get(
                SOURCE_URL,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("success") is not True or payload.get("code") != 0:
                raise RuntimeError(
                    f"Eastmoney returned code={payload.get('code')}: {payload.get('message')}"
                )
            result = payload.get("result") or {}
            if int(result.get("pages") or 0) > 1:
                raise RuntimeError(
                    f"page size 500 was insufficient for {stock_code}: {result.get('pages')} pages"
                )
            payload["request"] = {
                "url": SOURCE_URL,
                "report_name": REPORT_NAME,
                "stock_code": stock_code,
                "page_size": 500,
            }
            partial.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            if not valid_raw_file(partial, stock_code):
                raise RuntimeError(f"invalid raw response for {stock_code}")
            partial.replace(output)
            return output
        except (
            requests.RequestException,
            OSError,
            RuntimeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt + 1 < retries:
                time.sleep(min(2.0**attempt, 8.0))
    raise RuntimeError(f"failed to fetch financial data for {stock_code}") from last_error


def parse_raw_files(
    paths: list[Path],
    report_start_date: str,
    market_cutoff: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend((payload.get("result") or {}).get("data") or [])
    if not rows:
        raise RuntimeError("no financial statement rows were returned")
    raw = pd.DataFrame(rows)
    required = {
        "SECURITY_CODE",
        "SECURITY_NAME_ABBR",
        "ORG_TYPE",
        "REPORT_DATE",
        "REPORT_TYPE",
        "NOTICE_DATE",
        "UPDATE_DATE",
        *RAW_METRICS,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"financial response is missing columns: {missing}")

    output = pd.DataFrame()
    output["stock_code"] = raw["SECURITY_CODE"].map(normalize_stock_code)
    output["stock_name"] = raw["SECURITY_NAME_ABBR"].astype(str)
    output["org_type"] = raw["ORG_TYPE"].astype(str)
    output["report_date"] = pd.to_datetime(raw["REPORT_DATE"], errors="coerce").dt.normalize()
    output["report_type"] = raw["REPORT_TYPE"].astype(str)
    output["notice_date"] = pd.to_datetime(raw["NOTICE_DATE"], errors="coerce").dt.normalize()
    output["update_date"] = pd.to_datetime(raw["UPDATE_DATE"], errors="coerce").dt.normalize()
    output["available_date"] = output[["notice_date", "update_date"]].max(axis=1)
    for source_column in RAW_METRICS:
        output[source_column.lower()] = pd.to_numeric(raw[source_column], errors="coerce")

    output = output.dropna(subset=["report_date", "notice_date", "available_date"])
    output = output[
        (output["report_date"] >= pd.Timestamp(report_start_date))
        & (output["report_date"] <= pd.Timestamp(market_cutoff))
        & (output["available_date"] <= pd.Timestamp(market_cutoff))
    ].copy()
    output = output.sort_values(
        ["stock_code", "report_date", "available_date"],
        kind="stable",
    ).drop_duplicates(["stock_code", "report_date"], keep="last")
    for date_column in ["report_date", "notice_date", "update_date", "available_date"]:
        output[date_column] = output[date_column].dt.strftime("%Y-%m-%d")
    return output[OUTPUT_COLUMNS].sort_values(
        ["stock_code", "available_date", "report_date"]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze point-in-time-safe Eastmoney financial statement indicators."
    )
    parser.add_argument("--stock-train-csv", required=True)
    parser.add_argument("--report-start-date", default="2020-01-01")
    parser.add_argument("--market-cutoff", default="2026-07-03")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--download-retries", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.out_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    market = pd.read_csv(args.stock_train_csv)
    stock_column = market.columns[0]
    universe = sorted(
        {
            normalize_stock_code(value)
            for value in market[stock_column].dropna().astype(str).tolist()
        }
    )
    proxy_session = requests.Session()
    direct_session = requests.Session()
    direct_session.trust_env = False
    raw_paths: list[Path] = []
    for index, stock_code in enumerate(universe, start=1):
        raw_paths.append(
            fetch_stock(
                (proxy_session, direct_session),
                stock_code,
                raw_dir,
                float(args.timeout_seconds),
                max(1, int(args.download_retries)),
            )
        )
        if index % 25 == 0 or index == len(universe):
            print(f"fetched_or_verified={index}/{len(universe)}", flush=True)

    processed = parse_raw_files(
        raw_paths,
        args.report_start_date,
        args.market_cutoff,
    )
    processed_path = processed_dir / "financial_point_in_time.csv"
    processed.to_csv(processed_path, index=False, encoding="utf-8")

    available_stocks = set(processed["stock_code"])
    latest_by_stock = processed.groupby("stock_code")["available_date"].max()
    metric_coverage = {
        column: float(processed[column].notna().mean())
        for column in [metric.lower() for metric in RAW_METRICS]
    }
    future_revision_delayed = int(
        (
            pd.to_datetime(processed["update_date"])
            > pd.to_datetime(processed["notice_date"])
        ).sum()
    )
    validation = {
        "status": "pass"
        if len(available_stocks) / max(1, len(universe)) >= 0.99
        and not processed.duplicated(["stock_code", "report_date"]).any()
        else "fail",
        "report_start_date": args.report_start_date,
        "market_cutoff": args.market_cutoff,
        "universe_count": len(universe),
        "stock_with_statement_count": len(available_stocks),
        "stock_coverage": len(available_stocks) / max(1, len(universe)),
        "processed_row_count": int(len(processed)),
        "duplicate_stock_report_count": int(
            processed.duplicated(["stock_code", "report_date"]).sum()
        ),
        "future_availability_row_count": int(
            (
                pd.to_datetime(processed["available_date"])
                > pd.Timestamp(args.market_cutoff)
            ).sum()
        ),
        "revision_delayed_row_count": future_revision_delayed,
        "metric_non_null_coverage": metric_coverage,
        "latest_available_date_min": str(latest_by_stock.min()),
        "latest_available_date_max": str(latest_by_stock.max()),
        "availability_rule": "max(NOTICE_DATE, UPDATE_DATE); revised values are unavailable before the revision date",
    }
    validation_path = output_dir / "validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    files = [*raw_paths, processed_path, validation_path]
    manifest = {
        "schema_version": 1,
        "dataset_id": (
            f"eastmoney_financial_pit_{args.report_start_date}_{args.market_cutoff}"
        ),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "report_name": REPORT_NAME,
        "source_provider": "Eastmoney public data center",
        "competition_reporting_status": "UNVERIFIED_AFTER_2026-07-18_RESEARCH_ONLY",
        "point_in_time_rule": validation["availability_rule"],
        "files": {
            str(path.relative_to(output_dir)): {
                "bytes": path.stat().st_size,
                "md5": file_hash(path, "md5"),
                "sha256": file_hash(path, "sha256"),
            }
            for path in files
        },
        "validation": validation,
    }
    manifest_path = output_dir / "SOURCE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

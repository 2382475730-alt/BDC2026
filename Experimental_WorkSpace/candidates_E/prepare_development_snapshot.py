from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CUTOFF = "2026-03-27"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build(source: Path, output: Path, manifest: Path) -> None:
    """Stream-filter by the second CSV field without interpreting sealed prices."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    dates: set[str] = set()
    stocks: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as src, output.open("w", encoding="utf-8", newline="") as dst:
        header = src.readline()
        if not header:
            raise ValueError("empty source")
        dst.write(header.lstrip("\ufeff"))
        for line in src:
            prefix = line.split(",", 2)
            if len(prefix) < 3:
                raise ValueError("malformed source line")
            stock, date = prefix[0].strip(), prefix[1].strip()
            if date <= CUTOFF:
                dst.write(line)
                rows += 1
                dates.add(date)
                stocks.add(stock)
    payload = {
        "schema_version": "e_r1_development_snapshot_v1",
        "source_path": str(source.resolve()), "source_md5": digest(source, "md5"),
        "source_sha256": digest(source, "sha256"), "filter_mode": "STREAM_DATE_FIELD_ONLY_NO_SEALED_NUMERIC_PARSE",
        "numeric_cutoff_inclusive": CUTOFF, "output_path": str(output.resolve()),
        "output_md5": digest(output, "md5"), "output_sha256": digest(output, "sha256"),
        "row_count": rows, "date_start": min(dates), "date_end": max(dates),
        "trading_date_count": len(dates), "stock_count": len(stocks),
        "sealed_numeric_access": False,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    build(Path(args.source), Path(args.output), Path(args.manifest))


if __name__ == "__main__":
    main()

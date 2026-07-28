from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


def inspect_market_csv(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc
        row_count = 0
        stocks: set[str] = set()
        dates: set[str] = set()
        for row in reader:
            if not row:
                continue
            row_count += 1
            if len(row) > 0 and row[0].strip():
                stocks.add(row[0].strip())
            if len(row) > 1 and row[1].strip():
                dates.add(row[1].strip().split(" ")[0])
    return {
        "path": str(path),
        "columns": header,
        "rows": row_count,
        "stock_count": len(stocks),
        "date_count": len(dates),
        "start_date": min(dates) if dates else "",
        "end_date": max(dates) if dates else "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect local official CSV inputs before training.")
    parser.add_argument("--work-dir", default=str(Path(__file__).resolve().parents[2]))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_dir = Path(args.work_dir)
    for name in ["train.csv", "test.csv"]:
        summary = inspect_market_csv(work_dir / "data" / name)
        print(
            "{path}: rows={rows}, stocks={stock_count}, dates={date_count}, "
            "range={start_date}..{end_date}".format(**summary)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

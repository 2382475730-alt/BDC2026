from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


CN_STOCK_ID = "\u80a1\u7968\u4ee3\u7801"
CN_DATE = "\u65e5\u671f"
CN_OPEN = "\u5f00\u76d8"
CN_WEIGHT = "\u6743\u91cd"


def _resolve_path(value: str | Path, work_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return work_dir / path


def normalize_stock_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    lower = text.lower()
    if lower.startswith(("sh.", "sz.")) and len(text) >= 9:
        text = text.split(".", 1)[1]
    if lower.endswith((".sh", ".sz")) and len(text) >= 9:
        text = text.split(".", 1)[0]
    if text.isdigit():
        return text.zfill(6)
    return text


def read_prediction(path: Path) -> list[tuple[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("prediction CSV is empty")
        rows = []
        seen = set()
        for row in reader:
            stock_id = normalize_stock_code(row.get("stock_id") or row.get(CN_STOCK_ID) or "")
            weight = float(row.get("weight") or row.get(CN_WEIGHT) or "nan")
            if not stock_id.isdigit() or len(stock_id) != 6:
                raise ValueError(f"invalid stock_id: {stock_id}")
            if stock_id in seen:
                raise ValueError(f"duplicate stock_id: {stock_id}")
            if weight < 0:
                raise ValueError(f"negative weight for {stock_id}")
            seen.add(stock_id)
            rows.append((stock_id, weight))
    if len(rows) > 5:
        raise ValueError("prediction contains more than five stocks")
    if sum(weight for _, weight in rows) > 1.0 + 1e-10:
        raise ValueError("prediction weight sum exceeds 1.0")
    return rows


def read_test_opens(path: Path) -> dict[str, list[tuple[str, float]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("test CSV is empty")
        fieldnames = {name.strip(): name for name in reader.fieldnames}
        stock_col = fieldnames.get("stock_id") or fieldnames.get(CN_STOCK_ID) or reader.fieldnames[0]
        date_col = fieldnames.get("date") or fieldnames.get(CN_DATE) or reader.fieldnames[1]
        open_col = fieldnames.get("open") or fieldnames.get(CN_OPEN) or reader.fieldnames[2]
        data: dict[str, list[tuple[str, float]]] = {}
        for row in reader:
            stock_id = normalize_stock_code(row.get(stock_col, ""))
            date = str(row.get(date_col, "")).strip().split(" ")[0]
            try:
                open_price = float(row.get(open_col, "nan"))
            except ValueError:
                continue
            if not stock_id or not date or open_price <= 0:
                continue
            data.setdefault(stock_id, []).append((date, open_price))
    for rows in data.values():
        rows.sort(key=lambda item: item[0])
    return data


def calculate_score(
    prediction: list[tuple[str, float]],
    test_opens: dict[str, list[tuple[str, float]]],
) -> tuple[float, list[dict[str, object]]]:
    details = []
    score = 0.0
    for stock_id, weight in prediction:
        rows = test_opens.get(stock_id, [])[-5:]
        if len(rows) < 2:
            details.append(
                {
                    "stock_id": stock_id,
                    "weight": weight,
                    "start_date": "",
                    "end_date": "",
                    "start_open": "",
                    "end_open": "",
                    "return": 0.0,
                    "weighted_return": 0.0,
                }
            )
            continue
        start_date, start_open = rows[0]
        end_date, end_open = rows[-1]
        stock_return = (end_open - start_open) / start_open
        weighted_return = stock_return * weight
        score += weighted_return
        details.append(
            {
                "stock_id": stock_id,
                "weight": weight,
                "start_date": start_date,
                "end_date": end_date,
                "start_open": start_open,
                "end_open": end_open,
                "return": stock_return,
                "weighted_return": weighted_return,
            }
        )
    return score, details


def write_score(work_dir: Path, team_name: str, score: float) -> None:
    path = work_dir / "temp" / "tmp.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Team Name", "Final Score"])
        writer.writeheader()
        writer.writerow({"Team Name": team_name, "Final Score": score})


def write_details(work_dir: Path, details: list[dict[str, object]]) -> None:
    path = work_dir / "temp" / "evaluation_detail.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stock_id",
        "weight",
        "start_date",
        "end_date",
        "start_open",
        "end_open",
        "return",
        "weighted_return",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate a local official-style open-to-open score.")
    parser.add_argument("--work-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--team-name", default="team_name")
    parser.add_argument("--output", default="output/result.csv")
    parser.add_argument("--test-data", default="data/test.csv")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_dir = Path(args.work_dir).resolve()
    output_path = _resolve_path(args.output, work_dir)
    test_data_path = _resolve_path(args.test_data, work_dir)
    try:
        prediction = read_prediction(output_path)
        test_opens = read_test_opens(test_data_path)
        score, details = calculate_score(prediction, test_opens)
    except Exception as exc:
        print(f"Error reading test data or validating prediction: {exc}")
        write_score(work_dir, args.team_name, -999.0)
        return 0
    write_score(work_dir, args.team_name, score)
    write_details(work_dir, details)
    print(f"Weighted open-return score: {score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

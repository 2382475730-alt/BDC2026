from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from featurework import DATE_COL, OPEN_COL, STOCK_COL


def validate_result(path: Path) -> pd.DataFrame:
    result = pd.read_csv(path, dtype={"stock_id": str})
    if list(result.columns) != ["stock_id", "weight"]:
        raise ValueError("result columns must be exactly stock_id,weight")
    result["stock_id"] = result["stock_id"].str.zfill(6)
    if result.empty or len(result) > 5 or result["stock_id"].duplicated().any():
        raise ValueError("result must contain 1-5 distinct stocks")
    if (result["weight"] < 0).any() or float(result["weight"].sum()) > 1.0 + 1e-10:
        raise ValueError("result weights must be nonnegative and sum to at most 1")
    return result


def score_if_available(
    result: pd.DataFrame,
    market: pd.DataFrame,
    decision_date: str | None,
) -> dict[str, object]:
    market[DATE_COL] = pd.to_datetime(market[DATE_COL]).dt.normalize()
    dates = sorted(market[DATE_COL].drop_duplicates())
    if decision_date is None:
        return {"status": "format_valid", "score": None, "reason": "decision date not provided"}
    decision = pd.Timestamp(decision_date).normalize()
    future = [date for date in dates if date > decision]
    if len(future) < 5:
        return {"status": "format_valid", "score": None, "reason": "fewer than five future trading dates"}
    buy_date, sell_date = future[0], future[4]
    prices = market[market[DATE_COL].isin([buy_date, sell_date])].copy()
    prices[STOCK_COL] = prices[STOCK_COL].astype(str).str.zfill(6)
    pivot = prices.pivot_table(index=STOCK_COL, columns=DATE_COL, values=OPEN_COL, aggfunc="last")
    joined = result.set_index("stock_id").join(pivot, how="left")
    if joined[[buy_date, sell_date]].isna().any().any():
        return {"status": "format_valid", "score": None, "reason": "missing selected-stock future prices"}
    returns = joined[sell_date] / joined[buy_date] - 1.0
    score = float((joined["weight"] * returns).sum())
    return {
        "status": "scored",
        "score": score,
        "decision_date": decision.strftime("%Y-%m-%d"),
        "buy_date": pd.Timestamp(buy_date).strftime("%Y-%m-%d"),
        "sell_date": pd.Timestamp(sell_date).strftime("%Y-%m-%d"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or self-score Candidate-D result.csv")
    parser.add_argument("--work-dir", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--test-data", default="data/test.csv")
    parser.add_argument("--output", default="output/result.csv")
    parser.add_argument("--decision-date", default=None)
    args = parser.parse_args()
    work_dir = Path(args.work_dir).resolve()
    result = validate_result(work_dir / args.output)
    market = pd.read_csv(work_dir / args.test_data)
    report = score_if_available(result, market, args.decision_date)
    report["result_rows"] = len(result)
    report["weight_sum"] = float(result["weight"].sum())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

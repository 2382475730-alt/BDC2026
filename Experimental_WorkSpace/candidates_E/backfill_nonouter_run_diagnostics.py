from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "automation_runs"
DEV = ROOT / "data_snapshots" / "E_R1_DEVELOPMENT_THROUGH_20260327.csv"
DATE_COL = "日期"
STOCK_COL = "股票代码"
OPEN_COL = "开盘"
BUY_COMMISSION = 0.0005
SELL_COMMISSION_AND_STAMP = 0.0010


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_manifest(run_dir: Path) -> None:
    files = []
    for path in sorted(value for value in run_dir.rglob("*") if value.is_file()):
        if path.name == "artifact_manifest.json" or "__pycache__" in path.parts:
            continue
        files.append({
            "path": path.relative_to(run_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        })
    write_json(run_dir / "artifact_manifest.json", {
        "schema_version": "e_run_artifacts_v1",
        "created_at": now(),
        "files": files,
    })


def normalize_stock(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def metrics(weekly: pd.DataFrame) -> dict[str, Any]:
    if weekly.empty:
        return {"week_count": 0, "status": "NO_COMPLETE_WEEK_IN_REDUCED_DIAGNOSTIC"}
    net = weekly["weekly_return_net"].to_numpy(float)
    gross = weekly["weekly_return_gross"].to_numpy(float)
    nav = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    peak = np.maximum.accumulate(nav)
    drawdown = nav / peak - 1.0
    tail_count = max(1, int(math.ceil(0.05 * len(net))))
    return {
        "week_count": int(len(weekly)),
        "mean_weekly_return_gross": float(gross.mean()),
        "mean_weekly_return_net": float(net.mean()),
        "median_weekly_return_net": float(np.median(net)),
        "cumulative_return_net": float(np.prod(1.0 + net) - 1.0),
        "positive_week_rate": float(np.mean(net > 0)),
        "max_drawdown_abs": float(abs(drawdown.min())),
        "CVaR_95_loss_abs": float(max(0.0, np.sort(-net)[-tail_count:].mean())),
        "cost_drag": float((gross - net).sum()),
        "average_cash_ratio": float(weekly["cash_ratio"].mean()),
        "average_invested_hhi": float(weekly["invested_hhi"].mean()),
        "average_composition_turnover": float(weekly["composition_turnover"].mean()),
        "date_start": str(weekly["buy_date"].min()),
        "date_end": str(weekly["sell_date"].max()),
    }


def build_weekly(
    predictions: pd.DataFrame,
    dates: list[pd.Timestamp],
    date_index: dict[pd.Timestamp, int],
    week_dates: dict[tuple[int, int], list[pd.Timestamp]],
    opens: pd.Series,
    label_end_limit: pd.Timestamp,
) -> pd.DataFrame:
    pred = predictions.copy()
    pred["sample_end_date"] = pd.to_datetime(pred["sample_end_date"]).dt.normalize()
    pred["stock_id"] = normalize_stock(pred["stock_id"])
    pred_by_date = {pd.Timestamp(date).normalize(): day for date, day in pred.groupby("sample_end_date", sort=True)}
    eligible: list[tuple[pd.Timestamp, list[pd.Timestamp]]] = []
    for decision in sorted(pred_by_date):
        index = date_index.get(decision)
        if index is None or index + 1 >= len(dates):
            continue
        next_date = dates[index + 1]
        decision_iso = decision.isocalendar()
        next_iso = next_date.isocalendar()
        if (int(decision_iso.year), int(decision_iso.week)) == (int(next_iso.year), int(next_iso.week)):
            continue
        week = week_dates[(int(next_iso.year), int(next_iso.week))]
        if week[-1] <= label_end_limit:
            eligible.append((decision, week))
    rows: list[dict[str, Any]] = []
    previous: dict[str, float] = {}
    for decision, week in eligible:
        buy_date, sell_date = week[0], week[-1]
        day = pred_by_date[decision]
        selected = day[pd.to_numeric(day["portfolio_weight"], errors="coerce").fillna(0.0) > 1e-12]
        weights = selected["portfolio_weight"].to_numpy(float)
        stocks = selected["stock_id"].astype(str).tolist()
        returns = []
        for stock in stocks:
            buy_open = float(opens.get((stock, buy_date), np.nan))
            sell_open = float(opens.get((stock, sell_date), np.nan))
            if not np.isfinite(buy_open) or not np.isfinite(sell_open) or buy_open <= 0:
                raise AssertionError(f"missing open for {stock}: {buy_date}..{sell_date}")
            returns.append(sell_open / buy_open - 1.0)
        returns_array = np.asarray(returns, dtype=float)
        gross = float(np.dot(weights, returns_array))
        buy_notional = float(weights.sum())
        sell_notional = float(np.dot(weights, 1.0 + returns_array))
        cost = BUY_COMMISSION * buy_notional + SELL_COMMISSION_AND_STAMP * sell_notional
        current = dict(zip(stocks, weights))
        union = set(previous) | set(current)
        turnover = 0.5 * sum(abs(current.get(code, 0.0) - previous.get(code, 0.0)) for code in union)
        invested = weights / weights.sum()
        rows.append({
            "week_start": buy_date.strftime("%Y-%m-%d"),
            "buy_date": buy_date.strftime("%Y-%m-%d"),
            "sell_date": sell_date.strftime("%Y-%m-%d"),
            "decision_date": decision.strftime("%Y-%m-%d"),
            "weekly_return_gross": gross,
            "weekly_return_net": gross - cost,
            "cost": cost,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "weight_sum": buy_notional,
            "cash_ratio": 1.0 - buy_notional,
            "selected_count": len(stocks),
            "selected_stocks": ",".join(stocks),
            "invested_hhi": float(np.square(invested).sum()),
            "max_invested_name_weight": float(invested.max()),
            "composition_turnover": float(turnover),
        })
        previous = current
    return pd.DataFrame(rows)


def main() -> None:
    market = pd.read_csv(DEV, usecols=[STOCK_COL, DATE_COL, OPEN_COL], dtype={STOCK_COL: str})
    market[DATE_COL] = pd.to_datetime(market[DATE_COL]).dt.normalize()
    market[STOCK_COL] = normalize_stock(market[STOCK_COL])
    dates = [pd.Timestamp(value).normalize() for value in sorted(market[DATE_COL].dropna().unique())]
    date_index = {date: index for index, date in enumerate(dates)}
    week_dates: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for date in dates:
        iso = date.isocalendar()
        week_dates.setdefault((int(iso.year), int(iso.week)), []).append(date)
    opens = market.set_index([STOCK_COL, DATE_COL])[OPEN_COL].astype(float)
    targets = sorted(path for pattern in ("SMOKE_*", "QUICK_INNER_*") for path in RUNS.glob(pattern) if path.is_dir())
    results = []
    for run_dir in targets:
        predictions = pd.read_csv(run_dir / "best_val_predictions.csv", dtype={"stock_id": str})
        split = read_json(run_dir / "split_metadata.json")
        scaler = read_json(run_dir / "scalers.json")
        prediction_dates = pd.to_datetime(predictions["sample_end_date"]).dt.normalize()
        if "validation" in split:
            label_limit = pd.Timestamp(split["validation"]["label_end_end"]).normalize()
            label_boundary_pass = split.get("train_label_end_before_val_start") is True
            scaler_range_pass = scaler.get("fit_end_date") <= split.get("train_end")
            validation_sample_end = str(split["validation"]["sample_end"])
        else:
            # Smoke uses the explicit chronological 80/20 fallback after
            # --max-dates truncation. Its date boundary can still be audited
            # directly from the train-only scaler end and first validation date.
            last_index = date_index[pd.Timestamp(prediction_dates.max()).normalize()]
            label_limit = dates[min(last_index + 5, len(dates) - 1)]
            first_validation = pd.Timestamp(prediction_dates.min()).normalize()
            label_boundary_pass = pd.Timestamp(scaler.get("fit_end_date")) < first_validation
            scaler_range_pass = label_boundary_pass
            validation_sample_end = str(pd.Timestamp(prediction_dates.max()).date())
        weekly = build_weekly(predictions, dates, date_index, week_dates, opens, label_limit)
        weekly.to_csv(run_dir / "weekly_returns_exact_cost.csv", index=False)
        official_proxy = float(
            (predictions["portfolio_weight"].astype(float) * predictions["label"].astype(float))
            .groupby(predictions["sample_end_date"].astype(str)).sum().mean()
        )
        scope = "SMOKE_REDUCED_DIAGNOSTIC_NOT_GATE_EVIDENCE" if run_dir.name.startswith("SMOKE_") else "INNER_SELECTION_DIAGNOSTIC_NOT_OUTER_GATE_EVIDENCE"
        leakage = {
            "scope": scope,
            "label_boundary_pass": label_boundary_pass,
            "scaler_end_not_after_train_end": scaler_range_pass,
            "sealed_rows_absent": validation_sample_end <= "2026-03-27",
            "weekly_sell_end_not_after_development_cutoff": weekly.empty or str(weekly["sell_date"].max()) <= "2026-03-27",
            "prediction_row_count": int(len(predictions)),
            "weekly_diagnostic_count": int(len(weekly)),
        }
        leakage["pass"] = all(bool(value) for key, value in leakage.items() if key.endswith("pass") or key in {"sealed_rows_absent", "weekly_sell_end_not_after_development_cutoff"})
        write_json(run_dir / "leakage_audit.json", leakage)
        write_json(run_dir / "E_metrics.json", {
            "scope": scope,
            "weekly": metrics(weekly),
            "official_proxy": official_proxy,
            "leakage": leakage,
        })
        refresh_manifest(run_dir)
        results.append({"run_id": run_dir.name, "week_count": len(weekly), "leakage_pass": leakage["pass"]})
    payload = {
        "schema_version": "e_r1_nonouter_diagnostic_backfill_v1",
        "created_at": now(),
        "source": str(DEV),
        "source_sha256": file_hash(DEV),
        "sealed_access": False,
        "run_count": len(results),
        "all_leakage_pass": all(record["leakage_pass"] for record in results),
        "all_have_weekly_diagnostic": all(record["week_count"] > 0 for record in results),
        "results": results,
        "note": "Post-training diagnostic-only artifact completion; no model, score, checkpoint, gate, or selection result changed.",
    }
    write_json(RUNS / "E_R1_NONOUTER_DIAGNOSTIC_BACKFILL.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

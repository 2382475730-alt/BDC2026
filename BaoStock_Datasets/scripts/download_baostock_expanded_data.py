from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]

NUMERIC_COLUMNS = [column for column in TRAIN_COLUMNS if column not in {"股票代码", "日期"}]
BAOSTOCK_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"


@dataclass
class DownloadReport:
    symbol: str
    rows: int
    start_date: str | None
    end_date: str | None
    ok: bool
    error: str = ""


def normalize_stock_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"invalid stock code: {value!r}")
    return digits[-6:].zfill(6)


def to_baostock_code(stock_code: object) -> str:
    code = normalize_stock_code(stock_code)
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    if code.startswith(("4", "8")):
        return f"bj.{code}"
    return f"sz.{code}"


def normalize_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in TRAIN_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    output = df[TRAIN_COLUMNS].copy()
    output["股票代码"] = output["股票代码"].map(normalize_stock_code)
    output["日期"] = pd.to_datetime(output["日期"]).dt.strftime("%Y-%m-%d")
    for column in NUMERIC_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["开盘", "收盘", "最高", "最低"])
    output[NUMERIC_COLUMNS] = output[NUMERIC_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return output[TRAIN_COLUMNS].sort_values(["日期", "股票代码"]).reset_index(drop=True)


def baostock_rows_to_train_frame(rows: list[list[str]], fields: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TRAIN_COLUMNS)

    raw = pd.DataFrame(rows, columns=fields)
    required = {"date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"baostock response missing columns: {missing}")

    numeric = {}
    for column in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
        numeric[column] = pd.to_numeric(raw[column], errors="coerce")

    preclose = numeric["preclose"].replace(0.0, np.nan)
    output = pd.DataFrame(
        {
            "股票代码": raw["code"].map(normalize_stock_code),
            "日期": pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d"),
            "开盘": numeric["open"],
            "收盘": numeric["close"],
            "最高": numeric["high"],
            "最低": numeric["low"],
            "成交量": numeric["volume"],
            "成交额": numeric["amount"],
            "振幅": (numeric["high"] - numeric["low"]) / preclose * 100.0,
            "涨跌额": numeric["close"] - numeric["preclose"],
            "换手率": numeric["turn"],
            "涨跌幅": numeric["pctChg"],
        }
    )
    return normalize_train_frame(output)


def query_stock_history(bs_module, stock_code: str, start_date: str, end_date: str, adjustflag: str) -> pd.DataFrame:
    result = bs_module.query_history_k_data_plus(
        to_baostock_code(stock_code),
        BAOSTOCK_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjustflag,
    )
    if result.error_code != "0":
        raise RuntimeError(f"{stock_code} query failed: {result.error_code} {result.error_msg}")

    rows = []
    while result.next():
        rows.append(result.get_row_data())
    return baostock_rows_to_train_frame(rows, result.fields)


def login_baostock(bs_module, retries: int, retry_sec: float):
    last_login = None
    for attempt in range(1, retries + 1):
        last_login = bs_module.login()
        if last_login.error_code == "0":
            return last_login
        print(f"Baostock login attempt {attempt}/{retries} failed: {last_login.error_code} {last_login.error_msg}", flush=True)
        if attempt < retries and retry_sec > 0:
            time.sleep(retry_sec)
    return last_login


def stock_universe(base_df: pd.DataFrame, max_stocks: int | None) -> list[str]:
    codes = sorted(base_df["股票代码"].map(normalize_stock_code).unique())
    return codes[:max_stocks] if max_stocks is not None else codes


def download_universe(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    adjustflag: str,
    cache_dir: Path,
    sleep_sec: float,
    retries: int,
    retry_sec: float,
) -> tuple[pd.DataFrame, list[DownloadReport]]:
    import baostock as bs

    cache_dir.mkdir(parents=True, exist_ok=True)
    login = login_baostock(bs, retries, retry_sec)
    if login.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login.error_code} {login.error_msg}")

    frames: list[pd.DataFrame] = []
    reports: list[DownloadReport] = []
    try:
        for index, stock_code in enumerate(stock_codes, start=1):
            code = normalize_stock_code(stock_code)
            cache_path = cache_dir / f"{code}.csv"
            try:
                if cache_path.exists():
                    frame = normalize_train_frame(pd.read_csv(cache_path, dtype={"股票代码": str}))
                else:
                    frame = query_stock_history(bs, code, start_date, end_date, adjustflag)
                    frame.to_csv(cache_path, index=False, encoding="utf-8")
                if not frame.empty:
                    frames.append(frame)
                report = DownloadReport(
                    symbol=code,
                    rows=int(len(frame)),
                    start_date=str(frame["日期"].min()) if not frame.empty else None,
                    end_date=str(frame["日期"].max()) if not frame.empty else None,
                    ok=True,
                )
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - keep downloading remaining stocks
                report = DownloadReport(code, 0, None, None, False, f"{type(exc).__name__}: {exc}")
                status = "failed"
            reports.append(report)
            print(f"[{index:03d}/{len(stock_codes):03d}] {code} {status} {report.rows} rows", flush=True)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    finally:
        bs.logout()

    if not frames:
        return pd.DataFrame(columns=TRAIN_COLUMNS), reports
    return normalize_train_frame(pd.concat(frames, ignore_index=True)), reports


def merge_append_new(base_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    base = normalize_train_frame(base_df)
    history = normalize_train_frame(history_df)
    combined = pd.concat([history, base], ignore_index=True)
    combined = combined.drop_duplicates(subset=["股票代码", "日期"], keep="last")
    return normalize_train_frame(combined)


def merge_replace_overlap(base_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    base = normalize_train_frame(base_df)
    history = normalize_train_frame(history_df)
    combined = pd.concat([base, history], ignore_index=True)
    combined = combined.drop_duplicates(subset=["股票代码", "日期"], keep="last")
    return normalize_train_frame(combined)


def summarize_frame(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {"rows": 0, "stocks": 0, "date_min": None, "date_max": None}
    counts = df.groupby("股票代码")["日期"].count()
    return {
        "rows": int(len(df)),
        "stocks": int(df["股票代码"].nunique()),
        "date_min": str(df["日期"].min()),
        "date_max": str(df["日期"].max()),
        "rows_per_stock_min": int(counts.min()),
        "rows_per_stock_max": int(counts.max()),
    }


def overlap_quality(base_df: pd.DataFrame, history_df: pd.DataFrame) -> dict[str, object]:
    base = normalize_train_frame(base_df)
    history = normalize_train_frame(history_df)
    merged = base.merge(history, on=["股票代码", "日期"], suffixes=("_base", "_download"))
    if merged.empty:
        return {"overlap_rows": 0}

    result: dict[str, object] = {
        "overlap_rows": int(len(merged)),
        "overlap_stocks": int(merged["股票代码"].nunique()),
    }
    for column in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"]:
        ratio = merged[f"{column}_download"].astype(float) / merged[f"{column}_base"].astype(float).replace(0.0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.empty:
            continue
        result[f"{column}_ratio_median"] = float(ratio.median())
        result[f"{column}_ratio_p05"] = float(ratio.quantile(0.05))
        result[f"{column}_ratio_p95"] = float(ratio.quantile(0.95))
        result[f"{column}_ratio_abs_gt_5pct"] = float((ratio.sub(1.0).abs() > 0.05).mean())
        result[f"{column}_ratio_abs_gt_10pct"] = float((ratio.sub(1.0).abs() > 0.10).mean())

    return_diff = (merged["涨跌幅_download"].astype(float) - merged["涨跌幅_base"].astype(float)).abs()
    result["涨跌幅_abs_diff_median"] = float(return_diff.median())
    result["涨跌幅_abs_diff_p95"] = float(return_diff.quantile(0.95))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download expanded competition data from BaoStock.")
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--history-start", default="2021-01-04")
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--future-start", default="2026-03-20")
    parser.add_argument("--future-end", default="2026-06-30")
    parser.add_argument("--adjustflag", default="1", choices=["1", "2", "3"], help="BaoStock adjustflag: 1 post, 2 pre, 3 none.")
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.02)
    parser.add_argument("--login-retries", type=int, default=3)
    parser.add_argument("--retry-sec", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = normalize_train_frame(pd.read_csv(args.base_train, dtype={"股票代码": str}))
    train_end = pd.to_datetime(args.train_end or base["日期"].max()).strftime("%Y-%m-%d")
    future_start = pd.to_datetime(args.future_start).strftime("%Y-%m-%d")
    future_end = pd.to_datetime(args.future_end).strftime("%Y-%m-%d")
    codes = stock_universe(base, args.max_stocks)

    print(f"Universe: {len(codes)} stocks", flush=True)
    print(f"History window: {args.history_start} to {train_end}, adjustflag={args.adjustflag}", flush=True)
    history, history_reports = download_universe(
        codes,
        args.history_start,
        train_end,
        args.adjustflag,
        out_dir / "history_by_symbol",
        args.sleep_sec,
        args.login_retries,
        args.retry_sec,
    )
    history_path = out_dir / f"baostock_history_post_{args.history_start}_{train_end}.csv"
    history.to_csv(history_path, index=False, encoding="utf-8")

    append_new = merge_append_new(base, history)
    replace_overlap = merge_replace_overlap(base, history)
    append_path = out_dir / "train_extended_append_new.csv"
    replace_path = out_dir / "train_extended_replace_overlap.csv"
    append_new.to_csv(append_path, index=False, encoding="utf-8")
    replace_overlap.to_csv(replace_path, index=False, encoding="utf-8")

    print(f"Future window: {future_start} to {future_end}, adjustflag={args.adjustflag}", flush=True)
    future, future_reports = download_universe(
        codes,
        future_start,
        future_end,
        args.adjustflag,
        out_dir / "future_by_symbol",
        args.sleep_sec,
        args.login_retries,
        args.retry_sec,
    )
    future_path = out_dir / f"baostock_future_post_{future_start}_{future_end}.csv"
    future.to_csv(future_path, index=False, encoding="utf-8")

    report = {
        "source": "baostock",
        "adjustflag": args.adjustflag,
        "base_train": summarize_frame(base),
        "history": summarize_frame(history),
        "future": summarize_frame(future),
        "append_new_train": summarize_frame(append_new),
        "replace_overlap_train": summarize_frame(replace_overlap),
        "overlap_quality": overlap_quality(base, history),
        "paths": {
            "history": str(history_path),
            "future": str(future_path),
            "append_new_train": str(append_path),
            "replace_overlap_train": str(replace_path),
        },
        "history_reports": [asdict(item) for item in history_reports],
        "future_reports": [asdict(item) for item in future_reports],
    }
    (out_dir / "download_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["source", "base_train", "history", "future", "append_new_train", "replace_overlap_train", "overlap_quality", "paths"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

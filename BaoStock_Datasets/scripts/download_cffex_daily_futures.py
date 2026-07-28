from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PRODUCTS = ("IF", "IH", "IC", "IM")
SOURCE_TEMPLATE = "http://www.cffex.com.cn/sj/historysj/{month}/zip/{month}.zip"
RAW_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "open_interest",
    "open_interest_change",
    "settle",
    "pre_settle",
    "product",
]
OUTPUT_COLUMNS = [
    "date",
    "fut_if_return_1",
    "fut_if_return_5",
    "fut_if_volatility_20",
    "fut_if_volume_z20",
    "fut_if_oi_change",
    "fut_if_term_spread",
    "fut_if_ih_spread_5",
    "fut_ic_if_spread_5",
    "fut_im_if_spread_5",
    "fut_if_available",
    "fut_ih_available",
    "fut_ic_available",
    "fut_im_available",
]


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def months_between(start_date: str, end_date: str) -> list[str]:
    start = pd.Timestamp(start_date).to_period("M")
    end = pd.Timestamp(end_date).to_period("M")
    return [str(period).replace("-", "") for period in pd.period_range(start, end, freq="M")]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"--": np.nan, "-": np.nan}), errors="coerce")


def parse_daily_csv(data: bytes, member_name: str) -> pd.DataFrame:
    text = data.decode("gb2312", errors="replace")
    frame = pd.read_csv(io.StringIO(text))
    if frame.shape[1] < 11:
        raise ValueError(f"{member_name} has only {frame.shape[1]} columns")
    frame = frame.iloc[:, :14].copy()
    names = [
        "symbol",
        "open",
        "high",
        "low",
        "volume",
        "turnover",
        "open_interest",
        "open_interest_change",
        "close",
        "settle",
        "pre_settle",
        "change_1",
        "change_2",
        "delta",
    ][: frame.shape[1]]
    frame.columns = names
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["product"] = frame["symbol"].str.extract(r"^([A-Z]+)", expand=False)
    frame = frame[frame["product"].isin(PRODUCTS)]
    frame = frame[frame["symbol"].str.fullmatch(r"(?:IF|IH|IC|IM)\d{4}")]
    date_match = re.search(r"(\d{8})_1\.csv$", member_name)
    if not date_match:
        raise ValueError(f"cannot parse trading date from {member_name}")
    frame["date"] = pd.Timestamp(date_match.group(1)).strftime("%Y-%m-%d")
    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "open_interest",
        "open_interest_change",
        "settle",
        "pre_settle",
    ]:
        frame[column] = numeric(frame[column])
    return frame[RAW_COLUMNS]


def valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None and any(
                member.endswith("_1.csv") for member in archive.namelist()
            )
    except zipfile.BadZipFile:
        return False


def download_month(
    sessions: tuple[requests.Session, ...],
    month: str,
    raw_dir: Path,
    timeout: float,
    retries: int,
) -> Path:
    output = raw_dir / f"{month}.zip"
    if valid_zip(output):
        return output
    if output.exists():
        output.unlink()
    url = SOURCE_TEMPLATE.format(month=month)
    partial = raw_dir / f"{month}.zip.part"
    last_error: Exception | None = None
    for attempt in range(retries):
        session = sessions[attempt % len(sessions)]
        try:
            response = session.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            if not response.content.startswith(b"PK"):
                raise RuntimeError(f"{url} did not return a ZIP archive")
            partial.write_bytes(response.content)
            if not valid_zip(partial):
                raise RuntimeError(f"{url} returned an invalid ZIP archive")
            partial.replace(output)
            return output
        except (requests.RequestException, OSError, RuntimeError, zipfile.BadZipFile) as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt + 1 < retries:
                time.sleep(min(2.0**attempt, 8.0))
    raise RuntimeError(f"failed to download {url} after {retries} attempts") from last_error


def parse_archives(paths: list[Path], start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith("_1.csv"):
                    continue
                day = member[-14:-6]
                if start_date.replace("-", "") <= day <= end_date.replace("-", ""):
                    frames.append(parse_daily_csv(archive.read(member), member))
    if not frames:
        raise RuntimeError("no daily futures rows were parsed")
    output = pd.concat(frames, ignore_index=True)
    return output.sort_values(["date", "product", "symbol"]).reset_index(drop=True)


def contract_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = raw.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    working = working.sort_values(["symbol", "date"])
    working["contract_return_1"] = working.groupby("symbol")["close"].pct_change()
    working["previous_open_interest"] = working.groupby("symbol")["open_interest"].shift(1)
    working["selection_open_interest"] = working["previous_open_interest"].fillna(working["open_interest"])
    working["expiry_key"] = pd.to_numeric(working["symbol"].str[-4:], errors="coerce")

    selected_rows: list[pd.Series] = []
    term_rows: list[dict[str, object]] = []
    for (date, product), group in working.groupby(["date", "product"], sort=True):
        active = group.dropna(subset=["close", "selection_open_interest"]).copy()
        if active.empty:
            continue
        dominant = active.sort_values(
            ["selection_open_interest", "expiry_key", "symbol"],
            ascending=[False, True, True],
        ).iloc[0]
        selected_rows.append(dominant)
        by_expiry = active.sort_values(["expiry_key", "symbol"])
        term_spread = 0.0
        if len(by_expiry) >= 2 and float(by_expiry.iloc[0]["close"]) > 0:
            term_spread = float(by_expiry.iloc[1]["close"] / by_expiry.iloc[0]["close"] - 1.0)
        term_rows.append({"date": date, "product": product, "term_spread": term_spread})

    main = pd.DataFrame(selected_rows).sort_values(["product", "date"])
    term = pd.DataFrame(term_rows)
    main = main.merge(term, on=["date", "product"], how="left", validate="one_to_one")
    main["main_return_1"] = main["contract_return_1"].fillna(0.0)
    main["main_return_5"] = main.groupby("product")["main_return_1"].transform(
        lambda values: (1.0 + values).rolling(5, min_periods=2).apply(np.prod, raw=True) - 1.0
    )
    main["main_volatility_20"] = main.groupby("product")["main_return_1"].transform(
        lambda values: values.rolling(20, min_periods=5).std(ddof=0)
    )
    volume_log = np.log1p(main["volume"].clip(lower=0))
    volume_mean = volume_log.groupby(main["product"]).transform(
        lambda values: values.rolling(20, min_periods=5).mean()
    )
    volume_std = volume_log.groupby(main["product"]).transform(
        lambda values: values.rolling(20, min_periods=5).std(ddof=0)
    )
    main["volume_z20"] = (volume_log - volume_mean) / (volume_std + 1e-12)
    main["oi_change"] = (
        main["open_interest_change"]
        / (main["open_interest"].abs() + main["open_interest_change"].abs() + 1e-12)
    )
    return main, working


def wide_features(main: pd.DataFrame, stock_dates: pd.DatetimeIndex) -> pd.DataFrame:
    pivots: dict[str, pd.DataFrame] = {}
    for value in [
        "main_return_1",
        "main_return_5",
        "main_volatility_20",
        "volume_z20",
        "oi_change",
        "term_spread",
    ]:
        pivots[value] = main.pivot(index="date", columns="product", values=value).sort_index()
    output = pd.DataFrame(index=stock_dates)
    output["fut_if_return_1"] = pivots["main_return_1"].get("IF")
    output["fut_if_return_5"] = pivots["main_return_5"].get("IF")
    output["fut_if_volatility_20"] = pivots["main_volatility_20"].get("IF")
    output["fut_if_volume_z20"] = pivots["volume_z20"].get("IF")
    output["fut_if_oi_change"] = pivots["oi_change"].get("IF")
    output["fut_if_term_spread"] = pivots["term_spread"].get("IF")
    output["fut_if_ih_spread_5"] = pivots["main_return_5"].get("IF") - pivots["main_return_5"].get("IH")
    output["fut_ic_if_spread_5"] = pivots["main_return_5"].get("IC") - pivots["main_return_5"].get("IF")
    output["fut_im_if_spread_5"] = pivots["main_return_5"].get("IM") - pivots["main_return_5"].get("IF")
    for product in PRODUCTS:
        available_dates = set(main.loc[main["product"] == product, "date"])
        output[f"fut_{product.lower()}_available"] = output.index.to_series().isin(available_dates).astype(float)
    output = output.replace([np.inf, -np.inf], np.nan)
    value_columns = [column for column in OUTPUT_COLUMNS if column != "date" and not column.endswith("_available")]
    output[value_columns] = output[value_columns].fillna(0.0)
    output = output.reset_index(names="date")
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    return output[OUTPUT_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze CFFEX daily equity-index futures data.")
    parser.add_argument("--start-date", default="2021-01-04")
    parser.add_argument("--end-date", default="2026-07-03")
    parser.add_argument("--stock-train-csv", required=True)
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

    proxy_session = requests.Session()
    direct_session = requests.Session()
    direct_session.trust_env = False
    archives = [
        download_month(
            (proxy_session, direct_session),
            month,
            raw_dir,
            float(args.timeout_seconds),
            max(1, int(args.download_retries)),
        )
        for month in months_between(args.start_date, args.end_date)
    ]
    raw = parse_archives(archives, args.start_date, args.end_date)
    raw_path = processed_dir / "cffex_contract_daily.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")
    main_contracts, _ = contract_features(raw)
    main_path = processed_dir / "cffex_main_contract_daily.csv"
    main_contracts.to_csv(main_path, index=False, encoding="utf-8")

    market = pd.read_csv(args.stock_train_csv)
    date_column = market.columns[1]
    stock_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(market[date_column]).dt.normalize().unique()),
        name="date",
    )
    features = wide_features(main_contracts, stock_dates)
    feature_path = processed_dir / "cffex_market_features.csv"
    features.to_csv(feature_path, index=False, encoding="utf-8")

    coverage = {
        product: float(
            features.loc[features["date"] >= args.start_date, f"fut_{product.lower()}_available"].mean()
        )
        for product in PRODUCTS
    }
    validation = {
        "status": "pass" if coverage["IF"] >= 0.99 else "fail",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "archive_count": len(archives),
        "contract_row_count": int(len(raw)),
        "main_row_count": int(len(main_contracts)),
        "feature_row_count": int(len(features)),
        "stock_trading_date_count": int(len(stock_dates)),
        "coverage_by_product": coverage,
        "duplicate_contract_date_count": int(raw.duplicated(["date", "symbol"]).sum()),
        "duplicate_feature_date_count": int(features.duplicated(["date"]).sum()),
        "main_contract_selection": "highest prior-day open interest; current-day fallback on first observation",
        "return_rule": "same-contract close-to-close return before dominant-contract selection",
    }
    validation_path = output_dir / "validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    files = [*archives, raw_path, main_path, feature_path, validation_path]
    manifest = {
        "schema_version": 1,
        "dataset_id": f"cffex_equity_index_daily_{args.start_date}_{args.end_date}",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url_template": SOURCE_TEMPLATE,
        "source_provider": "China Financial Futures Exchange",
        "products": list(PRODUCTS),
        "competition_reporting_status": "UNVERIFIED_AFTER_2026-07-18_RESEARCH_ONLY",
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
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

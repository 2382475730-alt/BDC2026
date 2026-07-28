from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DATE_COL = "日期"
STOCK_COL = "股票代码"
OPEN_COL = "开盘"
CLOSE_COL = "收盘"
HIGH_COL = "最高"
LOW_COL = "最低"
VOLUME_COL = "成交量"
AMOUNT_COL = "成交额"
TURNOVER_COL = "换手率"

BASE_COLUMNS = [
    STOCK_COL,
    DATE_COL,
    OPEN_COL,
    CLOSE_COL,
    HIGH_COL,
    LOW_COL,
    VOLUME_COL,
    AMOUNT_COL,
    TURNOVER_COL,
]

STOCK_FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "return_60",
    "intraday_return",
    "high_low_spread",
    "open_close_spread",
    "volume_change",
    "volume_ratio_20",
    "amount_ratio_20",
    "turnover",
    "volatility_5",
    "volatility_20",
    "downside_volatility_20",
    "drawdown_20",
    "atr_proxy_14",
    "market_beta_20",
    "market_corr_20",
]

INDUSTRY_STOCK_FEATURE_COLUMNS = [
    "industry_return_1",
    "industry_return_5",
    "industry_return_20",
    "industry_breadth",
    "industry_dispersion_20",
    "industry_volume_ratio_20",
    "industry_relative_return_5",
    "industry_relative_return_20",
    "industry_momentum_rank_20",
    "stock_industry_rank_20",
]

FACTOR_FEATURE_COLUMNS = [
    "factor_momentum_short",
    "factor_momentum_medium",
    "factor_momentum_long",
    "factor_volatility",
    "factor_drawdown",
    "factor_liquidity",
    "factor_volume_shock",
    "factor_market_beta",
]

MARKET_FEATURE_COLUMNS = [
    "market_return_5",
    "market_return_20",
    "market_return_60",
    "market_volatility_20",
    "market_downside_volatility_20",
    "market_atr_proxy_20",
    "market_breadth",
    "market_advance_decline_ratio",
    "market_new_high_ratio_60",
    "market_volume_ratio_20",
    "market_turnover_mean",
    "market_return_dispersion_20",
]

FUTURES_MARKET_FEATURE_COLUMNS = [
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


@dataclass(frozen=True)
class FeatureColumns:
    stock: list[str]
    factor: list[str]
    market: list[str]


def feature_columns(
    enable_industry: bool = False,
    enable_futures: bool = False,
) -> FeatureColumns:
    return FeatureColumns(
        stock=list(STOCK_FEATURE_COLUMNS)
        + (list(INDUSTRY_STOCK_FEATURE_COLUMNS) if enable_industry else []),
        factor=list(FACTOR_FEATURE_COLUMNS),
        market=list(MARKET_FEATURE_COLUMNS)
        + (list(FUTURES_MARKET_FEATURE_COLUMNS) if enable_futures else []),
    )


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _safe_div(numerator: pd.Series, denominator: pd.Series | float, eps: float = 1e-12) -> pd.Series:
    return numerator / (denominator + eps)


def _rolling_compound_return(series: pd.Series, window: int) -> pd.Series:
    return (1.0 + series).rolling(window, min_periods=max(2, window // 3)).apply(np.prod, raw=True) - 1.0


def _rolling_downside_std(series: pd.Series, window: int) -> pd.Series:
    downside = series.where(series < 0.0, 0.0)
    return downside.rolling(window, min_periods=max(2, window // 3)).std(ddof=0)


def _build_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["is_new_high_60"] = (
        working[CLOSE_COL]
        >= working.groupby(STOCK_COL, sort=False)[CLOSE_COL].transform(
            lambda values: values.rolling(60, min_periods=20).max()
        )
    ).astype(float)

    grouped = working.groupby(DATE_COL, sort=True)
    daily = grouped.agg(
        market_return_1=("return_1", "mean"),
        market_cross_section_vol=("return_1", "std"),
        market_breadth=("return_1", lambda values: float((values > 0.0).mean())),
        market_turnover_mean=("turnover", "mean"),
        market_volume_sum=(VOLUME_COL, "sum"),
        market_atr_proxy_1=("atr_proxy_14", "mean"),
        market_new_high_ratio_60=("is_new_high_60", "mean"),
    ).sort_index()

    advances = grouped["return_1"].apply(lambda values: int((values > 0.0).sum()))
    declines = grouped["return_1"].apply(lambda values: int((values < 0.0).sum()))
    daily["market_advance_decline_ratio"] = advances / (declines + 1.0)

    daily["market_return_5"] = _rolling_compound_return(daily["market_return_1"], 5)
    daily["market_return_20"] = _rolling_compound_return(daily["market_return_1"], 20)
    daily["market_return_60"] = _rolling_compound_return(daily["market_return_1"], 60)
    daily["market_volatility_20"] = daily["market_return_1"].rolling(20, min_periods=5).std(ddof=0)
    daily["market_downside_volatility_20"] = _rolling_downside_std(daily["market_return_1"], 20)
    daily["market_atr_proxy_20"] = daily["market_atr_proxy_1"].rolling(20, min_periods=5).mean()
    daily["market_volume_ratio_20"] = (
        daily["market_volume_sum"] / (daily["market_volume_sum"].rolling(20, min_periods=5).mean() + 1e-12) - 1.0
    )
    daily["market_return_dispersion_20"] = daily["market_cross_section_vol"].rolling(20, min_periods=5).mean()

    daily = daily.replace([np.inf, -np.inf], np.nan)
    return daily[MARKET_FEATURE_COLUMNS]


def _normalize_industry_map(
    industry_map: pd.DataFrame,
    missing_label: str,
) -> pd.DataFrame:
    required = {"stock_code", "industry"}
    missing = sorted(required - set(industry_map.columns))
    if missing:
        raise KeyError(f"industry map missing required columns: {missing}")
    output = industry_map[["stock_code", "industry"]].copy()
    output["stock_code"] = (
        output["stock_code"].astype(str).str.replace(r"\.0$", "", regex=True).str[-6:].str.zfill(6)
    )
    output["industry"] = output["industry"].fillna(missing_label).astype(str).str.strip()
    output.loc[output["industry"] == "", "industry"] = missing_label
    if output["stock_code"].duplicated().any():
        duplicates = output.loc[output["stock_code"].duplicated(), "stock_code"].head(10).tolist()
        raise ValueError(f"industry map contains duplicate stock codes: {duplicates}")
    return output


def _attach_industry_features(
    df: pd.DataFrame,
    industry_map: pd.DataFrame,
    missing_label: str,
) -> pd.DataFrame:
    mapping = _normalize_industry_map(industry_map, missing_label)
    industry_by_code = mapping.set_index("stock_code")["industry"]
    working = df.copy()
    working["industry"] = working[STOCK_COL].map(industry_by_code).fillna(missing_label)

    daily = (
        working.groupby([DATE_COL, "industry"], sort=True)
        .agg(
            industry_return_1=("return_1", "mean"),
            industry_breadth=("return_1", lambda values: float((values > 0.0).mean())),
            industry_dispersion_1=("return_1", "std"),
            industry_volume_sum=(VOLUME_COL, "sum"),
        )
        .reset_index()
        .sort_values(["industry", DATE_COL])
    )
    by_industry = daily.groupby("industry", sort=False)
    daily["industry_return_5"] = by_industry["industry_return_1"].transform(
        lambda values: _rolling_compound_return(values, 5)
    )
    daily["industry_return_20"] = by_industry["industry_return_1"].transform(
        lambda values: _rolling_compound_return(values, 20)
    )
    daily["industry_dispersion_20"] = by_industry["industry_dispersion_1"].transform(
        lambda values: values.rolling(20, min_periods=5).mean()
    )
    volume_mean = by_industry["industry_volume_sum"].transform(
        lambda values: values.rolling(20, min_periods=5).mean()
    )
    daily["industry_volume_ratio_20"] = daily["industry_volume_sum"] / (volume_mean + 1e-12) - 1.0
    daily["industry_momentum_rank_20"] = daily.groupby(DATE_COL)["industry_return_20"].rank(
        pct=True,
        method="average",
    )

    merge_columns = [
        DATE_COL,
        "industry",
        "industry_return_1",
        "industry_return_5",
        "industry_return_20",
        "industry_breadth",
        "industry_dispersion_20",
        "industry_volume_ratio_20",
        "industry_momentum_rank_20",
    ]
    working = working.merge(daily[merge_columns], on=[DATE_COL, "industry"], how="left", validate="many_to_one")
    working["industry_relative_return_5"] = working["return_5"] - working["industry_return_5"]
    working["industry_relative_return_20"] = working["return_20"] - working["industry_return_20"]
    working["stock_industry_rank_20"] = working.groupby(
        [DATE_COL, "industry"],
        sort=False,
    )["return_20"].rank(pct=True, method="average")
    working[INDUSTRY_STOCK_FEATURE_COLUMNS] = working[INDUSTRY_STOCK_FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return working


def _attach_futures_features(
    market_frame: pd.DataFrame,
    futures_frame: pd.DataFrame,
    lag_days: int,
) -> pd.DataFrame:
    required = {"date", *FUTURES_MARKET_FEATURE_COLUMNS}
    missing = sorted(required - set(futures_frame.columns))
    if missing:
        raise KeyError(f"futures feature file missing required columns: {missing}")
    auxiliary = futures_frame[["date", *FUTURES_MARKET_FEATURE_COLUMNS]].copy()
    auxiliary["date"] = pd.to_datetime(auxiliary["date"]).dt.normalize()
    if auxiliary["date"].duplicated().any():
        raise ValueError("futures feature file contains duplicate dates")
    auxiliary = auxiliary.sort_values("date").set_index("date")
    if lag_days > 0:
        auxiliary[FUTURES_MARKET_FEATURE_COLUMNS] = auxiliary[FUTURES_MARKET_FEATURE_COLUMNS].shift(lag_days)
    output = market_frame.join(auxiliary, how="left")
    output[FUTURES_MARKET_FEATURE_COLUMNS] = output[FUTURES_MARKET_FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    availability = [column for column in FUTURES_MARKET_FEATURE_COLUMNS if column.endswith("_available")]
    values = [column for column in FUTURES_MARKET_FEATURE_COLUMNS if column not in availability]
    output[availability] = output[availability].fillna(0.0)
    output[values] = output[values].fillna(0.0)
    return output


def prepare_d_features(
    raw_df: pd.DataFrame,
    horizon: int = 5,
    buy_offset: int = 1,
    sell_offset: int = 5,
    industry_map: pd.DataFrame | None = None,
    futures_frame: pd.DataFrame | None = None,
    enable_industry: bool = False,
    enable_futures: bool = False,
    industry_missing_label: str = "UNKNOWN",
    futures_lag_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureColumns]:
    """Create D-series causal features and supervised labels.

    Feature columns use rows at or before sample date T. Labels are the official
    open-to-open forward return and should only be used inside the training fold.
    """
    if sell_offset != horizon:
        raise ValueError("D001 expects sell_offset to match horizon")
    if buy_offset < 1 or sell_offset <= buy_offset:
        raise ValueError("expected 1 <= buy_offset < sell_offset")

    _require_columns(raw_df, BASE_COLUMNS)
    df = raw_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.normalize()
    df[STOCK_COL] = df[STOCK_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    numeric_columns = [OPEN_COL, CLOSE_COL, HIGH_COL, LOW_COL, VOLUME_COL, AMOUNT_COL, TURNOVER_COL]
    df = _to_numeric(df, numeric_columns)
    df = df.sort_values([STOCK_COL, DATE_COL]).reset_index(drop=True)

    by_stock = df.groupby(STOCK_COL, sort=False)
    df["return_1"] = by_stock[CLOSE_COL].pct_change()
    for window in (5, 10, 20, 60):
        df[f"return_{window}"] = by_stock[CLOSE_COL].pct_change(window)

    df["intraday_return"] = _safe_div(df[CLOSE_COL] - df[OPEN_COL], df[OPEN_COL])
    df["high_low_spread"] = _safe_div(df[HIGH_COL] - df[LOW_COL], df[OPEN_COL])
    df["open_close_spread"] = _safe_div(df[CLOSE_COL] - df[OPEN_COL], df[OPEN_COL])
    df["volume_change"] = by_stock[VOLUME_COL].pct_change()
    df["volume_ratio_20"] = (
        df[VOLUME_COL] / (by_stock[VOLUME_COL].transform(lambda values: values.rolling(20, min_periods=5).mean()) + 1e-12)
        - 1.0
    )
    df["amount_ratio_20"] = (
        df[AMOUNT_COL] / (by_stock[AMOUNT_COL].transform(lambda values: values.rolling(20, min_periods=5).mean()) + 1e-12)
        - 1.0
    )
    df["turnover"] = df[TURNOVER_COL]
    for window in (5, 20):
        df[f"volatility_{window}"] = by_stock["return_1"].transform(
            lambda values, w=window: values.rolling(w, min_periods=max(2, w // 3)).std(ddof=0)
        )
    df["downside_volatility_20"] = by_stock["return_1"].transform(lambda values: _rolling_downside_std(values, 20))
    df["drawdown_20"] = (
        df[CLOSE_COL] / (by_stock[CLOSE_COL].transform(lambda values: values.rolling(20, min_periods=5).max()) + 1e-12)
        - 1.0
    )
    df["atr_proxy_14"] = by_stock["high_low_spread"].transform(
        lambda values: values.rolling(14, min_periods=5).mean()
    )

    market_frame = _build_market_frame(df)
    df["market_return_proxy"] = df[DATE_COL].map(market_frame["market_return_5"]).fillna(0.0)

    beta_values: list[pd.Series] = []
    corr_values: list[pd.Series] = []
    for _, group in df.groupby(STOCK_COL, sort=False):
        stock_ret = group["return_1"]
        market_ret = group["market_return_proxy"]
        covariance = stock_ret.rolling(20, min_periods=8).cov(market_ret)
        variance = market_ret.rolling(20, min_periods=8).var(ddof=0)
        beta_values.append(covariance / (variance + 1e-12))
        corr_values.append(stock_ret.rolling(20, min_periods=8).corr(market_ret))
    df["market_beta_20"] = pd.concat(beta_values).sort_index()
    df["market_corr_20"] = pd.concat(corr_values).sort_index()

    df["factor_momentum_short"] = df["return_5"]
    df["factor_momentum_medium"] = df["return_20"]
    df["factor_momentum_long"] = df["return_60"]
    df["factor_volatility"] = df["volatility_20"]
    df["factor_drawdown"] = df["drawdown_20"]
    df["factor_liquidity"] = df["turnover"]
    df["factor_volume_shock"] = df["volume_ratio_20"]
    df["factor_market_beta"] = df["market_beta_20"]

    if enable_industry:
        if industry_map is None:
            raise ValueError("industry features enabled but no industry map was supplied")
        df = _attach_industry_features(df, industry_map, industry_missing_label)
    if enable_futures:
        if futures_frame is None:
            raise ValueError("futures features enabled but no futures frame was supplied")
        market_frame = _attach_futures_features(market_frame, futures_frame, futures_lag_days)

    by_stock = df.groupby(STOCK_COL, sort=False)
    df["open_t_buy"] = by_stock[OPEN_COL].shift(-buy_offset)
    df["open_t_sell"] = by_stock[OPEN_COL].shift(-sell_offset)
    df["label"] = _safe_div(df["open_t_sell"] - df["open_t_buy"], df["open_t_buy"])
    df["label_end_date"] = by_stock[DATE_COL].shift(-sell_offset)
    df = df.drop(columns=["open_t_buy", "open_t_sell"])

    columns = feature_columns(enable_industry=enable_industry, enable_futures=enable_futures)
    all_feature_columns = columns.stock + columns.factor
    df[all_feature_columns] = df[all_feature_columns].replace([np.inf, -np.inf], np.nan)
    market_frame[columns.market] = market_frame[columns.market].replace([np.inf, -np.inf], np.nan)
    return df, market_frame, columns


def assert_market_truncation_equivalence(
    raw_df: pd.DataFrame,
    sample_dates: Iterable[pd.Timestamp | str],
    atol: float = 1e-10,
) -> bool:
    """Prove market features at T are unchanged when future rows are removed."""
    full_df, full_market, columns = prepare_d_features(raw_df)
    del full_df
    full_market = full_market.sort_index()
    raw = raw_df.copy()
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL]).dt.normalize()

    for value in sample_dates:
        sample_date = pd.Timestamp(value).normalize()
        truncated = raw[raw[DATE_COL] <= sample_date].copy()
        _, truncated_market, _ = prepare_d_features(truncated)
        if sample_date not in full_market.index or sample_date not in truncated_market.index:
            raise AssertionError(f"sample date {sample_date.date()} is missing from market features")
        left = full_market.loc[sample_date, columns.market].fillna(0.0).astype(float).to_numpy()
        right = truncated_market.loc[sample_date, columns.market].fillna(0.0).astype(float).to_numpy()
        if not np.allclose(left, right, atol=atol, rtol=0.0):
            diff = np.abs(left - right).max()
            raise AssertionError(f"market features use future information at {sample_date.date()}; max diff={diff}")
    return True

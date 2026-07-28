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
    STOCK_COL, DATE_COL, OPEN_COL, CLOSE_COL, HIGH_COL, LOW_COL,
    VOLUME_COL, AMOUNT_COL, TURNOVER_COL,
]

D02_STOCK_FEATURES = [
    "return_1", "return_5", "return_10", "return_20", "return_60",
    "intraday_return", "high_low_spread", "open_close_spread", "volume_change",
    "volume_ratio_20", "amount_ratio_20", "turnover", "volatility_5",
    "volatility_20", "downside_volatility_20", "drawdown_20", "atr_proxy_14",
    "market_beta_20", "market_corr_20",
]

DEDUP_STOCK_FEATURES = [name for name in D02_STOCK_FEATURES if name != "open_close_spread"]

FULL_EXTRA_FEATURES = [
    "drawdown_60", "return_5_xs_pct", "return_20_xs_pct", "return_60_xs_pct",
    "volatility_20_xs_pct", "drawdown_60_xs_pct", "turnover_xs_pct",
    "volume_ratio_20_xs_pct", "market_beta_20_xs_pct",
    "return_1_robust_z60", "return_1_robust_z120",
    "turnover_robust_z60", "turnover_robust_z120",
    "volume_ratio_20_robust_z60", "volume_ratio_20_robust_z120",
    "momentum_slope_5", "momentum_slope_20", "momentum_slope_60",
    "drawdown_repair_5", "vol5_over_vol20", "volume_price_confirmation",
    "history_valid_60", "history_valid_120",
]

MARKET_FEATURES = [
    "market_return_1", "market_return_5", "market_return_20", "market_return_60",
    "market_volatility_20", "market_downside_volatility_20", "market_atr_proxy_20",
    "market_breadth", "market_advance_decline_ratio", "market_new_high_ratio_60",
    "market_volume_ratio_20", "market_turnover_mean", "market_return_dispersion_20",
    "market_breadth_delta_5", "market_dispersion_delta_5",
    "market_turnover_delta_5", "market_volatility_delta_5",
]


@dataclass(frozen=True)
class FeatureColumns:
    stock: list[str]
    factor: list[str]
    market: list[str]


def feature_columns(version: str = "e1_full_v1") -> FeatureColumns:
    if version in {"e0_d02_exact", "d_market_v1"}:
        stock = list(D02_STOCK_FEATURES)
    elif version in {"e1_dedup_v1", "e1_causal_scale_v1"}:
        stock = list(DEDUP_STOCK_FEATURES)
    elif version == "e1_full_v1":
        stock = list(DEDUP_STOCK_FEATURES + FULL_EXTRA_FEATURES)
    else:
        raise ValueError(f"unsupported E feature version: {version}")
    return FeatureColumns(stock=stock, factor=[], market=list(MARKET_FEATURES))


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")


def _safe_div(numerator: pd.Series, denominator: pd.Series | float, eps: float = 1e-12) -> pd.Series:
    return numerator / (denominator + eps)


def _rolling_downside_std(series: pd.Series, window: int) -> pd.Series:
    return series.where(series < 0.0, 0.0).rolling(window, min_periods=max(2, window // 3)).std(ddof=0)


def _rolling_compound_return(series: pd.Series, window: int) -> pd.Series:
    return (1.0 + series).rolling(window, min_periods=max(2, window // 3)).apply(np.prod, raw=True) - 1.0


def _rolling_robust_z(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(10, window // 3)
    median = series.rolling(window, min_periods=min_periods).median()
    mad = (series - median).abs().rolling(window, min_periods=min_periods).median()
    return (series - median) / (1.4826 * mad + 1e-8)


def _rolling_log_slope(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(3, window // 2)

    def slope(values: np.ndarray) -> float:
        valid = np.isfinite(values) & (values > 0.0)
        if int(valid.sum()) < min_periods:
            return np.nan
        y = np.log(values[valid])
        x = np.arange(values.size, dtype=np.float64)[valid]
        x = x - x.mean()
        denom = float(np.dot(x, x))
        return float(np.dot(x, y - y.mean()) / denom) if denom > 0 else 0.0

    return series.rolling(window, min_periods=min_periods).apply(slope, raw=True)


def _build_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    rolling_high = working.groupby(STOCK_COL, sort=False)[CLOSE_COL].transform(
        lambda x: x.rolling(60, min_periods=20).max()
    )
    working["is_new_high_60"] = (working[CLOSE_COL] >= rolling_high).astype(float)
    grouped = working.groupby(DATE_COL, sort=True)
    daily = grouped.agg(
        market_return_1=("return_1", "mean"),
        market_cross_section_vol=("return_1", "std"),
        market_breadth=("return_1", lambda x: float((x > 0.0).mean())),
        market_turnover_mean=("turnover", "mean"),
        market_volume_sum=(VOLUME_COL, "sum"),
        market_atr_proxy_1=("atr_proxy_14", "mean"),
        market_new_high_ratio_60=("is_new_high_60", "mean"),
    ).sort_index()
    advances = grouped["return_1"].apply(lambda x: int((x > 0.0).sum()))
    declines = grouped["return_1"].apply(lambda x: int((x < 0.0).sum()))
    daily["market_advance_decline_ratio"] = advances / (declines + 1.0)
    for window in (5, 20, 60):
        daily[f"market_return_{window}"] = _rolling_compound_return(daily["market_return_1"], window)
    daily["market_volatility_20"] = daily["market_return_1"].rolling(20, min_periods=5).std(ddof=0)
    daily["market_downside_volatility_20"] = _rolling_downside_std(daily["market_return_1"], 20)
    daily["market_atr_proxy_20"] = daily["market_atr_proxy_1"].rolling(20, min_periods=5).mean()
    daily["market_volume_ratio_20"] = daily["market_volume_sum"] / (
        daily["market_volume_sum"].rolling(20, min_periods=5).mean() + 1e-12
    ) - 1.0
    daily["market_return_dispersion_20"] = daily["market_cross_section_vol"].rolling(20, min_periods=5).mean()
    daily["market_breadth_delta_5"] = daily["market_breadth"].diff(5)
    daily["market_dispersion_delta_5"] = daily["market_return_dispersion_20"].diff(5)
    daily["market_turnover_delta_5"] = daily["market_turnover_mean"].diff(5)
    daily["market_volatility_delta_5"] = daily["market_volatility_20"].diff(5)
    return daily[MARKET_FEATURES].replace([np.inf, -np.inf], np.nan)


def _attach_calendar_labels(df: pd.DataFrame, buy_offset: int, sell_offset: int) -> pd.DataFrame:
    """Join open(T+1)/open(T+5) on the global trading calendar.

    A suspended or not-yet-listed stock is not silently relabelled using its own
    next row: either target-calendar open missing makes the label unavailable.
    """
    calendar = pd.Index(sorted(df[DATE_COL].dropna().unique()))
    index_by_date = {pd.Timestamp(value): idx for idx, value in enumerate(calendar)}
    target = df[[STOCK_COL, DATE_COL, OPEN_COL]].copy()
    target["_calendar_index"] = target[DATE_COL].map(index_by_date)
    opens = target.set_index([STOCK_COL, "_calendar_index"])[OPEN_COL]
    current_index = target["_calendar_index"].to_numpy()
    stocks = target[STOCK_COL].to_numpy()
    buy_index = pd.MultiIndex.from_arrays([stocks, current_index + buy_offset])
    sell_index = pd.MultiIndex.from_arrays([stocks, current_index + sell_offset])
    target["open_t_buy"] = opens.reindex(buy_index).to_numpy()
    target["open_t_sell"] = opens.reindex(sell_index).to_numpy()
    label_end = np.full(len(target), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_end = current_index + sell_offset < len(calendar)
    label_end[valid_end] = calendar.to_numpy()[current_index[valid_end] + sell_offset]
    target["label_end_date"] = label_end
    target["label"] = (target["open_t_sell"] - target["open_t_buy"]) / (target["open_t_buy"] + 1e-12)
    return df.merge(
        target[[STOCK_COL, DATE_COL, "label", "label_end_date"]],
        on=[STOCK_COL, DATE_COL], how="left", validate="one_to_one",
    )


def prepare_e_features(
    raw_df: pd.DataFrame,
    feature_version: str = "e1_full_v1",
    horizon: int = 5,
    buy_offset: int = 1,
    sell_offset: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureColumns]:
    if sell_offset != horizon or buy_offset != 1 or sell_offset != 5:
        raise ValueError("E_R1 fixes label to open(T+5)/open(T+1)-1")
    columns = feature_columns(feature_version)
    _require_columns(raw_df, BASE_COLUMNS)
    df = raw_df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.normalize()
    df[STOCK_COL] = df[STOCK_COL].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    numeric = [OPEN_COL, CLOSE_COL, HIGH_COL, LOW_COL, VOLUME_COL, AMOUNT_COL, TURNOVER_COL]
    for name in numeric:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    if df.duplicated([STOCK_COL, DATE_COL]).any():
        raise ValueError("duplicate stock/date rows are forbidden")
    df = df.sort_values([STOCK_COL, DATE_COL]).reset_index(drop=True)
    by_stock = df.groupby(STOCK_COL, sort=False)
    df["return_1"] = by_stock[CLOSE_COL].pct_change(fill_method=None)
    for window in (5, 10, 20, 60):
        df[f"return_{window}"] = by_stock[CLOSE_COL].pct_change(window, fill_method=None)
    df["intraday_return"] = _safe_div(df[CLOSE_COL] - df[OPEN_COL], df[OPEN_COL])
    df["high_low_spread"] = _safe_div(df[HIGH_COL] - df[LOW_COL], df[OPEN_COL])
    df["open_close_spread"] = df["intraday_return"]  # retained only for E0 exact reproduction
    df["volume_change"] = by_stock[VOLUME_COL].pct_change(fill_method=None)
    for raw_name, out_name in ((VOLUME_COL, "volume_ratio_20"), (AMOUNT_COL, "amount_ratio_20")):
        mean20 = by_stock[raw_name].transform(lambda x: x.rolling(20, min_periods=5).mean())
        df[out_name] = df[raw_name] / (mean20 + 1e-12) - 1.0
    df["turnover"] = df[TURNOVER_COL]
    for window in (5, 20):
        df[f"volatility_{window}"] = by_stock["return_1"].transform(
            lambda x, w=window: x.rolling(w, min_periods=max(2, w // 3)).std(ddof=0)
        )
    df["downside_volatility_20"] = by_stock["return_1"].transform(lambda x: _rolling_downside_std(x, 20))
    df["drawdown_20"] = df[CLOSE_COL] / (
        by_stock[CLOSE_COL].transform(lambda x: x.rolling(20, min_periods=5).max()) + 1e-12
    ) - 1.0
    df["drawdown_60"] = df[CLOSE_COL] / (
        by_stock[CLOSE_COL].transform(lambda x: x.rolling(60, min_periods=20).max()) + 1e-12
    ) - 1.0
    df["atr_proxy_14"] = by_stock["high_low_spread"].transform(lambda x: x.rolling(14, min_periods=5).mean())

    # Build level features before beta/correlation; beta uses same-frequency
    # return_1 on both sides (D used return_1 versus market_return_5).
    market_frame = _build_market_frame(df)
    proxy_name = "market_return_5" if feature_version in {"e0_d02_exact", "e1_dedup_v1"} else "market_return_1"
    market_proxy = df[DATE_COL].map(market_frame[proxy_name])
    beta_parts: list[pd.Series] = []
    corr_parts: list[pd.Series] = []
    for _, group in df.assign(_market_proxy=market_proxy).groupby(STOCK_COL, sort=False):
        covariance = group["return_1"].rolling(20, min_periods=8).cov(group["_market_proxy"])
        variance = group["_market_proxy"].rolling(20, min_periods=8).var(ddof=0)
        beta_parts.append(covariance / (variance + 1e-12))
        corr_parts.append(group["return_1"].rolling(20, min_periods=8).corr(group["_market_proxy"]))
    df["market_beta_20"] = pd.concat(beta_parts).sort_index()
    df["market_corr_20"] = pd.concat(corr_parts).sort_index()

    if feature_version == "e1_full_v1":
        for source in ("return_5", "return_20", "return_60", "volatility_20", "drawdown_60", "turnover", "volume_ratio_20", "market_beta_20"):
            df[f"{source}_xs_pct"] = df.groupby(DATE_COL, sort=False)[source].rank(pct=True, method="average")
        for source in ("return_1", "turnover", "volume_ratio_20"):
            for window in (60, 120):
                df[f"{source}_robust_z{window}"] = df.groupby(STOCK_COL, sort=False)[source].transform(
                    lambda x, w=window: _rolling_robust_z(x, w)
                )
        for window in (5, 20, 60):
            df[f"momentum_slope_{window}"] = df.groupby(STOCK_COL, sort=False)[CLOSE_COL].transform(
                lambda x, w=window: _rolling_log_slope(x, w)
            )
        df["drawdown_repair_5"] = df["drawdown_20"] - by_stock["drawdown_20"].shift(5)
        df["vol5_over_vol20"] = np.log((df["volatility_5"] + 1e-8) / (df["volatility_20"] + 1e-8))
        df["volume_price_confirmation"] = np.sign(df["return_5"]) * np.log1p(df["volume_ratio_20"].clip(lower=-0.999999))
        history_count = by_stock.cumcount() + 1
        df["history_valid_60"] = (history_count >= 60).astype(float)
        df["history_valid_120"] = (history_count >= 120).astype(float)

    df = _attach_calendar_labels(df, buy_offset=buy_offset, sell_offset=sell_offset)
    df[columns.stock] = df[columns.stock].replace([np.inf, -np.inf], np.nan)
    market_frame[columns.market] = market_frame[columns.market].replace([np.inf, -np.inf], np.nan)
    return df, market_frame, columns


# Compatibility name used by copied lineage utilities.
prepare_d_features = prepare_e_features


def assert_truncation_equivalence(
    raw_df: pd.DataFrame,
    sample_dates: Iterable[pd.Timestamp | str],
    feature_version: str = "e1_full_v1",
    atol: float = 1e-7,
) -> bool:
    """Verify stock and market features at T do not depend on rows after T."""
    full, full_market, columns = prepare_e_features(raw_df, feature_version=feature_version)
    raw = raw_df.copy()
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL]).dt.normalize()
    for value in sample_dates:
        date = pd.Timestamp(value).normalize()
        truncated_raw = raw[raw[DATE_COL] <= date].copy()
        truncated, truncated_market, _ = prepare_e_features(truncated_raw, feature_version=feature_version)
        left_stock = full.loc[full[DATE_COL] == date, [STOCK_COL] + columns.stock].sort_values(STOCK_COL)
        right_stock = truncated.loc[truncated[DATE_COL] == date, [STOCK_COL] + columns.stock].sort_values(STOCK_COL)
        if left_stock[STOCK_COL].tolist() != right_stock[STOCK_COL].tolist():
            raise AssertionError(f"stock universe changed at {date.date()}")
        left = left_stock[columns.stock].fillna(0.0).to_numpy(float)
        right = right_stock[columns.stock].fillna(0.0).to_numpy(float)
        if not np.allclose(left, right, atol=atol, rtol=0.0):
            raise AssertionError(f"stock features use future rows at {date.date()}; max_diff={np.max(np.abs(left-right))}")
        lm = full_market.loc[date, columns.market].fillna(0.0).to_numpy(float)
        rm = truncated_market.loc[date, columns.market].fillna(0.0).to_numpy(float)
        if not np.allclose(lm, rm, atol=atol, rtol=0.0):
            raise AssertionError(f"market features use future rows at {date.date()}; max_diff={np.max(np.abs(lm-rm))}")
    return True


assert_market_truncation_equivalence = assert_truncation_equivalence

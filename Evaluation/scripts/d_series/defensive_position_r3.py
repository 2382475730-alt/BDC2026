from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import normalize_stock_code, read_market_csv, safe_div


MIN_INVESTED_WEIGHT = 0.10
MAX_INVESTED_WEIGHT = 0.30
MAX_INVESTED_HHI = 0.24
RANK_PRIOR = np.asarray([0.28, 0.24, 0.20, 0.16, 0.12], dtype=np.float64)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    files = sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts and item.suffix not in {".pyc", ".pyo"}
    )
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix()
        digest.update(f"{relative}\t{file_hash(item)}\n".encode("utf-8"))
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def _rolling_last_percentile(values: pd.Series, window: int = 252, minimum: int = 60) -> pd.Series:
    def percentile(array: np.ndarray) -> float:
        finite = array[np.isfinite(array)]
        if finite.size == 0 or not np.isfinite(array[-1]):
            return math.nan
        less = float(np.sum(finite < array[-1]))
        equal = float(np.sum(finite == array[-1]))
        return (less + 0.5 * equal) / float(finite.size)

    return values.rolling(window, min_periods=minimum).apply(percentile, raw=True)


@dataclass
class CausalRiskData:
    return_panel: pd.DataFrame
    market_state: pd.DataFrame

    @classmethod
    def from_market_path(
        cls,
        market_path: str | Path,
        *,
        cutoff: str | pd.Timestamp | None = None,
    ) -> "CausalRiskData":
        market, columns = read_market_csv(market_path)
        close_candidates = [name for name in ("收盘", "close", "收盘价") if name in market.columns]
        if not close_candidates:
            raise KeyError("market data has no recognized close column")
        frame = market[[columns.stock, columns.date, close_candidates[0]]].copy()
        frame.columns = ["stock_id", "date", "close"]
        frame["stock_id"] = frame["stock_id"].map(normalize_stock_code)
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        if cutoff is not None:
            frame = frame[frame["date"] <= pd.Timestamp(cutoff).normalize()].copy()
        frame = frame.sort_values(["stock_id", "date"]).reset_index(drop=True)
        frame["return_1"] = frame.groupby("stock_id", sort=False)["close"].pct_change(fill_method=None)
        panel = frame.pivot(index="date", columns="stock_id", values="return_1").sort_index()

        valid_count = panel.notna().sum(axis=1).replace(0, np.nan)
        daily = pd.DataFrame(index=panel.index)
        daily["market_return_1"] = panel.mean(axis=1, skipna=True)
        daily["market_breadth_1"] = (panel.gt(0.0) & panel.notna()).sum(axis=1) / valid_count
        daily["market_dispersion_1"] = panel.std(axis=1, ddof=0, skipna=True)
        daily["market_return_20"] = (
            np.log1p(daily["market_return_1"].clip(lower=-0.999999))
            .rolling(20, min_periods=10)
            .sum()
            .pipe(np.expm1)
        )
        daily["market_volatility_20"] = daily["market_return_1"].rolling(20, min_periods=10).std(ddof=0)
        daily["market_breadth_20"] = daily["market_breadth_1"].rolling(20, min_periods=10).mean()
        daily["market_dispersion_20"] = daily["market_dispersion_1"].rolling(20, min_periods=10).mean()
        daily["return20_percentile"] = _rolling_last_percentile(daily["market_return_20"])
        daily["vol20_percentile"] = _rolling_last_percentile(daily["market_volatility_20"])
        daily["breadth20_percentile"] = _rolling_last_percentile(daily["market_breadth_20"])
        daily["dispersion20_percentile"] = _rolling_last_percentile(daily["market_dispersion_20"])
        components = pd.concat(
            [
                1.0 - daily["return20_percentile"],
                daily["vol20_percentile"],
                1.0 - daily["breadth20_percentile"],
                daily["dispersion20_percentile"],
            ],
            axis=1,
        )
        daily["market_stress"] = components.mean(axis=1, skipna=True).fillna(0.5).clip(0.0, 1.0)
        return cls(return_panel=panel, market_state=daily)

    def history(self, date: str | pd.Timestamp, stocks: Sequence[str], window: int = 60) -> pd.DataFrame:
        target = pd.Timestamp(date).normalize()
        normalized = [normalize_stock_code(stock) for stock in stocks]
        return self.return_panel.loc[self.return_panel.index <= target].tail(window).reindex(columns=normalized)

    def downside_risk(self, date: str | pd.Timestamp, stocks: Sequence[str], window: int = 60) -> np.ndarray:
        history = self.history(date, stocks, window=window)
        downside = history.where(history < 0.0, 0.0)
        risk = np.sqrt(np.nanmean(np.square(downside.to_numpy(dtype=np.float64)), axis=0))
        finite = risk[np.isfinite(risk) & (risk > 1e-8)]
        fallback = float(np.median(finite)) if finite.size else 0.02
        return np.where(np.isfinite(risk) & (risk > 1e-8), risk, fallback)

    def downside_capture_score(
        self,
        date: str | pd.Timestamp,
        stocks: Sequence[str],
        *,
        window: int = 120,
    ) -> np.ndarray:
        history = self.history(date, stocks, window=window)
        market = self.market_state.loc[self.market_state.index <= pd.Timestamp(date).normalize(), "market_return_1"].tail(window)
        aligned = history.join(market.rename("market"), how="inner")
        if aligned.empty:
            return np.zeros(len(stocks), dtype=np.float64)
        threshold = float(aligned["market"].quantile(0.20))
        stress_days = aligned[aligned["market"] <= threshold].drop(columns="market")
        score = stress_days.mean(axis=0, skipna=True).to_numpy(dtype=np.float64)
        return np.where(np.isfinite(score), score, -1.0)

    def sign_reliability(
        self,
        date: str | pd.Timestamp,
        stocks: Sequence[str],
        *,
        window: int = 60,
    ) -> np.ndarray:
        history = self.history(date, stocks, window=window)
        valid = history.notna().sum(axis=0).to_numpy(dtype=np.float64)
        positive = history.gt(0.0).sum(axis=0).to_numpy(dtype=np.float64)
        return (positive + 1.0) / (valid + 2.0)

    def worst_block_residual_strength(
        self,
        date: str | pd.Timestamp,
        stocks: Sequence[str],
        *,
        window: int = 80,
        block: int = 20,
    ) -> np.ndarray:
        history = self.history(date, stocks, window=window)
        market_median = self.return_panel.loc[
            self.return_panel.index <= pd.Timestamp(date).normalize()
        ].tail(window).median(axis=1, skipna=True)
        residual = history.sub(market_median, axis=0).fillna(0.0)
        values = residual.to_numpy(dtype=np.float64)
        if len(values) < block:
            return np.full(len(stocks), -1.0, dtype=np.float64)
        blocks = []
        for start in range(0, len(values), block):
            piece = values[start : start + block]
            if len(piece) < block:
                continue
            blocks.append(np.expm1(np.log1p(np.clip(piece, -0.999999, None)).sum(axis=0)))
        return np.min(np.vstack(blocks), axis=0) if blocks else np.full(len(stocks), -1.0, dtype=np.float64)

    def covariance(
        self,
        date: str | pd.Timestamp,
        stocks: Sequence[str],
        *,
        window: int = 60,
        shrinkage: float = 0.75,
    ) -> np.ndarray:
        history = self.history(date, stocks, window=window)
        covariance = history.cov(min_periods=max(20, window // 3)).to_numpy(dtype=np.float64)
        variances = np.nanvar(history.to_numpy(dtype=np.float64), axis=0, ddof=0)
        finite_variances = variances[np.isfinite(variances) & (variances > 1e-10)]
        fallback = float(np.median(finite_variances)) if finite_variances.size else 0.0004
        variances = np.where(np.isfinite(variances) & (variances > 1e-10), variances, fallback)
        covariance = np.where(np.isfinite(covariance), covariance, 0.0)
        np.fill_diagonal(covariance, variances)
        diagonal = np.diag(np.diag(covariance))
        shrunk = (1.0 - float(shrinkage)) * covariance + float(shrinkage) * diagonal
        ridge = max(float(np.trace(shrunk)) / max(1, len(stocks)), fallback) * 1e-4
        return shrunk + np.eye(len(stocks), dtype=np.float64) * ridge

    def stress(self, date: str | pd.Timestamp) -> float:
        target = pd.Timestamp(date).normalize()
        rows = self.market_state.loc[self.market_state.index <= target]
        if rows.empty:
            return 0.5
        value = float(rows.iloc[-1]["market_stress"])
        return value if math.isfinite(value) else 0.5

    def state_row(self, date: str | pd.Timestamp) -> dict[str, float]:
        target = pd.Timestamp(date).normalize()
        rows = self.market_state.loc[self.market_state.index <= target]
        if rows.empty:
            return {"market_stress": 0.5}
        row = rows.iloc[-1]
        keys = (
            "market_return_20",
            "market_volatility_20",
            "market_breadth_20",
            "market_dispersion_20",
            "market_stress",
        )
        return {
            key: float(row[key]) if pd.notna(row[key]) and math.isfinite(float(row[key])) else 0.0
            for key in keys
        }


def project_invested_weights(
    values: Sequence[float],
    *,
    lower: float = MIN_INVESTED_WEIGHT,
    upper: float = MAX_INVESTED_WEIGHT,
    hhi_cap: float = MAX_INVESTED_HHI,
) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (5,) or not np.isfinite(raw).all():
        raise ValueError("five finite raw weights are required")
    if 5 * lower > 1.0 + 1e-12 or 5 * upper < 1.0 - 1e-12:
        raise ValueError("infeasible five-stock weight bounds")
    raw = np.maximum(raw, 0.0)
    if float(raw.sum()) <= 1e-12:
        raw = np.ones(5, dtype=np.float64)
    raw /= float(raw.sum())
    low = float(np.min(raw - upper) - 1.0)
    high = float(np.max(raw - lower) + 1.0)
    for _ in range(120):
        midpoint = 0.5 * (low + high)
        projected = np.clip(raw - midpoint, lower, upper)
        if projected.sum() > 1.0:
            low = midpoint
        else:
            high = midpoint
    weights = np.clip(raw - high, lower, upper)
    weights += (1.0 - float(weights.sum())) / len(weights)
    equal = np.full(5, 0.2, dtype=np.float64)
    hhi = float(np.square(weights).sum())
    if hhi > hhi_cap + 1e-14:
        shrink = math.sqrt((hhi_cap - 0.2) / (hhi - 0.2))
        weights = equal + shrink * (weights - equal)
    weights += (1.0 - float(weights.sum())) / len(weights)
    assert_position_contract(weights, total_exposure=1.0)
    return weights


def assert_position_contract(invested_weights: Sequence[float], *, total_exposure: float) -> None:
    fractions = np.asarray(invested_weights, dtype=np.float64)
    if fractions.shape != (5,) or not np.isfinite(fractions).all():
        raise AssertionError("exactly five finite invested fractions are required")
    if not np.isclose(float(fractions.sum()), 1.0, atol=1e-10):
        raise AssertionError("invested fractions must sum to one")
    if float(fractions.min()) < MIN_INVESTED_WEIGHT - 1e-10:
        raise AssertionError("minimum invested fraction failed")
    if float(fractions.max()) > MAX_INVESTED_WEIGHT + 1e-10:
        raise AssertionError("maximum invested fraction failed")
    if float(np.square(fractions).sum()) > MAX_INVESTED_HHI + 1e-10:
        raise AssertionError("invested HHI cap failed")
    if not 0.50 - 1e-10 <= float(total_exposure) <= 1.0 + 1e-10:
        raise AssertionError("total stock exposure must be in [0.50, 1.00]")


def _minimum_variance_prior(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (5, 5) or not np.isfinite(matrix).all():
        return np.full(5, 0.2, dtype=np.float64)
    inverse = np.linalg.pinv(matrix, hermitian=True)
    raw = inverse @ np.ones(5, dtype=np.float64)
    raw = np.maximum(raw, 0.0)
    if float(raw.sum()) <= 1e-12:
        return np.full(5, 0.2, dtype=np.float64)
    return raw / float(raw.sum())


def candidate_grid() -> list[dict[str, Any]]:
    return [
        {
            "id": "R3_CONTROL_TOP5_EQUAL",
            "family": "top5_control",
            "selection_mode": "top5",
            "weight_mode": "equal",
            "exposure_mode": "full",
        },
        {
            "id": "R3_DOWNSIDE_SATELLITES_FIXED",
            "family": "downside_satellites",
            "selection_mode": "downside_satellites",
            "weight_mode": "core_satellite_fixed",
            "exposure_mode": "full",
        },
        {
            "id": "R3_DOWNSIDE_SATELLITES_RISK",
            "family": "downside_satellites",
            "selection_mode": "downside_satellites",
            "weight_mode": "core_satellite_downside",
            "exposure_mode": "full",
        },
        {
            "id": "R3_SIGN_RELIABILITY_EQUAL",
            "family": "sign_reliability",
            "selection_mode": "sign_reliability",
            "weight_mode": "equal",
            "exposure_mode": "full",
        },
        {
            "id": "R3_SIGN_RELIABILITY_WEIGHTED",
            "family": "sign_reliability",
            "selection_mode": "sign_reliability",
            "weight_mode": "sign_reliability",
            "exposure_mode": "full",
        },
        {
            "id": "R3_WORST_BLOCK_EQUAL",
            "family": "worst_block",
            "selection_mode": "worst_block",
            "weight_mode": "equal",
            "exposure_mode": "full",
        },
        {
            "id": "R3_WORST_BLOCK_WEIGHTED",
            "family": "worst_block",
            "selection_mode": "worst_block",
            "weight_mode": "worst_block",
            "exposure_mode": "full",
        },
        {
            "id": "R3_TOP5_EMPIRICAL_CVAR05",
            "family": "empirical_cvar",
            "selection_mode": "top5",
            "weight_mode": "empirical_cvar",
            "tail_fraction": 0.05,
            "exposure_mode": "full",
        },
        {
            "id": "R3_TOP5_EMPIRICAL_CVAR10",
            "family": "empirical_cvar",
            "selection_mode": "top5",
            "weight_mode": "empirical_cvar",
            "tail_fraction": 0.10,
            "exposure_mode": "full",
        },
    ]


def _exposure(config: Mapping[str, Any], stress: float, boundary_gap: float) -> float:
    if config.get("exposure_mode", "full") == "full":
        return 1.0
    middle = float(config.get("stress_middle", 0.55))
    high = float(config.get("stress_high", 0.75))
    if stress >= high:
        exposure = float(config.get("q_floor", 0.55))
    elif stress >= middle:
        exposure = float(config.get("q_middle", 0.80))
    else:
        exposure = 1.0
    if config.get("exposure_mode") == "stress_confidence" and boundary_gap < 0.05:
        exposure = max(float(config.get("q_floor", 0.55)), exposure - 0.10)
    return float(np.clip(exposure, 0.50, 1.00))


def _invested_weights(
    config: Mapping[str, Any],
    risk_data: CausalRiskData,
    decision_date: str,
    stocks: Sequence[str],
    selection_signal: np.ndarray | None = None,
) -> np.ndarray:
    equal = np.full(5, 0.2, dtype=np.float64)
    mode = str(config.get("weight_mode", "equal"))
    if mode == "equal":
        return equal
    downside = risk_data.downside_risk(decision_date, stocks)
    downside_prior = project_invested_weights(1.0 / np.maximum(downside, 1e-8))
    if mode == "core_satellite_fixed":
        return np.asarray([0.25, 0.25, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)
    if mode == "core_satellite_downside":
        core = np.asarray([0.25, 0.25], dtype=np.float64)
        satellites = downside_prior[2:]
        satellites = satellites / max(float(satellites.sum()), 1e-12) * 0.50
        return project_invested_weights(np.concatenate([core, satellites]))
    if mode in {"sign_reliability", "worst_block"}:
        signal = np.asarray(selection_signal, dtype=np.float64)
        if signal.shape != (5,):
            return equal
        centered = signal - float(np.nanmedian(signal))
        scale = float(np.nanmedian(np.abs(centered))) * 1.4826 + 1e-12
        logits = np.clip(centered / scale, -2.0, 2.0)
        raw = np.exp(logits - float(np.max(logits)))
        return project_invested_weights(raw)
    if mode == "empirical_cvar":
        history = risk_data.history(decision_date, stocks, window=120).dropna(how="all").fillna(0.0)
        scenarios = history.to_numpy(dtype=np.float64)
        weights = np.full(5, 0.2, dtype=np.float64)
        tail_fraction = float(config.get("tail_fraction", 0.05))
        if len(scenarios) < 40:
            return weights
        tail_count = max(1, int(math.ceil(tail_fraction * len(scenarios))))
        for iteration in range(240):
            portfolio = scenarios @ weights
            tail = np.argsort(portfolio)[:tail_count]
            gradient = -scenarios[tail].mean(axis=0)
            step = 0.20 / math.sqrt(iteration + 1.0)
            weights = project_invested_weights(weights - step * gradient)
        return weights
    raise ValueError(f"unknown R3 weight mode: {mode}")


def _select_five(
    ranked: pd.DataFrame,
    config: Mapping[str, Any],
    risk_data: CausalRiskData,
    decision_date: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    mode = str(config.get("selection_mode", "top5"))
    pool = ranked.head(10).copy()
    if len(pool) < 10:
        raise ValueError(f"{decision_date} has fewer than ten ranked candidates")
    if mode == "top5":
        selected = pool.head(5).copy()
        signal = np.zeros(5, dtype=np.float64)
    elif mode == "downside_satellites":
        core = pool.head(2).copy()
        satellite_pool = pool.iloc[2:].copy()
        values = risk_data.downside_capture_score(
            decision_date, satellite_pool["stock_id"].astype(str).tolist()
        )
        satellite_pool["selection_signal"] = values
        satellites = satellite_pool.sort_values(
            ["selection_signal", "score", "stock_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(3)
        selected = pd.concat([core, satellites], axis=0)
        signal = risk_data.downside_capture_score(
            decision_date, selected["stock_id"].astype(str).tolist()
        )
    elif mode == "sign_reliability":
        core = pool.head(1).copy()
        candidates = pool.iloc[1:].copy()
        values = risk_data.sign_reliability(
            decision_date, candidates["stock_id"].astype(str).tolist()
        )
        candidates["selection_signal"] = values
        selected = pd.concat(
            [
                core,
                candidates.sort_values(
                    ["selection_signal", "score", "stock_id"],
                    ascending=[False, False, True],
                    kind="mergesort",
                ).head(4),
            ],
            axis=0,
        )
        signal = risk_data.sign_reliability(
            decision_date, selected["stock_id"].astype(str).tolist()
        )
    elif mode == "worst_block":
        core = pool.head(1).copy()
        candidates = pool.iloc[1:].copy()
        values = risk_data.worst_block_residual_strength(
            decision_date, candidates["stock_id"].astype(str).tolist()
        )
        candidates["selection_signal"] = values
        selected = pd.concat(
            [
                core,
                candidates.sort_values(
                    ["selection_signal", "score", "stock_id"],
                    ascending=[False, False, True],
                    kind="mergesort",
                ).head(4),
            ],
            axis=0,
        )
        signal = risk_data.worst_block_residual_strength(
            decision_date, selected["stock_id"].astype(str).tolist()
        )
    else:
        raise ValueError(f"unknown R3 selection mode: {mode}")
    if len(selected) != 5 or selected["stock_id"].nunique() != 5:
        raise AssertionError(f"{decision_date} did not produce five unique stocks")
    return selected, np.asarray(signal, dtype=np.float64)


def attach_candidate_weights(
    predictions: pd.DataFrame,
    config: Mapping[str, Any],
    risk_data: CausalRiskData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"sample_end_date", "stock_id", "score", "label"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction columns missing: {sorted(missing)}")
    frame = predictions.copy().reset_index(drop=True)
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
    frame["stock_id"] = frame["stock_id"].map(normalize_stock_code)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["portfolio_weight"] = 0.0
    frame["portfolio_selected"] = False
    frame["position_policy"] = str(config["id"])
    decisions: list[dict[str, Any]] = []
    for date, day in frame.groupby("sample_end_date", sort=True):
        ranked = day.dropna(subset=["score"]).sort_values(
            ["score", "stock_id"], ascending=[False, True], kind="mergesort"
        )
        if len(ranked) < 6:
            raise ValueError(f"{date} has fewer than six valid ranking scores")
        selected, selection_signal = _select_five(ranked, config, risk_data, date)
        stocks = selected["stock_id"].astype(str).tolist()
        fractions = _invested_weights(config, risk_data, date, stocks, selection_signal)
        score_scale = float(np.subtract(*np.percentile(ranked["score"].to_numpy(dtype=float), [75, 25])))
        boundary_gap = float((ranked.iloc[4]["score"] - ranked.iloc[5]["score"]) / (score_scale + 1e-12))
        state = risk_data.state_row(date)
        total_exposure = _exposure(config, state.get("market_stress", 0.5), boundary_gap)
        assert_position_contract(fractions, total_exposure=total_exposure)
        actual = total_exposure * fractions
        frame.loc[selected.index, "portfolio_weight"] = actual
        frame.loc[selected.index, "portfolio_selected"] = True
        decisions.append(
            {
                "decision_date": date,
                "strategy": str(config["id"]),
                "selected_stocks": ",".join(stocks),
                "selected_count": 5,
                "weight_sum": total_exposure,
                "cash_ratio": 1.0 - total_exposure,
                "invested_hhi": float(np.square(fractions).sum()),
                "max_invested_weight": float(fractions.max()),
                "max_actual_weight": float(actual.max()),
                "boundary_score_gap_robust": boundary_gap,
                "selection_mode": str(config.get("selection_mode", "top5")),
                "selection_signal": ",".join(f"{value:.17g}" for value in selection_signal),
                "weights": ",".join(f"{value:.17g}" for value in actual),
                **state,
            }
        )
    return frame, pd.DataFrame(decisions)


def attach_top2_control(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.copy().reset_index(drop=True)
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
    frame["stock_id"] = frame["stock_id"].map(normalize_stock_code)
    frame["portfolio_weight"] = 0.0
    frame["portfolio_selected"] = False
    frame["position_policy"] = "R3_DP0_TOP2_EQUAL"
    records: list[dict[str, Any]] = []
    for date, day in frame.groupby("sample_end_date", sort=True):
        selected = day.sort_values(["score", "stock_id"], ascending=[False, True], kind="mergesort").head(2)
        frame.loc[selected.index, "portfolio_weight"] = 0.5
        frame.loc[selected.index, "portfolio_selected"] = True
        records.append(
            {
                "decision_date": date,
                "strategy": "R3_DP0_TOP2_EQUAL",
                "selected_stocks": ",".join(selected["stock_id"].astype(str)),
                "selected_count": 2,
                "weight_sum": 1.0,
                "cash_ratio": 0.0,
                "invested_hhi": 0.5,
                "max_invested_weight": 0.5,
                "max_actual_weight": 0.5,
            }
        )
    return frame, pd.DataFrame(records)


def weekly_metrics(weekly: pd.DataFrame, decisions: pd.DataFrame | None = None) -> dict[str, Any]:
    scored = weekly.copy()
    if "skipped_missing_decision_prediction" in scored:
        scored = scored[scored["skipped_missing_decision_prediction"].astype(int) == 0].copy()
    scored = scored.sort_values([column for column in ("fold_id", "week_start") if column in scored.columns])
    net = scored["weekly_return_net"].astype(float).to_numpy(dtype=np.float64)
    gross = scored["weekly_return_gross"].astype(float).to_numpy(dtype=np.float64)
    if net.size == 0:
        raise ValueError("weekly metrics require scored weeks")
    nav = np.concatenate([[1.0], np.cumprod(1.0 + net)])
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    tail_count = max(1, int(math.ceil(0.05 * len(net))))
    downside = net[net < 0.0]
    best_indices = np.argsort(net)[::-1][: min(2, len(net))]
    remaining = np.delete(net, best_indices)
    positive_sum = float(net[net > 0.0].sum())
    best_positive = float(np.maximum(net[best_indices], 0.0).sum())
    worst4 = min(
        (float(np.prod(1.0 + net[index : index + 4]) - 1.0) for index in range(max(1, len(net) - 3))),
        default=float(net.min()),
    )
    output: dict[str, Any] = {
        "week_count": int(len(net)),
        "mean_weekly_return_net": float(net.mean()),
        "median_weekly_return_net": float(np.median(net)),
        "cumulative_return_net": float(np.prod(1.0 + net) - 1.0),
        "positive_week_rate": float(np.mean(net > 0.0)),
        "max_drawdown_abs": float(abs(drawdown.min())),
        "worst_week": float(net.min()),
        "worst_4_week_cumulative_return": worst4,
        "CVaR_95_loss_abs": float(max(0.0, np.sort(-net)[-tail_count:].mean())),
        "downside_deviation": float(np.std(downside, ddof=0)) if downside.size else 0.0,
        "cost_drag": float(np.sum(gross - net)),
        "cumulative_without_best_2_weeks": float(np.prod(1.0 + remaining) - 1.0) if remaining.size else 0.0,
        "best_2_weeks_contribution_ratio": safe_div(best_positive, positive_sum) if positive_sum > 0.0 else math.inf,
    }
    output["Sortino"] = safe_div(output["mean_weekly_return_net"], output["downside_deviation"])
    output["Calmar"] = safe_div(output["cumulative_return_net"], output["max_drawdown_abs"])
    if decisions is not None and not decisions.empty:
        output.update(
            {
                "average_weight_sum": float(decisions["weight_sum"].astype(float).mean()),
                "average_cash_ratio": float(decisions["cash_ratio"].astype(float).mean()),
                "mean_invested_hhi": float(decisions["invested_hhi"].astype(float).mean()),
                "max_invested_weight": float(decisions["max_invested_weight"].astype(float).max()),
                "max_actual_weight": float(decisions["max_actual_weight"].astype(float).max()),
                "mean_market_stress": (
                    float(decisions["market_stress"].astype(float).mean()) if "market_stress" in decisions else 0.0
                ),
            }
        )
    return output


def paired_weekly(candidate: pd.DataFrame, control: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = [column for column in ("fold_id", "week_start") if column in candidate.columns and column in control.columns]
    if "week_start" not in keys:
        keys = ["week_start"]
    merged = candidate.merge(
        control[keys + ["weekly_return_net"]].rename(columns={"weekly_return_net": "control_weekly_return_net"}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    merged["weekly_excess"] = merged["weekly_return_net"] - merged["control_weekly_return_net"]
    diff = merged["weekly_excess"].to_numpy(dtype=np.float64)
    best = np.sort(diff)[::-1][: min(2, len(diff))]
    return merged, {
        "week_count": int(len(diff)),
        "mean_weekly_excess": float(diff.mean()),
        "median_weekly_excess": float(np.median(diff)),
        "positive_diff_rate": float(np.mean(diff > 0.0)),
        "diff_without_best_2_weeks": float(diff.sum() - best.sum()),
        "excess_sum": float(diff.sum()),
    }


def moving_block_probability(
    values: Sequence[float],
    *,
    block_size: int = 4,
    draws: int = 5000,
    seed: int = 20260716,
) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("bootstrap requires a non-empty vector")
    generator = np.random.default_rng(seed)
    starts = np.arange(data.size)
    blocks = int(math.ceil(data.size / block_size))
    positive = 0
    for _ in range(draws):
        sampled_starts = generator.choice(starts, size=blocks, replace=True)
        sample = np.concatenate(
            [data[(start + np.arange(block_size)) % data.size] for start in sampled_starts]
        )[: data.size]
        positive += int(float(sample.mean()) > 0.0)
    return float(positive / draws)


def weighted_turnover(decisions: pd.DataFrame) -> float:
    if decisions.empty:
        return 0.0
    previous: dict[str, float] | None = None
    values: list[float] = []
    for row in decisions.sort_values("decision_date").itertuples(index=False):
        stocks = str(row.selected_stocks).split(",")
        weights = [float(value) for value in str(row.weights).split(",")] if hasattr(row, "weights") else []
        current = dict(zip(stocks, weights))
        if previous is not None:
            universe = set(previous) | set(current)
            values.append(0.5 * sum(abs(current.get(stock, 0.0) - previous.get(stock, 0.0)) for stock in universe))
        previous = current
    return float(np.mean(values)) if values else 0.0


def fold_ids_for_decisions(predictions: pd.DataFrame) -> dict[str, str]:
    if "fold_id" not in predictions:
        return {}
    frame = predictions[["sample_end_date", "fold_id"]].copy()
    frame["sample_end_date"] = pd.to_datetime(frame["sample_end_date"]).dt.strftime("%Y-%m-%d")
    return {
        str(date): str(group["fold_id"].iloc[0])
        for date, group in frame.groupby("sample_end_date", sort=True)
    }


def assign_fold_ids(weekly: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    result = weekly.copy()
    if mapping:
        result["fold_id"] = result["decision_date"].astype(str).map(mapping).fillna("unknown")
    return result


def evaluate_explicit_weeks(
    weighted_predictions: pd.DataFrame,
    weeks: Sequence[Mapping[str, Any]],
    market_path: str | Path,
    *,
    fold_id: str | None = None,
    buy_fee: float = 0.0005,
    sell_fee: float = 0.0005,
    stamp_tax: float = 0.0005,
) -> pd.DataFrame:
    market, columns = read_market_csv(market_path)
    open_by_key: dict[tuple[str, pd.Timestamp], float] = {}
    for stock, date, value in market[[columns.stock, columns.date, columns.open]].itertuples(index=False, name=None):
        number = float(value)
        if math.isfinite(number) and number > 0.0:
            open_by_key[(normalize_stock_code(stock), pd.Timestamp(date).normalize())] = number
    prediction = weighted_predictions.copy()
    prediction["sample_end_date"] = pd.to_datetime(prediction["sample_end_date"]).dt.strftime("%Y-%m-%d")
    prediction["stock_id"] = prediction["stock_id"].map(normalize_stock_code)
    prediction["portfolio_weight"] = pd.to_numeric(prediction["portfolio_weight"], errors="coerce").fillna(0.0)
    by_date = {date: group for date, group in prediction.groupby("sample_end_date", sort=True)}
    rows: list[dict[str, Any]] = []
    previous: dict[str, float] | None = None
    for week in weeks:
        decision = str(week["decision_date"])
        buy = pd.Timestamp(week["buy_date"]).normalize()
        sell = pd.Timestamp(week["sell_date"]).normalize()
        if decision not in by_date:
            raise AssertionError(f"missing decision prediction: {decision}")
        day = by_date[decision]
        day = day[day["portfolio_weight"] > 1e-12].sort_values(
            ["portfolio_weight", "score", "stock_id"], ascending=[False, False, True]
        )
        stocks = day["stock_id"].astype(str).tolist()
        weights = day["portfolio_weight"].to_numpy(dtype=np.float64)
        if not stocks:
            raise AssertionError(f"empty portfolio on {decision}")
        returns = []
        for stock in stocks:
            start = open_by_key.get((stock, buy))
            end = open_by_key.get((stock, sell))
            if start is None or end is None or start <= 0.0:
                raise AssertionError(f"missing open price for {stock} on {buy.date()}/{sell.date()}")
            returns.append(end / start - 1.0)
        stock_returns = np.asarray(returns, dtype=np.float64)
        gross = float(np.dot(weights, stock_returns))
        invested = float(weights.sum())
        sell_notional = float(np.dot(weights, 1.0 + stock_returns))
        cost = buy_fee * invested + (sell_fee + stamp_tax) * sell_notional
        current = dict(zip(stocks, weights))
        if previous is None:
            turnover = 0.0
        else:
            universe = set(previous) | set(current)
            turnover = 0.5 * sum(abs(current.get(stock, 0.0) - previous.get(stock, 0.0)) for stock in universe)
        previous = current
        invested_fractions = weights / max(invested, 1e-12)
        rows.append(
            {
                "fold_id": fold_id or str(week.get("fold_id", "unknown")),
                "week_start": buy.strftime("%Y-%m-%d"),
                "buy_date": buy.strftime("%Y-%m-%d"),
                "sell_date": sell.strftime("%Y-%m-%d"),
                "decision_date": decision,
                "weekly_return_gross": gross,
                "weekly_return_net": gross - cost,
                "cost": cost,
                "weight_sum": invested,
                "cash_ratio": 1.0 - invested,
                "selected_count": len(stocks),
                "selected_stocks": ",".join(stocks),
                "max_actual_weight": float(weights.max()),
                "invested_hhi": float(np.square(invested_fractions).sum()),
                "turnover": turnover,
                "boundary_week": int(week.get("trading_day_count", 5)) != 5,
                "skipped_missing_decision_prediction": 0,
            }
        )
    return pd.DataFrame(rows)


def official_proxy_summary(weighted_predictions: pd.DataFrame) -> dict[str, Any]:
    frame = weighted_predictions.copy()
    frame["portfolio_weight"] = pd.to_numeric(frame["portfolio_weight"], errors="coerce").fillna(0.0)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame["weighted_return"] = frame["portfolio_weight"] * frame["label"]
    daily = frame.groupby("sample_end_date", as_index=False).agg(
        portfolio_return=("weighted_return", "sum"),
        weight_sum=("portfolio_weight", "sum"),
        selected_count=("portfolio_weight", lambda values: int((np.asarray(values, dtype=float) > 1e-12).sum())),
    )
    return {
        "sample_count": int(len(daily)),
        "date_start": str(daily["sample_end_date"].min()),
        "date_end": str(daily["sample_end_date"].max()),
        "official_proxy_final_score": float(daily["portfolio_return"].mean()),
        "mean_portfolio_return": float(daily["portfolio_return"].mean()),
        "median_portfolio_return": float(daily["portfolio_return"].median()),
        "average_weight_sum": float(daily["weight_sum"].mean()),
        "average_selected_count": float(daily["selected_count"].mean()),
    }


def decision_config_hash(configs: Iterable[Mapping[str, Any]]) -> str:
    return canonical_hash(sorted((dict(config) for config in configs), key=lambda item: str(item["id"])))

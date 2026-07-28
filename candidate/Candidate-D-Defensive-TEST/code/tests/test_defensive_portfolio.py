from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from defensive_portfolio import build_defensive_result  # noqa: E402
from featurework import CLOSE_COL, DATE_COL, STOCK_COL  # noqa: E402


SETTINGS = {
    "policy_id": "EQ_V0.050_C0.150_T0.000_R0.100",
    "correlation_cluster_cut": 0.65,
    "correlation_lookback": 60,
    "correlation_min_observations": 45,
    "correlation_shrinkage": 0.25,
    "correlation_missing_value": 0.65,
    "pool_steps": [20, 50, 100, None],
    "max_cluster_holdings": 2,
    "lambda_volatility": 0.05,
    "lambda_correlation": 0.15,
    "lambda_turnover": 0.0,
    "lambda_reversal": 0.10,
    "regime_multiplier": 1.0,
    "weights": [0.2] * 5,
}


def synthetic_market() -> tuple[pd.DataFrame, pd.Timestamp, list[str], np.ndarray]:
    rng = np.random.default_rng(20260728)
    dates = pd.bdate_range("2025-01-02", periods=125)
    stocks = [f"{index:06d}" for index in range(1, 26)]
    rows: list[dict[str, object]] = []
    common = rng.normal(0.0002, 0.004, len(dates))
    for stock_index, stock in enumerate(stocks):
        noise = rng.normal(0.0, 0.012 + stock_index * 0.0001, len(dates))
        close = 20.0 * np.exp(np.cumsum(common * 0.2 + noise))
        for date, value in zip(dates, close, strict=True):
            rows.append(
                {
                    STOCK_COL: stock,
                    DATE_COL: date,
                    CLOSE_COL: value,
                }
            )
    scores = np.linspace(1.0, 0.0, len(stocks))
    return pd.DataFrame(rows), dates[-1], stocks, scores


class DefensivePortfolioTest(unittest.TestCase):
    def test_constraints_and_determinism(self) -> None:
        raw, decision_date, stocks, scores = synthetic_market()
        first, first_diag = build_defensive_result(
            raw=raw,
            decision_date=decision_date,
            stock_codes=stocks,
            scores=scores,
            settings=SETTINGS,
        )
        shuffled = raw.sample(frac=1.0, random_state=7).reset_index(drop=True)
        second, second_diag = build_defensive_result(
            raw=shuffled,
            decision_date=decision_date,
            stock_codes=list(reversed(stocks)),
            scores=np.asarray(list(reversed(scores))),
            settings=SETTINGS,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(first["stock_id"].nunique(), 5)
        self.assertAlmostEqual(float(first["weight"].sum()), 1.0, places=12)
        self.assertLessEqual(float(first["weight"].max()), 0.30)
        self.assertLessEqual(float(first_diag["hhi"]), 0.30)
        counts = pd.Series(
            [item["cluster_id"] for item in first_diag["selected"]]
        ).value_counts()
        self.assertLessEqual(int(counts.max()), 2)
        self.assertEqual(first_diag["selected"], second_diag["selected"])

    def test_future_append_cannot_change_decision(self) -> None:
        raw, decision_date, stocks, scores = synthetic_market()
        base, base_diag = build_defensive_result(
            raw=raw,
            decision_date=decision_date,
            stock_codes=stocks,
            scores=scores,
            settings=SETTINGS,
        )
        future = raw.copy()
        future[DATE_COL] = pd.to_datetime(future[DATE_COL]) + pd.Timedelta(days=365)
        future[CLOSE_COL] = pd.to_numeric(future[CLOSE_COL]) * 100.0
        appended, appended_diag = build_defensive_result(
            raw=pd.concat([raw, future], ignore_index=True),
            decision_date=decision_date,
            stock_codes=stocks,
            scores=scores,
            settings=SETTINGS,
        )
        pd.testing.assert_frame_equal(base, appended)
        self.assertEqual(base_diag["selected"], appended_diag["selected"])


if __name__ == "__main__":
    unittest.main()

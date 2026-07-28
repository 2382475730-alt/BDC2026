from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from Evaluation.scripts.d_series.defensive_position_r3 import (
    CausalRiskData,
    assert_position_contract,
    attach_candidate_weights,
    candidate_grid,
    evaluate_explicit_weeks,
    project_invested_weights,
)


class DefensivePositionR3Tests(unittest.TestCase):
    def test_candidate_grid_is_unique_and_preregistered_size(self) -> None:
        configs = candidate_grid()
        identifiers = [config["id"] for config in configs]
        self.assertEqual(len(configs), 9)
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(identifiers[0], "R3_CONTROL_TOP5_EQUAL")

    def test_projection_obeys_five_stock_contract(self) -> None:
        weights = project_invested_weights([10.0, 1.0, 0.2, 0.1, 0.01])
        assert_position_contract(weights, total_exposure=0.65)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertLessEqual(float(weights.max()), 0.30 + 1e-10)
        self.assertLessEqual(float(np.square(weights).sum()), 0.24 + 1e-10)

    @staticmethod
    def _market_frame() -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-02", periods=330)
        rows = []
        for stock_index in range(10):
            stock = f"{stock_index + 1:06d}"
            phase = stock_index * 0.4
            returns = 0.0002 + 0.008 * np.sin(np.arange(len(dates)) / 11.0 + phase)
            close = 10.0 * np.cumprod(1.0 + returns)
            for date, value in zip(dates, close):
                rows.append(
                    {
                        "股票代码": stock,
                        "日期": date.strftime("%Y-%m-%d"),
                        "开盘": value * 0.999,
                        "收盘": value,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _predictions(date: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_end_date": [date] * 10,
                "stock_id": [f"{index + 1:06d}" for index in range(10)],
                "score": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                "label": [0.02, 0.01, 0.0, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02, 0.0],
            }
        )

    def test_candidate_has_exactly_five_full_exposure_positions(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "market.csv"
            market = self._market_frame()
            market.to_csv(path, index=False, encoding="utf-8-sig")
            decision = str(pd.to_datetime(market["日期"]).max().date())
            risk = CausalRiskData.from_market_path(path)
            config = next(item for item in candidate_grid() if item["id"] == "R3_DOWNSIDE_SATELLITES_RISK")
            weighted, decisions = attach_candidate_weights(self._predictions(decision), config, risk)
            selected = weighted[weighted["portfolio_weight"] > 1e-12]
            self.assertEqual(len(selected), 5)
            self.assertEqual(selected["stock_id"].nunique(), 5)
            self.assertAlmostEqual(float(selected["portfolio_weight"].sum()), 1.0, places=12)
            self.assertEqual(int(decisions.iloc[0]["selected_count"]), 5)

    def test_future_append_does_not_change_historical_weights(self) -> None:
        with TemporaryDirectory() as temporary:
            full_path = Path(temporary) / "full.csv"
            truncated_path = Path(temporary) / "truncated.csv"
            market = self._market_frame()
            all_dates = sorted(pd.to_datetime(market["日期"]).unique())
            decision = pd.Timestamp(all_dates[-31]).strftime("%Y-%m-%d")
            market.to_csv(full_path, index=False, encoding="utf-8-sig")
            market[pd.to_datetime(market["日期"]) <= pd.Timestamp(decision)].to_csv(
                truncated_path, index=False, encoding="utf-8-sig"
            )
            full = CausalRiskData.from_market_path(full_path)
            truncated = CausalRiskData.from_market_path(truncated_path)
            config = next(item for item in candidate_grid() if item["id"] == "R3_DOWNSIDE_SATELLITES_RISK")
            left, _ = attach_candidate_weights(self._predictions(decision), config, full)
            right, _ = attach_candidate_weights(self._predictions(decision), config, truncated)
            np.testing.assert_allclose(
                left["portfolio_weight"].to_numpy(dtype=float),
                right["portfolio_weight"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )

    def test_missing_price_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "market.csv"
            market = self._market_frame()
            market = market[~((market["股票代码"] == "000001") & (market["日期"] == market["日期"].max()))]
            market.to_csv(path, index=False, encoding="utf-8-sig")
            dates = sorted(pd.to_datetime(market["日期"]).unique())
            decision = pd.Timestamp(dates[-2]).strftime("%Y-%m-%d")
            buy = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
            weighted = self._predictions(decision)
            weighted["portfolio_weight"] = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
            with self.assertRaises(AssertionError):
                evaluate_explicit_weeks(
                    weighted,
                    [{"decision_date": decision, "buy_date": buy, "sell_date": buy, "trading_day_count": 1}],
                    path,
                )


if __name__ == "__main__":
    unittest.main()

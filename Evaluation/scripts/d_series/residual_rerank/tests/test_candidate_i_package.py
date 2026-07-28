from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[5]
CANDIDATE_SRC = REPO_ROOT / "candidate" / "Candidate-I" / "code" / "src"
if str(CANDIDATE_SRC) not in sys.path:
    sys.path.insert(0, str(CANDIDATE_SRC))

from featurework import (  # noqa: E402
    AMOUNT_COL,
    CLOSE_COL,
    DATE_COL,
    HIGH_COL,
    LOW_COL,
    OPEN_COL,
    STOCK_COL,
    VOLUME_COL,
)
from residual_rerank import (  # noqa: E402
    build_lag1_residual_factors,
    build_residual_rerank_portfolio,
)


class CandidateIPackageTests(unittest.TestCase):
    def market_frame(self) -> pd.DataFrame:
        rows = []
        dates = pd.bdate_range("2025-01-02", periods=70)
        for stock_index in range(12):
            for date_index, date in enumerate(dates):
                close = 10.0 + stock_index + 0.03 * date_index
                rows.append(
                    {
                        STOCK_COL: f"{stock_index + 1:06d}",
                        DATE_COL: date,
                        OPEN_COL: close * 0.995,
                        CLOSE_COL: close,
                        HIGH_COL: close * 1.01,
                        LOW_COL: close * 0.99,
                        VOLUME_COL: 1_000_000 + 100 * date_index + stock_index,
                        AMOUNT_COL: close
                        * (1_000_000 + 100 * date_index + stock_index),
                    }
                )
        return pd.DataFrame(rows)

    def test_lag1_factors_ignore_rows_on_and_after_decision_date(self) -> None:
        raw = self.market_frame()
        decision = pd.Timestamp("2025-04-01")
        expected = build_lag1_residual_factors(raw, decision)
        distorted = raw.copy()
        mask = pd.to_datetime(distorted[DATE_COL]) >= decision
        distorted.loc[mask, CLOSE_COL] = 1_000_000.0
        observed = build_lag1_residual_factors(distorted, decision)
        columns = ["momentum_rank", "volume_rank", "reversal_rank"]
        np.testing.assert_allclose(
            expected[columns].to_numpy(),
            observed[columns].to_numpy(),
            atol=0.0,
            rtol=0.0,
        )
        self.assertTrue(
            (pd.to_datetime(observed["factor_asof_date"]) < decision).all()
        )

    def test_portfolio_has_five_equal_weights_and_hhi_point_two(self) -> None:
        raw = self.market_frame()
        result, diagnostics = build_residual_rerank_portfolio(
            raw=raw,
            decision_date="2025-04-01",
            stock_codes=[f"{value + 1:06d}" for value in range(12)],
            model_scores=np.arange(12, dtype=float),
            lambdas={
                "lambda_momentum": 0.05,
                "lambda_volume": 0.05,
                "lambda_reversal": 0.05,
            },
        )
        self.assertEqual(5, len(result))
        self.assertAlmostEqual(1.0, float(result["weight"].sum()))
        self.assertTrue((result["weight"] == 0.20).all())
        self.assertAlmostEqual(0.20, float(diagnostics["hhi"]))

    def test_lambda_outside_preregistered_grid_is_rejected(self) -> None:
        raw = self.market_frame()
        with self.assertRaisesRegex(ValueError, "outside the preregistered grid"):
            build_residual_rerank_portfolio(
                raw=raw,
                decision_date="2025-04-01",
                stock_codes=[f"{value + 1:06d}" for value in range(12)],
                model_scores=np.arange(12, dtype=float),
                lambdas={
                    "lambda_momentum": 0.07,
                    "lambda_volume": 0.05,
                    "lambda_reversal": 0.05,
                },
            )


if __name__ == "__main__":
    unittest.main()

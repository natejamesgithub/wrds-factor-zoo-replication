from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtest import (  # noqa: E402
    assign_quantiles,
    calculate_long_short_returns,
    calculate_portfolio_returns,
    read_table,
    run_quantile_backtest,
    write_table,
)


class BacktestTests(unittest.TestCase):
    def test_assign_quantiles_sorts_signal_within_each_month(self) -> None:
        panel = pd.DataFrame(
            {
                "month": ["2020-01-31"] * 5,
                "permno": [1, 2, 3, 4, 5],
                "ret_adj": [0.01, 0.02, 0.03, 0.04, 0.05],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        result = assign_quantiles(panel, signal="value", n_quantiles=5)

        self.assertEqual(result["quantile"].tolist(), [1, 2, 3, 4, 5])

    def test_assign_quantiles_drops_months_with_too_few_stocks(self) -> None:
        panel = pd.DataFrame(
            {
                "month": ["2020-01-31"] * 4,
                "permno": [1, 2, 3, 4],
                "ret_adj": [0.01, 0.02, 0.03, 0.04],
                "value": [10.0, 20.0, 30.0, 40.0],
            }
        )

        result = assign_quantiles(panel, signal="value", n_quantiles=5)

        self.assertTrue(result.empty)

    def test_calculate_equal_weight_portfolio_returns(self) -> None:
        quantile_panel = pd.DataFrame(
            {
                "month": ["2020-01-31", "2020-01-31", "2020-01-31"],
                "quantile": [1, 1, 2],
                "ret_adj": [0.02, 0.04, 0.10],
            }
        )

        result = calculate_portfolio_returns(quantile_panel, weighting="equal")

        q1 = result.loc[result["quantile"] == 1, "portfolio_return"].iloc[0]
        q2 = result.loc[result["quantile"] == 2, "portfolio_return"].iloc[0]
        self.assertAlmostEqual(q1, 0.03)
        self.assertAlmostEqual(q2, 0.10)

    def test_calculate_value_weight_portfolio_returns(self) -> None:
        quantile_panel = pd.DataFrame(
            {
                "month": ["2020-01-31", "2020-01-31"],
                "quantile": [1, 1],
                "ret_adj": [0.00, 0.10],
                "me_lag": [1.0, 3.0],
            }
        )

        result = calculate_portfolio_returns(quantile_panel, weighting="value")

        self.assertAlmostEqual(result.iloc[0]["portfolio_return"], 0.075)

    def test_calculate_long_short_returns_uses_requested_direction(self) -> None:
        portfolio_returns = pd.DataFrame(
            {
                "month": ["2020-01-31", "2020-01-31"],
                "quantile": [1, 5],
                "portfolio_return": [0.01, 0.06],
            }
        )

        result = calculate_long_short_returns(portfolio_returns, n_quantiles=5)
        reversed_result = calculate_long_short_returns(
            portfolio_returns,
            n_quantiles=5,
            high_minus_low=False,
        )

        self.assertAlmostEqual(result.iloc[0]["long_short_return"], 0.05)
        self.assertAlmostEqual(reversed_result.iloc[0]["long_short_return"], -0.05)

    def test_run_quantile_backtest_returns_monthly_long_short_series(self) -> None:
        panel = pd.DataFrame(
            {
                "month": ["2020-01-31"] * 5 + ["2020-02-29"] * 5,
                "permno": [1, 2, 3, 4, 5] * 2,
                "ret_adj": [0.01, 0.02, 0.03, 0.04, 0.05, 0.02, 0.03, 0.04, 0.05, 0.06],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0] * 2,
            }
        )

        result = run_quantile_backtest(panel, signal="value", n_quantiles=5)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["signal"].tolist(), ["value", "value"])
        self.assertAlmostEqual(result.iloc[0]["long_short_return"], 0.04)
        self.assertAlmostEqual(result.iloc[1]["long_short_return"], 0.04)

    def test_csv_round_trip_helpers(self) -> None:
        df = pd.DataFrame({"gvkey": ["001234"], "cusip": ["000001"], "value": [1.0]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.csv"
            write_table(df, path)
            result = read_table(path)

        pd.testing.assert_frame_equal(result, df)

    def test_missing_signal_raises_helpful_error(self) -> None:
        panel = pd.DataFrame(
            {
                "month": ["2020-01-31"],
                "permno": [1],
                "ret_adj": [0.01],
            }
        )

        with self.assertRaisesRegex(ValueError, "Missing required factor panel columns"):
            assign_quantiles(panel, signal="value")


if __name__ == "__main__":
    unittest.main()
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analytics import (  # noqa: E402
    add_cumulative_performance,
    cumulative_returns,
    drawdown_series,
    prepare_returns,
    read_table,
    summarize_backtest,
    summarize_many,
    write_table,
)


class AnalyticsTests(unittest.TestCase):
    def test_prepare_returns_sorts_deduplicates_and_drops_missing_returns(self) -> None:
        backtest = pd.DataFrame(
            {
                "month": ["2020-02-29", "2020-01-31", "2020-01-31", "2020-03-31"],
                "long_short_return": [0.02, 0.01, 0.03, None],
                "signal": ["value", "value", "value", "value"],
            }
        )

        result = prepare_returns(backtest)

        self.assertEqual(result["month"].tolist(), [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")])
        self.assertEqual(result["long_short_return"].tolist(), [0.03, 0.02])

    def test_cumulative_returns_compounds_periodic_returns(self) -> None:
        result = cumulative_returns(pd.Series([0.10, -0.10, 0.20]))

        self.assertAlmostEqual(result.iloc[-1], (1.10 * 0.90 * 1.20) - 1)

    def test_drawdown_series_tracks_declines_from_prior_peak(self) -> None:
        result = drawdown_series(pd.Series([0.10, -0.20, 0.05]))

        self.assertAlmostEqual(result.iloc[0], 0.0)
        self.assertAlmostEqual(result.iloc[1], -0.20)
        self.assertAlmostEqual(result.iloc[2], -0.16)

    def test_summarize_backtest_returns_expected_metrics(self) -> None:
        backtest = pd.DataFrame(
            {
                "signal": ["value", "value", "value"],
                "month": ["2020-01-31", "2020-02-29", "2020-03-31"],
                "long_short_return": [0.10, -0.05, 0.02],
                "weighting": ["equal", "equal", "equal"],
                "n_quantiles": [5, 5, 5],
            }
        )

        summary = summarize_backtest(backtest)
        row = summary.iloc[0]
        returns = pd.Series([0.10, -0.05, 0.02])
        total_return = (1.10 * 0.95 * 1.02) - 1
        annualized_return = returns.mean() * 12
        annualized_volatility = returns.std(ddof=1) * np.sqrt(12)

        self.assertEqual(row["signal"], "value")
        self.assertEqual(row["months"], 3)
        self.assertEqual(row["start_month"], pd.Timestamp("2020-01-31"))
        self.assertEqual(row["end_month"], pd.Timestamp("2020-03-31"))
        self.assertAlmostEqual(row["total_return"], total_return)
        self.assertAlmostEqual(row["cagr"], (1 + total_return) ** 4 - 1)
        self.assertAlmostEqual(row["annualized_return"], annualized_return)
        self.assertAlmostEqual(row["annualized_volatility"], annualized_volatility)
        self.assertAlmostEqual(row["sharpe"], annualized_return / annualized_volatility)
        self.assertAlmostEqual(row["max_drawdown"], -0.05)
        self.assertAlmostEqual(row["hit_rate"], 2 / 3)
        self.assertEqual(row["best_month"], 0.10)
        self.assertEqual(row["worst_month"], -0.05)
        self.assertEqual(row["weighting"], "equal")
        self.assertEqual(row["n_quantiles"], 5)

    def test_add_cumulative_performance_adds_report_columns(self) -> None:
        backtest = pd.DataFrame(
            {
                "month": ["2020-01-31", "2020-02-29"],
                "long_short_return": [0.10, -0.10],
            }
        )

        result = add_cumulative_performance(backtest)

        self.assertIn("cumulative_return", result.columns)
        self.assertIn("drawdown", result.columns)
        self.assertAlmostEqual(result.iloc[-1]["cumulative_return"], -0.01)

    def test_summarize_many_combines_backtests(self) -> None:
        first = pd.DataFrame(
            {"signal": ["value"], "month": ["2020-01-31"], "long_short_return": [0.01]}
        )
        second = pd.DataFrame(
            {"signal": ["momentum"], "month": ["2020-01-31"], "long_short_return": [0.02]}
        )

        result = summarize_many([first, second])

        self.assertEqual(result["signal"].tolist(), ["value", "momentum"])

    def test_csv_round_trip_helpers(self) -> None:
        df = pd.DataFrame({"signal": ["value"], "long_short_return": [0.01]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.csv"
            write_table(df, path)
            result = read_table(path)

        pd.testing.assert_frame_equal(result, df)

    def test_empty_returns_raise_helpful_error(self) -> None:
        backtest = pd.DataFrame(
            {
                "month": ["2020-01-31"],
                "long_short_return": [None],
            }
        )

        with self.assertRaisesRegex(ValueError, "no valid return observations"):
            summarize_backtest(backtest)


if __name__ == "__main__":
    unittest.main()
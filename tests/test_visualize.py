from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize import create_backtest_charts, read_table  # noqa: E402


class VisualizeTests(unittest.TestCase):
    def test_create_backtest_charts_writes_expected_pngs(self) -> None:
        backtest = pd.DataFrame(
            {
                "signal": ["value", "value", "value"],
                "month": ["2020-01-31", "2020-02-29", "2020-03-31"],
                "long_leg": [0.05, -0.01, 0.04],
                "short_leg": [0.01, 0.03, -0.02],
                "long_short_return": [0.04, -0.04, 0.06],
                "n_quantiles": [5, 5, 5],
                "weighting": ["equal", "equal", "equal"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            results = create_backtest_charts(backtest, output_dir, "value")
            names = [result.name for result in results]
            paths = [result.path for result in results]

            self.assertEqual(
                names,
                ["cumulative_return", "monthly_returns", "drawdown", "long_vs_short"],
            )
            for path in paths:
                self.assertTrue(path.exists())
                self.assertEqual(path.suffix, ".png")
                self.assertGreater(path.stat().st_size, 0)

    def test_read_table_rejects_unknown_format(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported input format"):
            read_table(Path("bad.txt"))


if __name__ == "__main__":
    unittest.main()
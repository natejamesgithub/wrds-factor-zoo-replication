from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clean_compustat import clean_compustat_annual, read_table, write_table  # noqa: E402


class CleanCompustatTests(unittest.TestCase):
    def test_clean_compustat_annual_builds_factor_inputs(self) -> None:
        raw = pd.DataFrame(
            {
                "gvkey": ["1234", "1234", "1234", "9999"],
                "datadate": ["2019-12-31", "2020-12-31", "2020-12-31", "2020-12-31"],
                "fyear": [2019, 2020, 2020, 2020],
                "tic": ["AAA", "AAA", "AAA", "BAD"],
                "cusip": ["000001", "000001", "000001", "999999"],
                "conm": ["Alpha Inc", "Alpha Inc", "Alpha Inc", "Bad Book Equity"],
                "at": [100.0, 150.0, 160.0, 100.0],
                "lt": [40.0, 60.0, 70.0, 120.0],
                "seq": [60.0, 90.0, 95.0, None],
                "ceq": [55.0, 85.0, 90.0, None],
                "txditc": [5.0, 6.0, 7.0, None],
                "pstkrv": [2.0, None, 4.0, None],
                "pstkl": [3.0, 4.0, 5.0, None],
                "pstk": [4.0, 5.0, 6.0, None],
                "sale": [120.0, 180.0, 200.0, 50.0],
                "revt": [125.0, 185.0, 205.0, 50.0],
                "cogs": [70.0, 100.0, 110.0, 20.0],
                "xsga": [20.0, 30.0, 35.0, 10.0],
                "xint": [5.0, 8.0, 9.0, 1.0],
                "capx": [10.0, 15.0, 16.0, 2.0],
                "oancf": [12.0, 18.0, 19.0, 1.0],
                "ib": [8.0, 14.0, 15.0, -5.0],
                "dp": [4.0, 5.0, 6.0, 1.0],
            }
        )

        cleaned = clean_compustat_annual(raw)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["gvkey"].tolist(), ["001234", "001234"])

        first = cleaned.iloc[0]
        second = cleaned.iloc[1]

        self.assertEqual(first["portfolio_date"], pd.Timestamp("2020-06-30"))
        self.assertEqual(second["portfolio_date"], pd.Timestamp("2021-06-30"))
        self.assertEqual(first["book_equity"], 63.0)
        self.assertEqual(second["book_equity"], 98.0)
        self.assertEqual(second["sales"], 200.0)
        self.assertEqual(second["gross_profit"], 90.0)
        self.assertEqual(second["operating_profit"], 46.0)
        self.assertAlmostEqual(second["gross_profitability"], 90.0 / 160.0)
        self.assertAlmostEqual(second["operating_profitability"], 46.0 / 98.0)
        self.assertAlmostEqual(second["asset_growth"], 0.6)

    def test_clean_compustat_annual_requires_core_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required Compustat columns"):
            clean_compustat_annual(pd.DataFrame({"gvkey": ["001234"]}))

    def test_csv_round_trip_helpers(self) -> None:
        df = pd.DataFrame({"gvkey": ["001234"], "at": [100.0]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.csv"
            write_table(df, path)
            result = read_table(path)

        pd.testing.assert_frame_equal(result, df)


if __name__ == "__main__":
    unittest.main()
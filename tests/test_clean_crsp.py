from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clean_crsp import clean_crsp_monthly, read_table, write_table  # noqa: E402


class CleanCRSPTests(unittest.TestCase):
    def test_clean_crsp_monthly_filters_and_builds_core_fields(self) -> None:
        raw = pd.DataFrame(
            {
                "permno": [10001, 10001, 10002, 10003, 10004],
                "permco": [5001, 5001, 5002, 5003, 5004],
                "date": [
                    "2020-01-31",
                    "2020-01-31",
                    "2020-01-31",
                    "2020-01-31",
                    "2020-01-31",
                ],
                "ticker": ["AAA", "AAA", "BBB", "CCC", "DDD"],
                "ncusip": ["000001", "000001", "000002", "000003", "000004"],
                "shrcd": [10, 10, 12, 11, 10],
                "exchcd": [1, 1, 1, 4, 3],
                "siccd": [2000, 2000, 3000, 4000, 5000],
                "ret": [0.10, 0.20, 0.05, 0.03, None],
                "retx": [0.09, 0.19, 0.04, 0.02, None],
                "dlret": [None, -0.50, None, None, None],
                "shrout": [1000, 1000, 2000, 3000, 4000],
                "prc": [-10.0, -11.0, 20.0, 30.0, 40.0],
                "vol": [100, 110, 200, 300, 400],
            }
        )

        cleaned = clean_crsp_monthly(raw)

        self.assertEqual(len(cleaned), 1)
        row = cleaned.iloc[0]
        self.assertEqual(row["permno"], 10001)
        self.assertEqual(row["ticker"], "AAA")
        self.assertEqual(row["month"], pd.Timestamp("2020-01-31"))
        self.assertEqual(row["price"], 11.0)
        self.assertEqual(row["market_equity"], 11.0)
        self.assertAlmostEqual(row["ret_adj"], -0.4)

    def test_clean_crsp_monthly_requires_core_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required CRSP columns"):
            clean_crsp_monthly(pd.DataFrame({"permno": [10001]}))

    def test_csv_round_trip_helpers(self) -> None:
        df = pd.DataFrame({"permno": [10001], "ret": [0.05]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.csv"
            write_table(df, path)
            result = read_table(path)

        pd.testing.assert_frame_equal(result, df)


if __name__ == "__main__":
    unittest.main()
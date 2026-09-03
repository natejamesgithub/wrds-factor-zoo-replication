from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factors import (  # noqa: E402
    add_market_signals,
    build_factor_panel,
    clean_ccm_links,
    link_compustat_to_crsp,
    read_table,
    write_table,
)


class FactorTests(unittest.TestCase):
    def test_clean_ccm_links_filters_and_normalizes_links(self) -> None:
        raw_links = pd.DataFrame(
            {
                "gvkey": ["1234", "5678", "9999"],
                "permno": [10001, 10002, 10003],
                "linktype": ["LU", "XX", "LC"],
                "linkprim": ["P", "P", "J"],
                "linkdt": ["2010-01-01", "2010-01-01", "2010-01-01"],
                "linkenddt": [None, None, None],
            }
        )

        links = clean_ccm_links(raw_links)

        self.assertEqual(len(links), 1)
        self.assertEqual(links.iloc[0]["gvkey"], "001234")
        self.assertEqual(links.iloc[0]["permno"], 10001)
        self.assertEqual(links.iloc[0]["linkenddt"], pd.Timestamp("2100-12-31"))

    def test_link_compustat_to_crsp_keeps_only_valid_link_dates(self) -> None:
        compustat = pd.DataFrame(
            {
                "gvkey": ["001234", "001234"],
                "datadate": ["2009-12-31", "2020-12-31"],
                "portfolio_date": ["2010-06-30", "2021-06-30"],
                "book_equity": [50.0, 100.0],
                "operating_profitability": [0.1, 0.2],
                "asset_growth": [None, 0.5],
            }
        )
        links = pd.DataFrame(
            {
                "gvkey": ["001234"],
                "permno": [10001],
                "linktype": ["LU"],
                "linkprim": ["P"],
                "linkdt": ["2010-01-01"],
                "linkenddt": ["2025-12-31"],
            }
        )

        linked = link_compustat_to_crsp(compustat, links)

        self.assertEqual(len(linked), 1)
        self.assertEqual(linked.iloc[0]["permno"], 10001)
        self.assertEqual(linked.iloc[0]["book_equity"], 100.0)

    def test_add_market_signals_builds_lagged_size_and_momentum(self) -> None:
        crsp = pd.DataFrame(
            {
                "permno": [10001] * 13,
                "month": pd.date_range("2020-01-31", periods=13, freq="ME"),
                "ret_adj": [0.01] * 13,
                "market_equity": [100.0 + i for i in range(13)],
            }
        )

        result = add_market_signals(crsp)

        self.assertTrue(pd.isna(result.iloc[0]["me_lag"]))
        self.assertEqual(result.iloc[1]["me_lag"], 100.0)
        self.assertAlmostEqual(result.iloc[1]["size"], 4.605170185988092)
        self.assertAlmostEqual(result.iloc[12]["momentum_12_2"], (1.01**11) - 1)

    def test_build_factor_panel_uses_latest_available_fundamentals(self) -> None:
        crsp = pd.DataFrame(
            {
                "permno": [10001, 10001, 10001],
                "permco": [5001, 5001, 5001],
                "month": ["2021-05-31", "2021-06-30", "2021-07-31"],
                "ticker": ["AAA", "AAA", "AAA"],
                "ret_adj": [0.01, 0.02, 0.03],
                "market_equity": [190.0, 200.0, 210.0],
            }
        )
        compustat = pd.DataFrame(
            {
                "gvkey": ["001234", "001234"],
                "datadate": ["2019-12-31", "2020-12-31"],
                "portfolio_date": ["2020-06-30", "2021-06-30"],
                "book_equity": [50.0, 100.0],
                "operating_profitability": [0.10, 0.25],
                "asset_growth": [None, 0.20],
            }
        )
        links = pd.DataFrame(
            {
                "gvkey": ["001234"],
                "permno": [10001],
                "linktype": ["LU"],
                "linkprim": ["P"],
                "linkdt": ["2010-01-01"],
                "linkenddt": [None],
            }
        )

        panel = build_factor_panel(crsp, compustat, links)

        may = panel.loc[panel["month"] == pd.Timestamp("2021-05-31")].iloc[0]
        june = panel.loc[panel["month"] == pd.Timestamp("2021-06-30")].iloc[0]
        july = panel.loc[panel["month"] == pd.Timestamp("2021-07-31")].iloc[0]

        self.assertEqual(may["book_equity"], 50.0)
        self.assertEqual(june["book_equity"], 100.0)
        self.assertEqual(july["book_equity"], 100.0)
        self.assertAlmostEqual(june["book_to_market"], 100.0 / 190.0)
        self.assertEqual(june["value"], june["book_to_market"])
        self.assertEqual(june["profitability"], 0.25)
        self.assertAlmostEqual(june["investment"], -0.20)

    def test_csv_round_trip_helpers_preserve_identifiers(self) -> None:
        df = pd.DataFrame({"gvkey": ["001234"], "cusip": ["000001"], "value": [1.0]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.csv"
            write_table(df, path)
            result = read_table(path)

        pd.testing.assert_frame_equal(result, df)


if __name__ == "__main__":
    unittest.main()
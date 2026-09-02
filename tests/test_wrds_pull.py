from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wrds_pull import (  # noqa: E402
    WRDSPuller,
    build_ccm_link_query,
    build_compustat_annual_query,
    build_crsp_monthly_query,
)


class FakeWRDSConnection:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def raw_sql(self, query: str) -> pd.DataFrame:
        self.queries.append(" ".join(query.split()))
        return pd.DataFrame({"id": [1, 2, 3]})


class WRDSPullTests(unittest.TestCase):
    def test_crsp_query_uses_monthly_returns_and_common_shares(self) -> None:
        query = " ".join(build_crsp_monthly_query("2020-01-01", "2020-12-31").split())

        self.assertIn("from crsp.msf as m", query)
        self.assertIn("left join crsp.msenames as n", query)
        self.assertIn("m.date between '2020-01-01' and '2020-12-31'", query)
        self.assertIn("n.shrcd in (10, 11)", query)

    def test_compustat_query_uses_standard_annual_fundamentals(self) -> None:
        query = " ".join(
            build_compustat_annual_query("2020-01-01", "2020-12-31").split()
        )

        self.assertIn("from comp.funda", query)
        self.assertIn("datadate between '2020-01-01' and '2020-12-31'", query)
        self.assertIn("indfmt = 'INDL'", query)
        self.assertIn("datafmt = 'STD'", query)
        self.assertIn("popsrc = 'D'", query)
        self.assertIn("consol = 'C'", query)

    def test_ccm_query_uses_research_standard_links(self) -> None:
        query = " ".join(build_ccm_link_query().split())

        self.assertIn("from crsp_a_ccm.ccmxpf_linktable", query)
        self.assertIn("linktype in ('LU', 'LC')", query)
        self.assertIn("linkprim in ('P', 'C')", query)

    def test_pull_all_writes_three_extracts(self) -> None:
        connection = FakeWRDSConnection()
        puller = WRDSPuller(connection=connection)

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            results = puller.pull_all(
                start_date="2020-01-01",
                end_date="2020-12-31",
                data_dir=data_dir,
                file_format="csv",
            )

            self.assertEqual(
                [result.name for result in results],
                ["crsp_monthly", "compustat_annual", "ccm_links"],
            )
            self.assertEqual([result.rows for result in results], [3, 3, 3])
            self.assertEqual(len(connection.queries), 3)
            self.assertTrue((data_dir / "crsp_monthly.csv").exists())
            self.assertTrue((data_dir / "compustat_annual.csv").exists())
            self.assertTrue((data_dir / "ccm_links.csv").exists())


if __name__ == "__main__":
    unittest.main()
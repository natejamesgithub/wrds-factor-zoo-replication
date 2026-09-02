"""Pull the core WRDS datasets needed for factor replication.

This module pulls three raw inputs:

1. CRSP monthly stock returns
2. Compustat annual fundamentals
3. CRSP/Compustat Merged linking table

Run from the repository root:

    python3 src/wrds_pull.py --start-date 2010-01-01 --end-date 2024-12-31

The script uses the local WRDS credentials through the official `wrds` package.
It writes raw extracts to `data/` and does not save credentials.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


FileFormat = Literal["csv", "parquet"]


@dataclass(frozen=True)
class PullResult:
    """Metadata for one completed WRDS extract."""

    name: str
    rows: int
    path: Path


def build_crsp_monthly_query(start_date: str, end_date: str) -> str:
    """Build SQL for monthly CRSP common-stock returns."""
    return f"""
        select
            m.permno,
            m.permco,
            m.date,
            n.shrcd,
            n.exchcd,
            n.siccd,
            n.ncusip,
            n.ticker,
            m.ret,
            m.retx,
            m.shrout,
            m.prc,
            m.vol
        from crsp.msf as m
        left join crsp.msenames as n
            on m.permno = n.permno
            and n.namedt <= m.date
            and m.date <= coalesce(n.nameendt, '9999-12-31')
        where m.date between '{start_date}' and '{end_date}'
            and n.shrcd in (10, 11)
    """


def build_compustat_annual_query(start_date: str, end_date: str) -> str:
    """Build SQL for annual Compustat fundamentals."""
    return f"""
        select
            gvkey,
            datadate,
            fyear,
            tic,
            cusip,
            conm,
            at,
            lt,
            seq,
            ceq,
            txditc,
            pstkrv,
            pstkl,
            pstk,
            sale,
            revt,
            cogs,
            xsga,
            xrd,
            capx,
            oancf,
            ib,
            dp
        from comp.funda
        where datadate between '{start_date}' and '{end_date}'
            and indfmt = 'INDL'
            and datafmt = 'STD'
            and popsrc = 'D'
            and consol = 'C'
    """


def build_ccm_link_query() -> str:
    """Build SQL for CRSP/Compustat linking metadata."""
    return """
        select
            gvkey,
            lpermno as permno,
            lpermco as permco,
            linktype,
            linkprim,
            linkdt,
            linkenddt
        from crsp_a_ccm.ccmxpf_linktable
        where linktype in ('LU', 'LC')
            and linkprim in ('P', 'C')
    """


def write_extract(df: pd.DataFrame, output_path: Path, file_format: FileFormat) -> Path:
    """Write one extract to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format == "csv":
        path = output_path.with_suffix(".csv")
        df.to_csv(path, index=False)
        return path

    if file_format == "parquet":
        path = output_path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path

    raise ValueError(f"Unsupported file format: {file_format}")


class WRDSPuller:
    """Small wrapper around a WRDS connection."""

    def __init__(self, connection=None) -> None:
        if connection is None:
            import wrds

            connection = wrds.Connection()
            self._owns_connection = True
        else:
            self._owns_connection = False

        self.connection = connection

    def close(self) -> None:
        if self._owns_connection:
            self.connection.close()

    def _read_sql(self, query: str) -> pd.DataFrame:
        return self.connection.raw_sql(query)

    def pull_crsp_monthly(
        self,
        start_date: str,
        end_date: str,
        data_dir: Path,
        file_format: FileFormat,
    ) -> PullResult:
        df = self._read_sql(build_crsp_monthly_query(start_date, end_date))
        path = write_extract(df, data_dir / "crsp_monthly", file_format)
        return PullResult(name="crsp_monthly", rows=len(df), path=path)

    def pull_compustat_annual(
        self,
        start_date: str,
        end_date: str,
        data_dir: Path,
        file_format: FileFormat,
    ) -> PullResult:
        df = self._read_sql(build_compustat_annual_query(start_date, end_date))
        path = write_extract(df, data_dir / "compustat_annual", file_format)
        return PullResult(name="compustat_annual", rows=len(df), path=path)

    def pull_ccm_links(self, data_dir: Path, file_format: FileFormat) -> PullResult:
        df = self._read_sql(build_ccm_link_query())
        path = write_extract(df, data_dir / "ccm_links", file_format)
        return PullResult(name="ccm_links", rows=len(df), path=path)

    def pull_all(
        self,
        start_date: str,
        end_date: str,
        data_dir: Path,
        file_format: FileFormat = "parquet",
    ) -> list[PullResult]:
        return [
            self.pull_crsp_monthly(start_date, end_date, data_dir, file_format),
            self.pull_compustat_annual(start_date, end_date, data_dir, file_format),
            self.pull_ccm_links(data_dir, file_format),
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull raw WRDS data for factor replication.")
    parser.add_argument("--start-date", required=True, help="Start date, formatted YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="End date, formatted YYYY-MM-DD.")
    parser.add_argument("--data-dir", default="data", help="Output directory for raw extracts.")
    parser.add_argument(
        "--file-format",
        choices=("csv", "parquet"),
        default="parquet",
        help="Output format. Use csv if parquet dependencies are not installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    puller = WRDSPuller()

    try:
        results = puller.pull_all(
            start_date=args.start_date,
            end_date=args.end_date,
            data_dir=Path(args.data_dir),
            file_format=args.file_format,
        )
    finally:
        puller.close()

    print("WRDS pull complete")
    print("=" * 18)
    for result in results:
        print(f"{result.name}: {result.rows:,} rows -> {result.path}")


if __name__ == "__main__":
    main()
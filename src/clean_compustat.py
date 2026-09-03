"""Clean raw Compustat annual fundamentals for factor construction.

Run from the repository root:

    python3 src/clean_compustat.py --input data/compustat_annual.csv --output data/compustat_annual_clean.csv

Use Parquet paths instead if you pulled Parquet files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"gvkey", "datadate", "at", "lt"}

NUMERIC_COLUMNS = [
    "fyear",
    "at",
    "lt",
    "seq",
    "ceq",
    "txditc",
    "pstkrv",
    "pstkl",
    "pstk",
    "sale",
    "revt",
    "cogs",
    "xsga",
    "xrd",
    "xint",
    "capx",
    "oancf",
    "ib",
    "dp",
]


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet table."""
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"gvkey": str, "cusip": str})
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_table(df: pd.DataFrame, path: Path) -> Path:
    """Write a CSV or Parquet table."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return path
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        return path
    raise ValueError(f"Unsupported output format: {path.suffix}")


def validate_compustat_columns(df: pd.DataFrame) -> None:
    """Raise a helpful error when required Compustat columns are missing."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required Compustat columns: {', '.join(missing)}")


def _first_available(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first non-null value across a list of columns."""
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)
    return df[available].bfill(axis=1).iloc[:, 0]


def _numeric_or_zero(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df[column].fillna(0)


def clean_compustat_annual(raw: pd.DataFrame) -> pd.DataFrame:
    """Return analysis-ready Compustat annual fundamentals."""
    df = raw.copy()
    df.columns = [column.lower() for column in df.columns]
    validate_compustat_columns(df)

    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df["datadate"] = pd.to_datetime(df["datadate"])

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["gvkey", "datadate"]).drop_duplicates(
        ["gvkey", "datadate"],
        keep="last",
    )

    stockholders_equity = _first_available(
        df,
        ["seq", "ceq_plus_pstk", "at_minus_lt"],
    )
    if "ceq" in df.columns and "pstk" in df.columns:
        stockholders_equity = stockholders_equity.fillna(df["ceq"] + df["pstk"].fillna(0))
    stockholders_equity = stockholders_equity.fillna(df["at"] - df["lt"])

    preferred_stock = _first_available(df, ["pstkrv", "pstkl", "pstk"]).fillna(0)
    deferred_taxes = _numeric_or_zero(df, "txditc")

    df["book_equity"] = stockholders_equity + deferred_taxes - preferred_stock
    df["sales"] = _first_available(df, ["sale", "revt"])
    df["gross_profit"] = df["sales"] - _numeric_or_zero(df, "cogs")
    df["operating_profit"] = (
        df["sales"]
        - _numeric_or_zero(df, "cogs")
        - _numeric_or_zero(df, "xsga")
        - _numeric_or_zero(df, "xint")
    )
    df["gross_profitability"] = df["gross_profit"] / df["at"]
    df["operating_profitability"] = df["operating_profit"] / df["book_equity"]

    df["asset_growth"] = df.groupby("gvkey")["at"].pct_change()
    df["portfolio_date"] = (df["datadate"] + pd.DateOffset(months=6)).dt.to_period(
        "M"
    ).dt.to_timestamp("M")

    df = df[df["at"].replace([np.inf, -np.inf], np.nan).notna()]
    df = df[df["book_equity"].replace([np.inf, -np.inf], np.nan).notna()]
    df = df[df["at"] > 0]
    df = df[df["book_equity"] > 0]

    ordered_columns = [
        "gvkey",
        "datadate",
        "portfolio_date",
        "fyear",
        "tic",
        "cusip",
        "conm",
        "at",
        "lt",
        "book_equity",
        "sales",
        "gross_profit",
        "operating_profit",
        "gross_profitability",
        "operating_profitability",
        "asset_growth",
        "seq",
        "ceq",
        "txditc",
        "pstkrv",
        "pstkl",
        "pstk",
        "sale",
        "revt",
        "cogs",
        "xsga",
        "xrd",
        "xint",
        "capx",
        "oancf",
        "ib",
        "dp",
    ]
    existing_columns = [column for column in ordered_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in existing_columns]

    return df[existing_columns + remaining_columns].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw Compustat annual data.")
    parser.add_argument("--input", required=True, help="Raw Compustat annual CSV or Parquet file.")
    parser.add_argument("--output", required=True, help="Cleaned Compustat output CSV or Parquet file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = read_table(Path(args.input))
    cleaned = clean_compustat_annual(raw)
    output_path = write_table(cleaned, Path(args.output))

    print("Compustat clean complete")
    print("=" * 23)
    print(f"Rows: {len(cleaned):,}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
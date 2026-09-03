"""Clean raw CRSP monthly data for factor construction.

Run from the repository root:

    python3 src/clean_crsp.py --input data/crsp_monthly.csv --output data/crsp_monthly_clean.csv

Uses Parquet paths instead if Parquet files were pulled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "permno",
    "permco",
    "date",
    "shrcd",
    "exchcd",
    "ret",
    "shrout",
    "prc",
}


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet table."""
    if path.suffix == ".csv":
        return pd.read_csv(path)
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


def validate_crsp_columns(df: pd.DataFrame) -> None:
    """Raise a helpful error when required CRSP columns are missing."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required CRSP columns: {', '.join(missing)}")


def combine_returns(ret: pd.Series, dlret: pd.Series | None = None) -> pd.Series:
    """Combine monthly return and optional delisting return."""
    monthly_ret = pd.to_numeric(ret, errors="coerce")

    if dlret is None:
        return monthly_ret

    delisting_ret = pd.to_numeric(dlret, errors="coerce")
    has_any_return = monthly_ret.notna() | delisting_ret.notna()
    combined = (1 + monthly_ret.fillna(0)) * (1 + delisting_ret.fillna(0)) - 1
    return combined.where(has_any_return)


def clean_crsp_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """Return analysis-ready CRSP monthly stock data."""
    validate_crsp_columns(raw)

    df = raw.copy()
    df.columns = [column.lower() for column in df.columns]

    numeric_columns = ["permno", "permco", "shrcd", "exchcd", "shrout", "prc", "vol"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")

    if "retx" in df.columns:
        df["retx"] = pd.to_numeric(df["retx"], errors="coerce")

    df["ret_adj"] = combine_returns(df["ret"], df["dlret"] if "dlret" in df.columns else None)
    df["price"] = df["prc"].abs()
    df["market_equity"] = df["price"] * df["shrout"] / 1000

    df = df[df["shrcd"].isin([10, 11])]
    df = df[df["exchcd"].isin([1, 2, 3])]
    df = df[df["ret_adj"].notna()]
    df = df[df["market_equity"].replace([np.inf, -np.inf], np.nan).notna()]
    df = df[df["market_equity"] > 0]

    sort_columns = ["permno", "month", "market_equity"]
    df = df.sort_values(sort_columns).drop_duplicates(["permno", "month"], keep="last")

    ordered_columns = [
        "permno",
        "permco",
        "month",
        "date",
        "ticker",
        "ncusip",
        "shrcd",
        "exchcd",
        "siccd",
        "ret",
        "retx",
        "ret_adj",
        "shrout",
        "price",
        "market_equity",
        "vol",
    ]
    existing_columns = [column for column in ordered_columns if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in existing_columns]

    return df[existing_columns + remaining_columns].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw CRSP monthly data.")
    parser.add_argument("--input", required=True, help="Raw CRSP monthly CSV or Parquet file.")
    parser.add_argument("--output", required=True, help="Cleaned CRSP output CSV or Parquet file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = read_table(Path(args.input))
    cleaned = clean_crsp_monthly(raw)
    output_path = write_table(cleaned, Path(args.output))

    print("CRSP clean complete")
    print("=" * 19)
    print(f"Rows: {len(cleaned):,}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
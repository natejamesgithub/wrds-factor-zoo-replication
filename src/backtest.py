"""Backtest factor-sorted portfolios from the monthly factor panel.

Run from the repository root:

    python3 src/backtest.py \
        --input data/factor_panel.csv \
        --signal value \
        --output data/value_backtest.csv

Use CSV paths instead if your pipeline is using CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


WeightingMethod = Literal["equal", "value"]

REQUIRED_COLUMNS = {"month", "permno", "ret_adj"}


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet table."""
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={"gvkey": str, "cusip": str, "ncusip": str})
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


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required {label} columns: {', '.join(missing)}")


def assign_quantiles(
    panel: pd.DataFrame,
    signal: str,
    n_quantiles: int = 5,
    min_stocks: int | None = None,
) -> pd.DataFrame:
    """Assign monthly signal quantiles."""
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")

    df = panel.copy()
    df.columns = [column.lower() for column in df.columns]
    signal = signal.lower()
    _require_columns(df, REQUIRED_COLUMNS | {signal}, "factor panel")

    df["month"] = pd.to_datetime(df["month"])
    df["ret_adj"] = pd.to_numeric(df["ret_adj"], errors="coerce")
    df[signal] = pd.to_numeric(df[signal], errors="coerce")
    df = df[df["ret_adj"].notna() & df[signal].notna()]

    min_stocks = min_stocks or n_quantiles

    def quantile_for_month(values: pd.Series) -> pd.Series:
        if values.notna().sum() < min_stocks or values.nunique(dropna=True) < n_quantiles:
            return pd.Series(np.nan, index=values.index)
        ranks = values.rank(method="first")
        return pd.qcut(ranks, q=n_quantiles, labels=False) + 1

    df["quantile"] = df.groupby("month", group_keys=False)[signal].apply(
        quantile_for_month
    )
    df = df[df["quantile"].notna()]
    df["quantile"] = df["quantile"].astype(int)

    return df.reset_index(drop=True)


def calculate_portfolio_returns(
    quantile_panel: pd.DataFrame,
    weighting: WeightingMethod = "equal",
    weight_column: str = "me_lag",
) -> pd.DataFrame:
    """Calculate monthly returns for each signal quantile."""
    df = quantile_panel.copy()
    _require_columns(df, {"month", "quantile", "ret_adj"}, "quantile panel")

    if weighting == "equal":
        returns = (
            df.groupby(["month", "quantile"], as_index=False)["ret_adj"]
            .mean()
            .rename(columns={"ret_adj": "portfolio_return"})
        )
        return returns.sort_values(["month", "quantile"]).reset_index(drop=True)

    if weighting == "value":
        _require_columns(df, {weight_column}, "quantile panel")
        df[weight_column] = pd.to_numeric(df[weight_column], errors="coerce")
        df = df[df[weight_column].notna() & (df[weight_column] > 0)]
        weight_sum = df.groupby(["month", "quantile"])[weight_column].transform("sum")
        df["weight"] = df[weight_column] / weight_sum
        df["weighted_return"] = df["weight"] * df["ret_adj"]
        returns = (
            df.groupby(["month", "quantile"], as_index=False)["weighted_return"]
            .sum()
            .rename(columns={"weighted_return": "portfolio_return"})
        )
        return returns.sort_values(["month", "quantile"]).reset_index(drop=True)

    raise ValueError(f"Unsupported weighting method: {weighting}")


def calculate_long_short_returns(
    portfolio_returns: pd.DataFrame,
    n_quantiles: int = 5,
    high_minus_low: bool = True,
) -> pd.DataFrame:
    """Calculate monthly long-short returns from quantile returns."""
    _require_columns(
        portfolio_returns,
        {"month", "quantile", "portfolio_return"},
        "portfolio returns",
    )

    wide = portfolio_returns.pivot(
        index="month",
        columns="quantile",
        values="portfolio_return",
    )
    low = wide[1]
    high = wide[n_quantiles]
    long_short = high - low if high_minus_low else low - high

    result = pd.DataFrame(
        {
            "month": long_short.index,
            "long_leg": high if high_minus_low else low,
            "short_leg": low if high_minus_low else high,
            "long_short_return": long_short,
        }
    )
    return result.dropna().reset_index(drop=True)


def run_quantile_backtest(
    panel: pd.DataFrame,
    signal: str,
    n_quantiles: int = 5,
    weighting: WeightingMethod = "equal",
    high_minus_low: bool = True,
) -> pd.DataFrame:
    """Run a complete monthly quantile factor backtest."""
    quantile_panel = assign_quantiles(panel, signal=signal, n_quantiles=n_quantiles)
    portfolio_returns = calculate_portfolio_returns(
        quantile_panel,
        weighting=weighting,
    )
    long_short = calculate_long_short_returns(
        portfolio_returns,
        n_quantiles=n_quantiles,
        high_minus_low=high_minus_low,
    )
    long_short["signal"] = signal
    long_short["n_quantiles"] = n_quantiles
    long_short["weighting"] = weighting

    ordered_columns = [
        "signal",
        "month",
        "long_leg",
        "short_leg",
        "long_short_return",
        "n_quantiles",
        "weighting",
    ]
    return long_short[ordered_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest monthly factor-sorted portfolios.")
    parser.add_argument("--input", required=True, help="Factor panel CSV or Parquet file.")
    parser.add_argument("--signal", required=True, help="Signal column to sort on.")
    parser.add_argument("--output", required=True, help="Backtest output CSV or Parquet file.")
    parser.add_argument("--n-quantiles", type=int, default=5, help="Number of portfolios.")
    parser.add_argument(
        "--weighting",
        choices=("equal", "value"),
        default="equal",
        help="Portfolio weighting method.",
    )
    parser.add_argument(
        "--low-minus-high",
        action="store_true",
        help="Use low-minus-high instead of high-minus-low.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = read_table(Path(args.input))
    result = run_quantile_backtest(
        panel,
        signal=args.signal,
        n_quantiles=args.n_quantiles,
        weighting=args.weighting,
        high_minus_low=not args.low_minus_high,
    )
    output_path = write_table(result, Path(args.output))

    print("Backtest complete")
    print("=" * 17)
    print(f"Rows: {len(result):,}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
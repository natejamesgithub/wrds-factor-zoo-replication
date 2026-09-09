"""Summarizes factor backtest results.

Run from the repository root:

    python3 src/analytics.py \
        --input data/value_backtest.csv \
        --output reports/value_metrics.csv

Use Parquet paths instead if your pipeline is using Parquet files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"month", "long_short_return"}
MONTHS_PER_YEAR = 12


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


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required {label} columns: {', '.join(missing)}")


def prepare_returns(backtest: pd.DataFrame) -> pd.DataFrame:
    """Normalize a backtest return series for analytics."""
    df = backtest.copy()
    df.columns = [column.lower() for column in df.columns]
    _require_columns(df, REQUIRED_COLUMNS, "backtest")

    df["month"] = pd.to_datetime(df["month"])
    df["long_short_return"] = pd.to_numeric(df["long_short_return"], errors="coerce")
    df = df[df["long_short_return"].notna()]
    df = df.sort_values("month").drop_duplicates("month", keep="last")

    return df.reset_index(drop=True)


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Convert periodic returns to cumulative total returns."""
    return (1 + returns).cumprod() - 1


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate drawdowns from a periodic return series."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1


def summarize_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row performance summary for a long-short backtest."""
    df = prepare_returns(backtest)
    returns = df["long_short_return"]

    if returns.empty:
        raise ValueError("Backtest contains no valid return observations")

    months = len(returns)
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (MONTHS_PER_YEAR / months) - 1
    annualized_return = returns.mean() * MONTHS_PER_YEAR
    annualized_volatility = returns.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    sharpe = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
    max_drawdown = drawdown_series(returns).min()
    hit_rate = (returns > 0).mean()

    summary = {
        "signal": df["signal"].iloc[0] if "signal" in df.columns else "unknown",
        "start_month": df["month"].min(),
        "end_month": df["month"].max(),
        "months": months,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "best_month": returns.max(),
        "worst_month": returns.min(),
    }

    if "weighting" in df.columns:
        summary["weighting"] = df["weighting"].iloc[0]
    if "n_quantiles" in df.columns:
        summary["n_quantiles"] = df["n_quantiles"].iloc[0]

    return pd.DataFrame([summary])


def add_cumulative_performance(backtest: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative return and drawdown columns to a backtest series."""
    df = prepare_returns(backtest)
    df["cumulative_return"] = cumulative_returns(df["long_short_return"])
    df["drawdown"] = drawdown_series(df["long_short_return"])
    return df


def summarize_many(backtests: list[pd.DataFrame]) -> pd.DataFrame:
    """Summarize multiple backtest outputs into one metrics table."""
    summaries = [summarize_backtest(backtest) for backtest in backtests]
    return pd.concat(summaries, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize factor backtest results.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more backtest CSV or Parquet files.",
    )
    parser.add_argument("--output", required=True, help="Metrics output CSV or Parquet file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backtests = [read_table(Path(path)) for path in args.input]
    summary = summarize_many(backtests)
    output_path = write_table(summary, Path(args.output))

    print("Analytics complete")
    print("=" * 18)
    print(f"Rows: {len(summary):,}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
"""Creates chart outputs for factor backtest results.

Run from the repository root:

    python3 src/visualize.py \
        --input data/value_backtest.csv \
        --output-dir reports \
        --prefix value
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analytics import add_cumulative_performance


@dataclass(frozen=True)
class ChartResult:
    """Metadata for one saved chart."""

    name: str
    path: Path


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet table."""
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def format_percent_axis(axis) -> None:
    """Format an axis as whole-percentage labels."""
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


def save_cumulative_return_chart(df: pd.DataFrame, output_dir: Path, prefix: str) -> ChartResult:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["month"], df["cumulative_return"], color="#1f77b4", linewidth=2)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Cumulative Long-Short Return")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative return")
    format_percent_axis(ax)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = output_dir / f"{prefix}_cumulative_return.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return ChartResult("cumulative_return", path)


def save_monthly_return_chart(df: pd.DataFrame, output_dir: Path, prefix: str) -> ChartResult:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = df["long_short_return"].map(lambda value: "#2ca02c" if value >= 0 else "#d62728")
    ax.bar(df["month"], df["long_short_return"], color=colors, width=20)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Monthly Long-Short Returns")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly return")
    format_percent_axis(ax)
    ax.grid(True, axis="y", alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = output_dir / f"{prefix}_monthly_returns.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return ChartResult("monthly_returns", path)


def save_long_short_leg_chart(df: pd.DataFrame, output_dir: Path, prefix: str) -> ChartResult:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["month"], df["long_leg"], label="Long leg", color="#1f77b4", linewidth=2)
    ax.plot(df["month"], df["short_leg"], label="Short leg", color="#ff7f0e", linewidth=2)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Long Leg vs Short Leg")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly return")
    ax.legend()
    format_percent_axis(ax)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = output_dir / f"{prefix}_long_vs_short.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return ChartResult("long_vs_short", path)


def save_drawdown_chart(df: pd.DataFrame, output_dir: Path, prefix: str) -> ChartResult:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(df["month"], df["drawdown"], 0, color="#d62728", alpha=0.35)
    ax.plot(df["month"], df["drawdown"], color="#d62728", linewidth=1.5)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Long-Short Drawdown")
    ax.set_xlabel("Month")
    ax.set_ylabel("Drawdown")
    format_percent_axis(ax)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = output_dir / f"{prefix}_drawdown.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return ChartResult("drawdown", path)


def create_backtest_charts(
    backtest: pd.DataFrame,
    output_dir: Path,
    prefix: str,
) -> list[ChartResult]:
    """Create standard chart PNGs for one factor backtest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    performance = add_cumulative_performance(backtest)

    results = [
        save_cumulative_return_chart(performance, output_dir, prefix),
        save_monthly_return_chart(performance, output_dir, prefix),
        save_drawdown_chart(performance, output_dir, prefix),
    ]

    if {"long_leg", "short_leg"}.issubset(performance.columns):
        results.append(save_long_short_leg_chart(performance, output_dir, prefix))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create factor backtest charts.")
    parser.add_argument("--input", required=True, help="Backtest CSV or Parquet file.")
    parser.add_argument("--output-dir", default="reports", help="Directory for PNG charts.")
    parser.add_argument("--prefix", default="factor", help="Filename prefix for charts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backtest = read_table(Path(args.input))
    results = create_backtest_charts(backtest, Path(args.output_dir), args.prefix)

    print("Visualization complete")
    print("=" * 22)
    for result in results:
        print(f"{result.name}: {result.path}")


if __name__ == "__main__":
    main()
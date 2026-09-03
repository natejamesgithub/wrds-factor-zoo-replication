"""Build monthly equity factor signals from cleaned CRSP and Compustat data.

Run from the repository root:

    python3 src/factors.py \
        --crsp data/crsp_monthly_clean.csv \
        --compustat data/compustat_annual_clean.csv \
        --ccm data/ccm_links.csv \
        --output data/factor_panel.csv

Use Parquet paths instead if you pulled Parquet files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CRSP_REQUIRED_COLUMNS = {"permno", "month", "ret_adj", "market_equity"}
COMPUSTAT_REQUIRED_COLUMNS = {
    "gvkey",
    "datadate",
    "portfolio_date",
    "book_equity",
    "operating_profitability",
    "asset_growth",
}
CCM_REQUIRED_COLUMNS = {"gvkey", "permno", "linkdt", "linkenddt"}


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


def clean_ccm_links(raw_links: pd.DataFrame) -> pd.DataFrame:
    """Prepare CCM links for merging Compustat records to CRSP permnos."""
    links = raw_links.copy()
    links.columns = [column.lower() for column in links.columns]
    _require_columns(links, CCM_REQUIRED_COLUMNS, "CCM")

    links["gvkey"] = links["gvkey"].astype(str).str.zfill(6)
    links["permno"] = pd.to_numeric(links["permno"], errors="coerce")
    links["linkdt"] = pd.to_datetime(links["linkdt"]).fillna(pd.Timestamp("1900-01-01"))
    links["linkenddt"] = pd.to_datetime(links["linkenddt"]).fillna(
        pd.Timestamp("2100-12-31")
    )

    if "linktype" in links.columns:
        links = links[links["linktype"].isin(["LU", "LC"])]
    if "linkprim" in links.columns:
        links = links[links["linkprim"].isin(["P", "C"])]

    links = links[links["permno"].notna()]
    links["permno"] = links["permno"].astype(int)

    return links.drop_duplicates().reset_index(drop=True)


def link_compustat_to_crsp(compustat: pd.DataFrame, ccm_links: pd.DataFrame) -> pd.DataFrame:
    """Attach CRSP permnos to cleaned Compustat rows through valid CCM links."""
    comp = compustat.copy()
    comp.columns = [column.lower() for column in comp.columns]
    _require_columns(comp, COMPUSTAT_REQUIRED_COLUMNS, "Compustat")

    comp["gvkey"] = comp["gvkey"].astype(str).str.zfill(6)
    comp["datadate"] = pd.to_datetime(comp["datadate"])
    comp["portfolio_date"] = pd.to_datetime(comp["portfolio_date"])

    links = clean_ccm_links(ccm_links)
    linked = comp.merge(links, on="gvkey", how="inner")
    linked = linked[
        (linked["datadate"] >= linked["linkdt"])
        & (linked["datadate"] <= linked["linkenddt"])
    ]

    return linked.reset_index(drop=True)


def add_market_signals(crsp: pd.DataFrame) -> pd.DataFrame:
    """Add market-only signals such as lagged size and 12-2 momentum."""
    df = crsp.copy()
    df.columns = [column.lower() for column in df.columns]
    _require_columns(df, CRSP_REQUIRED_COLUMNS, "CRSP")

    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df["month"] = pd.to_datetime(df["month"])
    df["ret_adj"] = pd.to_numeric(df["ret_adj"], errors="coerce")
    df["market_equity"] = pd.to_numeric(df["market_equity"], errors="coerce")
    df = df.sort_values(["permno", "month"])

    df["me_lag"] = df.groupby("permno")["market_equity"].shift(1)
    df["size"] = np.log(df["me_lag"])

    shifted_returns = df.groupby("permno")["ret_adj"].shift(2)
    df["momentum_12_2"] = shifted_returns.groupby(df["permno"]).transform(
        lambda returns: (1 + returns).rolling(11, min_periods=8).apply(np.prod, raw=True)
        - 1
    )

    return df


def build_factor_panel(
    crsp: pd.DataFrame,
    compustat: pd.DataFrame,
    ccm_links: pd.DataFrame,
) -> pd.DataFrame:
    """Create a monthly panel with equity factor signals."""
    market_panel = add_market_signals(crsp)
    linked_compustat = link_compustat_to_crsp(compustat, ccm_links)

    panel = market_panel.merge(linked_compustat, on="permno", how="left")
    panel = panel[panel["portfolio_date"].isna() | (panel["portfolio_date"] <= panel["month"])]
    panel = panel.sort_values(["permno", "month", "portfolio_date"])
    panel = panel.drop_duplicates(["permno", "month"], keep="last")

    panel["book_to_market"] = panel["book_equity"] / panel["me_lag"]
    panel["value"] = panel["book_to_market"]
    panel["profitability"] = panel["operating_profitability"]
    panel["investment"] = -panel["asset_growth"]

    ordered_columns = [
        "permno",
        "permco",
        "gvkey",
        "month",
        "ticker",
        "ret_adj",
        "market_equity",
        "me_lag",
        "size",
        "book_equity",
        "book_to_market",
        "value",
        "profitability",
        "investment",
        "momentum_12_2",
        "portfolio_date",
        "datadate",
    ]
    existing_columns = [column for column in ordered_columns if column in panel.columns]
    remaining_columns = [column for column in panel.columns if column not in existing_columns]

    return panel[existing_columns + remaining_columns].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build monthly factor signal panel.")
    parser.add_argument("--crsp", required=True, help="Cleaned CRSP monthly CSV or Parquet file.")
    parser.add_argument(
        "--compustat",
        required=True,
        help="Cleaned Compustat annual CSV or Parquet file.",
    )
    parser.add_argument("--ccm", required=True, help="Raw CCM links CSV or Parquet file.")
    parser.add_argument("--output", required=True, help="Factor panel output CSV or Parquet file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crsp = read_table(Path(args.crsp))
    compustat = read_table(Path(args.compustat))
    ccm_links = read_table(Path(args.ccm))

    panel = build_factor_panel(crsp, compustat, ccm_links)
    output_path = write_table(panel, Path(args.output))

    print("Factor panel complete")
    print("=" * 21)
    print(f"Rows: {len(panel):,}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
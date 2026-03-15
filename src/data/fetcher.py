"""Data fetching and alignment for Brent crude and ITO-S3 basket levels.

Loads Brent futures (BZ=F) from Yahoo Finance and the S3 basket level
from the internal CSV, then aligns them on a common date axis and
computes daily log returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import BASKET_CODE, DATA_DIR


def fetch_brent(start: str = "2024-01-01") -> pd.Series:
    """Download Brent crude daily close prices from Yahoo Finance.

    Parameters
    ----------
    start : str
        ISO-format start date for the download window.

    Returns
    -------
    pd.Series
        Daily close prices with a DatetimeIndex named 'date'.
    """
    ticker = yf.Ticker("BZ=F")
    hist = ticker.history(start=start)
    if hist.empty:
        raise ValueError(f"No data returned for BZ=F from {start}")
    close = hist["Close"].copy()
    close.index = pd.DatetimeIndex(close.index.date, name="date")
    close.name = "brent_price"
    return close


def load_s3_basket(
    csv_path: str = f"{DATA_DIR}/basket_level_monthly.csv",
    basket_code: str = BASKET_CODE,
) -> pd.Series:
    """Load the S3 basket level from the internal CSV.

    The CSV contains daily rows for multiple baskets. This function
    filters to the target basket code and returns the basket_level
    column as a Series with a daily DatetimeIndex.

    Parameters
    ----------
    csv_path : str
        Path to the basket-level CSV file.
    basket_code : str
        Internal basket identifier (default: ADIT-S3).

    Returns
    -------
    pd.Series
        Daily basket level with a DatetimeIndex named 'date'.
    """
    df = pd.read_csv(csv_path, parse_dates=["rebalance_date"])
    mask = df["basket_code"] == basket_code
    if not mask.any():
        available = df["basket_code"].unique().tolist()
        raise ValueError(
            f"Basket code '{basket_code}' not found. Available: {available}"
        )
    subset = df.loc[mask, ["rebalance_date", "basket_level"]].copy()
    subset = subset.sort_values("rebalance_date").drop_duplicates(
        subset="rebalance_date", keep="last"
    )
    subset.set_index("rebalance_date", inplace=True)
    subset.index = pd.DatetimeIndex(subset.index.date, name="date")
    series = subset["basket_level"].astype(float)
    series.name = "s3_level"
    return series


def align_data(brent: pd.Series, s3: pd.Series) -> pd.DataFrame:
    """Inner-join Brent and S3 on date, then compute daily returns.

    Parameters
    ----------
    brent : pd.Series
        Brent close prices with DatetimeIndex.
    s3 : pd.Series
        S3 basket levels with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Columns: brent_price, s3_level, brent_ret, s3_ret.
        Index: DatetimeIndex named 'date'.
    """
    merged = pd.DataFrame({"brent_price": brent, "s3_level": s3}).dropna()
    merged.sort_index(inplace=True)
    merged["brent_ret"] = np.log(merged["brent_price"]).diff()
    merged["s3_ret"] = np.log(merged["s3_level"]).diff()
    merged.dropna(inplace=True)
    return merged


def fetch_all(
    data_dir: str = DATA_DIR,
    start: str = "2024-01-01",
    basket_code: str = BASKET_CODE,
) -> pd.DataFrame:
    """Convenience wrapper: fetch Brent, load S3, align, return DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory containing the basket CSV.
    start : str
        Start date for the Brent download.
    basket_code : str
        Internal basket identifier.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with prices, levels, and returns.
    """
    brent = fetch_brent(start=start)
    csv_path = str(Path(data_dir) / "basket_level_monthly.csv")
    s3 = load_s3_basket(csv_path=csv_path, basket_code=basket_code)
    return align_data(brent, s3)

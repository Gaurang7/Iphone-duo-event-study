"""
data.py — download daily stock prices and save them to a file.

Why save to a file? Downloading takes time and the data source can be slow
or down. Once prices are saved in data/prices.csv, the analysis and the
dashboard read that file instantly, and anyone who clones the repo gets
the exact same numbers you did.
"""

from pathlib import Path

import pandas as pd

# The folder where the price file lives (project_root/data).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_FILE = DATA_DIR / "prices.csv"


def download_prices(tickers, start, end=None):
    """
    Download daily closing prices from Yahoo Finance.

    auto_adjust=True gives prices adjusted for dividends and stock splits.
    That matters: Apple did a 4-for-1 split in 2020, and without adjusting,
    the price would look like it crashed 75% overnight.

    Returns a table with one row per date and one column per ticker.
    """
    # Imported here (not at the top) so the rest of the project still works
    # on a computer that doesn't have yfinance installed.
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]

    # With a single ticker, yfinance returns a Series instead of a table.
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()

    # Drop rows where every ticker is missing (e.g. a holiday everywhere).
    prices = prices.dropna(how="all")

    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise RuntimeError(f"No data came back for: {missing}. Check the ticker symbols.")

    return prices[tickers]


def save_prices(prices):
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(PRICES_FILE)


def load_prices(tickers, start, refresh=False):
    """
    Load prices from data/prices.csv, downloading them only if needed.

    refresh=True forces a fresh download (use it to pick up new days).
    """
    if PRICES_FILE.exists() and not refresh:
        cached = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
        if set(tickers).issubset(cached.columns):
            return cached[tickers]

    prices = download_prices(tickers, start)
    save_prices(prices)
    return prices


def pair_returns(prices, stock, benchmark):
    """
    Daily returns for one stock and its benchmark, on the days BOTH traded.

    A daily return is the percent change from yesterday's close to today's:
        return_t = price_t / price_(t-1) - 1

    Samsung trades in Korea and Apple in the US. They have different
    holidays, so the combined price table has gaps. Dropping rows where
    either one is missing keeps each comparison apples-to-apples.
    """
    pair = prices[[stock, benchmark]].dropna()
    returns = pair.pct_change().dropna()
    returns.columns = ["stock", "market"]
    return returns

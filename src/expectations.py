"""
expectations.py — what did traders EXPECT before the event?

Stock prices tell us what happened. Options prices tell us what traders
thought MIGHT happen. An option pays off when a stock moves a lot, so when
traders expect big news, options get more expensive. From option prices you
can back out the "implied volatility" (IV): the volatility the market is
pricing in for the next 30 days.

Cboe publishes this for Apple as an index, VXAPL (daily history from Cboe's
website, with Yahoo Finance as a backup). A reading of 30 means the
options market expects Apple to move about 30% (annualized) over the next month.

Three questions:
  1. Expected vs. actual move: from IV the day before, how big a one-day move
     was "normal" to expect, and how big was Apple's actual move?
  2. Did IV fall after the event? Uncertainty gets resolved once the news is
     out, so IV often drops right after a big event ("volatility crush").
  3. Was the options market right? Compare the IV the day before the event
     with the volatility Apple actually had over the following month.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMPLIED_VOL_FILE = DATA_DIR / "implied_vol.csv"


# Cboe publishes the full daily history of its volatility indexes as CSV files.
CBOE_HISTORY_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"


def _from_cboe(symbol):
    table = pd.read_csv(CBOE_HISTORY_URL.format(symbol=symbol))
    table.columns = [c.strip().upper() for c in table.columns]
    # Files have either DATE,OPEN,HIGH,LOW,CLOSE or just DATE,<SYMBOL>.
    value_col = "CLOSE" if "CLOSE" in table.columns else table.columns[-1]
    iv = pd.Series(pd.to_numeric(table[value_col], errors="coerce").to_numpy(),
                   index=pd.to_datetime(table["DATE"]))
    return iv


def _from_yahoo(ticker, start):
    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    iv = raw["Close"]
    if isinstance(iv, pd.DataFrame):
        iv = iv.iloc[:, 0]
    iv.index = pd.to_datetime(iv.index).tz_localize(None)
    return iv


def download_implied_vol(ticker, start):
    """
    Daily implied volatility history. Tries Cboe's own file first (complete
    history), then Yahoo Finance. Keeps whichever source has the most days.
    """
    candidates = []
    for name, fetch in [("Cboe", lambda: _from_cboe(ticker.lstrip("^"))),
                        ("Yahoo", lambda: _from_yahoo(ticker, start))]:
        try:
            series = fetch().dropna()
            candidates.append(series[series.index >= pd.Timestamp(start)])
        except Exception as err:  # noqa: BLE001 - try the next source
            print(f"  {name} implied volatility download failed: {err}")
    candidates = [c for c in candidates if len(c) > 0]
    if not candidates:
        raise RuntimeError(f"No implied volatility data for {ticker}.")
    best = max(candidates, key=len).sort_index()
    if len(best) < 60:
        raise RuntimeError(f"Only {len(best)} days of implied volatility for {ticker}; need history.")
    return best.rename("implied_vol")


def load_implied_vol(ticker, start, refresh=False):
    """Load implied volatility from data/implied_vol.csv, downloading if needed."""
    if IMPLIED_VOL_FILE.exists() and not refresh:
        return pd.read_csv(IMPLIED_VOL_FILE, index_col=0, parse_dates=True)["implied_vol"]
    iv = download_implied_vol(ticker, start)
    DATA_DIR.mkdir(exist_ok=True)
    iv.to_frame().to_csv(IMPLIED_VOL_FILE)
    return iv


def implied_vs_realized(implied_vol, stock_returns, day0_date, horizon_days=21,
                        trading_days=252):
    """
    Compare what options traders expected with what happened.

    implied_vol: index level in percent (e.g. 28.5), one value per day.
    stock_returns: daily returns of the stock.
    horizon_days: about one month of trading days (VXAPL looks 30 calendar days ahead).
    """
    iv = implied_vol.dropna() / 100  # 28.5 -> 0.285
    day0 = pd.Timestamp(day0_date)

    before = iv[iv.index < day0]
    after = iv[iv.index >= day0]
    if before.empty:
        raise ValueError("No implied volatility data before the event.")

    iv_day_before = before.iloc[-1]
    # One "standard" daily move implied by the options market.
    expected_daily_move = iv_day_before / np.sqrt(trading_days)

    ret_after = stock_returns[stock_returns.index >= day0]
    actual_day0 = ret_after.iloc[0] if len(ret_after) else np.nan
    realized_next_month = (ret_after.iloc[:horizon_days].std() * np.sqrt(trading_days)
                           if len(ret_after) >= 5 else np.nan)

    return {
        "iv_30d_before_avg": before.iloc[-30:].mean(),
        "iv_day_before": iv_day_before,
        "iv_day_after": after.iloc[1] if len(after) > 1 else np.nan,
        "iv_30d_after_avg": after.iloc[:30].mean() if len(after) else np.nan,
        "expected_daily_move": expected_daily_move,
        "actual_day0_return": actual_day0,
        "move_vs_expected": abs(actual_day0) / expected_daily_move if pd.notna(actual_day0) else np.nan,
        "realized_vol_next_month": realized_next_month,
        "realized_days": min(len(ret_after), horizon_days),
    }

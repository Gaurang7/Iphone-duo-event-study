"""
event_study.py — measure how much a stock moved because of an event.

THE BIG IDEA
------------
On any given day, a stock moves for two kinds of reasons:
  1. The whole market moved (interest rates, the economy, general mood).
  2. Something specific to the company happened.

An event study separates the two. It answers:
  "On the day of the event, how much did the stock move BEYOND what the
   overall market would have predicted?"
That extra part is the ABNORMAL RETURN.

THE STEPS
---------
1. Estimation window: look at a calm period before the event and learn the
   stock's normal relationship with the market, using a simple regression:

        stock_return = alpha + beta * market_return + noise

   beta  = how strongly the stock follows the market. beta 1.2 means when
           the market rises 1%, the stock usually rises about 1.2%.
   alpha = the stock's typical daily drift on top of that (usually ~0).

2. Event window: for each day around the event, predict the "normal" return
   from that day's market move, then compare it with what actually happened:

        expected return = alpha + beta * market_return
        abnormal return = actual return - expected return

   A second, simpler version is the MARKET-ADJUSTED model, which assumes
   alpha = 0 and beta = 1:  abnormal return = stock return - market return.
   It is cruder, but it can't be fooled by a hot streak: if a stock rose
   0.5% a day before the event, the market model's alpha "expects" that to
   continue, and when it stops, the model reports a large abnormal loss.
   Over long windows we report both, and trust the story only when they agree.

   A third version is a MULTI-FACTOR model, in the spirit of Fama and French.
   Besides the market, it controls for three well-known "styles" that move
   groups of stocks together: size (small vs. large companies), value (cheap
   vs. expensive stocks) and momentum (recent winners vs. losers). If tech
   growth stocks all rallied the week of the launch, the market model would
   credit that to the Duo; the factor model would not.

3. Add up the abnormal returns over a window to get the CUMULATIVE ABNORMAL
   RETURN (CAR). A CAR of -2% over days -1..+1 means the stock did 2% worse
   than the market predicted over those three days.

4. Check significance: is the CAR big compared with the stock's usual
   day-to-day noise? A t-statistic around 2 or more (p-value under 0.05)
   means a move this big would be unusual if the event didn't matter.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MarketModel:
    """The fitted 'normal behaviour' of a stock relative to the market."""

    alpha: float       # average daily return not explained by the market
    beta: float        # sensitivity to the market
    resid_std: float   # typical size of the stock's day-to-day "noise"
    r_squared: float   # share of the stock's moves explained by the market (0 to 1)
    n_obs: int         # number of days used to fit the model
    adj_std: float = float("nan")  # daily noise of (stock - market), for the market-adjusted model
    ff_std: float = float("nan")   # daily noise left over after the multi-factor model
    ff_r_squared: float = float("nan")
    factor_betas: dict = None      # the multi-factor model's coefficients


# The two ways of defining "normal" (see the top of this file).
# name -> (abnormal-return column, CAR column, readable label, noise attribute)
METHODS = {
    "market_model": ("abnormal", "car_from_day0", "Market model", "resid_std"),
    "market_adjusted": ("abnormal_adj", "car_adj_from_day0", "Market-adjusted", "adj_std"),
    "factor_model": ("abnormal_ff", "car_ff_from_day0", "Multi-factor", "ff_std"),
}


@dataclass
class EventStudyResult:
    stock: str
    event_date: pd.Timestamp   # the calendar date of the event
    day0_date: pd.Timestamp    # the first trading day on/after it (day 0)
    model: MarketModel
    daily: pd.DataFrame        # one row per day in the event window


def fit_market_model(stock_returns, market_returns):
    """
    Fit  stock = alpha + beta * market  with ordinary least squares (OLS).

    OLS picks the alpha and beta that make the squared prediction errors as
    small as possible. This is exactly what Excel's SLOPE/INTERCEPT do.
    """
    y = np.asarray(stock_returns, dtype=float)
    x = np.asarray(market_returns, dtype=float)
    n = len(y)
    if n < 30:
        raise ValueError(f"Only {n} days to fit the market model; need at least 30.")

    # Design matrix: a column of 1s (for alpha) and the market returns (for beta).
    X = np.column_stack([np.ones(n), x])
    (alpha, beta), *_ = np.linalg.lstsq(X, y, rcond=None)

    residuals = y - (alpha + beta * x)
    # Divide by n - 2 because we estimated two numbers (alpha and beta).
    resid_std = np.sqrt(np.sum(residuals**2) / (n - 2))
    r_squared = 1 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)

    return MarketModel(float(alpha), float(beta), float(resid_std), float(r_squared), n)


def fit_factor_model(stock_returns, factor_returns):
    """
    Multi-factor version of the market model:

        stock = alpha + b1*market + b2*size + b3*value + b4*momentum + noise

    Same least-squares idea as fit_market_model, just with more columns.
    Returns (coefficients as a dict, noise std, R-squared).
    """
    y = np.asarray(stock_returns, dtype=float)
    F = np.asarray(factor_returns, dtype=float)
    n, k = F.shape
    X = np.column_stack([np.ones(n), F])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coef
    noise = np.sqrt(np.sum(residuals**2) / (n - k - 1))
    r_squared = 1 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)
    names = ["alpha", *factor_returns.columns]
    return dict(zip(names, map(float, coef))), float(noise), float(r_squared)


def find_day0(dates, event_date, offset_days=0):
    """
    Position of day 0: the first trading day on or after the event.

    If the event happened on a weekend, day 0 is the following Monday.
    offset_days shifts the event for markets in other time zones (Korea).
    """
    target = pd.Timestamp(event_date) + pd.Timedelta(days=offset_days)
    pos = int(dates.searchsorted(target))
    if pos >= len(dates):
        raise ValueError(f"No trading data on or after {target.date()} yet.")
    return pos


def run_event_study(returns, event_date, estimation_window, event_window,
                    offset_days=0, stock_name="stock", factors=None):
    """
    Run a full event study for one stock and one event.

    returns: table with columns 'stock' and 'market' (daily returns),
             as produced by data.pair_returns().
    estimation_window / event_window: (start, end) in trading days vs day 0.
    factors: optional table of extra factor returns (size, value, momentum)
             for the multi-factor model, as produced by data.factor_returns().
    """
    if factors is not None:
        # Keep only days where we have the stock, the market and every factor.
        returns = returns.join(factors, how="inner")
    dates = returns.index
    day0 = find_day0(dates, event_date, offset_days)

    # --- Step 1: learn normal behaviour from the estimation window ---------
    est_start, est_end = day0 + estimation_window[0], day0 + estimation_window[1]
    if est_start < 0:
        raise ValueError("Not enough history before the event for the estimation window.")
    est = returns.iloc[est_start:est_end + 1]
    model = fit_market_model(est["stock"], est["market"])
    model.adj_std = float((est["stock"] - est["market"]).std())
    if factors is not None:
        factor_cols = ["market", *factors.columns]
        model.factor_betas, model.ff_std, model.ff_r_squared = fit_factor_model(
            est["stock"], est[factor_cols])

    # --- Step 2: abnormal returns in the event window ----------------------
    # The window may be cut short if the event is recent (no future data yet).
    ev_start = max(day0 + event_window[0], 0)
    ev_end = min(day0 + event_window[1], len(dates) - 1)
    ev = returns.iloc[ev_start:ev_end + 1]

    daily = pd.DataFrame(index=ev.index)
    daily["day"] = np.arange(ev_start - day0, ev_end - day0 + 1)
    daily["actual"] = ev["stock"]
    daily["market"] = ev["market"]
    daily["expected"] = model.alpha + model.beta * ev["market"]
    daily["abnormal"] = daily["actual"] - daily["expected"]
    # CAR path measured from day 0, so the chart reads "since the event".
    # Days before the event count backwards (pre-event drift shows up as
    # the line arriving at day 0 from above or below zero).
    daily["car_from_day0"] = _car_path_from_day0(daily, "abnormal")
    # The market-adjusted version: simply stock minus market.
    daily["abnormal_adj"] = daily["actual"] - daily["market"]
    daily["car_adj_from_day0"] = _car_path_from_day0(daily, "abnormal_adj")
    # The multi-factor version (only when factor data was supplied).
    if factors is not None:
        b = model.factor_betas
        expected_ff = b["alpha"] + sum(b[c] * ev[c] for c in factor_cols)
        daily["abnormal_ff"] = daily["actual"] - expected_ff
        daily["car_ff_from_day0"] = _car_path_from_day0(daily, "abnormal_ff")
    else:
        daily["abnormal_ff"] = np.nan
        daily["car_ff_from_day0"] = np.nan

    return EventStudyResult(
        stock=stock_name,
        event_date=pd.Timestamp(event_date),
        day0_date=dates[day0],
        model=model,
        daily=daily,
    )


def _car_path_from_day0(daily, column):
    """
    Running total of abnormal returns, shifted so the line is at 0 on day -1
    (the last day before the event). Then the value on day 5, for example,
    reads as "the total abnormal return from the event through day 5".
    """
    running_total = daily[column].cumsum()
    before_event = daily["day"] == -1
    anchor = running_total[before_event].iloc[0] if before_event.any() else 0.0
    return running_total - anchor


def car_for_window(result, window, method="market_model"):
    """
    CAR and its significance for one (start, end) window.

    t-statistic = CAR / (noise_per_day * sqrt(number_of_days))
    The sqrt is there because independent daily noise adds up like that:
    over 4 days, typical noise is about 2x one day's, not 4x.
    """
    abnormal_col, _, _, noise_attr = METHODS[method]
    noise = getattr(result.model, noise_attr)
    start, end = window
    d = result.daily
    sub = d[(d["day"] >= start) & (d["day"] <= end)]
    n_days = len(sub)
    expected_days = end - start + 1

    if n_days == 0 or sub[abnormal_col].isna().all():
        return {"window": f"[{start:+d}, {end:+d}]", "days": 0, "complete": False,
                "car": np.nan, "t_stat": np.nan, "p_value": np.nan}

    car = sub[abnormal_col].sum()
    t_stat = car / (noise * np.sqrt(n_days))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=result.model.n_obs - 2))

    return {
        "window": f"[{start:+d}, {end:+d}]",
        "days": n_days,
        "complete": n_days == expected_days,
        "car": car,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def car_table(result, windows, method="market_model"):
    """CAR for several windows at once, as a table."""
    rows = [car_for_window(result, w, method) for w in windows]
    table = pd.DataFrame(rows)
    table.insert(0, "stock", result.stock)
    table.insert(1, "method", METHODS[method][2])
    table["significant_5pct"] = table["p_value"] < 0.05
    return table

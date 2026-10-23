"""
volatility.py — did the stock get calmer or jumpier after the event?

WHAT IS VOLATILITY?
-------------------
Volatility measures how much a stock's price bounces around. Technically it
is the standard deviation of daily returns. We "annualize" it (multiply by
the square root of 252 trading days) so it reads like the numbers traders
quote: "Apple's volatility is 25%" means a typical year's swing is about
plus or minus 25%.

We look at it three ways:
  1. Rolling realized volatility: volatility over the last 21 trading days,
     recalculated every day. Easy to read on a chart.
  2. Before vs. after: volatility in the 30 days before the event vs. the
     30 days after, with a statistical test of whether they differ.
  3. GARCH(1,1): a classic finance model of how volatility evolves. It
     captures "volatility clustering": calm days tend to follow calm days,
     and wild days follow wild days.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats


def annualize(daily_std, trading_days=252):
    return daily_std * np.sqrt(trading_days)


def rolling_volatility(returns, window=21, trading_days=252):
    """Annualized volatility over a rolling window of trading days."""
    return annualize(returns.rolling(window).std(), trading_days)


def volatility_before_after(returns, day0_date, n_days=30, trading_days=252):
    """
    Compare volatility in the n_days before day 0 with the n_days from day 0 on.

    Levene's test asks: "Do these two groups of returns have the same spread?"
    A p-value under 0.05 suggests volatility genuinely changed.
    """
    pos = int(returns.index.searchsorted(pd.Timestamp(day0_date)))
    before = returns.iloc[max(pos - n_days, 0):pos].dropna()
    after = returns.iloc[pos:pos + n_days].dropna()

    result = {
        "days_before": len(before),
        "days_after": len(after),
        "vol_before": annualize(before.std(), trading_days),
        "vol_after": annualize(after.std(), trading_days) if len(after) > 1 else np.nan,
    }
    result["change"] = result["vol_after"] - result["vol_before"]

    if len(after) >= 5:
        result["levene_p_value"] = stats.levene(before, after).pvalue
    else:
        result["levene_p_value"] = np.nan  # too few days after the event to test
    return result


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------
# The model says today's variance (volatility squared) depends on:
#
#   variance_t = omega + alpha * (yesterday's surprise)^2 + beta * variance_(t-1)
#
#   omega: a small constant, the long-run "floor"
#   alpha: how much a big surprise yesterday raises volatility today
#   beta:  how much of yesterday's volatility carries over (memory)
#
# alpha + beta is called PERSISTENCE. Close to 1 means shocks fade slowly.
# The HALF-LIFE, log(0.5) / log(alpha + beta), is how many days it takes for
# half of a volatility shock to wear off.
#
# We fit it by maximum likelihood: try many (omega, alpha, beta) values and
# keep the ones that make the returns we actually saw most probable.
# (Python's `arch` package does the same thing; writing it out makes the
# model easy to explain.)


def _garch_variances(params, r):
    omega, alpha, beta = params
    var = np.empty_like(r)
    var[0] = r.var()  # start at the sample variance
    for t in range(1, len(r)):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
    return var


def _negative_log_likelihood(params, r):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10  # outside the allowed region: return a huge penalty
    var = _garch_variances(params, r)
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(var) + r**2 / var)


def fit_garch(returns, trading_days=252):
    """
    Fit GARCH(1,1) to daily returns.

    Returns a dict with the parameters and a Series of the model's
    annualized volatility for every day.
    """
    clean = returns.dropna()
    # Work in percent (1.5 instead of 0.015): the optimizer behaves better.
    r = (clean - clean.mean()).to_numpy() * 100

    sample_var = r.var()
    start = [sample_var * 0.05, 0.08, 0.90]  # typical values for stocks
    best = optimize.minimize(
        _negative_log_likelihood, start, args=(r,), method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )
    omega, alpha, beta = best.x
    persistence = alpha + beta

    variances = _garch_variances(best.x, r)
    vol = pd.Series(annualize(np.sqrt(variances) / 100, trading_days), index=clean.index)

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "half_life_days": np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.nan,
        "long_run_vol": annualize(np.sqrt(omega / (1 - persistence)) / 100, trading_days),
        "converged": bool(best.success),
        "conditional_vol": vol,
    }

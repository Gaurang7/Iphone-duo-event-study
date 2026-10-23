"""
Tests: check the math on fake data where we KNOW the right answer.

We invent a stock with beta = 1.3 and plant a -3% surprise on the event
day. If the code is correct, it should find beta near 1.3 and an abnormal
return near -3% on day 0.

Run with:  python -m pytest tests     (or simply: python tests/test_analysis.py)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.event_study import car_table, fit_market_model, run_event_study  # noqa: E402
from src.volatility import fit_garch, volatility_before_after  # noqa: E402

TRUE_BETA = 1.3
SHOCK = -0.03
EVENT = "2026-09-09"


def make_fake_returns(seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-06-02", "2026-10-23")
    market = rng.normal(0.0004, 0.01, len(dates))
    noise = rng.normal(0, 0.008, len(dates))
    stock = 0.0002 + TRUE_BETA * market + noise
    returns = pd.DataFrame({"stock": stock, "market": market}, index=dates)
    returns.loc[pd.Timestamp(EVENT), "stock"] += SHOCK
    return returns


def test_market_model_recovers_beta():
    r = make_fake_returns()
    model = fit_market_model(r["stock"], r["market"])
    assert abs(model.beta - TRUE_BETA) < 0.1
    assert 0 < model.r_squared < 1


def test_event_study_finds_planted_shock():
    r = make_fake_returns()
    result = run_event_study(r, EVENT, (-150, -31), (-10, 30))
    day0 = result.daily[result.daily["day"] == 0]
    assert result.day0_date == pd.Timestamp(EVENT)
    assert abs(day0["abnormal"].iloc[0] - SHOCK) < 0.02
    # The CAR path is anchored at zero on the day before the event.
    assert result.daily.loc[result.daily["day"] == -1, "car_from_day0"].iloc[0] == 0


def test_event_day_shock_is_significant():
    r = make_fake_returns()
    result = run_event_study(r, EVENT, (-150, -31), (-10, 30))
    table = car_table(result, [(0, 0)])
    assert table["t_stat"].iloc[0] < -2
    assert table["significant_5pct"].iloc[0]


def test_weekend_event_rolls_to_monday():
    r = make_fake_returns()
    result = run_event_study(r, "2026-09-12", (-150, -31), (-10, 30))  # a Saturday
    assert result.day0_date == pd.Timestamp("2026-09-14")


def test_recent_event_window_is_marked_incomplete():
    r = make_fake_returns()
    result = run_event_study(r, "2026-10-21", (-150, -31), (-10, 30))
    table = car_table(result, [(0, 30)])
    assert not table["complete"].iloc[0]
    assert table["days"].iloc[0] == 3


def test_volatility_before_after_detects_jump():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2026-01-01", periods=120)
    calm, wild = rng.normal(0, 0.01, 60), rng.normal(0, 0.03, 60)
    returns = pd.Series(np.concatenate([calm, wild]), index=dates)
    out = volatility_before_after(returns, dates[60], n_days=30)
    assert out["vol_after"] > out["vol_before"]
    assert out["levene_p_value"] < 0.05


def test_garch_recovers_reasonable_parameters():
    # Simulate a GARCH(1,1) with known parameters and fit it back.
    rng = np.random.default_rng(2)
    omega, alpha, beta, n = 0.05, 0.10, 0.85, 3000
    var, r = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        r[t] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    fit = fit_garch(pd.Series(r / 100, index=pd.bdate_range("2010-01-01", periods=n)))
    assert abs(fit["alpha"] - alpha) < 0.05
    assert abs(fit["beta"] - beta) < 0.07
    assert fit["persistence"] < 1


if __name__ == "__main__":
    tests = [obj for name, obj in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"passed  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")

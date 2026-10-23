"""
config.py — every setting for the project lives here.

If you want to add a stock, add an event, or change a window, this is the
only file you need to edit. Everything else reads from these values.
"""

# ---------------------------------------------------------------------------
# 1. How much price history to download
# ---------------------------------------------------------------------------
# We go back to 2017 so we can compare the Duo with the iPhone X launch.
START_DATE = "2017-01-01"


# ---------------------------------------------------------------------------
# 2. The events we study
# ---------------------------------------------------------------------------
# Each event is a date when the market learned something new about Apple.
# Apple's launch events start at 10am Pacific (1pm Eastern), so US stocks
# can react the same day. That makes the event date "day 0".
EVENTS = {
    "duo_announce": {
        "date": "2026-09-09",
        "label": "iPhone Duo announced",
    },
    "duo_sales": {
        "date": "2026-10-23",
        "label": "iPhone Duo goes on sale",
    },
    "iphone_17": {
        "date": "2025-09-09",
        "label": "iPhone 17 / iPhone Air announced",
    },
    "iphone_16": {
        "date": "2024-09-09",
        "label": "iPhone 16 announced",
    },
    "vision_pro": {
        "date": "2023-06-05",
        "label": "Vision Pro announced (WWDC)",
    },
    "iphone_x": {
        "date": "2017-09-12",
        "label": "iPhone X announced",
    },
}

# The event the whole project is about.
MAIN_EVENT = "duo_announce"

# Past Apple launches we compare the Duo against (Apple stock only).
COMPARISON_EVENTS = ["duo_announce", "iphone_17", "iphone_16", "vision_pro", "iphone_x"]


# ---------------------------------------------------------------------------
# 3. The stocks we study, and the "market" each one is compared with
# ---------------------------------------------------------------------------
# benchmark = the index that represents "the whole market" for that stock.
#   SPY   = an ETF that tracks the S&P 500 (US market)
#   ^KS11 = the KOSPI index (Korean market), because Samsung trades in Seoul
#
# reaction_offset_days: Apple's event ends around 2am the next day in Korea,
# so Korean stocks can only react one calendar day later.
STOCKS = {
    "AAPL": {
        "name": "Apple",
        "role": "The company launching the Duo",
        "benchmark": "SPY",
        "reaction_offset_days": 0,
    },
    "QCOM": {
        "name": "Qualcomm",
        "role": "Supplier: the Duo uses Apple's own C2 modem instead of Qualcomm's",
        "benchmark": "SPY",
        "reaction_offset_days": 0,
    },
    "MU": {
        "name": "Micron",
        "role": "Supplier: memory chips; its CEO appeared at the launch",
        "benchmark": "SPY",
        "reaction_offset_days": 0,
    },
    "005930.KS": {
        "name": "Samsung Electronics",
        "role": "Competitor: the leading foldable phone maker",
        "benchmark": "^KS11",
        "reaction_offset_days": 1,
    },
}


# ---------------------------------------------------------------------------
# 4. Event-study windows (measured in TRADING days relative to day 0)
# ---------------------------------------------------------------------------
# Estimation window: a "normal" period well before the event, used to learn
# how the stock usually moves with the market. Days -150 to -31 = 120 days.
# We stop at -31 so rumours right before the event don't distort "normal".
ESTIMATION_WINDOW = (-150, -31)

# The stretch of days around the event we look at day by day.
EVENT_WINDOW = (-10, 30)

# The windows we report a cumulative abnormal return (CAR) for.
#   (0, 0)   = just the event day
#   (-1, 1)  = the day before, the day of, and the day after
#   (-5, 5)  = a week either side (catches leaks and slow reactions)
#   (0, 30)  = the six weeks after (did the effect last or fade?)
CAR_WINDOWS = [(0, 0), (-1, 1), (-5, 5), (0, 10), (0, 30)]


# ---------------------------------------------------------------------------
# 5. Volatility settings
# ---------------------------------------------------------------------------
# Number of trading days before and after the event to compare.
VOL_COMPARE_DAYS = 30

# Rolling window for "realized volatility" (21 trading days = about 1 month).
ROLLING_VOL_DAYS = 21

# Trading days in a year, used to turn daily volatility into yearly.
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# 6. Extra data
# ---------------------------------------------------------------------------
# ETFs used to build the size, value and momentum factors (see src/data.py).
# The multi-factor model applies to US stocks only.
FACTOR_ETFS = ["IWM", "IWD", "IWF", "MTUM"]
FACTOR_MODEL_STOCKS = ["AAPL", "QCOM", "MU"]

# Cboe's implied volatility index for Apple: the volatility that Apple's
# options prices imply for the next 30 days, i.e. what traders EXPECT.
IMPLIED_VOL_TICKER = "^VXAPL"

# Wikipedia articles whose daily page views measure public attention.
ATTENTION_ARTICLES = {
    "Foldable_smartphone": "Foldable smartphone",
    "IPhone": "iPhone",
}
ATTENTION_START = "2026-05-01"


def all_tickers():
    """Every ticker we need to download: stocks, benchmarks and factor ETFs."""
    tickers = list(STOCKS)
    for info in STOCKS.values():
        if info["benchmark"] not in tickers:
            tickers.append(info["benchmark"])
    return tickers + [t for t in FACTOR_ETFS if t not in tickers]

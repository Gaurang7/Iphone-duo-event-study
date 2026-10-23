"""
app.py — the interactive dashboard.

Run it on your computer with:
    streamlit run app.py
It opens in your browser. Streamlit turns this Python script into a web page:
every st.something(...) call below draws one piece of the page.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.charts import COLORS, HIGHLIGHT, MUTED
from src.data import PRICES_FILE, download_prices, pair_returns, save_prices
from src.event_study import car_table, run_event_study
from src.volatility import fit_garch, rolling_volatility, volatility_before_after

st.set_page_config(page_title="iPhone Duo Event Study", layout="wide")

NAMES = {t: info["name"] for t, info in config.STOCKS.items()}
EVENT_LABELS = {k: v["label"] for k, v in config.EVENTS.items()}
WINDOW_NAMES = {
    "[+0, +0]": "Event day",
    "[-1, +1]": "Days -1 to +1",
    "[-5, +5]": "Days -5 to +5",
    "[+0, +10]": "Days 0 to +10",
    "[+0, +30]": "Days 0 to +30",
}


# ---------------------------------------------------------------------------
# Loading data (cached, so it only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading prices...")
def get_prices(force_download=False):
    if PRICES_FILE.exists() and not force_download:
        return pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
    prices = download_prices(config.all_tickers(), config.START_DATE)
    save_prices(prices)
    return prices


@st.cache_data
def get_study(_prices, last_date, ticker, event_key):
    """Event study for one stock/event. last_date is only there to refresh the cache."""
    info = config.STOCKS[ticker]
    returns = pair_returns(_prices, ticker, info["benchmark"])
    return run_event_study(returns, config.EVENTS[event_key]["date"],
                           config.ESTIMATION_WINDOW, config.EVENT_WINDOW,
                           offset_days=info["reaction_offset_days"], stock_name=ticker)


def car_chart(lines, title):
    """lines: list of (name, daily_table, color, width). One line per series."""
    fig = go.Figure()
    for name, daily, color, width in lines:
        fig.add_trace(go.Scatter(
            x=daily["day"], y=daily["car_from_day0"], name=name, mode="lines",
            line={"color": color, "width": width},
            hovertemplate="Day %{x}: %{y:+.2%}<extra>" + name + "</extra>",
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="#52514e",
                  annotation_text="Event", annotation_position="top")
    fig.add_hline(y=0, line_color="#52514e", line_width=1)
    fig.update_layout(title=title, hovermode="x unified", height=430,
                      yaxis_tickformat=".0%", xaxis_title="Trading days relative to the event",
                      yaxis_title="Cumulative abnormal return", margin={"t": 60, "b": 40},
                      legend={"orientation": "h", "y": -0.2})
    return fig


def format_car_table(table):
    out = table.copy()
    out["Window"] = out["window"].map(WINDOW_NAMES).fillna(out["window"])
    out["Abnormal return"] = out["car"].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "n/a")
    out["t-stat"] = out["t_stat"].round(2)
    out["p-value"] = out["p_value"].round(3)
    out["Significant (5%)"] = out["significant_5pct"].map({True: "Yes", False: "No"})
    out["Days of data"] = out["days"]
    return out


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("How did the market react to Apple's iPhone Duo?")
st.caption(
    "An event study of Apple's first foldable iPhone, announced September 9, 2026: "
    "how much Apple, its suppliers, and a competitor moved beyond what the overall market "
    "predicted, and whether volatility changed afterward."
)

with st.sidebar:
    st.header("Settings")
    event_key = st.selectbox("Event", list(config.EVENTS), format_func=EVENT_LABELS.get,
                             index=list(config.EVENTS).index(config.MAIN_EVENT))
    tickers = st.multiselect("Companies", list(config.STOCKS), default=list(config.STOCKS),
                             format_func=NAMES.get)
    refresh = st.button("Download latest prices")
    st.caption("Prices: Yahoo Finance, adjusted for splits and dividends.")

if refresh:
    get_prices.clear()  # forget the cached prices so we download fresh ones
try:
    prices = get_prices(force_download=refresh)
except Exception as err:  # e.g. no internet and no saved file
    st.error(f"Couldn't load prices: {err}. Run `python run_analysis.py --refresh` first.")
    st.stop()

last_date = str(prices.index.max().date())
st.caption(f"Data through **{last_date}**.")

if not tickers:
    st.info("Pick at least one company in the sidebar.")
    st.stop()

studies = {}
for t in tickers:
    try:
        studies[t] = get_study(prices, last_date, t, event_key)
    except ValueError as err:
        st.warning(f"{NAMES[t]}: {err}")

tab_reaction, tab_launches, tab_vol, tab_method = st.tabs(
    ["Market reaction", "Duo vs. past launches", "Volatility", "How it works"])

# --- Tab 1: market reaction -------------------------------------------------
with tab_reaction:
    if studies:
        cols = st.columns(len(studies))
        for col, (t, result) in zip(cols, studies.items()):
            row = car_table(result, [(-1, 1)]).iloc[0]
            col.metric(f"{NAMES[t]}, days -1 to +1",
                       f"{row['car']:+.2%}" if pd.notna(row["car"]) else "n/a",
                       help=config.STOCKS[t]["role"])

        lines = [(NAMES[t], r.daily, COLORS.get(t, MUTED), 2.5) for t, r in studies.items()]
        st.plotly_chart(car_chart(lines, f"Cumulative abnormal return: {EVENT_LABELS[event_key]}"),
                        use_container_width=True)

        table = pd.concat([car_table(r, config.CAR_WINDOWS) for r in studies.values()])
        table["Company"] = table["stock"].map(NAMES)
        st.dataframe(format_car_table(table)[["Company", "Window", "Abnormal return", "t-stat",
                                              "p-value", "Significant (5%)", "Days of data"]],
                     hide_index=True, use_container_width=True)

        with st.expander("Each company's normal relationship with its market"):
            st.dataframe(pd.DataFrame([
                {"Company": NAMES[t], "Market": config.STOCKS[t]["benchmark"],
                 "Beta": round(r.model.beta, 2), "R-squared": round(r.model.r_squared, 2),
                 "Day 0": r.day0_date.date()}
                for t, r in studies.items()]), hide_index=True)

# --- Tab 2: Apple's launches --------------------------------------------------
with tab_launches:
    launch_studies = {k: get_study(prices, last_date, "AAPL", k) for k in config.COMPARISON_EVENTS}
    lines = [(EVENT_LABELS[k], r.daily, HIGHLIGHT if k == config.MAIN_EVENT else MUTED,
              3 if k == config.MAIN_EVENT else 1.5) for k, r in launch_studies.items()]
    st.plotly_chart(car_chart(lines, "Apple's abnormal return around each launch"),
                    use_container_width=True)

    rows = []
    for k, r in launch_studies.items():
        t = car_table(r, [(-1, 1), (0, 30)]).set_index("window")
        rows.append({"Launch": EVENT_LABELS[k], "Date": config.EVENTS[k]["date"],
                     "Days -1 to +1": f"{t.loc['[-1, +1]', 'car']:+.2%}",
                     "t-stat": round(t.loc["[-1, +1]", "t_stat"], 2),
                     "Days 0 to +30": f"{t.loc['[+0, +30]', 'car']:+.2%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Grey lines are past launches; the blue line is the iPhone Duo.")

# --- Tab 3: volatility --------------------------------------------------------
with tab_vol:
    vol_ticker = st.selectbox("Company", tickers, format_func=NAMES.get, key="vol_ticker")
    if vol_ticker in studies:
        stock_returns = pair_returns(prices, vol_ticker, config.STOCKS[vol_ticker]["benchmark"])["stock"]
        day0 = studies[vol_ticker].day0_date
        ba = volatility_before_after(stock_returns, day0, config.VOL_COMPARE_DAYS)

        c1, c2, c3 = st.columns(3)
        c1.metric("Volatility, 30 days before", f"{ba['vol_before']:.1%}")
        c2.metric("Volatility, 30 days after",
                  f"{ba['vol_after']:.1%}" if pd.notna(ba["vol_after"]) else "n/a",
                  f"{ba['change']:+.1%}" if pd.notna(ba["change"]) else None, delta_color="inverse")
        c3.metric("Levene test p-value",
                  f"{ba['levene_p_value']:.3f}" if pd.notna(ba["levene_p_value"]) else "n/a",
                  help="Below 0.05 suggests volatility genuinely changed.")

        recent = stock_returns[stock_returns.index >= stock_returns.index.max() - pd.DateOffset(years=3)]
        garch = fit_garch(recent)
        rolling = rolling_volatility(stock_returns, config.ROLLING_VOL_DAYS)
        start = day0 - pd.DateOffset(months=9)
        r, g = rolling[rolling.index >= start], garch["conditional_vol"]
        g = g[g.index >= start]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r.index, y=r, name="21-day realized volatility",
                                 line={"color": HIGHLIGHT, "width": 2.5}))
        fig.add_trace(go.Scatter(x=g.index, y=g, name="GARCH(1,1) volatility",
                                 line={"color": COLORS["QCOM"], "width": 1.5}))
        fig.add_vline(x=day0, line_dash="dash", line_color="#52514e")
        fig.update_layout(title=f"{NAMES[vol_ticker]}: annualized volatility", height=420,
                          hovermode="x unified", yaxis_tickformat=".0%", yaxis_rangemode="tozero",
                          legend={"orientation": "h", "y": -0.15}, margin={"t": 60})
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"GARCH persistence {garch['persistence']:.3f}: a volatility shock loses half its "
            f"effect in about {garch['half_life_days']:.0f} trading days."
        )

# --- Tab 4: method ------------------------------------------------------------
with tab_method:
    st.markdown(
        """
**The question.** Did the iPhone Duo announcement move these stocks, beyond what the
overall market was doing that day?

**1. Learn "normal."** For each company, I fit a market model over 120 trading days
well before the event (days -150 to -31):
`stock return = alpha + beta × market return`.
Beta measures how strongly the stock follows the market.

**2. Measure the surprise.** Around the event, the *expected* return is what the market
model predicts from that day's market move. The *abnormal return* is actual minus expected.

**3. Add it up.** The cumulative abnormal return (CAR) sums abnormal returns over a window,
for example the day before through the day after.

**4. Test it.** The t-statistic compares the CAR with the stock's normal daily noise.
Roughly, |t| above 2 means the move would be unusual if the event didn't matter.

**5. Volatility.** I compare volatility 30 days before vs. after (Levene's test) and fit a
GARCH(1,1) model, which captures how calm and turbulent periods cluster.

**Caveats.** Other news overlaps with the event: John Ternus became CEO on September 1, 2026,
and the iPhone 18 Pro launched at the same event. An event study measures the reaction to
*everything* the market learned that day, not the Duo alone. Samsung is compared with the
Korean market (KOSPI) and its day 0 is shifted one day, because the event ended overnight in Seoul.
"""
    )

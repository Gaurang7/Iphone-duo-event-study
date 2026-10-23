"""
run_analysis.py — run the whole project from start to finish.

    python run_analysis.py            # use saved prices if we have them
    python run_analysis.py --refresh  # download the latest prices first

What it does:
  1. Loads daily prices for Apple, Qualcomm, Micron, Samsung and their markets.
  2. Event study of the iPhone Duo announcement for every stock.
  3. Compares Apple's reaction to the Duo with past launches.
  4. Volatility before vs. after, plus a GARCH(1,1) model.
  5. Saves tables to results/, charts to figures/, and rewrites the
     "Results" section of README.md with the latest numbers.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

import config
from src import charts
from src.data import load_prices, pair_returns
from src.event_study import METHODS, car_table, run_event_study
from src.volatility import fit_garch, rolling_volatility, volatility_before_after

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
README = ROOT / "README.md"


def study(prices, ticker, event_key):
    """Run the event study for one stock and one event from config."""
    info = config.STOCKS[ticker]
    returns = pair_returns(prices, ticker, info["benchmark"])
    return run_event_study(
        returns,
        config.EVENTS[event_key]["date"],
        config.ESTIMATION_WINDOW,
        config.EVENT_WINDOW,
        offset_days=info["reaction_offset_days"],
        stock_name=ticker,
    )


def main(refresh=False):
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    names = {t: info["name"] for t, info in config.STOCKS.items()}

    # 1. Prices ----------------------------------------------------------------
    prices = load_prices(config.all_tickers(), config.START_DATE, refresh=refresh)
    last_date = prices.index.max().date()
    print(f"Prices loaded: {prices.index.min().date()} to {last_date}")

    # 2. The Duo announcement, every stock -------------------------------------
    duo = {t: study(prices, t, config.MAIN_EVENT) for t in config.STOCKS}
    # Both definitions of "normal", so readers can see whether they agree.
    duo_table = pd.concat([car_table(r, config.CAR_WINDOWS, method)
                           for r in duo.values() for method in METHODS])
    duo_table.insert(1, "company", duo_table["stock"].map(names))
    duo_table.to_csv(RESULTS / "duo_car_by_stock.csv", index=False)

    models = pd.DataFrame([
        {"stock": t, "company": names[t], "benchmark": config.STOCKS[t]["benchmark"],
         "alpha": r.model.alpha, "beta": r.model.beta, "r_squared": r.model.r_squared,
         "estimation_days": r.model.n_obs, "day0": r.day0_date.date()}
        for t, r in duo.items()
    ])
    models.to_csv(RESULTS / "market_models.csv", index=False)

    charts.plot_car_paths(
        duo, names, FIGURES / "duo_car_by_stock.png",
        "How the market reacted to the iPhone Duo announcement",
        "Cumulative return of each stock minus its market, from the event onward",
        column="car_adj_from_day0",
    )

    # 3. Apple: the Duo vs. past launches --------------------------------------
    launches = {k: study(prices, "AAPL", k) for k in config.COMPARISON_EVENTS}
    labels = {k: config.EVENTS[k]["label"] for k in launches}
    comparison = pd.concat([
        car_table(r, config.CAR_WINDOWS, method).assign(event=labels[k], date=config.EVENTS[k]["date"])
        for k, r in launches.items() for method in METHODS
    ])
    comparison.to_csv(RESULTS / "aapl_launch_comparison.csv", index=False)
    charts.plot_launch_comparison(launches, labels, config.MAIN_EVENT,
                                  FIGURES / "aapl_launch_comparison.png", column="car_adj_from_day0",
                                  subtitle="Apple's cumulative return minus the S&P 500's, from the event onward")

    # 4. Volatility --------------------------------------------------------------
    vol_rows = []
    for t, r in duo.items():
        stock_returns = pair_returns(prices, t, config.STOCKS[t]["benchmark"])["stock"]
        row = volatility_before_after(stock_returns, r.day0_date, config.VOL_COMPARE_DAYS,
                                      config.TRADING_DAYS_PER_YEAR)
        vol_rows.append({"stock": t, "company": names[t], **row})
    vol_table = pd.DataFrame(vol_rows)
    vol_table.to_csv(RESULTS / "volatility_before_after.csv", index=False)

    # GARCH for Apple, fit on the last ~3 years so old regimes don't dominate.
    aapl_returns = prices["AAPL"].dropna().pct_change().dropna()
    recent = aapl_returns[aapl_returns.index >= aapl_returns.index.max() - pd.DateOffset(years=3)]
    garch = fit_garch(recent, config.TRADING_DAYS_PER_YEAR)
    pd.DataFrame([{k: v for k, v in garch.items() if k != "conditional_vol"}]).to_csv(
        RESULTS / "aapl_garch.csv", index=False)

    rolling = rolling_volatility(aapl_returns, config.ROLLING_VOL_DAYS, config.TRADING_DAYS_PER_YEAR)
    charts.plot_volatility(rolling, garch["conditional_vol"], duo["AAPL"].day0_date,
                           FIGURES / "aapl_volatility.png",
                           start=duo["AAPL"].day0_date - pd.DateOffset(months=9))

    # 5. Write the findings into the README ------------------------------------
    summary = results_markdown(duo_table, comparison, vol_table, garch, last_date)
    (RESULTS / "summary.md").write_text(summary)
    update_readme(summary)
    print(summary)


# ---------------------------------------------------------------------------
# Turning the numbers into readable text
# ---------------------------------------------------------------------------

def pct(x):
    return "n/a" if pd.isna(x) else f"{x:+.2%}"


def stars(p):
    """Common shorthand: *** p<0.01, ** p<0.05, * p<0.10."""
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def results_markdown(duo_table, comparison, vol_table, garch, last_date):
    lines = [f"*Auto-generated by `run_analysis.py` with prices through **{last_date}**.*", ""]

    lines += ["#### Cumulative abnormal return after the Duo announcement", "",
              "| Company | Method | Event day | Days -1 to +1 | Days -5 to +5 | Days 0 to +30 |",
              "|---|---|---|---|---|---|"]
    for (stock, method), g in duo_table.groupby(["stock", "method"], sort=False):
        cells = []
        for w in ["[+0, +0]", "[-1, +1]", "[-5, +5]", "[+0, +30]"]:
            row = g[g["window"] == w].iloc[0]
            cell = f"{pct(row['car'])}{stars(row['p_value'])}"
            if not row["complete"] and row["days"] > 0:
                cell += f" ({row['days']}d so far)"
            cells.append(cell)
        company = g["company"].iloc[0] if method == "Market model" else ""
        lines.append(f"| {company} | {method} | " + " | ".join(cells) + " |")
    lines += ["",
              "Stars mark statistical significance: \\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.10.",
              "",
              "**Two definitions of \"normal.\"** The *market model* predicts each stock from its market "
              "using a fitted alpha and beta. The *market-adjusted* version is simply the stock's return "
              "minus its market's. Over a few days they agree closely. Over 30 days they can differ, because "
              "the market model's alpha assumes a stock's pre-event trend continues: a stock on a hot "
              "streak (like Micron) then shows a large \"abnormal\" loss just for cooling off. "
              "Conclusions here rely only on results where both methods agree.", ""]

    mm = comparison[comparison["method"] == "Market model"]
    ma = comparison[comparison["method"] == "Market-adjusted"]
    lines += ["#### Apple: the Duo vs. past launches", "",
              "| Launch | Date | Days -1 to +1 | t-stat | Days 0 to +30 (market-adjusted) |",
              "|---|---|---|---|---|"]
    for (_, row), (_, row30) in zip(mm[mm["window"] == "[-1, +1]"].iterrows(),
                                    ma[ma["window"] == "[+0, +30]"].iterrows()):
        lines.append(f"| {row['event']} | {row['date']} | {pct(row['car'])}{stars(row['p_value'])} "
                     f"| {row['t_stat']:.2f} | {pct(row30['car'])}{stars(row30['p_value'])} |")
    lines.append("")

    lines += ["#### Volatility, 30 trading days before vs. after (annualized)", "",
              "| Company | Before | After | Change | Levene p-value |", "|---|---|---|---|---|"]
    for _, row in vol_table.iterrows():
        p = "n/a" if pd.isna(row["levene_p_value"]) else f"{row['levene_p_value']:.3f}"
        lines.append(f"| {row['company']} | {row['vol_before']:.1%} | {row['vol_after']:.1%} "
                     f"| {row['change']:+.1%} | {p} |")
    lines += ["",
              f"Apple GARCH(1,1): alpha = {garch['alpha']:.3f}, beta = {garch['beta']:.3f}, "
              f"persistence = {garch['persistence']:.3f} (a volatility shock loses half its effect "
              f"in about {garch['half_life_days']:.0f} trading days).", ""]
    return "\n".join(lines)


def update_readme(summary):
    """Replace the text between the RESULTS markers in README.md."""
    if not README.exists():
        return
    text = README.read_text()
    pattern = r"(<!-- RESULTS:START -->)(.*?)(<!-- RESULTS:END -->)"
    if re.search(pattern, text, flags=re.S):
        new = re.sub(pattern, lambda m: f"{m.group(1)}\n{summary}\n{m.group(3)}", text, flags=re.S)
        README.write_text(new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="download the latest prices first")
    main(refresh=parser.parse_args().refresh)

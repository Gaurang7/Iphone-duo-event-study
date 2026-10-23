# How did the market react to Apple's iPhone Duo?

An event study of Apple's first foldable iPhone. On **September 9, 2026**, Apple announced the
iPhone Duo; it went on sale on **October 23, 2026**. This project measures how much Apple, two of
its suppliers, and its main foldable competitor moved **beyond what the overall market
predicted**, compares the reaction with Apple's past launches, and checks whether volatility
changed afterward.

**Live dashboard:** *(add your Streamlit link here)*

![Cumulative abnormal returns after the iPhone Duo announcement](figures/duo_car_by_stock.png)

## The question

A launch is news. If investors thought the Duo changed Apple's prospects, it should show up in
the stock price. But stocks also move every day for reasons that have nothing to do with Apple.
An event study separates the two by asking: *how much did each stock move beyond what the market
alone would have predicted?*

| Company | Why it's included |
|---|---|
| **Apple** (AAPL) | The company launching the Duo |
| **Qualcomm** (QCOM) | The Duo uses Apple's own C2 modem instead of Qualcomm's chips |
| **Micron** (MU) | Memory supplier; its CEO appeared at the launch, which is unusual for Apple |
| **Samsung Electronics** (005930.KS) | The leading foldable phone maker and the Duo's main rival |

## Results

<!-- RESULTS:START -->
*Results appear here automatically after the first run of `run_analysis.py`.*
<!-- RESULTS:END -->

### The Duo vs. Apple's past launches

![Apple's abnormal return after each product launch](figures/aapl_launch_comparison.png)

### Volatility

![Apple's volatility before and after the announcement](figures/aapl_volatility.png)

## Method

1. **Learn "normal."** For each stock, fit a market model over 120 trading days well before the
   event (days −150 to −31): `stock return = α + β × market return`. US stocks are compared with
   the S&P 500 (SPY); Samsung with the Korean market (KOSPI).
2. **Abnormal returns.** Around the event, abnormal return = actual return − (α + β × market return).
3. **Cumulative abnormal return (CAR).** Sum abnormal returns over windows such as days −1 to +1
   and 0 to +30, and test significance with a t-statistic based on the stock's normal daily noise.
4. **Volatility.** Compare annualized volatility 30 trading days before vs. after (Levene's test),
   and fit a GARCH(1,1) model by maximum likelihood to see how persistent volatility shocks are.

The math is checked against simulated data with known answers in `tests/`.

## Limitations

- **Overlapping news.** John Ternus became CEO on September 1, 2026, and the iPhone 18 Pro was
  announced at the same event. The event study captures the reaction to everything the market
  learned that day, not the Duo alone.
- **Expectations.** A foldable iPhone was rumored for years, so much of the news may have been
  priced in before the announcement. Small abnormal returns can mean "expected," not "unimportant."
- **Sales data is new.** The Duo went on sale October 23, 2026, so the post-launch window is still
  filling in. A GitHub Action re-runs the analysis every weekday, so the numbers update over time.
- **Time zones.** Apple's event ended overnight in Seoul, so Samsung's day 0 is the next Korean
  trading day.

## Run it yourself

```bash
git clone https://github.com/Gaurang7/iphone-duo-event-study.git
cd iphone-duo-event-study
pip install -r requirements.txt

python run_analysis.py --refresh   # download prices, run everything, update this README
streamlit run app.py               # open the interactive dashboard
python -m pytest tests             # check the math on simulated data
```

## Project structure

```
config.py              every setting: stocks, events, windows
run_analysis.py        runs the whole project and writes results/ and figures/
app.py                 interactive Streamlit dashboard
src/data.py            downloads and caches prices
src/event_study.py     market model, abnormal returns, CAR, t-tests
src/volatility.py      realized volatility, before/after test, GARCH(1,1)
src/charts.py          the static charts in this README
tests/                 checks the math against simulated data
.github/workflows/     refreshes the data and results every weekday
```

## Next steps

- Add options-implied volatility to see what the market *expected* before the event.
- Use a multi-factor model (Fama-French) instead of the single-market model.
- Add Google Trends interest in "foldable phone" as a non-price signal.

---

Built by **Gaurang Agarwal** · MS in Quantitative Management, Duke University (Fuqua)

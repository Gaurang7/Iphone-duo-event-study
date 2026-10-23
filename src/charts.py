"""
charts.py — the static charts saved to figures/ and shown in the README.

(The dashboard in app.py draws interactive versions of the same charts.)
"""

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402

# One consistent, colorblind-checked palette across every chart.
# Colors follow the company, never its rank, so Apple is always blue.
COLORS = {
    "AAPL": "#2a78d6",       # blue
    "QCOM": "#eb6834",       # orange
    "MU": "#1baf7a",         # aqua
    "005930.KS": "#eda100",  # yellow
}
HIGHLIGHT = "#2a78d6"
MUTED = "#b4b2ab"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"


def _style(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.set_title(title, loc="left", fontsize=13, color=TEXT, fontweight="bold", pad=24)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=TEXT_SECONDARY)


def _event_line(ax, label="Event (day 0)"):
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.text(0.3, ax.get_ylim()[1], label, fontsize=8.5, color=TEXT_SECONDARY, va="top")


def _spread(values, min_gap):
    """Nudge end-of-line label positions apart so they never overlap."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed


def _end_labels(ax, ends):
    """ends: list of (x, y, text, style dict). Draw labels at the line ends, de-overlapped."""
    low, high = ax.get_ylim()
    ys = _spread([y for _, y, _, _ in ends], (high - low) * 0.045)
    for (x, y, text, style), label_y in zip(ends, ys):
        ax.annotate(text, (x, y), xytext=(x, label_y), textcoords="data", fontsize=8.5,
                    va="center", annotation_clip=False, **style)


def plot_car_paths(results, names, path, title, subtitle):
    """One line per stock: cumulative abnormal return around the event."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ends = []
    for ticker, result in results.items():
        d = result.daily
        ax.plot(d["day"], d["car_from_day0"], color=COLORS.get(ticker, MUTED), linewidth=2,
                label=names[ticker])
        # Direct label at the end of each line, in text color (the line carries the color).
        last = d.dropna(subset=["car_from_day0"]).iloc[-1]
        ends.append((last["day"] + 0.8, last["car_from_day0"],
                     f"{names[ticker]} {last['car_from_day0']:+.1%}", {"color": TEXT}))
    _style(ax, title, subtitle)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Trading days relative to the event", color=TEXT_SECONDARY, fontsize=9)
    _event_line(ax)
    ax.margins(x=0.02)
    _end_labels(ax, ends)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=TEXT)
    fig.subplots_adjust(right=0.8)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_launch_comparison(results, labels, highlight_key, path):
    """Apple's CAR after each launch; the Duo highlighted, the rest muted."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ends = []
    for key, result in results.items():
        d = result.daily
        is_main = key == highlight_key
        ax.plot(d["day"], d["car_from_day0"], color=HIGHLIGHT if is_main else MUTED,
                linewidth=2.5 if is_main else 1.5, zorder=3 if is_main else 2)
        last = d.dropna(subset=["car_from_day0"]).iloc[-1]
        ends.append((last["day"] + 0.8, last["car_from_day0"],
                     f"{labels[key]} {last['car_from_day0']:+.1%}",
                     {"color": TEXT if is_main else TEXT_SECONDARY,
                      "fontweight": "bold" if is_main else "normal"}))
    _style(ax, "Apple's abnormal return after each product launch",
           "Cumulative return beyond what the S&P 500 predicted, from the event onward")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Trading days relative to the launch event", color=TEXT_SECONDARY, fontsize=9)
    _event_line(ax, "Launch event")
    ax.margins(x=0.02)
    _end_labels(ax, ends)
    fig.subplots_adjust(right=0.72)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_volatility(rolling, garch_vol, event_date, path, start):
    """Apple's volatility over time, with the event marked."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    r, g = rolling[rolling.index >= start], garch_vol[garch_vol.index >= start]
    ax.plot(r.index, r, color=HIGHLIGHT, linewidth=2, label="21-day realized volatility")
    ax.plot(g.index, g, color=COLORS["QCOM"], linewidth=1.5, label="GARCH(1,1) model volatility")
    _style(ax, "Apple's volatility before and after the iPhone Duo announcement",
           "Annualized; the dashed line marks the announcement")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.axvline(event_date, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=TEXT)
    fig.autofmt_xdate()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

"""
attention.py — did the public care? Daily Wikipedia page views.

Stock prices show what investors thought. Page views show how much ordinary
people were paying attention: when something is in the news, more people
look it up. Wikipedia publishes exact daily view counts for every article
through a free, official API, which makes it a reliable attention signal.
(Google Trends measures something similar, but it has no official API and
often blocks automated requests.)
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ATTENTION_FILE = DATA_DIR / "attention.csv"

API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
       "en.wikipedia/all-access/user/{article}/daily/{start}/{end}")
# Wikimedia asks every API user to identify themselves.
HEADERS = {"User-Agent": "iphone-duo-event-study (https://github.com/Gaurang7/Iphone-duo-event-study)"}


def download_pageviews(articles, start, end=None):
    """
    Daily page views for each article.

    articles: {"Wikipedia_article_title": "Readable label", ...}
    Returns a table with one column per article (skipping any that fail).
    """
    end = pd.Timestamp(end) if end else pd.Timestamp.today()
    columns = {}
    for article, label in articles.items():
        url = API.format(article=urllib.parse.quote(article, safe=""),
                         start=pd.Timestamp(start).strftime("%Y%m%d"),
                         end=end.strftime("%Y%m%d"))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as resp:
                items = json.load(resp)["items"]
        except (urllib.error.URLError, KeyError, ValueError) as err:
            print(f"  Skipping Wikipedia article {article}: {err}")
            continue
        series = pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items})
        columns[label] = series
    if not columns:
        raise RuntimeError("No Wikipedia page views could be downloaded.")
    return pd.DataFrame(columns).sort_index()


def load_pageviews(articles, start, refresh=False):
    if ATTENTION_FILE.exists() and not refresh:
        return pd.read_csv(ATTENTION_FILE, index_col=0, parse_dates=True)
    views = download_pageviews(articles, start)
    DATA_DIR.mkdir(exist_ok=True)
    views.to_csv(ATTENTION_FILE)
    return views


def attention_before_after(views, event_date, days=30):
    """
    Average daily views in the 30 days before the event vs. the 30 days from
    the event on, plus the busiest day. Page views count every calendar day.
    """
    event = pd.Timestamp(event_date)
    rows = []
    for label in views.columns:
        s = views[label].dropna()
        before = s[(s.index < event) & (s.index >= event - pd.Timedelta(days=days))]
        after = s[(s.index >= event) & (s.index < event + pd.Timedelta(days=days))]
        rows.append({
            "article": label,
            "avg_views_before": before.mean(),
            "avg_views_after": after.mean(),
            "ratio_after_vs_before": after.mean() / before.mean() if before.mean() else float("nan"),
            "peak_date": s.idxmax().date(),
            "peak_views": int(s.max()),
            "days_after": len(after),
        })
    return pd.DataFrame(rows)


def index_to_baseline(views, event_date, days=30):
    """Rescale each article so its pre-event 30-day average = 100 (comparable on one chart)."""
    event = pd.Timestamp(event_date)
    base = views[(views.index < event) & (views.index >= event - pd.Timedelta(days=days))].mean()
    return views / base * 100

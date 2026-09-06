"""Fundamentals overlay for the shortlist (Minervini's SEPA screens accelerating
EPS/sales growth alongside the chart, per 01_ミネルヴィニ投資手法_調査レポート.md §0).

Uses yfinance's free quarterly income statement. Only run this on an already
technically-qualifying shortlist (Trend Template 8/8), not the full universe —
each ticker is a separate API call (~0.3-0.4s), so it doesn't scale to
thousands of tickers the way the batched price download does.

Known limitations:
  - yfinance's free quarterly statements only go back 5-7 quarters: enough for
    one reliable YoY growth figure and, when 6+ quarters are available, a
    second one for a basic "is it accelerating" check — not the multi-quarter
    acceleration trend a paid fundamentals feed (EODHD, etc. — see
    02_自動化システム設計書.md §2.3) would support.
  - Banks/insurers (PNC, BNY, ALL, AIZ, ...) often don't report a "Total
    Revenue" line in this schema, so revenue_growth_yoy_pct comes back None
    for much of the financial sector — verified directly on a live run.
  - Raw single-quarter EPS YoY % is noisy off a small or one-time-charge-
    distorted prior-year base — seen directly on a live run: EIX -63%,
    VTRS -106%, ALL +338%, DOC +367% in one snapshot, all real yfinance
    numbers but not representative of steady underlying growth. Treat any
    |eps_growth_yoy_pct| this large as a "check the actual filing" flag,
    not a genuine acceleration signal.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _yoy_growth(q: pd.DataFrame, row_name: str, cols: list, offset: int) -> float | None:
    """% change of row_name at cols[-1-offset] vs. 4 quarters earlier."""
    if row_name not in q.index or len(cols) < 5 + offset:
        return None
    latest_v = q.loc[row_name, cols[-1 - offset]]
    prior_v = q.loc[row_name, cols[-5 - offset]]
    if pd.isna(latest_v) or pd.isna(prior_v) or prior_v == 0:
        return None
    return round((float(latest_v) / float(prior_v) - 1) * 100, 1)


def fetch_fundamentals(ticker: str) -> dict | None:
    try:
        q = yf.Ticker(ticker).quarterly_income_stmt
    except Exception:
        return None
    if q is None or q.empty or "Total Revenue" not in q.index:
        return None

    cols = sorted(q.columns)  # oldest -> newest
    if len(cols) < 5:
        return None
    eps_row = "Diluted EPS" if "Diluted EPS" in q.index else "Basic EPS"

    rev_growth = _yoy_growth(q, "Total Revenue", cols, offset=0)
    eps_growth = _yoy_growth(q, eps_row, cols, offset=0)
    rev_growth_prev = _yoy_growth(q, "Total Revenue", cols, offset=1)
    eps_growth_prev = _yoy_growth(q, eps_row, cols, offset=1)

    accelerating = None
    if rev_growth is not None and rev_growth_prev is not None:
        accelerating = rev_growth > rev_growth_prev

    # Grade only from metrics that actually exist — a missing value (None)
    # must never silently count as "declining" (a real bug this had: using
    # `x or -1` as a None-fallback made every ticker with missing revenue
    # data, e.g. most banks/insurers whose statements don't use a "Total
    # Revenue" line, grade as 弱い regardless of how strong its EPS growth was).
    available = [v for v in (eps_growth, rev_growth) if v is not None]
    if not available:
        grade = "データ不足"
    elif any(v < 0 for v in available):
        grade = "弱い"
    elif eps_growth is not None and rev_growth is not None and eps_growth >= 20 and rev_growth >= 20:
        grade = "強い"
    else:
        grade = "普通"

    return {
        "symbol": ticker,
        "quarter": str(pd.Timestamp(cols[-1]).date()),
        "revenue_growth_yoy_pct": rev_growth,
        "eps_growth_yoy_pct": eps_growth,
        "revenue_growth_accelerating": accelerating,
        "fundamental_grade": grade,
    }


def fetch_fundamentals_bulk(tickers: list[str]) -> pd.DataFrame:
    """Cached per-day (like company_info.py/data_fetch.py's own caches) —
    generate_report.py, generate_dashboard.py, and generate_company_site.py
    each call this independently for largely-overlapping ticker pools.
    Beyond the redundant work, yfinance's quarterly_income_stmt shares the
    same crumb-authenticated endpoint .info uses, and re-fetching the same
    tickers 2-3x in one job measurably eats into that endpoint's per-run
    rate-limit budget (seen live: company_info.py got 0/300 profiles with
    "Crumb fetch rate-limited (HTTP 429)" once nyse_nasdaq-scale volume
    across scripts added up)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    cache_file = CACHE_DIR / f"fundamentals_{today}.json"

    cached: dict[str, dict] = {}
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))

    out = dict(cached)
    missing = [t for t in tickers if t not in cached]
    if missing:
        for sym in missing:
            info = fetch_fundamentals(sym)
            out[sym] = info  # cache the miss too (None), so it isn't retried all day
        cache_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    rows = [out[t] for t in tickers if out.get(t) is not None]
    return pd.DataFrame(rows)

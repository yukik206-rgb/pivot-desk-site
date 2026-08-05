"""Company profile + valuation metrics + annual financials, for the linked
"company info" site (separate page from the chart dashboard, per user request
to keep company research on its own page, cross-linked by ticker).

Uses yfinance's free `.info` + annual `.income_stmt`. Each ticker is a
separate API call (~0.8-1.3s), same scaling caveat as fundamentals.py — only
run this on an already-screened shortlist, not the full universe. Cached
per-day (like data_fetch.py's OHLCV cache) since a 172-ticker fetch takes
several minutes and this data doesn't change intraday.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def fetch_annual_financials(ticker_obj) -> list[dict]:
    try:
        inc = ticker_obj.income_stmt
    except Exception:
        return []
    if inc is None or inc.empty or "Total Revenue" not in inc.index:
        return []
    cols = sorted(inc.columns)  # oldest -> newest
    rows = []
    for c in cols:
        rev = _num(inc.loc["Total Revenue", c]) if "Total Revenue" in inc.index else None
        ni = _num(inc.loc["Net Income", c]) if "Net Income" in inc.index else None
        if rev is None and ni is None:
            continue
        rows.append({"year": pd.Timestamp(c).year, "revenue": rev, "netIncome": ni})
    return rows


def fetch_company_info(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info
    except Exception:
        return None
    if not info or not info.get("longName"):
        return None

    return {
        "symbol": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "summary": info.get("longBusinessSummary"),
        "employees": info.get("fullTimeEmployees"),
        "website": info.get("website"),
        "city": info.get("city"),
        "country": info.get("country"),
        "marketCap": _num(info.get("marketCap")),
        "trailingPE": _num(info.get("trailingPE")),
        "forwardPE": _num(info.get("forwardPE")),
        "priceToBook": _num(info.get("priceToBook")),
        "priceToSales": _num(info.get("priceToSalesTrailing12Months")),
        "evToEbitda": _num(info.get("enterpriseToEbitda")),
        "dividendYield": _num(info.get("dividendYield")),
        "payoutRatio": _num(info.get("payoutRatio")),
        "beta": _num(info.get("beta")),
        "roe": _num(info.get("returnOnEquity")),
        "roa": _num(info.get("returnOnAssets")),
        "profitMargin": _num(info.get("profitMargins")),
        "operatingMargin": _num(info.get("operatingMargins")),
        "debtToEquity": _num(info.get("debtToEquity")),
        "currentRatio": _num(info.get("currentRatio")),
        "week52High": _num(info.get("fiftyTwoWeekHigh")),
        "week52Low": _num(info.get("fiftyTwoWeekLow")),
        "avgVolume": _num(info.get("averageVolume")),
        "sharesOut": _num(info.get("sharesOutstanding")),
        "instOwnPct": _num(info.get("heldPercentInstitutions")),
        "insiderOwnPct": _num(info.get("heldPercentInsiders")),
        "earningsGrowth": _num(info.get("earningsGrowth")),
        "revenueGrowth": _num(info.get("revenueGrowth")),
        "annual": fetch_annual_financials(t),
    }


def fetch_company_info_bulk(tickers: list[str]) -> dict[str, dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    cache_file = CACHE_DIR / f"company_info_{today}.json"

    cached: dict[str, dict] = {}
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))

    out = dict(cached)
    missing = [t for t in tickers if t not in cached]
    if missing:
        print(f"fetching company info for {len(missing)} tickers not in today's cache...")
        for sym in missing:
            info = fetch_company_info(sym)
            if info is not None:
                out[sym] = info
        cache_file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    return {t: out[t] for t in tickers if t in out}

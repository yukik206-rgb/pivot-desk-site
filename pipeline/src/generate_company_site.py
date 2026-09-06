"""Company info site — separate page from the chart dashboard (per user
request: 別のサイトを作成してそのサイトとリンクさせる形で). Cross-linked by
ticker via URL hash: this page links to dashboard.html#SYM and vice versa.

Usage:
    python generate_company_site.py --universe sp500
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from company_info import fetch_company_info_bulk
from fundamentals import fetch_fundamentals_bulk
from ja_labels import sector_ja, industry_ja
import screen

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
TEMPLATE_PATH = Path(__file__).resolve().parent / "company_template.html"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUMMARY_TRANSLATIONS_PATH = DATA_DIR / "summary_translations_ja.json"
MAX_DETAIL_POOL = 300  # keep in sync with generate_dashboard.py's cap, so the
                       # two sites cross-link to the same set of tickers


def load_summary_translations() -> dict[str, str]:
    """Cached Japanese translations of company business summaries, keyed by
    ticker. Business-summary text is free-form (unlike sector/industry, which
    are a small fixed vocabulary translated via ja_labels.py), so it isn't
    practical to hand-translate on every run — this cache is built once
    (see README: 企業情報の日本語化) and only needs new entries for tickers
    that weren't in a previous run."""
    if SUMMARY_TRANSLATIONS_PATH.exists():
        return json.loads(SUMMARY_TRANSLATIONS_PATH.read_text(encoding="utf-8"))
    return {}


def sanitize(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def run(universe_name: str = "sp500"):
    print("=== Phase 1: trend template (for symbol pool) ===")
    tt_out = screen.run(universe_name)
    qualifiers = tt_out[tt_out["qualifies"]].copy()
    watchlist = tt_out[(tt_out["passed"] == 7) & (~tt_out["qualifies"])].copy()

    if len(qualifiers) + len(watchlist) > MAX_DETAIL_POOL:
        combined = pd.concat([qualifiers, watchlist]).sort_values("rs_rating", ascending=False)
        keep_syms = set(combined["symbol"].head(MAX_DETAIL_POOL))
        qualifiers = qualifiers[qualifiers["symbol"].isin(keep_syms)]
        watchlist = watchlist[watchlist["symbol"].isin(keep_syms)]

    pool = qualifiers["symbol"].tolist() + watchlist["symbol"].tolist()
    in_main = {s: True for s in qualifiers["symbol"]}

    print(f"\n=== company profiles + valuation ({len(pool)} tickers) ===")
    companies = fetch_company_info_bulk(pool)
    print(f"got profiles for {len(companies)} / {len(pool)}")

    summary_ja = load_summary_translations()
    n_translated = 0
    for sym, c in companies.items():
        c["sectorJa"] = sector_ja(c.get("sector"))
        c["industryJa"] = industry_ja(c.get("industry"))
        c["summaryJa"] = summary_ja.get(sym)
        if c["summaryJa"]:
            n_translated += 1
    print(f"business summary translations available: {n_translated} / {len(companies)}")

    print(f"\n=== quarterly fundamentals ({len(pool)} tickers) ===")
    fund = fetch_fundamentals_bulk(pool)
    fund_by_sym = fund.set_index("symbol").to_dict("index") if not fund.empty else {}
    for sym, c in companies.items():
        f = fund_by_sym.get(sym)
        c["quarterlyGrowth"] = {
            "revGrowth": f["revenue_growth_yoy_pct"], "epsGrowth": f["eps_growth_yoy_pct"],
            "grade": f["fundamental_grade"],
        } if f else None

    company_list = [
        {"sym": sym, "name": companies[sym]["name"] if sym in companies else sym,
         "sector": companies[sym].get("sector") if sym in companies else None,
         "inMainList": in_main.get(sym, False)}
        for sym in pool
    ]

    payload = {
        "meta": {"date": dt.date.today().isoformat(), "universe": universe_name},
        "companies": companies,
        "list": company_list,
    }

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__COMPANY_JSON__", json.dumps(sanitize(payload), ensure_ascii=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"company_{universe_name}_{dt.date.today().isoformat()}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nsaved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["sp500", "nyse_nasdaq"], default="nyse_nasdaq")
    args = parser.parse_args()
    run(args.universe)

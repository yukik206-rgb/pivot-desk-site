"""MARKET DESK — market sentiment / overheating / positioning, as a page
separate from the chart dashboard (same rationale as generate_company_site.py:
keeps index.html from growing further, and lets this page load fast on its
own since it doesn't carry any per-ticker chart series).

Usage:
    python generate_market_dashboard.py --universe sp500
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

from data_fetch import fetch_ohlcv
from universe import get_sp500_tickers
import sentiment
import shock_detection

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
TEMPLATE_PATH = Path(__file__).resolve().parent / "market_template.html"


def sanitize(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def run(universe_name: str = "sp500"):
    print("=== VIX ===")
    vix = sentiment.vix_snapshot()
    print(vix)

    print("\n=== VIX term structure (VIX/VIX3M) ===")
    vix_term = sentiment.vix_term_structure()
    print(vix_term)

    print("\n=== SKEW (tail risk) ===")
    skew = sentiment.skew_snapshot()
    print(skew)

    print("\n=== breadth (S&P500 constituents) ===")
    tickers = get_sp500_tickers()
    ohlcv = fetch_ohlcv(tickers)  # default period="2y" — same cache key screen.py already populates
    breadth = sentiment.breadth_snapshot(ohlcv)
    print(breadth)

    print("\n=== AAII investor sentiment ===")
    aaii = sentiment.aaii_sentiment()
    print(aaii)

    print("\n=== JPX margin ratio (信用倍率) ===")
    jpx_margin = sentiment.jpx_margin_ratio()
    print(jpx_margin)

    print("\n=== Nikkei 225 margin evaluation P&L ratio (信用評価損益率) ===")
    nikkei_margin_pnl = sentiment.nikkei_margin_pnl()
    print(nikkei_margin_pnl)

    print("\n=== next SQ dates ===")
    sq_dates = sentiment.next_sq_dates()
    print(sq_dates)

    print("\n=== CFTC COT (E-mini S&P500, institutional) ===")
    cot = sentiment.cot_institutional()
    print(cot)

    print("\n=== CFTC COT (E-mini S&P500, leveraged funds) ===")
    cot_leveraged = sentiment.cot_leveraged_funds()
    print(cot_leveraged)

    overheat_label = sentiment.overheat_label(vix, breadth, aaii, vix_term, skew)
    print(f"\noverheat label: {overheat_label}")

    print("\n=== shock detection (watchlist: indices/sector baskets) ===")
    shocks = shock_detection.run_watchlist()
    for s in shocks:
        print(f"  {s['id']}: severity={s['severity']} breadth_pct={s.get('breadthDownPct')}")

    payload = {
        "meta": {"date": dt.date.today().isoformat(), "universe": universe_name},
        "vix": vix,
        "vixTerm": vix_term,
        "skew": skew,
        "breadth": breadth,
        "aaii": aaii,
        "jpxMargin": jpx_margin,
        "nikkeiMarginPnl": nikkei_margin_pnl,
        "sqDates": sq_dates,
        "cot": cot,
        "cotLeveraged": cot_leveraged,
        "overheatLabel": overheat_label,
        "shocks": shocks,
    }

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__MARKET_JSON__", json.dumps(sanitize(payload), ensure_ascii=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"market_{universe_name}_{dt.date.today().isoformat()}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nsaved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["sp500"], default="sp500")
    args = parser.parse_args()
    run(args.universe)

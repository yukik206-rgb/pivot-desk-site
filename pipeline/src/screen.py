"""Phase 1 prototype: Trend Template screener over a US equity universe.

Usage:
    python screen.py --universe smoke   # 16-ticker smoke test
    python screen.py --universe sp500   # full S&P 500

Data source: Yahoo Finance via yfinance (free, no API key). The system
design doc (投資/チャート分析/02_自動化システム設計書.md) recommends
Tiingo/EODHD for production use; Yahoo is used here for a zero-cost,
zero-setup Phase 1 prototype and can be swapped later behind fetch_prices().
"""
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from universe import get_sp500_tickers, get_nasdaq_nyse_universe, SMOKE_TEST_TICKERS
from data_fetch import fetch_ohlcv
from trend_template import compute_features, rs_raw_score, evaluate_trend_template

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Same liquidity gate backtest.py already uses (design doc §2.2) — without it,
# a broad universe's RS-rating percentile ranking gets dominated by thin,
# volatile micro-caps/low-price speculative names, crowding out real,
# tradable quality names further down the list (found live: with nyse_nasdaq
# and no liquidity gate, 1309 tickers passed 7-8/8 trend template, and a
# genuine large-cap (FTI, RS 89.5) ranked 382nd by RS — well outside any
# reasonably-sized "top N" detail cutoff — because dozens of sub-$10
# speculative names outranked it). Not a Trend Template rule — a pre-filter
# on which tickers are even eligible to be screened, so it doesn't change
# the meaning of "8/8" anywhere downstream.
MIN_PRICE = 5
MIN_DOLLAR_VOL = 10_000_000


def _is_liquid(df: pd.DataFrame) -> bool:
    if len(df) < 50:
        return False
    close, volume = df["Close"], df["Volume"]
    dollar_vol_50 = (close * volume).iloc[-50:].mean()
    return float(close.iloc[-1]) >= MIN_PRICE and float(dollar_vol_50) >= MIN_DOLLAR_VOL


def run(universe_name: str) -> pd.DataFrame:
    # generate_report.py, generate_dashboard.py, and generate_company_site.py
    # each independently call screen.run() for the same (universe, date) —
    # fine at S&P500 scale (seconds) but at nyse_nasdaq scale this screen
    # alone takes ~8 minutes, so redoing it 3x in one daily batch is real
    # wasted time. The result is a pure function of (universe, date), so the
    # CSV this same function already writes doubles as a same-day cache.
    cached_path = OUTPUT_DIR / f"screen_{universe_name}_{dt.date.today().isoformat()}.csv"
    if cached_path.exists():
        print(f"reusing cached screen result: {cached_path}")
        return pd.read_csv(cached_path)

    if universe_name == "smoke":
        tickers = SMOKE_TEST_TICKERS
    elif universe_name == "sp500":
        tickers = get_sp500_tickers()
    elif universe_name == "nyse_nasdaq":
        tickers = get_nasdaq_nyse_universe()
    else:
        raise ValueError(f"unknown universe: {universe_name}")

    print(f"universe: {len(tickers)} tickers ({universe_name})")
    ohlcv = fetch_ohlcv(tickers)
    print(f"usable price history: {len(ohlcv)} / {len(tickers)} tickers")

    rows = []
    n_illiquid = 0
    for sym, df in ohlcv.items():
        if not _is_liquid(df):
            n_illiquid += 1
            continue
        feat = compute_features(df["Close"])
        if feat is None:
            continue
        rows.append({"symbol": sym, **feat})
    print(f"excluded {n_illiquid} tickers below the liquidity gate "
          f"(price>=${MIN_PRICE}, 50d avg $ volume>=${MIN_DOLLAR_VOL:,.0f})")

    if not rows:
        print("no tickers had enough history to evaluate.")
        return pd.DataFrame()

    fdf = pd.DataFrame(rows).set_index("symbol")
    fdf["rs_raw"] = rs_raw_score(fdf.r3, fdf.r6, fdf.r9, fdf.r12)
    fdf["rs_rating"] = fdf["rs_raw"].rank(pct=True) * 99

    results = []
    for sym, row in fdf.iterrows():
        feat = row.to_dict()
        ev = evaluate_trend_template(feat, feat["rs_rating"])
        results.append({
            "symbol": sym,
            "price": feat["price"],
            "passed": ev["passed"],
            "qualifies": ev["qualifies"],
            "rs_rating": round(feat["rs_rating"], 1),
            "pct_above_52w_low": round((feat["price"] / feat["low52"] - 1) * 100, 1),
            "pct_below_52w_high": round((1 - feat["price"] / feat["high52"]) * 100, 1),
            **{k: v for k, v in ev["rules"].items()},
        })

    out = pd.DataFrame(results).sort_values(["qualifies", "rs_rating"], ascending=[False, False])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"screen_{universe_name}_{dt.date.today().isoformat()}.csv"
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")

    qualifying = out[out.qualifies]
    print(f"\nTrend Template 8/8 qualifiers: {len(qualifying)} / {len(out)}")
    if not qualifying.empty:
        print(qualifying[["symbol", "price", "rs_rating", "pct_below_52w_high"]].to_string(index=False))

    near = out[(out.passed == 7) & (~out.qualifies)]
    if not near.empty:
        print(f"\n7/8 watchlist: {len(near)}")
        print(near[["symbol", "price", "rs_rating", "passed"]].to_string(index=False))

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["smoke", "sp500", "nyse_nasdaq"], default="smoke")
    args = parser.parse_args()
    run(args.universe)

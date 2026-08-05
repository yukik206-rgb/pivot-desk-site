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

from universe import get_sp500_tickers, SMOKE_TEST_TICKERS
from data_fetch import fetch_ohlcv
from trend_template import compute_features, rs_raw_score, evaluate_trend_template

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def run(universe_name: str) -> pd.DataFrame:
    if universe_name == "smoke":
        tickers = SMOKE_TEST_TICKERS
    elif universe_name == "sp500":
        tickers = get_sp500_tickers()
    else:
        raise ValueError(f"unknown universe: {universe_name}")

    print(f"universe: {len(tickers)} tickers ({universe_name})")
    ohlcv = fetch_ohlcv(tickers)
    print(f"usable price history: {len(ohlcv)} / {len(tickers)} tickers")

    rows = []
    for sym, df in ohlcv.items():
        feat = compute_features(df["Close"])
        if feat is None:
            continue
        rows.append({"symbol": sym, **feat})

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
    parser.add_argument("--universe", choices=["smoke", "sp500"], default="smoke")
    args = parser.parse_args()
    run(args.universe)

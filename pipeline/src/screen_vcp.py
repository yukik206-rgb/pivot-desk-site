"""Phase 2 prototype: run VCP detection over the Trend Template 8/8 qualifiers
produced by screen.py, and rank by VCP score (design doc §6/§7 "①スクリーニング結果").

Usage:
    python screen.py --universe sp500      # Phase 1: produces output/screen_sp500_<date>.csv
    python screen_vcp.py --universe sp500   # Phase 2: reads that file, adds VCP scores
"""
import argparse
import datetime as dt
import glob
from pathlib import Path

import pandas as pd

from data_fetch import fetch_ohlcv
from vcp import analyze

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def latest_screen_file(universe_name: str) -> Path:
    pattern = str(OUTPUT_DIR / f"screen_{universe_name}_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no screen output found for pattern {pattern}; run screen.py first")
    return Path(matches[-1])


def run(universe_name: str) -> pd.DataFrame:
    src = latest_screen_file(universe_name)
    print(f"reading Trend Template qualifiers from: {src}")
    base = pd.read_csv(src)
    qualifiers = base[base["qualifies"]].copy()
    print(f"Trend Template 8/8 qualifiers: {len(qualifiers)}")

    tickers = qualifiers["symbol"].tolist()
    ohlcv = fetch_ohlcv(tickers)

    rows = []
    for sym in tickers:
        df = ohlcv.get(sym)
        if df is None or len(df) < 130:
            continue
        vcp = analyze(df)
        rows.append({"symbol": sym, "vcp_score": vcp["score"],
                      "num_contractions": len(vcp["contractions"]),
                      "depths_pct": ",".join(f"{d:.0f}" for d in vcp["depths"]),
                      "pivot_price": vcp["pivot_price"], "last_price": vcp["last_price"],
                      "dist_to_pivot_pct": vcp["dist_to_pivot_pct"],
                      "vol_ratio": vcp["vol_ratio"], "breakout": vcp["breakout"]})

    vcp_df = pd.DataFrame(rows)
    merged = qualifiers.merge(vcp_df, on="symbol", how="inner")
    merged = merged.sort_values(["breakout", "vcp_score"], ascending=[False, False])

    out_path = OUTPUT_DIR / f"screen_vcp_{universe_name}_{dt.date.today().isoformat()}.csv"
    cols = ["symbol", "last_price", "vcp_score", "num_contractions", "depths_pct",
            "vol_ratio", "pivot_price", "dist_to_pivot_pct", "breakout", "rs_rating"]
    merged[cols].to_csv(out_path, index=False)
    print(f"saved: {out_path}\n")

    print(merged[cols].to_string(index=False))
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["smoke", "sp500"], default="sp500")
    args = parser.parse_args()
    run(args.universe)

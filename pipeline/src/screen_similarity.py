"""Phase 3 prototype: rank candidates by similarity to a reference ticker.

Combines:
  - DTW shape similarity  (normalized last-90-day close, hand-rolled DTW)
  - Feature-vector cosine similarity (RS rating, VCP score, contractions, volume, pivot distance)

Candidate pool = Trend Template 8/8 qualifiers + 7/8 watchlist from the latest
screen.py run (the same "near-miss" idea shown in the PIVOT DESK mockup, so
Stage-1/near-qualifying names can still surface if their chart shape matches).

Usage:
    python screen_similarity.py --universe sp500 --ref AAPL --top 10
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from data_fetch import fetch_ohlcv
from vcp import analyze, volatility_profile
from similarity import dtw_distance, build_feature_vector, cosine_sim

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def latest(pattern: str) -> Path:
    matches = sorted(glob.glob(str(OUTPUT_DIR / pattern)))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern}; run screen.py first")
    return Path(matches[-1])


def build_pool(universe_name: str) -> pd.DataFrame:
    tt = pd.read_csv(latest(f"screen_{universe_name}_*.csv"))
    pool = tt[(tt["qualifies"]) | (tt["passed"] == 7)].copy()
    return pool


def run(universe_name: str, ref: str, top_n: int) -> pd.DataFrame:
    pool = build_pool(universe_name)
    if ref not in pool["symbol"].values:
        pool = pd.concat([pool, pd.DataFrame([{"symbol": ref}])], ignore_index=True)
    tickers = pool["symbol"].tolist()
    print(f"candidate pool (8/8 + 7/8 watchlist{' + ref' if ref not in tt_syms(pool) else ''}): {len(tickers)} tickers")

    ohlcv = fetch_ohlcv(tickers)
    if ref not in ohlcv or len(ohlcv[ref]) < 130:
        raise ValueError(f"not enough price history for reference ticker {ref}")

    vcp_by_sym = {}
    profile_by_sym = {}
    for sym in tickers:
        df = ohlcv.get(sym)
        if df is None or len(df) < 130:
            continue
        vcp_res = analyze(df)
        vcp_by_sym[sym] = vcp_res
        profile_by_sym[sym] = volatility_profile(df, vcp_res["base_start_idx"])

    pool_idx = pool.set_index("symbol")
    ref_profile = profile_by_sym[ref]
    ref_tt = pool_idx.loc[ref]
    ref_vcp = vcp_by_sym[ref]
    ref_feat = build_feature_vector(ref_tt, ref_vcp)

    rows = []
    for sym in tickers:
        if sym == ref or sym not in vcp_by_sym:
            continue
        dtw = dtw_distance(ref_profile, profile_by_sym[sym])
        tt_row = pool_idx.loc[sym]
        feat = build_feature_vector(tt_row, vcp_by_sym[sym])
        feat_sim = cosine_sim(ref_feat, feat)
        rows.append({"symbol": sym, "dtw_distance": dtw, "feature_cosine_sim": round(feat_sim, 3),
                      "vcp_score": vcp_by_sym[sym]["score"], "breakout": vcp_by_sym[sym]["breakout"]})

    out = pd.DataFrame(rows)
    max_dtw = out["dtw_distance"].max() or 1.0
    out["shape_similarity_pct"] = ((1 - out["dtw_distance"] / max_dtw) * 100).round(1)
    out["feature_similarity_pct"] = (out["feature_cosine_sim"] * 100).round(1)
    out = out.sort_values("shape_similarity_pct", ascending=False).head(top_n)

    out_path = OUTPUT_DIR / f"similar_to_{ref}_{universe_name}.csv"
    cols = ["symbol", "shape_similarity_pct", "feature_similarity_pct", "vcp_score", "breakout"]
    out[cols].to_csv(out_path, index=False)
    print(f"\nreference: {ref}  (VCP score {ref_vcp['score']}, breakout={ref_vcp['breakout']})")
    print(f"saved: {out_path}\n")
    print(out[cols].to_string(index=False))
    return out


def tt_syms(pool: pd.DataFrame) -> set:
    return set(pool["symbol"].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["smoke", "sp500"], default="sp500")
    parser.add_argument("--ref", required=True, help="reference ticker symbol")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    run(args.universe, args.ref.upper(), args.top)

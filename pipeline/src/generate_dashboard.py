"""Real-data interactive dashboard — charts + volume for every Trend Template
8/8 candidate, built on the same pipeline as generate_report.py but rendered
as a self-contained HTML file (no server needed) instead of Markdown.

Usage:
    python generate_dashboard.py --universe sp500
"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path

import pandas as pd

from data_fetch import fetch_ohlcv
from market_health import classify_market, index_health
from fundamentals import fetch_fundamentals_bulk
from vcp import analyze as vcp_analyze, volatility_profile
from similarity import dtw_distance, build_feature_vector, cosine_sim
import screen

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
TEMPLATE_PATH = Path(__file__).resolve().parent / "dashboard_template.html"

RULE_LABELS = {
    "1_price_above_sma150_200": "条件1: 価格が150日線・200日線の上",
    "2_sma150_above_sma200": "条件2: 150日線が200日線の上",
    "3_sma200_trending_up": "条件3: 200日線が上昇トレンド(1ヶ月以上)",
    "4_sma50_above_sma150_200": "条件4: 50日線が150日線・200日線の上",
    "5_price_above_sma50": "条件5: 価格が50日線の上",
    "6_at_least_25pct_above_52w_low": "条件6: 52週安値から25%以上上昇",
    "7_within_25pct_of_52w_high": "条件7: 52週高値の25%以内",
    "8_rs_rating_ge_70": "条件8: RSレーティング70以上",
}


def _nums(s: pd.Series, nd: int = 4):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


def _dates(idx):
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in idx]


def sanitize(obj):
    """Recursively replace float NaN with None before json.dumps.

    Needed because pandas silently turns a dict's `None` into NaN when it
    round-trips through a DataFrame (fetch_fundamentals_bulk builds one, then
    .to_dict("index") hands back NaN instead of the original None). Verified
    on a live run: PNC's missing revenue_growth_yoy_pct came back as a literal
    `NaN` token in the embedded JSON — technically still valid JS (NaN is a
    global), but `NaN == null` is false in JS, so the template's "show '—' if
    null" checks silently failed and rendered the literal text "NaN%" instead.
    """
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def missing_rule_label(row) -> str:
    for key, label in RULE_LABELS.items():
        if key in row and not row[key]:
            return label
    return "—"


def build_full_series(df: pd.DataFrame, vres: dict) -> dict:
    close, high, low, open_, vol = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    sma50, sma150, sma200 = close.rolling(50).mean(), close.rolling(150).mean(), close.rolling(200).mean()
    return {
        "dates": _dates(df.index),
        "open": _nums(open_, 2), "high": _nums(high, 2), "low": _nums(low, 2), "close": _nums(close, 2),
        "vol": [int(v) for v in vol],
        "sma50": _nums(sma50), "sma150": _nums(sma150), "sma200": _nums(sma200),
        "baseStartIdx": vres["base_start_idx"],
        "contractions": [{"startIdx": c["start_idx"], "endIdx": c["end_idx"], "depth": round(c["depth"])}
                          for c in vres["contractions"]],
        "pivotPrice": vres["pivot_price"],
        "breakout": vres["breakout"],
        "breakoutIdx": (len(close) - 1) if vres["breakout"] else -1,
        "lastClose": vres["last_price"],
        "distToPivot": vres["dist_to_pivot_pct"],
        "volRatio": vres["vol_ratio"] if vres["vol_ratio"] is not None else 0,
    }


def build_mini_series(df: pd.DataFrame) -> dict:
    tail = df.tail(150)
    return {"dates": _dates(tail.index), "close": _nums(tail["Close"], 2)}


def build_market_payload(sym: str, df: pd.DataFrame, name: str) -> dict:
    h = index_health(df)
    from market_health import compute_distribution_days
    dist_flags = compute_distribution_days(df["Close"], df["Volume"]).tail(25).tolist()
    return {
        "sym": sym, "name": name, **h,
        "distFlags25": [bool(x) for x in dist_flags],
        "sparkClose": _nums(df["Close"].tail(90), 2),
    }


def run(universe_name: str = "sp500", top_similarity_refs: int = 3):
    print("=== market health ===")
    idx = fetch_ohlcv(["SPY", "QQQ"], period="1y")
    market = classify_market(idx["SPY"], idx["QQQ"])
    market_payload = {
        "label": market["label"],
        "spy": build_market_payload("SPY", idx["SPY"], "S&P 500"),
        "qqq": build_market_payload("QQQ", idx["QQQ"], "Nasdaq総合"),
    }

    print("\n=== Phase 1: trend template ===")
    tt_out = screen.run(universe_name)
    tt_idx = tt_out.set_index("symbol")
    qualifiers = tt_out[tt_out["qualifies"]].copy()
    watchlist = tt_out[(tt_out["passed"] == 7) & (~tt_out["qualifies"])].copy()
    print(f"qualifiers: {len(qualifiers)}, watchlist(7/8): {len(watchlist)}")

    pool_syms = qualifiers["symbol"].tolist() + watchlist["symbol"].tolist()
    print(f"\n=== fetching OHLCV + VCP analysis for pool ({len(pool_syms)} tickers) ===")
    pool_ohlcv = fetch_ohlcv(pool_syms)

    vres_by_sym, profile_by_sym = {}, {}
    for sym in pool_syms:
        df = pool_ohlcv.get(sym)
        if df is None or len(df) < 130:
            continue
        vres = vcp_analyze(df)
        vres_by_sym[sym] = vres
        profile_by_sym[sym] = volatility_profile(df, vres["base_start_idx"])

    print(f"\n=== fundamentals ({len(qualifiers)} qualifiers) ===")
    fund = fetch_fundamentals_bulk(qualifiers["symbol"].tolist())
    fund_by_sym = fund.set_index("symbol").to_dict("index") if not fund.empty else {}

    print("\n=== building chart payloads (qualifiers) ===")
    series_payload, mini_payload, tickers_table = {}, {}, []
    for _, row in qualifiers.iterrows():
        sym = row["symbol"]
        if sym not in vres_by_sym:
            continue
        df = pool_ohlcv[sym]
        vres = vres_by_sym[sym]
        series_payload[sym] = build_full_series(df, vres)
        frow = fund_by_sym.get(sym, {})
        tickers_table.append({
            "sym": sym, "price": vres["last_price"], "tt": 8, "stage": "Stage 2",
            "vcp": vres["score"], "rs": round(float(row["rs_rating"]), 1),
            "distToPivot": vres["dist_to_pivot_pct"], "breakout": vres["breakout"],
            "volRatio": vres["vol_ratio"] if vres["vol_ratio"] is not None else 0,
            "pivotPrice": vres["pivot_price"], "depths": [round(d) for d in vres["depths"]],
            "revGrowth": frow.get("revenue_growth_yoy_pct"),
            "epsGrowth": frow.get("eps_growth_yoy_pct"),
            "fundGrade": frow.get("fundamental_grade", "データ不足"),
        })
    tickers_table.sort(key=lambda t: (not t["breakout"], -t["vcp"]))

    print("\n=== mini series for similarity sparklines (pool) ===")
    for sym in pool_syms:
        if sym in vres_by_sym and sym not in series_payload:
            mini_payload[sym] = build_mini_series(pool_ohlcv[sym])

    watchlist_table = []
    for _, row in watchlist.iterrows():
        watchlist_table.append({
            "sym": row["symbol"], "price": round(float(row["price"]), 2),
            "rs": round(float(row["rs_rating"]), 1), "missing": missing_rule_label(row),
        })
    watchlist_table.sort(key=lambda r: -r["rs"])

    print(f"\n=== similarity search (top {top_similarity_refs} VCP-score references) ===")
    ref_syms = sorted(
        (s for s in qualifiers["symbol"] if s in profile_by_sym),
        key=lambda s: -vres_by_sym[s]["score"],
    )[:top_similarity_refs]

    similarity_payload = {}
    for ref in ref_syms:
        ref_feat = build_feature_vector(tt_idx.loc[ref], vres_by_sym[ref])
        rows = []
        for sym in pool_syms:
            if sym == ref or sym not in profile_by_sym:
                continue
            dtw = dtw_distance(profile_by_sym[ref], profile_by_sym[sym])
            feat = build_feature_vector(tt_idx.loc[sym], vres_by_sym[sym])
            rows.append({"sym": sym, "dtw": dtw, "feat": cosine_sim(ref_feat, feat)})
        max_dtw = max((r["dtw"] for r in rows), default=1.0) or 1.0
        for r in rows:
            r["dtw"] = round((1 - r["dtw"] / max_dtw) * 100, 1)
            r["feat"] = round(r["feat"] * 100, 1)
        rows.sort(key=lambda r: -r["dtw"])
        breakout_tag = "・本日ブレイク" if vres_by_sym[ref]["breakout"] else ""
        similarity_payload[ref] = {
            "label": f"{ref}(VCPスコア{vres_by_sym[ref]['score']}{breakout_tag})",
            "matches": rows[:9],
        }

    payload = {
        "meta": {"date": dt.date.today().isoformat(), "universe": universe_name},
        "market": market_payload,
        "tickers": tickers_table,
        "watchlist": watchlist_table,
        "series": series_payload,
        "miniSeries": mini_payload,
        "similarity": similarity_payload,
    }

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__DASHBOARD_JSON__", json.dumps(sanitize(payload), ensure_ascii=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"dashboard_{universe_name}_{dt.date.today().isoformat()}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nsaved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["sp500"], default="sp500")
    args = parser.parse_args()
    run(args.universe)

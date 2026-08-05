"""VCP (Volatility Contraction Pattern) detection — Phase 2 prototype.

Reference: 投資/チャート分析/01_ミネルヴィニ投資手法_調査レポート.md §3
           投資/チャート分析/02_自動化システム設計書.md §6

VCP is a qualitative pattern in Minervini's own writing, not a rigid formula.
This scoring is a heuristic first pass for ranking/screening purposes — final
judgment should still involve a human chart review (per the design doc's
"known limitations" section).
"""
import numpy as np
import pandas as pd


def zigzag(high: pd.Series, low: pd.Series, pct: float = 0.05) -> list[dict]:
    """Detect alternating swing highs('H')/lows('L') using a % reversal threshold.

    Each pivot carries both its own extreme location ("idx") and the bar index
    at which the subsequent reversal *confirmed* it ("confirmed_at" — the loop
    iteration where the threshold-breaking move away from the extreme actually
    happened). A pivot is only knowable as of "confirmed_at", not "idx" — using
    "idx" alone for point-in-time analysis would leak future information,
    since you can't know bar i was the top of a swing until the price has
    since fallen pct% below it. See find_breakout_days() for why this matters.
    """
    n = len(high)
    if n < 3:
        return []
    direction = 1
    last_ext_idx, last_ext_price = 0, high.iloc[0]
    pivots = []
    for i in range(1, n):
        if direction == 1:
            if high.iloc[i] > last_ext_price:
                last_ext_price, last_ext_idx = high.iloc[i], i
            elif low.iloc[i] < last_ext_price * (1 - pct):
                pivots.append({"idx": last_ext_idx, "price": last_ext_price, "kind": "H", "confirmed_at": i})
                direction = -1
                last_ext_price, last_ext_idx = low.iloc[i], i
        else:
            if low.iloc[i] < last_ext_price:
                last_ext_price, last_ext_idx = low.iloc[i], i
            elif high.iloc[i] > last_ext_price * (1 + pct):
                pivots.append({"idx": last_ext_idx, "price": last_ext_price, "kind": "L", "confirmed_at": i})
                direction = 1
                last_ext_price, last_ext_idx = high.iloc[i], i
    pivots.append({"idx": last_ext_idx, "price": last_ext_price,
                    "kind": "H" if direction == 1 else "L", "confirmed_at": n - 1})
    return pivots


def find_contractions(pivots: list[dict], lookback_bars: int, total_len: int) -> list[dict]:
    """H -> L legs (pullbacks) whose start falls inside the recent base window."""
    contractions = []
    for p1, p2 in zip(pivots, pivots[1:]):
        if p1["kind"] == "H" and p2["kind"] == "L" and p1["idx"] >= total_len - lookback_bars:
            depth = (p1["price"] - p2["price"]) / p1["price"] * 100
            contractions.append({"start_idx": p1["idx"], "start_price": p1["price"],
                                  "end_idx": p2["idx"], "end_price": p2["price"], "depth": depth})
    return contractions


def score_vcp(contractions: list[dict], volume: pd.Series) -> dict:
    if len(contractions) < 2:
        return {"score": 0, "contractions": contractions, "depths": [],
                "monotonic_penalty": None, "vol_avgs": []}

    depths = [c["depth"] for c in contractions]
    monotonic_penalty = sum(1 for a, b in zip(depths, depths[1:]) if b > a * 1.05)

    depth_ratio = depths[-1] / depths[0] if depths[0] > 0 else 1
    depth_score = max(0.0, 30 * (1 - min(depth_ratio, 1))) - monotonic_penalty * 8

    n = len(contractions)
    count_score = {2: 5, 3: 10}.get(n, 15 if n >= 4 else 0)

    vol_avgs = [volume.iloc[c["start_idx"]:c["end_idx"] + 1].mean() for c in contractions]
    vol_score = 0.0
    if len(vol_avgs) >= 2 and vol_avgs[0]:
        vol_ratio = vol_avgs[-1] / vol_avgs[0]
        vol_score = max(0.0, 15 * (1 - min(vol_ratio, 1)))

    score = int(max(0, min(100, round(40 + depth_score + count_score + vol_score))))
    return {"score": score, "contractions": contractions, "depths": depths,
            "monotonic_penalty": monotonic_penalty, "vol_avgs": vol_avgs}


def volatility_profile(df: pd.DataFrame, base_start_idx: int, roll: int = 5) -> np.ndarray:
    """Rolling (High-Low)/Close %, from the detected base start to the last bar,
    min-max normalized to [0,1]. This is what actually differs between a clean
    VCP (narrowing envelope) and a choppy or flat chart — unlike raw normalized
    price, which mostly just encodes 'trending up' and looks similar across any
    Stage-2 name (see README "既知の課題")."""
    seg_high = df["High"].iloc[base_start_idx:]
    seg_low = df["Low"].iloc[base_start_idx:]
    seg_close = df["Close"].iloc[base_start_idx:]
    rng_pct = ((seg_high - seg_low) / seg_close * 100).rolling(roll, min_periods=1).mean()
    arr = rng_pct.to_numpy(dtype=float)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


def analyze(df: pd.DataFrame, pct: float = 0.05, lookback_bars: int = 130) -> dict:
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    pivots = zigzag(high, low, pct)
    contractions = find_contractions(pivots, lookback_bars, len(df))
    result = score_vcp(contractions, volume)

    window_start = max(0, len(df) - lookback_bars)
    base_start_idx = contractions[0]["start_idx"] if contractions else window_start
    base_highs = [p["price"] for p in pivots if p["kind"] == "H" and p["idx"] >= window_start]
    pivot_price = max(base_highs) if base_highs else float(high.iloc[window_start:].max())

    last_price = float(close.iloc[-1])
    avg_vol_50 = volume.iloc[-51:-1].mean()
    last_vol = float(volume.iloc[-1])
    vol_ratio = (last_vol / avg_vol_50) if avg_vol_50 else np.nan
    breakout = bool(last_price >= pivot_price and not np.isnan(vol_ratio) and vol_ratio >= 1.4)

    result.update({
        "pivot_price": round(pivot_price, 2),
        "last_price": round(last_price, 2),
        "dist_to_pivot_pct": round((last_price / pivot_price - 1) * 100, 1),
        "vol_ratio": round(vol_ratio, 2) if not np.isnan(vol_ratio) else None,
        "breakout": breakout,
        "base_start_idx": int(base_start_idx),
    })
    return result


def find_breakout_days(df: pd.DataFrame, pct: float = 0.05, lookback_bars: int = 130,
                        min_day: int = 260) -> pd.Series:
    """Point-in-time VCP pivot-breakout detection for backtesting.

    Runs zigzag() ONCE over the full series (cheap, O(n)), then for each day d
    only uses pivots already "confirmed_at" <= d — never a pivot the algorithm
    could not yet have known about — to compute that day's pivot price and
    check for a fresh breakout (today's close crosses above the pivot with
    volume >= 1.4x its 50-day average, having been below the pivot the prior
    session). This avoids re-running zigzag for every single day, which would
    be the same computation repeated ~n times for no benefit, since pivots
    below the "confirmed_at <= d" cutoff never change.
    """
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    n = len(df)
    pivots = zigzag(high, low, pct)
    flags = np.zeros(n, dtype=bool)

    for d in range(min_day, n):
        window_start = max(0, d - lookback_bars)
        known = [p for p in pivots if p["confirmed_at"] <= d]
        base_highs = [p["price"] for p in known if p["kind"] == "H" and p["idx"] >= window_start]
        if base_highs:
            pivot_price = max(base_highs)
        else:
            pivot_price = float(high.iloc[window_start:d + 1].max())

        avg_vol_50 = volume.iloc[max(0, d - 50):d].mean()
        if not avg_vol_50 or d < 51:
            continue
        vol_ratio = volume.iloc[d] / avg_vol_50
        crossed_today = close.iloc[d] >= pivot_price and close.iloc[d - 1] < pivot_price
        flags[d] = bool(crossed_today and vol_ratio >= 1.4)

    return pd.Series(flags, index=df.index)

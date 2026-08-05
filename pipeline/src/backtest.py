"""Phase 4 prototype: backtest the screening pipeline.

Reference: 投資/チャート分析/02_自動化システム設計書.md §9

Two signal definitions are compared side by side:

  A) trend_template — the day a ticker newly qualifies on all 8 Trend
     Template rules (a proxy for "confirmed Stage 2", cheap to compute).
  B) vcp_breakout — trend_template-qualifying AND a genuine point-in-time
     VCP pivot breakout that day (vcp.find_breakout_days — uses only pivots
     already confirmed as of that day, no lookahead). This is what the
     system is actually meant to trade; A is kept as a baseline to see
     whether the extra VCP timing filter changes the picture.

Both use a 5-day-smoothed cross-sectional RS rating + a per-ticker cooldown
to collapse near-daily threshold jitter into one event (see git history /
comments below for why: raw day-over-day RS>=70 transitions produced 25,857
"signals" out of noise alone before this fix).

Known limitations (be upfront about these, do not paper over them):
  - Universe = CURRENT S&P 500 constituents only -> survivorship bias
    (tickers removed/delisted during the window are invisible here).
  - Benchmark comparison is date-matched SPY forward return (same signal
    date, same horizon) -> controls for market regime, not sector/size.
  - History length is whatever --period covers; multi-regime coverage
    (bear markets, not just the recent bull run) needs a period long enough
    to include one, e.g. 8y+.

Usage:
    python backtest.py --period 8y --horizons 5,21,63
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from universe import get_sp500_tickers, get_broad_growth_universe
from data_fetch import fetch_ohlcv
from trend_template import TRADING_DAYS
from vcp import find_breakout_days

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def rolling_trend_template_bools(df: pd.DataFrame, min_price: float = 5,
                                  min_dollar_vol: float = 10_000_000) -> pd.DataFrame:
    close, volume = df["Close"], df["Volume"]
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    high52 = close.rolling(252).max()
    low52 = close.rolling(252).min()
    sma200_21d_ago = sma200.shift(21)
    dollar_vol_50 = (close * volume).rolling(50).mean()

    r = pd.DataFrame(index=close.index)
    r["r1"] = (close > sma150) & (close > sma200)
    r["r2"] = sma150 > sma200
    r["r3"] = sma200 > sma200_21d_ago
    r["r4"] = (sma50 > sma150) & (sma50 > sma200)
    r["r5"] = close > sma50
    r["r6"] = close >= low52 * 1.25
    r["r7"] = close >= high52 * 0.75
    # liquidity/price gate (design doc §2.2) — keeps thin SPAC shells and
    # illiquid micro-caps that slipped through the universe filter from
    # generating meaningless "signals" once a broader universe is used.
    r["liquidity"] = (close >= min_price) & (dollar_vol_50 >= min_dollar_vol)
    return r


def rolling_return(close: pd.Series, days: int) -> pd.Series:
    return close / close.shift(days) - 1


def compute_qualifies(tickers: list[str], ohlcv: dict[str, pd.DataFrame],
                       min_price: float = 5, min_dollar_vol: float = 10_000_000
                       ) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Shared groundwork for both signal definitions: per-ticker Trend
    Template qualification (bool Series) + the Close price Series."""
    closes = {t: ohlcv[t]["Close"] for t in tickers if t in ohlcv and len(ohlcv[t]) > 260}
    rules_by_t = {t: rolling_trend_template_bools(ohlcv[t], min_price, min_dollar_vol) for t in closes}

    r3 = pd.DataFrame({t: rolling_return(c, TRADING_DAYS["3m"]) for t, c in closes.items()})
    r6 = pd.DataFrame({t: rolling_return(c, TRADING_DAYS["6m"]) for t, c in closes.items()})
    r9 = pd.DataFrame({t: rolling_return(c, TRADING_DAYS["9m"]) for t, c in closes.items()})
    r12 = pd.DataFrame({t: rolling_return(c, TRADING_DAYS["12m"]) for t, c in closes.items()})
    rs_raw = 0.4 * r3 + 0.2 * r6 + 0.2 * r9 + 0.2 * r12
    rs_rating = (rs_raw.rank(axis=1, pct=True) * 99).rolling(5, min_periods=1).mean()

    qualifies_by_t = {}
    for t in closes:
        if t not in rs_rating.columns:
            continue
        r = rules_by_t[t]
        core = r["r1"] & r["r2"] & r["r3"] & r["r4"] & r["r5"] & r["r6"] & r["r7"] & r["liquidity"]
        rs8 = rs_rating[t] >= 70
        qualifies_by_t[t] = (core & rs8).fillna(False)

    return qualifies_by_t, closes


def debounced_events(flag: pd.Series, cooldown_days: int) -> list:
    """'newly True' transitions on a boolean Series, collapsing re-crossings
    within cooldown_days of the last kept event into a single signal."""
    newly = flag & (~flag.shift(1).fillna(False))
    positions = np.flatnonzero(newly.to_numpy())
    kept = []
    last_pos = -10**9
    for pos in positions:
        if pos - last_pos >= cooldown_days:
            kept.append(flag.index[pos])
            last_pos = pos
    return kept


def find_trend_template_signals(qualifies_by_t: dict[str, pd.Series], cooldown_days: int = 60
                                 ) -> pd.DataFrame:
    signals = []
    for t, qualifies in qualifies_by_t.items():
        for d in debounced_events(qualifies, cooldown_days):
            signals.append({"symbol": t, "date": d})
    return pd.DataFrame(signals)


def find_vcp_breakout_signals(qualifies_by_t: dict[str, pd.Series], ohlcv: dict[str, pd.DataFrame],
                               cooldown_days: int = 20) -> pd.DataFrame:
    signals = []
    for t, qualifies in qualifies_by_t.items():
        breakout = find_breakout_days(ohlcv[t])
        combined = (qualifies & breakout.reindex(qualifies.index).fillna(False))
        for d in debounced_events(combined, cooldown_days):
            signals.append({"symbol": t, "date": d})
    return pd.DataFrame(signals)


def evaluate(signals: pd.DataFrame, closes: dict[str, pd.Series], spy_close: pd.Series,
             horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()

    spy_fwd = {h: (spy_close.shift(-h) / spy_close - 1) for h in horizons}

    rows = []
    for row in signals.itertuples(index=False):
        c = closes[row.symbol]
        if row.date not in c.index:
            continue
        i = c.index.get_loc(row.date)
        rec = {"symbol": row.symbol, "date": row.date}
        for h in horizons:
            if i + h < len(c):
                stock_fwd = c.iloc[i + h] / c.iloc[i] - 1
                bench_fwd = spy_fwd[h].get(row.date, np.nan)
                rec[f"fwd_{h}d"] = stock_fwd
                rec[f"excess_{h}d"] = stock_fwd - bench_fwd if pd.notna(bench_fwd) else np.nan
        rows.append(rec)

    detail = pd.DataFrame(rows)
    summary_rows = []
    for h in horizons:
        col, ecol = f"fwd_{h}d", f"excess_{h}d"
        if col not in detail or detail[col].dropna().empty:
            continue
        vals = detail[col].dropna()
        exc = detail[ecol].dropna()
        summary_rows.append({
            "horizon_days": h,
            "n": len(vals),
            "win_rate_pct": round((vals > 0).mean() * 100, 1),
            "avg_return_pct": round(vals.mean() * 100, 2),
            "median_return_pct": round(vals.median() * 100, 2),
            "avg_excess_vs_spy_pct": round(exc.mean() * 100, 2) if len(exc) else None,
            "excess_win_rate_pct": round((exc > 0).mean() * 100, 1) if len(exc) else None,
        })
    summary = pd.DataFrame(summary_rows)
    return detail, summary


def run(universe_name: str, horizons: list[int], period: str, sample_n: int = 700, seed: int = 42,
        min_price: float = 5, min_dollar_vol: float = 10_000_000, tag: str = ""):
    if universe_name == "sp500":
        tickers = get_sp500_tickers()
    elif universe_name == "broad":
        tickers = get_broad_growth_universe(sample_n=sample_n, seed=seed)
    else:
        raise ValueError(f"unknown universe: {universe_name}")

    ohlcv = fetch_ohlcv(tickers, period=period)
    spy = fetch_ohlcv(["SPY"], period=period)["SPY"]["Close"]
    print(f"usable history: {sum(1 for t in tickers if t in ohlcv)} / {len(tickers)} tickers, period={period}")
    print(f"liquidity gate: price>={min_price}, 50d avg dollar volume>={min_dollar_vol:,.0f}")

    qualifies_by_t, closes = compute_qualifies(tickers, ohlcv, min_price, min_dollar_vol)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, sig_fn in [
        ("trend_template", lambda: find_trend_template_signals(qualifies_by_t)),
        ("vcp_breakout", lambda: find_vcp_breakout_signals(qualifies_by_t, ohlcv)),
    ]:
        print(f"\n=== signal definition: {name} ===")
        signals = sig_fn()
        print(f"signal events: {len(signals)}")
        if signals.empty:
            continue
        detail, summary = evaluate(signals, closes, spy, horizons)
        suffix = f"{name}_{universe_name}{('_' + tag) if tag else ''}"
        detail.to_csv(OUTPUT_DIR / f"backtest_detail_{suffix}.csv", index=False)
        summary.to_csv(OUTPUT_DIR / f"backtest_summary_{suffix}.csv", index=False)
        print(summary.to_string(index=False))
        results[name] = summary

    print("\n(caveats: current-constituent survivorship bias; date-matched SPY excess only "
          "controls for market regime, not sector/size — see backtest.py docstring)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["sp500", "broad"], default="sp500")
    parser.add_argument("--horizons", default="5,21,63")
    parser.add_argument("--period", default="8y", help="yfinance period, e.g. 2y/5y/8y/max")
    parser.add_argument("--sample-n", type=int, default=700,
                         help="for --universe broad: random sample size from the non-S&P500 NASDAQ/NYSE listing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-price", type=float, default=5)
    parser.add_argument("--min-dollar-vol", type=float, default=10_000_000)
    parser.add_argument("--tag", default="", help="suffix for output filenames, e.g. 'relaxed'")
    args = parser.parse_args()
    horizons = [int(h) for h in args.horizons.split(",")]
    run(args.universe, horizons, args.period, args.sample_n, args.seed,
        args.min_price, args.min_dollar_vol, args.tag)

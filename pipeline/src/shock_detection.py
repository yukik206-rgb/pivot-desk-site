"""Phase 7: shock/anomaly detection for a configurable watchlist of indices
and sector baskets — a different layer from market_health.py's SPY/QQQ
distribution-day gate (which judges the *overall US market's* trend) and from
Minervini SEPA individual-stock screening. This flags "this index/sector just
moved in a way that's rare relative to its own recent history", e.g. a
multi-day Nikkei 225 drop or a broad, simultaneous semiconductor-sector
selloff, so those show up on MARKET DESK instead of requiring the user to
notice them elsewhere first.

Descriptive only, same philosophy as the rest of this module's sentiment.py
sibling: flags "statistically unusual", not a buy/sell signal.

Extend WATCHLIST to add more indices/sector baskets (e.g. European equities,
China ADRs, bank stocks) — nothing else in this file needs to change.
"""
import pandas as pd

from data_fetch import fetch_ohlcv

WATCHLIST = [
    {"id": "nikkei225", "label": "日経平均株価", "tickers": ["^N225"]},
    {
        "id": "semiconductors",
        "label": "半導体セクター(米欧日 主要14銘柄)",
        "tickers": [
            "NVDA", "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU",
            "AMAT", "LRCX", "KLAC", "TSM", "ASML", "8035.T", "6857.T",
        ],
    },
]

RETURN_WINDOWS = {"1d": 1, "3d": 3, "5d": 5}
# A basket constituent's own 3-day return at or below this is counted as
# "notably down" for the breadth-of-decline figure — a fixed, simple
# threshold (unlike the percentile-based severity below) since breadth needs
# a shared bar across constituents to be comparable.
INDIVIDUAL_DECLINE_THRESHOLD_PCT = -5.0
MIN_HISTORY_BARS = 60


def _pct_return(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(window) * 100


def _percentile_of_latest(series: pd.Series) -> float | None:
    """Where the most recent value ranks within its own trailing history —
    same "rank within own distribution" framing sentiment.py already uses
    for VIX/SKEW. Low = rare on the downside, high = rare on the upside."""
    s = series.dropna()
    if len(s) < MIN_HISTORY_BARS:
        return None
    latest = s.iloc[-1]
    return float((s < latest).mean() * 100)


def _severity_label(pct: float | None) -> str:
    if pct is None:
        return "データ不足"
    if pct <= 5:
        return "急落(過去の下位域)"
    if pct <= 15:
        return "弱含み"
    if pct >= 95:
        return "急騰(過去の上位域)"
    if pct >= 85:
        return "強含み"
    return "平常範囲"


def _most_extreme_percentile(returns: dict) -> float | None:
    """Severity is driven by whichever window (1d/3d/5d) is currently the
    most statistically rare, not a fixed 3d window — otherwise a sharp
    1-day move gets diluted/hidden by calmer preceding days once it's
    averaged into the 3-day figure, even though the 1-day number alone
    would clearly be worth flagging."""
    pcts = [w["percentile"] for w in returns.values() if w["percentile"] is not None]
    if not pcts:
        return None
    return min(pcts, key=lambda p: min(p, 100 - p))


def _returns_block(close: pd.Series) -> dict:
    out = {}
    for key, window in RETURN_WINDOWS.items():
        ret_series = _pct_return(close, window)
        latest = ret_series.iloc[-1] if len(ret_series) else float("nan")
        out[key] = {
            "pct": None if pd.isna(latest) else round(float(latest), 2),
            "percentile": _percentile_of_latest(ret_series),
        }
    return out


def _index_result(entry: dict, df: pd.DataFrame) -> dict:
    close = df["Close"]
    returns = _returns_block(close)
    return {
        "id": entry["id"], "label": entry["label"], "kind": "index",
        "lastClose": round(float(close.iloc[-1]), 2),
        "returns": returns,
        "severity": _severity_label(_most_extreme_percentile(returns)),
        "breadthNote": None,
    }


def _basket_result(entry: dict, ohlcv: dict[str, pd.DataFrame]) -> dict | None:
    closes = {t: df["Close"] for t, df in ohlcv.items() if df is not None and len(df) > MIN_HISTORY_BARS}
    n_available = len(closes)
    if n_available == 0:
        return None

    returns = {}
    ret_3d_now_by_ticker = {}
    for key, window in RETURN_WINDOWS.items():
        per_ticker = {t: _pct_return(c, window) for t, c in closes.items()}
        avg_series = pd.concat(per_ticker, axis=1).mean(axis=1)  # equal-weight basket average
        latest = avg_series.iloc[-1] if len(avg_series) else float("nan")
        returns[key] = {
            "pct": None if pd.isna(latest) else round(float(latest), 2),
            "percentile": _percentile_of_latest(avg_series),
        }
        if window == 3:
            ret_3d_now_by_ticker = {t: s.iloc[-1] for t, s in per_ticker.items()}

    n_down = sum(1 for v in ret_3d_now_by_ticker.values()
                 if pd.notna(v) and v <= INDIVIDUAL_DECLINE_THRESHOLD_PCT)
    breadth_pct = round(n_down / n_available * 100, 1)

    breadth_note = None
    if breadth_pct >= 50:
        breadth_note = (f"構成銘柄{n_available}銘柄中{n_down}銘柄({breadth_pct}%)が3営業日で"
                         f"{INDIVIDUAL_DECLINE_THRESHOLD_PCT:.0f}%超下落 — セクター全体で同時安")
    elif breadth_pct >= 25:
        breadth_note = (f"構成銘柄{n_available}銘柄中{n_down}銘柄({breadth_pct}%)が3営業日で"
                         f"{INDIVIDUAL_DECLINE_THRESHOLD_PCT:.0f}%超下落 — 一部銘柄で下落")

    return {
        "id": entry["id"], "label": entry["label"], "kind": "basket",
        "nConstituents": n_available, "nDown": n_down, "breadthDownPct": breadth_pct,
        "returns": returns,
        "severity": _severity_label(_most_extreme_percentile(returns)),
        "breadthNote": breadth_note,
    }


def run_watchlist() -> list[dict]:
    results = []
    for entry in WATCHLIST:
        tickers = entry["tickers"]
        # 2y — enough trailing history for the percentile ranking to mean
        # something (same period sentiment.py's vix_snapshot/skew_snapshot
        # use, just longer since this ranks N-day *returns*, not levels).
        ohlcv = fetch_ohlcv(tickers, period="2y")
        if len(tickers) == 1:
            df = ohlcv.get(tickers[0])
            if df is None or len(df) < MIN_HISTORY_BARS:
                continue
            results.append(_index_result(entry, df))
        else:
            result = _basket_result(entry, ohlcv)
            if result is not None:
                results.append(result)
    return results

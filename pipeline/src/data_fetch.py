"""Shared OHLCV fetch + local cache for Phase 1 (Trend Template) and Phase 2 (VCP)."""
import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _drop_incomplete_session(df: pd.DataFrame) -> pd.DataFrame:
    """The design doc assumes the daily batch runs after US market close
    (投資/チャート分析/02_自動化システム設計書.md §2.4). If it runs mid-session
    instead, yfinance's last bar carries a partial day's volume — e.g. seen
    directly on a live run: 2026-07-30's bar showed ~5.2M shares vs. a ~50M
    daily average, because it was fetched 11 minutes after the US open. That
    silently wrecks anything volume-ratio-based (VCP breakout detection,
    distribution days). Drop the last bar if it looks like an in-progress
    session: today's date, and volume far below the recent normal range.
    """
    if len(df) < 21:
        return df
    last = df.iloc[-1]
    if last.name.date() != dt.date.today():
        return df
    recent_median_vol = df["Volume"].iloc[-21:-1].median()
    if recent_median_vol and last["Volume"] < 0.3 * recent_median_vol:
        return df.iloc[:-1]
    return df


def fetch_ohlcv(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    out: dict[str, pd.DataFrame] = {}
    to_download = []
    for t in tickers:
        cache_file = CACHE_DIR / f"{t}_{period}_{today}.parquet"
        if cache_file.exists():
            out[t] = _drop_incomplete_session(pd.read_parquet(cache_file))
        else:
            to_download.append(t)

    if to_download:
        print(f"downloading {len(to_download)} tickers from Yahoo Finance...")
        raw = yf.download(
            tickers=to_download, period=period, interval="1d",
            group_by="ticker", auto_adjust=True, threads=True, progress=False,
        )
        for t in to_download:
            try:
                # yfinance always returns ticker-keyed MultiIndex columns with
                # group_by="ticker", even for a single-ticker request.
                df = raw[t].dropna()
            except (KeyError, TypeError):
                continue
            if df.empty or "Close" not in df:
                continue
            df = df[COLUMNS]
            df.to_parquet(CACHE_DIR / f"{t}_{period}_{today}.parquet")
            out[t] = _drop_incomplete_session(df)

    return out

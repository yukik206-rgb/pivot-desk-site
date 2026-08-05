"""Phase 5: market health gate (real data).

Reference: 投資/マーケット分析/01_マーケット環境判断フレームワーク.md

Computes, per index (SPY/QQQ as S&P500/Nasdaq proxies):
  - distribution day count over the trailing 25 sessions (down day + volume
    above the prior day's volume)
  - the most recent follow-through day (day 4+ of a rally attempt off a
    correction low, +1.5% on volume above the prior day), and whether a
    distribution day struck within 10 sessions after it (the "especially
    dangerous" case the reference doc calls out)
  - a simple Stage-2-like check (price > SMA50 > SMA200)

Then rolls these into one BULL/CAUTION/BEAR label used to gate/weight the
daily report (design doc §8 "マクロゲート連携").
"""
import pandas as pd


def compute_distribution_days(close: pd.Series, volume: pd.Series) -> pd.Series:
    down = close.diff() < 0
    vol_up = volume > volume.shift(1)
    return down & vol_up


def find_last_follow_through(close: pd.Series, volume: pd.Series) -> int | None:
    n = len(close)
    low_so_far = close.iloc[0]
    rally_start = None
    last_ftd_idx = None
    for i in range(1, n):
        if close.iloc[i] < low_so_far:
            low_so_far = close.iloc[i]
            rally_start = None
        elif rally_start is None and close.iloc[i] > close.iloc[i - 1]:
            rally_start = i
        if rally_start is not None:
            day_num = i - rally_start + 1
            if day_num >= 4:
                gain = close.iloc[i] / close.iloc[i - 1] - 1
                if gain >= 0.015 and volume.iloc[i] > volume.iloc[i - 1]:
                    last_ftd_idx = i
                    low_so_far = close.iloc[i]
                    rally_start = None
    return last_ftd_idx


def index_health(df: pd.DataFrame) -> dict:
    close, volume = df["Close"], df["Volume"]
    dist_series = compute_distribution_days(close, volume)
    dist_count = int(dist_series.tail(25).sum())

    ftd_idx = find_last_follow_through(close, volume)
    ftd_date = df.index[ftd_idx] if ftd_idx is not None else None
    days_since_ftd = (len(df) - 1 - ftd_idx) if ftd_idx is not None else None
    ftd_warning = False
    if ftd_idx is not None:
        ftd_warning = bool(dist_series.iloc[ftd_idx + 1: ftd_idx + 11].any())

    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    stage2_like = bool(pd.notna(sma50) and pd.notna(sma200) and close.iloc[-1] > sma50 > sma200)

    return {
        "last_close": round(float(close.iloc[-1]), 2),
        "dist_count": dist_count,
        "ftd_date": str(pd.Timestamp(ftd_date).date()) if ftd_date is not None else None,
        "days_since_ftd": days_since_ftd,
        "ftd_warning": ftd_warning,
        "stage2_like": stage2_like,
    }


def classify_market(spy_df: pd.DataFrame, qqq_df: pd.DataFrame) -> dict:
    spy, qqq = index_health(spy_df), index_health(qqq_df)
    dist_max = max(spy["dist_count"], qqq["dist_count"])

    if dist_max >= 7 or not (spy["stage2_like"] or qqq["stage2_like"]):
        label = "BEAR"
    elif dist_max >= 4 or spy["ftd_warning"] or qqq["ftd_warning"]:
        label = "CAUTION"
    else:
        label = "BULL"

    return {"label": label, "spy": spy, "qqq": qqq}

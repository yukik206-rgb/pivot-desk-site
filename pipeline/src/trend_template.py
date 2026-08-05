"""Mark Minervini's Trend Template (8 rules) + RS Rating proxy.

Reference: 投資/チャート分析/01_ミネルヴィニ投資手法_調査レポート.md
The RS proxy formula (0.4/0.2/0.2/0.2 weighted trailing returns) is an
approximation of IBD's proprietary Relative Strength Rating and must be
percentile-ranked across the screened universe before use.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = {"3m": 63, "6m": 126, "9m": 189, "12m": 252}


def compute_features(close: pd.Series) -> dict | None:
    if len(close) < 252 + 21:
        return None

    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    high52 = close.rolling(252).max()
    low52 = close.rolling(252).min()

    price = close.iloc[-1]

    def ret(days: int) -> float:
        if len(close) <= days:
            return np.nan
        return price / close.iloc[-1 - days] - 1

    return {
        "price": price,
        "sma50": sma50.iloc[-1],
        "sma150": sma150.iloc[-1],
        "sma200": sma200.iloc[-1],
        "sma200_21d_ago": sma200.iloc[-22] if len(sma200) > 22 else np.nan,
        "high52": high52.iloc[-1],
        "low52": low52.iloc[-1],
        "r3": ret(TRADING_DAYS["3m"]),
        "r6": ret(TRADING_DAYS["6m"]),
        "r9": ret(TRADING_DAYS["9m"]),
        "r12": ret(TRADING_DAYS["12m"]),
    }


def rs_raw_score(r3: float, r6: float, r9: float, r12: float) -> float:
    return 0.4 * r3 + 0.2 * r6 + 0.2 * r9 + 0.2 * r12


def evaluate_trend_template(feat: dict, rs_rating: float) -> dict:
    rules = {
        "1_price_above_sma150_200": feat["price"] > feat["sma150"] and feat["price"] > feat["sma200"],
        "2_sma150_above_sma200": feat["sma150"] > feat["sma200"],
        "3_sma200_trending_up": feat["sma200"] > feat["sma200_21d_ago"],
        "4_sma50_above_sma150_200": feat["sma50"] > feat["sma150"] and feat["sma50"] > feat["sma200"],
        "5_price_above_sma50": feat["price"] > feat["sma50"],
        "6_at_least_25pct_above_52w_low": feat["price"] >= feat["low52"] * 1.25,
        "7_within_25pct_of_52w_high": feat["price"] >= feat["high52"] * 0.75,
        "8_rs_rating_ge_70": rs_rating >= 70,
    }
    passed = sum(rules.values())
    return {"rules": rules, "passed": passed, "qualifies": passed == 8}

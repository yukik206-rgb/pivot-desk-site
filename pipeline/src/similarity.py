"""Phase 3 prototype: chart similarity — feature-vector cosine similarity + DTW shape similarity.

Reference: 投資/チャート分析/02_自動化システム設計書.md §7
"""
import numpy as np
import pandas as pd


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic O(n*m) DTW on normalized series; hand-rolled to avoid a heavy
    dependency for a prototype of this size. Swap for dtaidistance/tslearn
    if this needs to scale to the full universe (design doc §11)."""
    n, m = len(a), len(b)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        row = d[i]
        prev = d[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            row[j] = cost + min(prev[j], row[j - 1], prev[j - 1])
    return d[n, m] / (n + m)


def build_feature_vector(tt_row: pd.Series, vcp_result: dict) -> np.ndarray:
    """tt_row: a row from screen.py's Trend Template output.
    vcp_result: the dict returned by vcp.analyze() (keys: score, contractions,
    vol_ratio, dist_to_pivot_pct, ...)."""
    return np.array([
        tt_row["rs_rating"] / 100,
        tt_row["pct_below_52w_high"] / 100,
        tt_row["pct_above_52w_low"] / 100,
        vcp_result["score"] / 100,
        min(len(vcp_result["contractions"]), 10) / 10,
        min(vcp_result["vol_ratio"] or 1.0, 3.0) / 3,
        max(min(vcp_result["dist_to_pivot_pct"], 0), -30) / -30,
    ])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / denom)

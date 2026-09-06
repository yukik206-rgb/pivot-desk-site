"""Phase 5: daily report — market health gate + Trend Template/VCP screening,
combined into one Markdown file (design doc §8 マクロゲート連携, §10 出力層).

Usage:
    python generate_report.py --universe sp500
"""
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from data_fetch import fetch_ohlcv
from market_health import classify_market
from fundamentals import fetch_fundamentals_bulk
import screen
import screen_vcp

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

LABEL_JA = {"BULL": "強気(BULL)", "CAUTION": "警戒(CAUTION)", "BEAR": "弱気(BEAR)"}


def df_to_md(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def market_section(market: dict) -> list[str]:
    lines = ["## マーケット環境", ""]
    lines.append(f"**総合判定: {LABEL_JA[market['label']]}**")
    lines.append("")
    for name, key in [("S&P500(SPY)", "spy"), ("Nasdaq(QQQ)", "qqq")]:
        d = market[key]
        ftd = f"{d['ftd_date']}({d['days_since_ftd']}営業日前)" if d["ftd_date"] else "検出なし"
        warn = " ⚠フォロースルー後10日以内に分配日あり" if d["ftd_warning"] else ""
        stage = "Stage2相当" if d["stage2_like"] else "Stage2条件を満たさず"
        lines.append(f"- **{name}**: 終値{d['last_close']} / 分配日 {d['dist_count']}/25 / "
                      f"フォロースルーデー {ftd}{warn} / {stage}")
    if market["label"] != "BULL":
        lines.append("")
        lines.append("> ⚠ 地合いが強気ではないため、新規エントリーは厳選・ポジションサイズ縮小を推奨"
                      "(`マーケット分析/01_マーケット環境判断フレームワーク.md`参照)")
    lines.append("")
    return lines


def run(universe_name: str = "sp500"):
    print("=== market health ===")
    idx = fetch_ohlcv(["SPY", "QQQ"], period="1y")
    market = classify_market(idx["SPY"], idx["QQQ"])
    print(f"label: {market['label']}")

    print("\n=== Phase 1: trend template ===")
    tt_out = screen.run(universe_name)
    n_qualify = int(tt_out["qualifies"].sum())
    n_watch = int(((tt_out["passed"] == 7) & (~tt_out["qualifies"])).sum())

    print("\n=== Phase 2: VCP scoring ===")
    vcp_out = screen_vcp.run(universe_name)

    lines = [f"# PIVOT DESK 日次レポート — {dt.date.today().isoformat()}", ""]
    lines += market_section(market)

    lines.append(f"## ① スクリーニング結果 — トレンドテンプレート8/8 適合({n_qualify}銘柄) × VCPスコア降順")
    lines.append("")
    top = vcp_out.sort_values(["breakout", "vcp_score"], ascending=[False, False]).head(20).copy()

    print(f"\n=== fundamentals overlay (top {len(top)}) ===")
    fund = fetch_fundamentals_bulk(top["symbol"].tolist())
    if not fund.empty:
        top = top.merge(fund, on="symbol", how="left")
        top["fundamental_grade"] = top["fundamental_grade"].fillna("データ不足")

    cols = ["symbol", "last_price", "vcp_score", "num_contractions", "depths_pct",
            "vol_ratio", "pivot_price", "dist_to_pivot_pct", "breakout", "rs_rating"]
    fund_cols = ["revenue_growth_yoy_pct", "eps_growth_yoy_pct", "fundamental_grade"]
    if all(c in top.columns for c in fund_cols):
        cols += fund_cols
    lines.append(df_to_md(top[cols]))
    lines.append("")
    lines.append("(ファンダメンタルズはyfinance無料データに基づく直近四半期の前年同期比。"
                 "銀行・保険株は`Total Revenue`行が存在せず売上成長率が欠損することがある。"
                 "EPS成長率は少数四半期の単純比較のため、前年同期が小さい/一時的要因がある場合"
                 "極端な数値(±100%超)が出ることがある — その場合は個別に開示資料を確認してください。"
                 "詳細は`README.md`参照)")
    lines.append("")

    lines.append(f"## ② 準合格ウォッチリスト(7/8) — {n_watch}銘柄")
    lines.append("")
    watch = tt_out[(tt_out["passed"] == 7) & (~tt_out["qualifies"])].sort_values("rs_rating", ascending=False)
    lines.append(df_to_md(watch[["symbol", "price", "rs_rating"]].head(20)))
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("これはPhase 1-5プロトタイプの自動生成レポートです。データはYahoo Finance(無料)。"
                  "RSレーティングは近似式、VCPスコアはヒューリスティックであり、最終判断には人間による"
                  "チャート確認を併用してください(`投資/チャート分析/README.md`参照)。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"report_{universe_name}_{dt.date.today().isoformat()}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["smoke", "sp500", "nyse_nasdaq"], default="nyse_nasdaq")
    args = parser.parse_args()
    run(args.universe)

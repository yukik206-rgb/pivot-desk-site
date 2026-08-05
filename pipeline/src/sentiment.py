"""Phase 6: market sentiment / overheating / positioning — a layer that
market_health.py (分配日・フォロースルーデー・Stage判定)does not cover.

Independent metrics, each isolated behind its own try/except so a single
source breaking (page markup change, network hiccup) degrades that one card
to "取得できませんでした" instead of failing the whole daily run (same
fallback philosophy as company_info.py's handling of missing .info data):

  1. 過熱度        — VIX水準+1年percentile、S&P500の幅(%>50MA/%>200MA)、新高値/新安値、
                     VIX期間構造(VIX/VIX3M)、CBOE SKEW指数(テールリスクの織り込み度)
  2. 個人投資家スタンス — AAII Investor Sentiment Survey(強気/中立/弱気%)、
                        JPX信用倍率(東証+名証、信用買残÷信用売残)
  3. 機関投資家動向   — CFTC Commitments of Traders、E-mini S&P500の
                        Asset Manager/Institutional区分、Leveraged Funds区分
                        (ヘッジファンド等の投機筋)の各ネットポジション

AAII・CFTCは無料の一括ヒストリカルDLが無いため、日次実行のたびにその日の値を
ローカルCSV(data/aaii_history.csv, data/cot_history.csv, data/cot_leveraged_history.csv)
に追記して自前で時系列を蓄積する(data/cache/と同じ考え方)。JPXの信用取引データは
逆に毎回のファイル自体に2002年からの全履歴が入っているため、蓄積の必要がない。

検討したが採用しなかったデータソース:
  - CBOE Put/Callレシオ: 数値はページ内の静的HTMLに存在せず、クライアントサイド
    JSでAPIから取得後に描画される方式(確認済み: 素のHTTP取得では該当箇所が
    空)。ヘッドレスブラウザなしでは安定して取れない。
  - FINRA信用残(マージンデット): 公式ページがCloudflareのボット判定チャレンジ
    で保護されている(確認済み)。突破は意図的な検知回避になるため見送り。
  - NAAIM Exposure Index: 2026年にサブスクリプション制へ移行し、無料の公開
    数値が無くなった(確認済み、公式ページに移行の告知あり)。代わりにCFTCの
    Leveraged Funds区分(実際の建玉データ)を使う。
  - Investors Intelligence ブルベア指数: 元データ自体が同社の有料調査で、
    McClellan Financial・Yardeni Researchの無料ページもチャート画像のみで
    生の数値がHTML中に存在しない(確認済み)。代わりにVIX期間構造(市場の
    値付けから出る客観的指標)を使う。
"""
import datetime
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from data_fetch import fetch_ohlcv

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AAII_HISTORY = DATA_DIR / "aaii_history.csv"
COT_HISTORY = DATA_DIR / "cot_history.csv"
COT_LEVERAGED_HISTORY = DATA_DIR / "cot_leveraged_history.csv"
JPX_MARGIN_CACHE = DATA_DIR / "jpx_margin_cache.xls"

AAII_URL = "https://www.aaii.com/sentimentsurvey"
CFTC_URL = "https://www.cftc.gov/dea/futures/financial_lf.htm"
JPX_MARGIN_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/margin/06.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

_cftc_text_cache: dict[str, str] = {}


def _append_history(path: Path, record: dict, dedupe_key: str) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        hist = pd.read_csv(path)
        if dedupe_key in hist.columns and (hist[dedupe_key].astype(str) == str(record[dedupe_key])).any():
            return hist
        hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
    else:
        hist = pd.DataFrame([record])
    hist.to_csv(path, index=False)
    return hist


def vix_snapshot() -> dict | None:
    """VIX水準 + 直近1年のパーセンタイル順位。低いほど「楽観・過熱注意」。"""
    try:
        idx = fetch_ohlcv(["^VIX"], period="1y")
        df = idx.get("^VIX")
        if df is None or df.empty:
            return None
        close = df["Close"].dropna()
        level = float(close.iloc[-1])
        percentile = float((close < level).mean() * 100)
        if level < 15:
            label = "低ボラ・楽観(過熱注意)"
        elif level < 25:
            label = "平常"
        elif level < 30:
            label = "警戒感上昇"
        else:
            label = "恐怖・パニック"
        return {"level": round(level, 2), "percentile_1y": round(percentile, 1), "label": label}
    except Exception as e:
        print(f"[sentiment] vix_snapshot failed: {e}")
        return None


def vix_term_structure() -> dict | None:
    """VIX(短期)とVIX3M(3ヶ月物)の比率。オプション市場の期間構造で、
    順イールド(VIX3M>VIX、比率<1、平常時のデフォルト状態)が崩れて
    逆イールド(VIX>=VIX3M、比率>=1)になると、プロのオプション市場が
    近い将来の急落リスクを高く織り込んでいるサイン。ブルベア指数のような
    意見調査ではなく値付けそのものから出るため、調査対象者の偏りがない。
    """
    try:
        idx = fetch_ohlcv(["^VIX", "^VIX3M"], period="1y")
        vix_df, vix3m_df = idx.get("^VIX"), idx.get("^VIX3M")
        if vix_df is None or vix3m_df is None or vix_df.empty or vix3m_df.empty:
            return None
        vix = float(vix_df["Close"].iloc[-1])
        vix3m = float(vix3m_df["Close"].iloc[-1])
        ratio = vix / vix3m
        label = "逆イールド(警戒)" if ratio >= 1.0 else "順イールド(平常)"
        return {"vix": round(vix, 2), "vix3m": round(vix3m, 2), "ratio": round(ratio, 3), "label": label}
    except Exception as e:
        print(f"[sentiment] vix_term_structure failed: {e}")
        return None


def skew_snapshot() -> dict | None:
    """CBOE SKEW指数。アウト・オブ・ザ・マネーのS&P500オプション価格から算出
    される「テールリスク(今後30日で2σ超の急落が起きる確率)」の織り込み度。
    VIXが値付けされたボラティリティの大きさを測るのに対し、SKEWは分布の歪み
    (暴落方向への保険=プロテクティブプットの需要)を測る、VIXとは独立した
    軸のオプション市場指標。目安として100が理論上のフラット(歪みなし)、
    135以上でテールリスクへの警戒が強いとされる(この閾値もVIXの15/25/30と
    同様に業界の目安であり、厳密な統計的根拠があるわけではない)。
    """
    try:
        idx = fetch_ohlcv(["^SKEW"], period="1y")
        df = idx.get("^SKEW")
        if df is None or df.empty:
            return None
        close = df["Close"].dropna()
        level = float(close.iloc[-1])
        percentile = float((close < level).mean() * 100)
        if level >= 145:
            label = "テールリスク警戒(強い)"
        elif level >= 135:
            label = "テールリスク警戒(やや強い)"
        else:
            label = "平常"
        return {"level": round(level, 1), "percentile_1y": round(percentile, 1), "label": label}
    except Exception as e:
        print(f"[sentiment] skew_snapshot failed: {e}")
        return None


def breadth_snapshot(ohlcv: dict[str, pd.DataFrame]) -> dict | None:
    """S&P500構成銘柄のうち50日線/200日線より上にいる比率と、52週新高値/新安値の数。
    幅が極端に高い(ほぼ全銘柄が50日線の上)状態が長引くと過熱のサイン、
    逆に指数は堅調でも幅が縮む(値がさ株頼み)状態は地合いの脆さのサインとされる。
    """
    try:
        above50 = above200 = new_high = new_low = n = 0
        for df in ohlcv.values():
            if df is None or len(df) < 200:
                continue
            close = df["Close"]
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = float(close.rolling(200).mean().iloc[-1])
            if pd.isna(sma50) or pd.isna(sma200):
                continue
            last = float(close.iloc[-1])
            n += 1
            above50 += int(last > sma50)
            above200 += int(last > sma200)
            window = close.tail(252)
            new_high += int(last >= float(window.max()))
            new_low += int(last <= float(window.min()))
        if n == 0:
            return None
        return {
            "n_total": n,
            "pct_above_50ma": round(above50 / n * 100, 1),
            "pct_above_200ma": round(above200 / n * 100, 1),
            "new_highs": new_high,
            "new_lows": new_low,
        }
    except Exception as e:
        print(f"[sentiment] breadth_snapshot failed: {e}")
        return None


def aaii_sentiment() -> dict | None:
    """AAII Investor Sentiment Survey(個人投資家、週次・木曜更新)の最新週。
    ページの最初のchartWrapperが直近4週分、2つ目が「Historical Averages」を
    含む比較ブロックなので、最初のブロックの先頭(=最新週)だけを使う。
    """
    try:
        resp = requests.get(AAII_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        wrappers = soup.select("div.chartWrapper")
        if not wrappers:
            return None
        latest = None
        for week in wrappers[0].select("div.weekending"):
            date_el = week.select_one(".date")
            bars = week.select(".bar")
            if not date_el or not bars:
                continue
            vals = {}
            for b in bars:
                classes = b.get("class", [])
                for tag in ("bullish", "neutral", "bearish"):
                    if tag in classes:
                        vals[tag] = float(b.get_text(strip=True).replace("%", ""))
            if {"bullish", "neutral", "bearish"} <= vals.keys():
                latest = {"date": date_el.get_text(strip=True), **vals}
                break
        if latest is None:
            return None
        record = {
            "date": latest["date"], "bull_pct": latest["bullish"],
            "neutral_pct": latest["neutral"], "bear_pct": latest["bearish"],
        }
        hist = _append_history(AAII_HISTORY, record, dedupe_key="date")
        return {**record, "history": hist.tail(26).to_dict("records")}
    except Exception as e:
        print(f"[sentiment] aaii_sentiment failed: {e}")
        return None


def jpx_margin_ratio() -> dict | None:
    """東証+名証の信用取引現在高(週次、JPX公式・無料Excel)から算出する
    信用倍率(信用買残÷信用売残、共に株数ベース)。JPXのファイルには2002年
    からの全履歴がそのまま入っているため、AAII/CFTCと違って自前でのローカル
    蓄積は不要 — 毎回ファイル全体をそのまま使う。ページ上のExcelリンクは
    JPXが更新するたびにファイル名(ハッシュ)が変わるため、ページを都度
    スクレイピングして「.xlsで終わる最初のリンク」を取得する
    (=「信用取引現在高(一般信用取引・制度信用取引別)」、確認済み)。
    """
    try:
        page = requests.get(JPX_MARGIN_PAGE, headers=HEADERS, timeout=20)
        page.raise_for_status()
        m = re.search(r'href="([^"]+\.xls)"', page.text)
        if not m:
            return None
        xls_url = "https://www.jpx.co.jp" + m.group(1)
        xls_resp = requests.get(xls_url, headers=HEADERS, timeout=30)
        xls_resp.raise_for_status()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        JPX_MARGIN_CACHE.write_bytes(xls_resp.content)

        df = pd.read_excel(JPX_MARGIN_CACHE, engine="xlrd", header=None)
        # Real data rows carry an actual datetime cell in column 0; header/
        # footnote rows carry text. Filtering to datetime-typed cells first
        # (rather than pd.to_datetime(..., errors="coerce") on the raw mixed
        # column) avoids pandas falling back to slow, warning-prone
        # per-element string parsing for the whole column.
        is_date = df[0].apply(lambda v: isinstance(v, (pd.Timestamp, datetime.datetime)))
        df = df[is_date].copy()
        df["date"] = pd.to_datetime(df[0]).dt.strftime("%Y-%m-%d")
        df["sold_short"] = pd.to_numeric(df[1], errors="coerce")
        df["bought_margin"] = pd.to_numeric(df[3], errors="coerce")
        df["ratio"] = df["bought_margin"] / df["sold_short"]
        df = df.dropna(subset=["ratio"])
        if df.empty:
            return None

        tail = df.tail(52)
        latest = tail.iloc[-1]
        return {
            "date": latest["date"],
            "bought_thousand_shares": int(latest["bought_margin"]),
            "sold_thousand_shares": int(latest["sold_short"]),
            "ratio": round(float(latest["ratio"]), 2),
            "history": [{"date": r["date"], "ratio": round(float(r["ratio"]), 2)} for _, r in tail.iterrows()],
        }
    except Exception as e:
        print(f"[sentiment] jpx_margin_ratio failed: {e}")
        return None


def _parse_number_row(line: str) -> list[int]:
    return [int(x.replace(",", "")) for x in re.findall(r"-?[\d,]+", line)]


def _fetch_cftc_text() -> str:
    if CFTC_URL not in _cftc_text_cache:
        resp = requests.get(CFTC_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        _cftc_text_cache[CFTC_URL] = resp.text
    return _cftc_text_cache[CFTC_URL]


def _cot_category(long_idx: int, short_idx: int, history_path: Path) -> dict | None:
    """Shared parser for one Long:Short:Spreading category out of the E-mini
    S&P500 block of the CFTC Financial Futures COT report (fixed-format
    plain text). Category order in the "Positions" row is Dealer(0,1,2) /
    Asset Manager(3,4,5) / Leveraged Funds(6,7,8) / Other Reportables(9,10,11)
    / Nonreportable(12,13) as Long:Short[:Spreading].
    """
    text = _fetch_cftc_text()
    start = text.find("E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE")
    if start == -1:
        return None
    block = text[start:start + 3000]

    date_m = re.search(r"as of (\w+ \d{1,2}, \d{4})", block)
    pos_m = re.search(r"Positions\s*\n([^\n]+)", block)
    chg_m = re.search(r"Total Change is:[^\n]*\n([^\n]+)", block)
    if not (date_m and pos_m):
        return None

    pos = _parse_number_row(pos_m.group(1))
    if len(pos) <= short_idx:
        return None
    long_, short_ = pos[long_idx], pos[short_idx]
    net = long_ - short_

    net_change_wow = None
    if chg_m:
        chg = _parse_number_row(chg_m.group(1))
        if len(chg) > short_idx:
            net_change_wow = chg[long_idx] - chg[short_idx]

    record = {
        "date": date_m.group(1), "long": long_, "short": short_,
        "net": net, "net_change_wow": net_change_wow,
    }
    hist = _append_history(history_path, record, dedupe_key="date")
    return {**record, "history": hist.tail(26).to_dict("records")}


def cot_institutional() -> dict | None:
    """CFTC Commitments of Traders(週次・金曜更新)、E-mini S&P500先物の
    Asset Manager/Institutional区分ネットポジション(Long-Short)と前週比。"""
    try:
        return _cot_category(3, 4, COT_HISTORY)
    except Exception as e:
        print(f"[sentiment] cot_institutional failed: {e}")
        return None


def cot_leveraged_funds() -> dict | None:
    """CFTC Commitments of Traders、E-mini S&P500先物のLeveraged Funds区分
    (ヘッジファンド等の投機筋)ネットポジション。NAAIM Exposure Index(2026年に
    サブスクリプション制へ移行し無料取得不可になった、確認済み)の代替として、
    実際の建玉データからアクティブ運用者に近い層のポジションを見る。
    """
    try:
        return _cot_category(6, 7, COT_LEVERAGED_HISTORY)
    except Exception as e:
        print(f"[sentiment] cot_leveraged_funds failed: {e}")
        return None


def overheat_label(vix: dict | None, breadth: dict | None, aaii: dict | None,
                    vix_term: dict | None = None, skew: dict | None = None) -> str:
    """VIX・幅・AAII・VIX期間構造・SKEWを組み合わせた3段階の参考ラベル。
    market_health.classify_marketのBULL/CAUTION/BEAR(分配日ベースのマクロ
    ゲート)とは別軸の情報であり、それ自体を新規/既存ポジションの機械的な
    オン/オフ判定には使わない(このリポジトリの一貫した方針: ヒューリス
    ティックは人間の確認と併用)。JPX信用倍率は市場規模に対する絶対水準の
    妥当な閾値をまだ検証できていないため、このスコアには含めずカードでの
    表示のみに留める。
    """
    score = 0
    signals = 0
    if vix is not None:
        signals += 1
        if vix["level"] < 15:
            score += 1
        elif vix["level"] >= 30:
            score -= 1
    if breadth is not None:
        signals += 1
        if breadth["pct_above_50ma"] >= 85:
            score += 1
        elif breadth["pct_above_50ma"] <= 30:
            score -= 1
    if aaii is not None:
        signals += 1
        spread = aaii["bull_pct"] - aaii["bear_pct"]
        if spread >= 20:
            score += 1
        elif spread <= -20:
            score -= 1
    if vix_term is not None:
        signals += 1
        if vix_term["ratio"] >= 1.0:
            score -= 1
    if skew is not None:
        signals += 1
        if skew["level"] >= 145:
            score -= 1

    if signals == 0:
        return "データ不足"
    if score >= 2:
        return "過熱注意"
    if score <= -1:
        return "弱気寄り(押し目狙いの検討域)"
    return "中立"

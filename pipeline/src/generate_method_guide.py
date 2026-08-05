"""Renders the Markdown methodology docs (投資/チャート分析 + マーケット分析) into
one styled, linkable HTML guide — so PIVOT DESK's "投資手法を学ぶ" link lands
somewhere readable instead of raw .md text in a browser tab.

Always converts from the current .md files on disk, so edits to the research
docs show up next time this regenerates (no manual re-transcription).

Usage:
    python generate_method_guide.py
"""
import datetime as dt
import os
from pathlib import Path

import markdown

INVEST_DIR = Path(os.environ.get(
    "PIVOT_DESK_INVEST_DIR",
    r"C:\Users\user\OneDrive\デスクトップ\投資",
))
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
TEMPLATE_PATH = Path(__file__).resolve().parent / "method_guide_template.html"

DOCS = [
    {"id": "method", "nav": "① ミネルヴィニ投資手法",
     "path": INVEST_DIR / "チャート分析" / "01_ミネルヴィニ投資手法_調査レポート.md"},
    {"id": "market-health", "nav": "② マーケット環境判断",
     "path": INVEST_DIR / "マーケット分析" / "01_マーケット環境判断フレームワーク.md"},
    {"id": "system-design", "nav": "③ 自動化システム設計",
     "path": INVEST_DIR / "チャート分析" / "02_自動化システム設計書.md"},
    {"id": "deep-research", "nav": "④ ディープリサーチ",
     "path": INVEST_DIR / "チャート分析" / "03_ディープリサーチ_実証根拠と限界.md"},
]

MD_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "toc"]


def run():
    sections = []
    nav_items = []
    missing = []
    for doc in DOCS:
        if not doc["path"].exists():
            missing.append(str(doc["path"]))
            continue
        text = doc["path"].read_text(encoding="utf-8")
        html_body = markdown.markdown(text, extensions=MD_EXTENSIONS)
        sections.append(f'<section class="doc" id="{doc["id"]}">{html_body}</section>')
        nav_items.append(f'<a class="nav-item" href="#{doc["id"]}">{doc["nav"]}</a>')

    if missing:
        print("warning: missing doc files:", missing)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = (template
            .replace("__NAV_ITEMS__", "\n".join(nav_items))
            .replace("__SECTIONS__", "\n".join(sections))
            .replace("__DATE__", dt.date.today().isoformat()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "method_guide.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"saved: {out_path}")
    return out_path


if __name__ == "__main__":
    run()

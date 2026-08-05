"""Japanese translations for yfinance's fixed-vocabulary GICS sector/industry
strings. These are a small, closed set (11 sectors, ~70 industries across the
S&P 500), so a static lookup table is more reliable than a translation API —
free-form text (business summaries) is handled separately via a cached
translation file (see company_info.load_summary_translations)."""

SECTOR_JA = {
    "Basic Materials": "素材",
    "Communication Services": "通信サービス",
    "Consumer Cyclical": "一般消費財",
    "Consumer Defensive": "生活必需品",
    "Energy": "エネルギー",
    "Financial Services": "金融",
    "Healthcare": "ヘルスケア",
    "Industrials": "資本財",
    "Real Estate": "不動産",
    "Technology": "情報技術",
    "Utilities": "公益事業",
}

INDUSTRY_JA = {
    "Aerospace & Defense": "航空宇宙・防衛",
    "Agricultural Inputs": "農業用資材",
    "Airlines": "航空",
    "Apparel Retail": "アパレル小売",
    "Asset Management": "資産運用",
    "Auto Manufacturers": "自動車製造",
    "Banks - Diversified": "銀行(総合)",
    "Banks - Regional": "地方銀行",
    "Beverages - Non-Alcoholic": "飲料(ノンアルコール)",
    "Biotechnology": "バイオテクノロジー",
    "Building Products & Equipment": "建材・建築設備",
    "Capital Markets": "資本市場",
    "Communication Equipment": "通信機器",
    "Computer Hardware": "コンピューターハードウェア",
    "Consumer Electronics": "民生用電子機器",
    "Copper": "銅",
    "Diagnostics & Research": "診断・研究サービス",
    "Discount Stores": "ディスカウントストア",
    "Drug Manufacturers - General": "医薬品製造(総合)",
    "Drug Manufacturers - Specialty & Generic": "医薬品製造(専門薬・ジェネリック)",
    "Electronic Components": "電子部品",
    "Electronic Gaming & Multimedia": "ゲーム・マルチメディア",
    "Engineering & Construction": "エンジニアリング・建設",
    "Farm & Heavy Construction Machinery": "農業機械・建設機械",
    "Farm Products": "農産物",
    "Healthcare Plans": "医療保険プラン",
    "Household & Personal Products": "家庭用品・パーソナルケア",
    "Industrial Distribution": "産業用製品卸売",
    "Insurance - Life": "生命保険",
    "Insurance - Property & Casualty": "損害保険",
    "Integrated Freight & Logistics": "総合物流",
    "Internet Content & Information": "インターネットコンテンツ・情報",
    "Internet Retail": "インターネット小売",
    "Lodging": "宿泊業",
    "Luxury Goods": "高級品",
    "Medical Care Facilities": "医療施設",
    "Medical Devices": "医療機器",
    "Medical Distribution": "医薬品・医療用品卸売",
    "Medical Instruments & Supplies": "医療用器具・用品",
    "Oil & Gas E&P": "石油・ガス探査開発",
    "Oil & Gas Equipment & Services": "石油・ガス関連設備・サービス",
    "Oil & Gas Integrated": "石油・ガス(統合型)",
    "Oil & Gas Midstream": "石油・ガス(中流:輸送・貯蔵)",
    "Oil & Gas Refining & Marketing": "石油精製・販売",
    "Packaged Foods": "加工食品",
    "Packaging & Containers": "包装・容器",
    "REIT - Healthcare Facilities": "REIT(医療施設)",
    "REIT - Hotel & Motel": "REIT(ホテル)",
    "REIT - Industrial": "REIT(産業用施設)",
    "REIT - Retail": "REIT(商業施設)",
    "REIT - Specialty": "REIT(特殊施設)",
    "Railroads": "鉄道",
    "Rental & Leasing Services": "レンタル・リースサービス",
    "Resorts & Casinos": "リゾート・カジノ",
    "Restaurants": "外食",
    "Scientific & Technical Instruments": "科学・技術機器",
    "Semiconductor Equipment & Materials": "半導体製造装置・材料",
    "Semiconductors": "半導体",
    "Software - Application": "ソフトウェア(アプリケーション)",
    "Software - Infrastructure": "ソフトウェア(インフラ)",
    "Specialty Chemicals": "特殊化学品",
    "Specialty Industrial Machinery": "特殊産業機械",
    "Specialty Retail": "専門小売",
    "Steel": "鉄鋼",
    "Tobacco": "たばこ",
    "Tools & Accessories": "工具・アクセサリー",
    "Travel Services": "旅行サービス",
    "Trucking": "陸運(トラック輸送)",
    "Utilities - Regulated Electric": "電力(規制対象)",
}


def sector_ja(s: str | None) -> str | None:
    if s is None:
        return None
    return SECTOR_JA.get(s, s)


def industry_ja(s: str | None) -> str | None:
    if s is None:
        return None
    return INDUSTRY_JA.get(s, s)

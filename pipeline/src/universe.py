"""Ticker universe helpers for the Trend Template screener (Phase 1 prototype)."""
import re

import pandas as pd

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

_VALID_SYMBOL = re.compile(r"^[A-Z]{1,5}$")


def get_sp500_tickers() -> list[str]:
    df = pd.read_csv(SP500_URL)
    tickers = df["Symbol"].str.replace(".", "-", regex=False).unique().tolist()
    return sorted(tickers)


def _clean_symbols(symbols: pd.Series) -> list[str]:
    """Keep plain common-stock-looking tickers: pure letters, <=5 chars.
    Drops warrant/unit/rights suffixes and other special-class symbols that
    NASDAQ Trader's listing files don't cleanly flag on their own."""
    return sorted(s for s in symbols.dropna().unique() if _VALID_SYMBOL.match(s))


_NON_COMMON_NAME = re.compile(
    r"warrant|right|unit|preferred|depositary|trust pfd|subordinated|notes?\b",
    re.IGNORECASE,
)


def get_nasdaq_nyse_universe() -> list[str]:
    """Full current NASDAQ + NYSE/AMEX common-stock-ish listing (free, public,
    no API key — design doc §2.2). ETFs, test issues, and non-common-stock
    listings (warrants/units/rights/preferred — identified from the Security
    Name text, since ticker-suffix heuristics alone miss plenty of real
    common-stock tickers that happen to end in the same letters) excluded."""
    nas = pd.read_csv(NASDAQ_LISTED_URL, sep="|")
    nas = nas[: -1]  # drop the "File Creation Time" footer row
    nas = nas[(nas["Test Issue"] == "N") & (nas["ETF"] == "N")
              & (~nas["Security Name"].str.contains(_NON_COMMON_NAME, na=False))]

    oth = pd.read_csv(OTHER_LISTED_URL, sep="|")
    oth = oth[: -1]
    oth = oth[(oth["Test Issue"] == "N") & (oth["ETF"] == "N")
              & (~oth["Security Name"].str.contains(_NON_COMMON_NAME, na=False))]

    symbols = pd.concat([nas["Symbol"], oth["ACT Symbol"]], ignore_index=True)
    return _clean_symbols(symbols)


def get_broad_growth_universe(sample_n: int = 700, seed: int = 42) -> list[str]:
    """S&P 500 (large caps) + a random sample of the rest of the NASDAQ/NYSE
    listing (where most small/mid-cap growth names — the classic Minervini
    hunting ground — actually live, per the design doc's own universe
    definition). Sampled rather than exhaustive to keep an 8-year backtest
    over thousands of tickers within a reasonable runtime; re-run with a
    different --seed or larger --sample-n for a fuller pass."""
    sp500 = set(get_sp500_tickers())
    full = get_nasdaq_nyse_universe()
    rest = [t for t in full if t not in sp500]
    sample = pd.Series(rest).sample(n=min(sample_n, len(rest)), random_state=seed).tolist()
    return sorted(sp500.union(sample))


SMOKE_TEST_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "COST",
    "LLY", "NFLX", "AMD", "CRM", "PANW", "NOW", "ANET", "FICO",
]

"""Download daily ETF bars for the real-data case study.

Pulls split/dividend-adjusted OHLCV for a small, deliberately diverse basket
(equity / bonds / gold) and writes one CSV per ticker to ``data/``. The CSVs are
gitignored — this script is the reproducible source of truth, not the data.

    pip install -e ".[data]"
    python examples/fetch_data.py
"""

from __future__ import annotations

import pathlib

import yfinance as yf

TICKERS = ["SPY", "TLT", "GLD", "QQQ", "EFA"]  # stocks, long bonds, gold, tech, intl
START = "2010-01-01"
END = "2024-12-31"

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def fetch() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for ticker in TICKERS:
        df = yf.download(
            ticker, start=START, end=END, progress=False, auto_adjust=True
        )
        if df.empty:
            print(f"{ticker}: no data, skipping")
            continue
        # yfinance returns a (field, ticker) MultiIndex; flatten to lowercase OHLCV.
        df.columns = [str(c[0]).lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "date"
        out = DATA_DIR / f"{ticker}.csv"
        df.to_csv(out)
        print(f"{ticker}: {len(df)} rows -> {out.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    fetch()

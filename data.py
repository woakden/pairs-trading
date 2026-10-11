"""Download daily closing prices (adjusted for splits and dividends) and cache them as CSV."""
from pathlib import Path

import pandas as pd


def download(tickers, start, end=None):
    """Adjusted daily closes from Yahoo Finance, one column per ticker."""
    import yfinance as yf  # imported here so the tests and --csv runs don't need it

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise SystemExit("No prices downloaded. Check the tickers and your internet connection.")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    closes = closes[list(tickers)]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes.dropna()


def load_csv(path):
    """Prices saved by save_csv (a Date column, then one column per ticker)."""
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.dropna()


def save_csv(prices, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="Date")
    return path

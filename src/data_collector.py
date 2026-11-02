"""
Data collection module for fetching stock price data and computing indicators.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


def download_stock_data(ticker: str = "AAPL", period: str = "5y") -> pd.DataFrame:
    """Download OHLCV historical stock data using yfinance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception as e:
        print(f"[WARN] yfinance download failed ({e}), generating synthetic stock data...")
        dates = pd.date_range(end=pd.Timestamp.today(), periods=1260, freq="B")
        price = 150.0 + np.cumsum(np.random.normal(0.05, 1.5, size=len(dates)))
        df = pd.DataFrame({
            "Open": price + np.random.normal(0, 0.5, size=len(dates)),
            "High": price + np.abs(np.random.normal(1.0, 0.5, size=len(dates))),
            "Low": price - np.abs(np.random.normal(1.0, 0.5, size=len(dates))),
            "Close": price,
            "Volume": np.random.randint(10000000, 50000000, size=len(dates)),
        }, index=dates)

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA, EMA, RSI, MACD, and Bollinger Bands."""
    df = df.copy()

    # Simple Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # Exponential Moving Average
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Bollinger Bands
    df["BB_upper"] = df["SMA_20"] + 2 * df["Close"].rolling(window=20).std()
    df["BB_lower"] = df["SMA_20"] - 2 * df["Close"].rolling(window=20).std()

    # Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df.bfill()

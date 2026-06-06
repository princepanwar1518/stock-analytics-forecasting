"""
analytics.py — Technical indicators and return statistics
"""
import pandas as pd
import numpy as np


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MACD, Bollinger Bands, and EMA."""
    close = df['Close']

    # EMA 20
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']   = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['BB_Mid']   = sma20

    df.dropna(inplace=True)
    return df


def compute_returns_stats(df: pd.DataFrame) -> dict:
    """Compute return statistics."""
    returns = df['Close'].pct_change().dropna()
    return {
        'mean_return': returns.mean(),
        'volatility':  returns.std(),
        'sharpe':      (returns.mean() / returns.std()) * np.sqrt(252),
        'max_drawdown': ((df['Close'] / df['Close'].cummax()) - 1).min(),
    }

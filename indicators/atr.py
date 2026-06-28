import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14):
    """
    Average True Range (volatility)
    """

    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    df["ATR"] = true_range.rolling(window=period).mean()

    return df
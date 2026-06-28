import pandas as pd


def calculate_rsi(df: pd.DataFrame, period: int = 14):
    """
    Calculate RSI (Relative Strength Index)
    Adds a new column: 'RSI'
    """

    delta = df["Close"].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df
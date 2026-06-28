import pandas as pd


def calculate_ema(df: pd.DataFrame, period: int = 20):
    """
    Exponential Moving Average
    """
    df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
    return df
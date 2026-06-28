import pandas as pd
import numpy as np
from data.downloader import download_stock_data
from indicators.rsi import calculate_rsi
from indicators.ema import calculate_ema
from indicators.macd import calculate_macd
from indicators.atr import calculate_atr


def create_dataset(symbol="AAPL"):
    """
    Build ML dataset from historical data
    """

    df = download_stock_data(symbol, years=5)

    if df is None or len(df) < 100:
        return None

    # indicators
    df = calculate_rsi(df)
    df = calculate_ema(df, 20)
    df = calculate_macd(df)
    df = calculate_atr(df)

    # target: will price go UP next day?
    df["future_return"] = df["Close"].shift(-1) - df["Close"]
    df["target"] = (df["future_return"] > 0).astype(int)

    # clean dataset
    df = df.dropna()

    features = [
        "Close",
        "Volume",
        "RSI",
        "EMA_20",
        "MACD",
        "MACD_signal",
        "ATR"
    ]

    X = df[features]
    y = df["target"]

    return X, y


if __name__ == "__main__":
    X, y = create_dataset("AAPL")

    print("Dataset shape:")
    print(X.shape)
    print("\nSample:")
    print(X.head())
    print("\nLabels:")
    print(y.head())
import joblib
import pandas as pd

from data.downloader import download_stock_data
from indicators.rsi import calculate_rsi
from indicators.ema import calculate_ema
from indicators.macd import calculate_macd
from indicators.atr import calculate_atr

from xgboost import XGBClassifier


FEATURES = [
    "Close", "Volume",
    "RSI", "EMA_20",
    "MACD", "MACD_signal",
    "ATR",
    "close_mean_10",
    "close_std_10",
    "momentum"
]


def prepare_data(df):

    df = calculate_rsi(df)
    df = calculate_ema(df, 20)
    df = calculate_macd(df)
    df = calculate_atr(df)

    df["close_mean_10"] = df["Close"].rolling(10).mean()
    df["close_std_10"] = df["Close"].rolling(10).std()
    df["momentum"] = df["Close"] - df["Close"].rolling(10).mean()

    df = df.dropna()
    return df


def build_dataset(symbol="AAPL"):

    df = download_stock_data(symbol)
    df = prepare_data(df)

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    X = df[FEATURES]
    y = df["target"]

    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    print(f"\nModel trained for AAPL")
    print(f"Accuracy: {round(acc, 4)}")

    joblib.dump(model, "models/xgb_model.pkl")
    print("Model saved to models/xgb_model.pkl")


if __name__ == "__main__":
    build_dataset("AAPL")
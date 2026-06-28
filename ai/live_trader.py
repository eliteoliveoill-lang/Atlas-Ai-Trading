import time
import joblib
import numpy as np

from data.downloader import download_stock_data
from ai.predict_ml import prepare_data

MODEL_PATH = "models/xgb_model.pkl"

FEATURES = [
    "Close", "Volume",
    "RSI", "EMA_20",
    "MACD", "MACD_signal",
    "ATR",
    "close_mean_10",
    "close_std_10",
    "momentum"
]


class LiveTrader:

    def __init__(self, symbol="AAPL"):
        self.symbol = symbol
        self.model = joblib.load(MODEL_PATH)

        self.position = 0
        self.entry_price = 0
        self.equity = 1.0

        self.trades = []

    def step(self):

        df = download_stock_data(self.symbol)
        df = prepare_data(df)

        row = df.iloc[-1]

        X = row[FEATURES].values.reshape(1, -1)
        prob = self.model.predict_proba(X)[0][1]

        price = row["Close"]

        signal = "HOLD"

        # BUY
        if prob > 0.62 and self.position == 0:
            self.position = 1
            self.entry_price = price
            signal = "BUY"

        # SELL
        elif prob < 0.45 and self.position == 1:
            self.position = 0

            pnl = (price - self.entry_price) / self.entry_price
            pnl -= 0.001

            self.equity *= (1 + pnl)
            self.trades.append(pnl)

            signal = "SELL"

        return {
            "price": price,
            "prob": round(prob, 3),
            "signal": signal,
            "equity": round(self.equity, 4),
            "trades": len(self.trades)
        }


def run_live(symbol="AAPL"):

    trader = LiveTrader(symbol)

    print("\nSTARTING LIVE PAPER TRADING...\n")

    while True:

        result = trader.step()

        print(result)

        time.sleep(60)  # runs every 1 minute


if __name__ == "__main__":
    run_live("AAPL")
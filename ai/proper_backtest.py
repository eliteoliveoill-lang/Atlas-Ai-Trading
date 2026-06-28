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


STOCKS = ["AAPL", "MSFT", "TSLA", "SPY"]


def run_single_backtest(model, symbol):

    df = download_stock_data(symbol)

    if df is None or len(df) < 200:
        print(f"{symbol}: not enough data")
        return None

    df = prepare_data(df).reset_index(drop=True)

    equity = 1.0
    trades = []
    position = 0
    entry_price = 0

    for i in range(100, len(df)):

        row = df.iloc[i]

        if row[FEATURES].isnull().any():
            continue

        X = row[FEATURES].values.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        price = row["Close"]

        # simple consistent logic
        if prob > 0.62 and position == 0:
            position = 1
            entry_price = price

        elif prob < 0.45 and position == 1:
            position = 0

            pnl = (price - entry_price) / entry_price
            pnl -= 0.001  # fee

            equity *= (1 + pnl)
            trades.append(pnl)

    # close position
    if position == 1:
        pnl = (df.iloc[-1]["Close"] - entry_price) / entry_price
        equity *= (1 + pnl)
        trades.append(pnl)

    if len(trades) == 0:
        return {
            "symbol": symbol,
            "return": 0,
            "trades": 0,
            "winrate": 0
        }

    wins = [t for t in trades if t > 0]

    return {
        "symbol": symbol,
        "return": (equity - 1) * 100,
        "trades": len(trades),
        "winrate": len(wins) / len(trades) * 100
    }


def stress_test():

    print("\nRUNNING STRESS TEST...\n")

    model = joblib.load(MODEL_PATH)

    results = []

    for stock in STOCKS:
        print(f"Testing {stock}...")
        res = run_single_backtest(model, stock)
        if res:
            results.append(res)

    print("\n=== STRESS TEST RESULTS ===")

    total_return = 0
    total_winrate = 0
    total_trades = 0

    for r in results:
        print(r)
        total_return += r["return"]
        total_winrate += r["winrate"]
        total_trades += r["trades"]

    n = len(results)

    if n == 0:
        print("No valid results")
        return

    print("\n=== AVERAGE PERFORMANCE ===")
    print(f"Avg return: {round(total_return / n, 2)}%")
    print(f"Avg winrate: {round(total_winrate / n, 2)}%")
    print(f"Total trades: {total_trades}")


if __name__ == "__main__":
    stress_test()
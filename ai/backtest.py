import pandas as pd
from data.downloader import download_stock_data
from ai.predict_ml import predict


def backtest(symbol="AAPL"):
    df = download_stock_data(symbol)

    if df is None or len(df) < 50:
        print("Not enough data")
        return

    df = df.tail(200).copy()

    position = 0  # 0 = flat, 1 = holding
    entry_price = 0

    trades = []
    equity = 1.0

    for i in range(len(df)):

        row = df.iloc[i]
        price = row["Close"]

        try:
            result = predict(symbol)
            prob = result["probability"]
        except:
            prob = 0.5

        # BUY
        if prob > 0.65 and position == 0:
            position = 1
            entry_price = price

        # SELL
        elif prob < 0.35 and position == 1:
            position = 0

            pnl = (price - entry_price) / entry_price
            equity *= (1 + pnl)

            trades.append(pnl)

    # close final trade if still open
    if position == 1:
        pnl = (df.iloc[-1]["Close"] - entry_price) / entry_price
        equity *= (1 + pnl)
        trades.append(pnl)

    wins = [t for t in trades if t > 0]

    win_rate = len(wins) / len(trades) if trades else 0

    print("\n=== BACKTEST RESULTS ===")
    print(f"Trades: {len(trades)}")
    print(f"Win rate: {round(win_rate * 100, 2)}%")
    print(f"Final return: {round((equity - 1) * 100, 2)}%")

if __name__ == "__main__":
    backtest("AAPL")
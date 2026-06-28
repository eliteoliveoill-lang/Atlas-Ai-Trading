import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indicators.rsi import calculate_rsi
from indicators.rsi import calculate_rsi
from indicators.ema import calculate_ema
from indicators.macd import calculate_macd
from indicators.atr import calculate_atr
from ai.predict import generate_signal

def download_stock_data(symbol: str, years: int = None):
    """
    Downloads historical stock data from Yahoo Finance
    """

    if years is None:
        years = config.LOOKBACK_YEARS

    end = datetime.today()
    start = end - timedelta(days=365 * years)

    print(f"Downloading {symbol} from {start.date()} to {end.date()}")

    df = yf.download(symbol, start=start, end=end, interval=config.TIMEFRAME)

    if df.empty:
        print(f"No data found for {symbol}")
        return None

    df.reset_index(inplace=True)

    from indicators.rsi import calculate_rsi
    df = calculate_rsi(df)
    df = calculate_ema(df, 20)
    df = calculate_macd(df)
    df = calculate_atr(df)

    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index(drop=True)

    return df


def download_watchlist():
    """
    Downloads all stocks in watchlist
    """

    data = {}

    for symbol in config.WATCHLIST:
        df = download_stock_data(symbol)

        if df is not None:
            data[symbol] = df

            print(f"\n{symbol} preview:")
            print(data[symbol].head())

    for symbol in data:
        result = generate_signal(data[symbol])
        print(f"\n{symbol} AI RESULT:")
        print(result)

    return data


if __name__ == "__main__":
    data = download_watchlist()
    print("Downloaded:", list(data.keys()))
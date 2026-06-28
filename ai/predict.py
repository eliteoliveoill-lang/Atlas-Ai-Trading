import pandas as pd


def generate_signal(df: pd.DataFrame):
    """
    Simple rule-based probability engine (first version of your AI brain)
    """

    latest = df.tail(1).squeeze()

    score = 0

    # Trend (EMA)
    if latest["Close"] > latest["EMA_20"]:
        score += 2
    else:
        score -= 2

    # Momentum (RSI)
    if 40 < latest["RSI"] < 70:
        score += 1
    elif latest["RSI"] >= 70:
        score -= 2
    elif latest["RSI"] <= 30:
        score += 2
    
    # MACD
    if latest["MACD"] > latest["MACD_signal"]:
        score += 2
    else:
        score -= 2

    # Volatility (ATR - simple filter)
    if latest["ATR"] > df["ATR"].mean():
        score -= 1  # too volatile = risk

    # Convert score → probability (simple mapping)
    probability = (score + 5) / 10  # scales roughly 0–1
    probability = max(0, min(1, probability))

    if probability > 0.65:
        signal = "BULLISH"
    elif probability < 0.35:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "signal": signal,
        "probability": round(probability, 2),
        "score": score
    }
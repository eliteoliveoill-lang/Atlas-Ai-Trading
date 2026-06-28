import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

from ai.predict_ml import predict_from_row

# =========================
# LIVE REFRESH
# =========================
st_autorefresh(interval=5000, key="live_refresh")

# =========================
# APP SETUP
# =========================
st.set_page_config(page_title="Atlas AI Live Analyst", layout="wide")
st.title("📊 Atlas AI Live Market Analyst")

symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")

model = joblib.load("models/xgb_model.pkl")

# =========================
# DATA
# =========================
df = yf.download(symbol, period="1d", interval="1m")

if df is None or len(df) < 50:
    st.warning("Not enough data yet.")
    st.stop()

# Flatten MultiIndex columns if Yahoo returns them
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

df = df.dropna().tail(150).copy().reset_index(drop=True)

# =========================
# FEATURES
# =========================
df["RSI"] = df["Close"].rolling(14).mean()
df["EMA_20"] = df["Close"].ewm(span=20).mean()
df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
df["ATR"] = df["High"] - df["Low"]

df["close_mean_10"] = df["Close"].rolling(10).mean()
df["close_std_10"] = df["Close"].rolling(10).std()
df["momentum"] = df["Close"] - df["close_mean_10"]

df = df.dropna().reset_index(drop=True)

# =========================
# STORAGE
# =========================
probs = []
signals = []
reasons = []

# =========================
# LOOP
# =========================
for i in range(len(df)):

    row = df.iloc[i]

    # AI prediction
    prob = predict_from_row(row)
    probs.append(prob)

    # base score = model output
    score = prob

    # IMPORTANT: use row, NOT latest
    if row["Close"] > row["EMA_20"]:
        score += 0.05
    else:
        score -= 0.05

    if row["momentum"] > 0:
        score += 0.03
    else:
        score -= 0.03

    if row["MACD"] > row["MACD_signal"]:
        score += 0.04
    else:
        score -= 0.04

    # clamp score so it doesn't break confidence
    score = max(0, min(1, score))

    # signal
    if score > 0.58:
        signal = "BUY"
    elif score < 0.42:
        signal = "SELL"
    else:
        signal = "HOLD"

    signals.append(signal)

    # explanation (use row only)
    explanation = []

    if row["Close"] > row["EMA_20"]:
        explanation.append("Price above EMA")
    else:
        explanation.append("Price below EMA")

    if row["momentum"] > 0:
        explanation.append("Positive momentum")
    else:
        explanation.append("Negative momentum")

    if row["MACD"] > row["MACD_signal"]:
        explanation.append("MACD bullish")
    else:
        explanation.append("MACD bearish")

    reasons.append(" | ".join(explanation))

# =========================
# OUTPUT DATA
# =========================
df["AI_Prob"] = probs
df["Signal"] = signals
df["Reason"] = reasons

buy_points = df[df["Signal"] == "BUY"]
sell_points = df[df["Signal"] == "SELL"]

latest = df.iloc[-1]

# Save latest AI results for sidebar chat
st.session_state.latest_signal = signals[-1]
st.session_state.latest_confidence = probs[-1]
st.session_state.latest_reason = reasons[-1]
st.session_state.latest_price = latest["Close"]

# =========================
# ATLAS AI REPORT
# =========================

raw = probs[-1]

# amplify signal clarity
confidence = min(100, max(0, (raw - 0.5) * 200 + 50))

# -------- Trend Score --------
trend_score = 100 if latest["Close"] > latest["EMA_20"] else 25

# -------- Momentum Score --------
if latest["momentum"] > 0:
    momentum_score = min(100, 50 + abs(latest["momentum"]) * 10)
else:
    momentum_score = max(0, 50 - abs(latest["momentum"]) * 10)

# -------- Volatility Score --------
atr_percent = latest["ATR"] / latest["Close"]

if atr_percent < 0.01:
    volatility_score = 95
    volatility_text = "Low"
elif atr_percent < 0.02:
    volatility_score = 75
    volatility_text = "Moderate"
elif atr_percent < 0.03:
    volatility_score = 55
    volatility_text = "High"
else:
    volatility_score = 30
    volatility_text = "Very High"

# -------- Overall AI Rating --------
overall_score = (
    confidence * 0.45 +
    trend_score * 0.25 +
    momentum_score * 0.20 +
    volatility_score * 0.10
)

overall_score = round(overall_score, 1)

# =========================
# HEADER METRICS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💲 Live Price", f"${latest['Close']:.2f}")

with col2:
    st.metric("🤖 AI Signal", signals[-1])

with col3:
    st.metric("🎯 Confidence", f"{confidence:.1f}%")

st.divider()

# =========================
# AI REPORT
# =========================
st.subheader("🧠 Atlas AI Market Report")

st.metric("⭐ Overall Rating", f"{overall_score}/100")

c1, c2 = st.columns(2)

with c1:
    st.metric("📈 Trend Score", f"{round(trend_score)}/100")
    st.metric("⚡ Momentum Score", f"{round(momentum_score)}/100")

with c2:
    st.metric("🌊 Volatility", volatility_text)
    st.metric("🎯 Model Confidence", f"{confidence:.1f}%")

# =========================
# RECOMMENDATION
# =========================
st.subheader("📋 Recommendation")

if overall_score >= 85:
    st.success("🟢 Strong BUY")
elif overall_score >= 70:
    st.info("🟢 BUY")
elif overall_score >= 55:
    st.warning("🟡 HOLD")
elif overall_score >= 40:
    st.warning("🟠 Weak SELL")
else:
    st.error("🔴 Strong SELL")

# =========================
# WHY
# =========================
st.subheader("🔍 Why Atlas AI Thinks This")

for reason in reasons[-1].split(","):
    st.write(f"• {reason.strip()}")

# =========================
# CHART
# =========================
fig = go.Figure()

# Candlestick chart
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
))

# BUY markers
fig.add_trace(go.Scatter(
    x=buy_points.index,
    y=buy_points["Close"],
    mode="markers",
    marker=dict(
        color="lime",
        size=12,
        symbol="triangle-up"
    ),
    name="BUY"
))

# SELL markers
fig.add_trace(go.Scatter(
    x=sell_points.index,
    y=sell_points["Close"],
    mode="markers",
    marker=dict(
        color="red",
        size=12,
        symbol="triangle-down"
    ),
    name="SELL"
))

# EMA Trend
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA_20"],
    mode="lines",
    line=dict(color="orange", width=2),
    name="EMA 20"
))

fig.update_layout(
    title=f"{symbol} • Atlas AI Live Analyst",
    height=700,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)
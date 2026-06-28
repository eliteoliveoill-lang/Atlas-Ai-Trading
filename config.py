# Atlas AI Analyst - Config File

# =========================
# GENERAL SETTINGS
# =========================

PROJECT_NAME = "Atlas AI Analyst"

# Timeframe for analysis (1d, 1h, 5m, etc.)
TIMEFRAME = "1d"

# How many years of historical data to pull
LOOKBACK_YEARS = 5

# =========================
# STOCK UNIVERSE
# =========================

# Start small (we expand later)
WATCHLIST = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "TSLA",
    "META",
    "GOOGL"
]

# =========================
# DATA SETTINGS
# =========================

DATA_SOURCE = "yfinance"

# =========================
# AI SETTINGS (later phase)
# =========================

MODEL_PATH = "models/atlas_model.pkl"

TRAIN_TEST_SPLIT = 0.8

# Probability threshold for "bullish signal"
BULLISH_THRESHOLD = 0.65
BEARISH_THRESHOLD = 0.35

# =========================
# RISK SETTINGS (future trading use)
# =========================

MAX_RISK_PER_TRADE = 0.01  # 1%
STOP_LOSS_PCT = 0.02       # 2%
TAKE_PROFIT_PCT = 0.04     # 4%
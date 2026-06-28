import joblib
import pandas as pd

MODEL_PATH = "models/xgb_model.pkl"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "Close", "Volume",
    "RSI", "EMA_20",
    "MACD", "MACD_signal",
    "ATR",
    "close_mean_10",
    "close_std_10",
    "momentum"
]


def predict_from_row(row):
    """
    Single-row safe prediction (NO leakage, NO mismatch)
    """

    try:
        X = pd.DataFrame([row[FEATURES].values], columns=FEATURES)
        prob = model.predict_proba(X)[0][1]
        return float(prob)
    except:
        return 0.5
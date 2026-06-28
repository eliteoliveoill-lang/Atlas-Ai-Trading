import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from ai.train import create_dataset


MODEL_PATH = "models/xgb_model.pkl"


def train_model(symbol="AAPL"):
    """
    Train XGBoost model on your dataset
    """

    X, y = create_dataset(symbol)

    if X is None:
        print("No data available")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    print(f"\nModel trained for {symbol}")
    print(f"Accuracy: {round(accuracy, 4)}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_model("AAPL")
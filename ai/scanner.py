from ai.predict_ml import predict
import config


def scan_market():
    results = []

    for symbol in config.WATCHLIST:
        try:
            result = predict(symbol)
            results.append(result)
        except Exception as e:
            print(f"Error on {symbol}: {e}")

    # sort by probability (highest first)
    results = sorted(results, key=lambda x: x["probability"], reverse=True)

    print("\n=== AI MARKET SCAN ===\n")

    for r in results:
        print(f"{r['symbol']}: {r['signal']} ({r['probability']})")

    return results


if __name__ == "__main__":
    scan_market()


import argparse
import torch

from app.data import download_stock_data, create_sequences
from app.model import LSTMForecaster


SEQUENCE_LENGTH = 60


def forecast_next_close(ticker: str):
    df = download_stock_data(ticker)

    X, y, scaler = create_sequences(
        df,
        sequence_length=SEQUENCE_LENGTH
    )

    last_sequence = X[-1]

    last_sequence_tensor = torch.tensor(
        last_sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    model = LSTMForecaster()

    model.load_state_dict(
        torch.load("lstm_model.pth")
    )

    model.eval()

    with torch.no_grad():
        next_scaled_prediction = model(
            last_sequence_tensor
        ).numpy()

    next_price_prediction = scaler.inverse_transform(
        next_scaled_prediction
    )[0][0]

    latest_actual_price = df["Close"].iloc[-1]

    change = next_price_prediction - latest_actual_price

    percent_change = (
        change / latest_actual_price
    ) * 100

    return {
        "ticker": ticker.upper(),
        "latest_close": float(latest_actual_price),
        "predicted_next_close": float(next_price_prediction),
        "predicted_change": float(change),
        "predicted_percent_change": float(percent_change),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forecast the next stock closing price."
    )

    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol"
    )

    args = parser.parse_args()

    result = forecast_next_close(args.ticker)

    print(f"Ticker: {result['ticker']}")
    print(f"Latest close: ${result['latest_close']:.2f}")
    print(
        f"Predicted next close: "
        f"${result['predicted_next_close']:.2f}"
    )

    print(
        f"Predicted change: "
        f"${result['predicted_change']:.2f}"
    )

    print(
        f"Predicted percent change: "
        f"{result['predicted_percent_change']:.2f}%"
    )
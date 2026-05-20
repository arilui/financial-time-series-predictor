import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def download_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    data = yf.download(ticker, period=period, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    data = data.reset_index()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if col[1] == "" else col[0] for col in data.columns]

    return data


def create_sequences(data: pd.DataFrame, sequence_length: int = 60):
    """
    Uses previous 60 closing prices to predict the next closing price.
    """
    close_prices = data[["Close"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    X = []
    y = []

    for i in range(sequence_length, len(scaled_prices)):
        X.append(scaled_prices[i - sequence_length:i])
        y.append(scaled_prices[i])

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


if __name__ == "__main__":
    df = download_stock_data("AAPL")
    X, y, scaler = create_sequences(df)

    print("Data shape:", df.shape)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First sequence shape:", X[0].shape)
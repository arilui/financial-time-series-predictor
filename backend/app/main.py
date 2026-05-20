from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.forecast import forecast_next_close
from app.data import download_stock_data


app = FastAPI(
    title="Financial Time Series Predictor API",
    description="LSTM-based stock price forecasting API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Financial Time Series Predictor API is running"
    }


@app.get("/forecast/{ticker}")
def get_forecast(ticker: str):
    result = forecast_next_close(ticker)
    return result


@app.get("/history/{ticker}")
def get_history(ticker: str):
    df = download_stock_data(ticker)

    history = df[["Date", "Close"]].tail(90)

    return {
        "ticker": ticker.upper(),
        "history": [
            {
                "date": str(row["Date"].date()),
                "close": float(row["Close"]),
            }
            for _, row in history.iterrows()
        ],
    }
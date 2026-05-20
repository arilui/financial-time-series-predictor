# Financial Time Series Predictor

A full-stack deep learning financial forecasting platform built with PyTorch, FastAPI, and Next.js.

## Features

- LSTM-based stock price forecasting
- Historical stock price visualization
- Interactive forecasting dashboard
- FastAPI backend API
- Next.js frontend
- Real-time market data from Yahoo Finance

## Tech Stack

### Backend
- Python
- PyTorch
- FastAPI
- Pandas
- NumPy
- yfinance

### Frontend
- Next.js
- React
- Tailwind CSS
- Recharts

## Project Structure

```txt
backend/
  app/
    data.py
    forecast.py
    main.py
    model.py
    predict.py
    train.py

frontend/
```

## Running the Backend

```bash
cd backend
uvicorn app.main:app --reload
```

## Running the Frontend

```bash
cd frontend
npm run dev
```

## Future Improvements

- Multi-day forecasting
- Transformer-based forecasting models
- Sentiment analysis integration
- Authentication
- Deployment to AWS/Vercel
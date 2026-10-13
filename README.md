# Weekly Volatility Forecaster

How far is a stock likely to move over the next five trading days? This app forecasts next-week volatility for 15 major stocks and ETFs, shows the expected price range, and reports how often those ranges have actually held.

**Live app:** https://financial-time-series-predictor.vercel.app

Forecasts update automatically every weekday after the US market close.

---

## Results

### Can an LSTM predict tomorrow's price? No.

This project began as a PyTorch LSTM that predicted the next day's closing price. A walk-forward backtest (2015 onward, six stocks: AAPL, MSFT, AMZN, JPM, XOM, SPY) found no real edge:

| Model | Error vs. "tomorrow = today" | Directional accuracy |
|---|---|---|
| LSTM | –0.0% to +0.2% | 52.6% |
| Linear model (last 20 returns) | –0.0% to +0.1% | 53.1% |
| Always guess "up" | n/a | 53.0% |

The LSTM called up-vs-down correctly 52.6% of the time, below the 53.0% you get by always guessing "up". Charts of LSTM price predictions that seem to track the real price closely usually just repeat yesterday's price.

### Can it predict next week's volatility? Yes, but a simpler model does better.

Volatility clusters (rough weeks tend to follow rough weeks), so it is partly predictable. Error reduction vs. the naive forecast "next week will be as volatile as last week":

| Ticker | EWMA | LSTM | HAR |
|---|---|---|---|
| AAPL | +11.6% | +16.5% | **+17.6%** |
| MSFT | +9.7% | +12.8% | **+15.9%** |
| AMZN | +7.6% | +13.1% | **+14.8%** |
| JPM | +13.3% | +18.5% | **+19.3%** |
| XOM | +16.6% | +15.2% | **+18.5%** |
| SPY | +7.4% | +7.5% | **+14.1%** |

The HAR model (a linear model on the past day's, week's and month's volatility; Corsi, 2009) beat the LSTM on all six stocks, so it powers the live app. Full results: [research/results.md](research/results.md).

### Are the forecast ranges honest?

Across 15 tickers and roughly 350 non-overlapping test weeks each, the real five-day move landed inside the forecast's **68% range 64–69% of the time** and inside its **90% range 85–91% of the time**. The 90% range runs slightly narrow because real returns have fatter tails than a normal distribution. Each ticker's own coverage is shown in the app.

---

## How It Avoids Look-Ahead Bias

- **Walk-forward evaluation:** train on all data up to a date, predict the next year, roll forward, retrain. Models never see the future.
- **Train-only scaling:** features are standardized with training-window statistics only.
- **Chronological early stopping:** the LSTM's validation set is the last 15% of each training window, never the test period.
- **Synthetic self-check:** `python research/backtest.py --synthetic` runs the same pipeline on simulated GARCH data, where direction is unpredictable and volatility is predictable by construction, to confirm the backtest finds signal only where it exists.

---

## Architecture

```text
GitHub Actions (weekdays, after market close)
        ↓
pipeline/forecast.py
  download prices (yfinance) → fit HAR → walk-forward accuracy + range coverage
        ↓
frontend/public/forecasts.json  (committed by the workflow)
        ↓
Vercel redeploys the Next.js app on each commit
```

There is no live backend: forecasts are precomputed once a day, so the site is fast, free to host, and unaffected by market-data rate limits from cloud servers.

---

## Tech Stack

**Modeling:** Python, PyTorch (LSTM), scikit-learn (HAR, linear baselines), pandas, NumPy, yfinance

**Pipeline:** GitHub Actions (scheduled)

**Frontend:** Next.js, React, TypeScript, Tailwind CSS, Recharts, deployed on Vercel

---

## Project Structure

```text
.github/workflows/daily-forecast.yml   scheduled forecast update
pipeline/forecast.py                   production HAR forecasts → forecasts.json
research/backtest.py                   walk-forward comparison: LSTM vs. baselines
research/results.md                    backtest output
frontend/                              Next.js app
backend/                               original LSTM price-prediction API (superseded)
```

---

## Running Locally

```bash
pip install -r pipeline/requirements.txt torch
python research/backtest.py          # full comparison, ~15–30 min (add --no-lstm for seconds)
python pipeline/forecast.py          # regenerate frontend/public/forecasts.json

cd frontend
npm install
npm run dev                          # http://localhost:3000
```

---

This project is for learning and research, not investment advice.

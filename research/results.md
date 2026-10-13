# Walk-Forward Backtest Results

Tickers: AAPL, MSFT, AMZN, JPM, XOM, SPY. Expanding-window walk-forward: first 5 years train only, then retrain every 252 trading days and predict the next block.

## 1. Predicting tomorrow's return (price)

*Improvement in error vs. the naive "tomorrow = today" forecast (positive = better).*

| Ticker | Naive (tomorrow = today) | Historical mean return | Linear (last 20 returns) | LSTM |
|---|---|---|---|---|
| AAPL | +0.0% | +0.1% | +0.1% | +0.2% |
| MSFT | +0.0% | +0.1% | +0.1% | +0.2% |
| AMZN | +0.0% | +0.1% | +0.1% | +0.0% |
| JPM | +0.0% | +0.1% | +0.1% | +0.1% |
| XOM | +0.0% | +0.0% | -0.0% | +0.0% |
| SPY | +0.0% | +0.1% | +0.1% | -0.0% |

### Directional accuracy (did it call up vs. down correctly?)

| Ticker | Days up (always-guess-up baseline) | Historical mean return | Linear (last 20 returns) | LSTM |
|---|---|---|---|---|
| AAPL | 53.0% | 53.1% | 53.2% | 51.8% |
| MSFT | 53.4% | 53.7% | 53.6% | 52.6% |
| AMZN | 53.2% | 53.3% | 53.2% | 52.9% |
| JPM | 52.4% | 52.6% | 52.5% | 52.0% |
| XOM | 51.5% | 52.1% | 51.4% | 51.5% |
| SPY | 54.8% | 54.9% | 54.8% | 54.6% |

Pooled across tickers (± 95% interval):

- Historical mean return: 53.3% ± 0.7% over 17,253 days
- Linear (last 20 returns): 53.1% ± 0.7% over 17,253 days
- LSTM: 52.6% ± 0.7% over 17,253 days

## 2. Predicting next week's volatility

*Improvement in error (log volatility) vs. assuming next week is as volatile as last week.*

| Ticker | Naive (last week's volatility) | EWMA (RiskMetrics) | HAR linear model | LSTM |
|---|---|---|---|---|
| AAPL | +0.0% | +11.6% | +17.6% | +16.5% |
| MSFT | +0.0% | +9.7% | +15.9% | +12.8% |
| AMZN | +0.0% | +7.6% | +14.8% | +13.1% |
| JPM | +0.0% | +13.3% | +19.3% | +18.5% |
| XOM | +0.0% | +16.6% | +18.5% | +15.2% |
| SPY | +0.0% | +7.4% | +14.1% | +7.5% |

Correlation between forecast and actual volatility:

| Ticker | Naive (last week's volatility) | EWMA (RiskMetrics) | HAR linear model | LSTM |
|---|---|---|---|---|
| AAPL | 0.40 | 0.44 | 0.44 | 0.41 |
| MSFT | 0.43 | 0.46 | 0.46 | 0.39 |
| AMZN | 0.43 | 0.43 | 0.43 | 0.39 |
| JPM | 0.36 | 0.43 | 0.42 | 0.39 |
| XOM | 0.53 | 0.64 | 0.63 | 0.59 |
| SPY | 0.58 | 0.60 | 0.61 | 0.53 |

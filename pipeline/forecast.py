"""
Daily volatility forecast pipeline.

For each ticker, fits a HAR (heterogeneous autoregressive) model of
next-week realised volatility, the model that beat both the naive baseline and
the LSTM in research/backtest.py, and writes everything the website needs to
frontend/public/forecasts.json:

  - next week's expected move (68% and 90% ranges) from the latest close
  - how that compares with the stock's usual volatility over the past year
  - walk-forward accuracy: error vs. the naive forecast, and how often real
    weekly moves actually landed inside the predicted ranges
  - one year of forecast-vs-actual history for the chart

Run locally:        python pipeline/forecast.py
Self-check:         python pipeline/forecast.py --synthetic
In production it runs every weekday via .github/workflows/daily-forecast.yml.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "frontend" / "public" / "forecasts.json"

TICKERS = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq-100 ETF",
    "IWM": "Russell 2000 ETF",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart",
    "KO": "Coca-Cola",
}

HORIZON = 5            # trading days ahead ("next week")
YEAR = 252
MIN_TRAIN = 3 * YEAR   # walk-forward: first refit after three years of data
Z68, Z90 = 1.0, 1.645


# --------------------------------------------------------------- data --------

def download(ticker: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()


def synthetic(seed: int, n: int = 2520) -> pd.Series:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    var = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        r[t] = math.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2016-01-01", periods=n))


# ------------------------------------------------------------ features -------

def log_vol(r: np.ndarray, end: int, window: int) -> float:
    """Annualised RMS volatility of the `window` returns ending at index `end` (inclusive)."""
    seg = r[end - window + 1 : end + 1]
    return math.log(math.sqrt(np.mean(seg ** 2) * YEAR) + 1e-8)


def build_rows(r: np.ndarray):
    """HAR features known at day t, and the realised log volatility over days t+1..t+5."""
    t_all = np.arange(21, len(r))
    X = np.array([[log_vol(r, t, 1), log_vol(r, t, 5), log_vol(r, t, 22)] for t in t_all])
    y = np.full(len(t_all), np.nan)
    for i, t in enumerate(t_all):
        if t + HORIZON < len(r):
            y[i] = log_vol(r, t + HORIZON, HORIZON)
    return t_all, X, y


# ------------------------------------------------------------- per ticker -----

def analyse(prices: pd.Series) -> dict:
    p = prices.to_numpy(dtype=float)
    dates = prices.index
    r = np.diff(np.log(p))                       # r[i]: close i -> close i+1
    t_all, X, y = build_rows(r)
    known = ~np.isnan(y)

    # Walk-forward predictions: refit once a year on everything known so far.
    wf_pred = np.full(len(t_all), np.nan)
    wf_resid_sd = np.full(len(t_all), np.nan)
    start = MIN_TRAIN
    while start < len(t_all):
        # Only rows whose 5-day outcome was already known by the refit date.
        train = np.arange(0, start - HORIZON)
        train = train[known[train]]
        model = Ridge(alpha=1e-3).fit(X[train], y[train])
        resid_sd = float(np.std(y[train] - model.predict(X[train])))
        block = np.arange(start, min(start + YEAR, len(t_all)))
        wf_pred[block] = model.predict(X[block])
        wf_resid_sd[block] = resid_sd
        start += YEAR

    # Accuracy vs. the naive "next week = last week" forecast.
    scored = known & ~np.isnan(wf_pred)
    naive = X[:, 1]
    rmse = lambda a: float(np.sqrt(np.mean((y[scored] - a[scored]) ** 2)))
    improvement = (rmse(naive) - rmse(wf_pred)) / rmse(naive)

    # Range coverage on non-overlapping weeks: did the real 5-day move land inside the band?
    hits68 = hits90 = weeks = 0
    for i in np.flatnonzero(scored)[::HORIZON]:
        t = t_all[i]
        week_sd = expected_vol(wf_pred[i], wf_resid_sd[i]) * math.sqrt(HORIZON / YEAR)
        move = abs(math.log(p[t + 1 + HORIZON] / p[t + 1]))
        hits68 += move <= Z68 * week_sd
        hits90 += move <= Z90 * week_sd
        weeks += 1

    # Final model on all data, forecasting from the latest close.
    final = Ridge(alpha=1e-3).fit(X[known], y[known])
    resid_sd = float(np.std(y[known] - final.predict(X[known])))
    vol_now = expected_vol(float(final.predict(X[-1:])[0]), resid_sd)
    week_sd = vol_now * math.sqrt(HORIZON / YEAR)
    last_close = float(p[-1])

    # "Usual" volatility: median realised 5-day volatility over the past year.
    recent = np.exp(X[-YEAR:, 1])
    usual = float(np.median(recent))
    percentile = float(np.mean(recent <= vol_now))

    # Chart history: last year of walk-forward forecasts vs. what actually happened.
    history = []
    for i in range(max(0, len(t_all) - YEAR), len(t_all)):
        t = t_all[i]
        history.append({
            "date": dates[t + 1].strftime("%Y-%m-%d"),
            "close": round(float(p[t + 1]), 2),
            "forecast": round(expected_vol(wf_pred[i], wf_resid_sd[i]) * 100, 2)
            if not np.isnan(wf_pred[i]) else None,
            "actual": round(math.exp(y[i]) * 100, 2) if known[i] else None,
        })

    return {
        "last_date": dates[-1].strftime("%Y-%m-%d"),
        "last_close": round(last_close, 2),
        "forecast_vol": round(vol_now * 100, 2),            # annualised, %
        "usual_vol": round(usual * 100, 2),
        "percentile_vs_past_year": round(percentile, 3),
        "move_68_pct": round(Z68 * week_sd * 100, 2),        # ± % over next 5 trading days
        "move_90_pct": round(Z90 * week_sd * 100, 2),
        "range_68": [round(last_close * math.exp(-Z68 * week_sd), 2),
                     round(last_close * math.exp(Z68 * week_sd), 2)],
        "range_90": [round(last_close * math.exp(-Z90 * week_sd), 2),
                     round(last_close * math.exp(Z90 * week_sd), 2)],
        "accuracy": {
            "improvement_vs_naive": round(improvement, 3),
            "coverage_68": round(hits68 / weeks, 3),
            "coverage_90": round(hits90 / weeks, 3),
            "weeks_tested": weeks,
        },
        "history": history,
    }


def expected_vol(log_pred: float, resid_sd: float) -> float:
    """Convert a log-volatility forecast to an expected volatility (lognormal mean)."""
    return math.exp(log_pred + 0.5 * resid_sd ** 2)


# ------------------------------------------------------------------ main -----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    tickers = {f"SIM{i}": f"Simulated {i}" for i in range(3)} if args.synthetic else TICKERS
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "model": "HAR (1-day, 1-week, 1-month realised volatility)",
           "horizon_days": HORIZON,
           "tickers": {}}
    failed = []

    for i, (ticker, name) in enumerate(tickers.items()):
        try:
            prices = synthetic(100 + i) if args.synthetic else download(ticker)
            out["tickers"][ticker] = {"name": name, **analyse(prices)}
            a = out["tickers"][ticker]
            print(f"{ticker:<6} ±{a['move_68_pct']:.1f}% next week | "
                  f"{a['accuracy']['improvement_vs_naive']:+.0%} vs naive | "
                  f"68% band hit {a['accuracy']['coverage_68']:.0%}, "
                  f"90% band hit {a['accuracy']['coverage_90']:.0%}", flush=True)
        except Exception as e:  # one bad download shouldn't sink the whole run
            failed.append(ticker)
            print(f"{ticker:<6} FAILED: {e}", flush=True)

    if not out["tickers"]:
        raise SystemExit("No tickers succeeded; not overwriting forecasts.json")

    target = OUTPUT.with_name("forecasts_synthetic.json") if args.synthetic else OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"\nWrote {target.relative_to(ROOT)} ({target.stat().st_size // 1024} KB)"
          + (f"; failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()

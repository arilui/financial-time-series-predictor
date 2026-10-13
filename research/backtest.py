"""
Walk-forward backtest for the Financial Time Series Predictor.

Answers two questions honestly, across several tickers:

1. RETURNS: can the LSTM predict tomorrow's price better than assuming
   "tomorrow's price = today's price"?
2. VOLATILITY: can it predict how much a stock will move over the next week
   better than assuming "next week will be as volatile as last week"?

How it avoids look-ahead bias
-----------------------------
- Walk-forward evaluation: train on everything up to a date, predict the next
  year, roll forward a year, retrain, repeat. The model never sees the future.
- Features are standardised with statistics from the training window only.
- The LSTM's early-stopping set is the last 15% of the training window
  (chronological), never the test year.

Usage (from the repo root, with the backend virtual environment active):

    python research/backtest.py              # full run, downloads data on first use
    python research/backtest.py --no-lstm    # baselines only (fast)
    python research/backtest.py --synthetic  # self-check on simulated data

Writes research/results.md. Price data is cached in research/data/.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

TICKERS = ["AAPL", "MSFT", "AMZN", "JPM", "XOM", "SPY"]
START, END = "2010-01-01", "2026-09-30"

LOOKBACK = 60        # days of history each prediction sees
VOL_HORIZON = 5      # volatility target: next 5 trading days
INITIAL_TRAIN = 5 * 252
TEST_BLOCK = 252     # retrain once per year of test data
SEED = 42


# ---------------------------------------------------------------- data -------

def load_prices(ticker: str) -> pd.Series:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{ticker}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)["Close"]

    import yfinance as yf

    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df[["Close"]].to_csv(path)
    return df["Close"]


def synthetic_prices(seed: int, n: int = 4000) -> pd.Series:
    """GARCH(1,1) returns: direction is unpredictable, volatility is predictable."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    var = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        r[t] = math.sqrt(var) * rng.standard_normal()
        var = omega + alpha * r[t] ** 2 + beta * var
    prices = 100 * np.exp(np.cumsum(r + 0.0003))
    return pd.Series(prices, index=pd.bdate_range("2010-01-01", periods=n), name="Close")


# ------------------------------------------------------------ datasets -------

def make_dataset(prices: pd.Series, target: str):
    """Rows are decision days t; features use data up to t, the target is after t."""
    p = prices.to_numpy(dtype=float)
    r = np.diff(np.log(p))           # r[i] is the return from day i to day i+1
    rows = []
    for t in range(LOOKBACK, len(r) - VOL_HORIZON):
        past = r[t - LOOKBACK + 1 : t + 1]          # returns up to and including day t
        if target == "returns":
            y = r[t + 1]
        else:
            future = r[t + 1 : t + 1 + VOL_HORIZON]
            y = np.log(np.sqrt(np.mean(future ** 2) * 252))
        rows.append((t, past, y))

    idx = np.array([row[0] for row in rows])
    seq = np.stack([row[1] for row in rows])
    y = np.array([row[2] for row in rows])

    # The price at the decision day and the next day, for price-error metrics.
    price_now = p[idx + 1]
    price_next = p[idx + 2]
    return seq, y, price_now, price_next


def har_features(seq: np.ndarray) -> np.ndarray:
    """Heterogeneous autoregressive (HAR) volatility features: 1-day, 1-week, 1-month."""
    def log_vol(window):
        return np.log(np.sqrt(np.mean(seq[:, -window:] ** 2, axis=1) * 252) + 1e-8)
    return np.column_stack([log_vol(1), log_vol(5), log_vol(22)])


def ewma_log_vol(seq: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics exponentially weighted volatility."""
    weights = lam ** np.arange(seq.shape[1])[::-1]
    weights /= weights.sum()
    return np.log(np.sqrt((seq ** 2) @ weights * 252) + 1e-8)


# -------------------------------------------------------------- models -------

def baseline_predictions(target, seq, train_idx, test_idx, y):
    preds = {}
    if target == "returns":
        preds["Naive (tomorrow = today)"] = np.zeros(len(test_idx))
        preds["Historical mean return"] = np.full(len(test_idx), y[train_idx].mean())
        lin = Ridge(alpha=10.0).fit(seq[train_idx][:, -20:], y[train_idx])
        preds["Linear (last 20 returns)"] = lin.predict(seq[test_idx][:, -20:])
    else:
        preds["Naive (last week's volatility)"] = har_features(seq[test_idx])[:, 1]
        preds["EWMA (RiskMetrics)"] = ewma_log_vol(seq[test_idx])
        har = Ridge(alpha=1e-3).fit(har_features(seq[train_idx]), y[train_idx])
        preds["HAR linear model"] = har.predict(har_features(seq[test_idx]))
    return preds


def lstm_predictions(target, seq, train_idx, test_idx, y):
    import torch
    from torch import nn

    torch.manual_seed(SEED)

    # Volatility models see the size of each day's move, not its direction.
    inputs = np.log(np.abs(seq) + 1e-4) if target == "volatility" else seq
    mu, sd = inputs[train_idx].mean(), inputs[train_idx].std() + 1e-8
    y_mu, y_sd = y[train_idx].mean(), y[train_idx].std() + 1e-8

    def tensors(rows):
        x = torch.tensor((inputs[rows] - mu) / sd, dtype=torch.float32).unsqueeze(-1)
        t = torch.tensor((y[rows] - y_mu) / y_sd, dtype=torch.float32)
        return x, t

    split = int(len(train_idx) * 0.85)   # chronological early-stopping split
    x_tr, y_tr = tensors(train_idx[:split])
    x_va, y_va = tensors(train_idx[split:])
    x_te, _ = tensors(test_idx)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    model = Net()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best, best_state, patience = float("inf"), None, 0

    for _ in range(30):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(perm), 64):
            b = perm[i : i + 64]
            opt.zero_grad()
            loss_fn(model(x_tr[b]), y_tr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(x_va), y_va).item()
        if val < best - 1e-5:
            best, patience = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(x_te).numpy() * y_sd + y_mu


# ------------------------------------------------------------- backtest ------

def walk_forward(target, seq, y, use_lstm):
    folds = []
    start = INITIAL_TRAIN
    while start < len(y):
        folds.append((np.arange(0, start), np.arange(start, min(start + TEST_BLOCK, len(y)))))
        start += TEST_BLOCK

    preds, test_rows = {}, []
    for train_idx, test_idx in folds:
        fold = baseline_predictions(target, seq, train_idx, test_idx, y)
        if use_lstm:
            fold["LSTM"] = lstm_predictions(target, seq, train_idx, test_idx, y)
        for name, p in fold.items():
            preds.setdefault(name, []).append(p)
        test_rows.append(test_idx)

    rows = np.concatenate(test_rows)
    return rows, {name: np.concatenate(p) for name, p in preds.items()}


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate_returns(rows, preds, y, price_now, price_next):
    actual = y[rows]
    naive = rmse(actual, preds["Naive (tomorrow = today)"])
    up_rate = float(np.mean(actual > 0))
    results = {}
    for name, p in preds.items():
        price_pred = price_now[rows] * np.exp(p)
        price_err = np.mean(np.abs(price_pred - price_next[rows]) / price_next[rows])
        moved = actual != 0
        direction = float(np.mean(np.sign(p[moved]) == np.sign(actual[moved]))) if np.any(p != 0) else float("nan")
        results[name] = {
            "vs_naive": (naive - rmse(actual, p)) / naive,
            "price_mape": price_err,
            "direction": direction,
        }
    return results, up_rate, len(rows)


def evaluate_volatility(rows, preds, y):
    actual = y[rows]
    naive = rmse(actual, preds["Naive (last week's volatility)"])
    results = {}
    for name, p in preds.items():
        results[name] = {
            "vs_naive": (naive - rmse(actual, p)) / naive,
            "corr": float(np.corrcoef(actual, p)[0, 1]),
        }
    return results


def pct(x):
    return "n/a" if x != x else f"{x * 100:+.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-lstm", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    use_lstm = not args.no_lstm

    tickers = [f"SIM{i}" for i in range(3)] if args.synthetic else TICKERS
    ret_tables, vol_tables, dir_rows = {}, {}, []
    pooled = {"returns": {}, "volatility": {}}

    for i, ticker in enumerate(tickers):
        prices = synthetic_prices(SEED + i) if args.synthetic else load_prices(ticker)
        print(f"{ticker}: {len(prices)} days", flush=True)

        seq, y, p_now, p_next = make_dataset(prices, "returns")
        rows, preds = walk_forward("returns", seq, y, use_lstm)
        res, up_rate, n = evaluate_returns(rows, preds, y, p_now, p_next)
        ret_tables[ticker] = res
        dir_rows.append((ticker, up_rate, n))
        for name, p in preds.items():
            pooled["returns"].setdefault(name, []).append((np.sign(p) == np.sign(y[rows]))[y[rows] != 0])

        seq_v, y_v, _, _ = make_dataset(prices, "volatility")
        rows_v, preds_v = walk_forward("volatility", seq_v, y_v, use_lstm)
        vol_tables[ticker] = evaluate_volatility(rows_v, preds_v, y_v)

    # ---- report ---------------------------------------------------------------
    lines = ["# Walk-Forward Backtest Results", ""]
    if args.synthetic:
        lines += ["**Synthetic self-check** (GARCH data: direction unpredictable, volatility predictable).", ""]
    lines += [
        f"Tickers: {', '.join(tickers)}. Expanding-window walk-forward: first {INITIAL_TRAIN // 252} years "
        f"train only, then retrain every {TEST_BLOCK} trading days and predict the next block.",
        "",
        "## 1. Predicting tomorrow's return (price)",
        "",
        "*Improvement in error vs. the naive \"tomorrow = today\" forecast (positive = better).*",
        "",
    ]
    names = list(next(iter(ret_tables.values())).keys())
    lines += ["| Ticker | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for t, res in ret_tables.items():
        lines.append(f"| {t} | " + " | ".join(pct(res[n]["vs_naive"]) for n in names) + " |")

    lines += ["", "### Directional accuracy (did it call up vs. down correctly?)", "",
              "| Ticker | Days up (always-guess-up baseline) | " +
              " | ".join(n for n in names if "Naive" not in n) + " |",
              "|---" * (len(names) + 1) + "|"]
    for (t, up, n), res in zip(dir_rows, ret_tables.values()):
        lines.append(f"| {t} | {up:.1%} | " + " | ".join(
            f"{res[nm]['direction']:.1%}" for nm in names if "Naive" not in nm) + " |")

    lines += ["", "Pooled across tickers (± 95% interval):", ""]
    for name, hits in pooled["returns"].items():
        if "Naive" in name:
            continue
        h = np.concatenate(hits)
        acc, se = h.mean(), h.std() / math.sqrt(len(h))
        lines.append(f"- {name}: {acc:.1%} ± {1.96 * se:.1%} over {len(h):,} days")

    lines += ["", "## 2. Predicting next week's volatility", "",
              "*Improvement in error (log volatility) vs. assuming next week is as volatile as last week.*", ""]
    vnames = list(next(iter(vol_tables.values())).keys())
    lines += ["| Ticker | " + " | ".join(vnames) + " |", "|---" * (len(vnames) + 1) + "|"]
    for t, res in vol_tables.items():
        lines.append(f"| {t} | " + " | ".join(pct(res[n]["vs_naive"]) for n in vnames) + " |")
    lines += ["", "Correlation between forecast and actual volatility:", "",
              "| Ticker | " + " | ".join(vnames) + " |", "|---" * (len(vnames) + 1) + "|"]
    for t, res in vol_tables.items():
        lines.append(f"| {t} | " + " | ".join(f"{res[n]['corr']:.2f}" for n in vnames) + " |")

    out = HERE / ("results_synthetic.md" if args.synthetic else "results.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()

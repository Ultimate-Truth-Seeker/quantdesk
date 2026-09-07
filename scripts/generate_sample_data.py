"""
Generate synthetic daily OHLCV price data for a fixed universe of tickers.

Why synthetic data: this sandbox has no network access to financial data
providers (Yahoo Finance, Alpha Vantage, etc.), so real historical prices
can't be downloaded here. This script generates realistic-looking price
series using Geometric Brownian Motion (GBM) with per-ticker drift/volatility
parameters, seeded for reproducibility.

TO USE REAL DATA INSTEAD (recommended before the final submission):
    pip install yfinance
    python scripts/fetch_real_data.py   # see README "Using real market data"

Run:
    python scripts/generate_sample_data.py
Output:
    servers/quant_mcp/data/<TICKER>.csv  (one file per ticker)
"""

import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "servers", "quant_mcp", "data")
TRADING_DAYS = 504  # ~2 years of trading days
SEED = 42

# (ticker, starting_price, annual_drift, annual_volatility)
# Parameters are illustrative, chosen to give each ticker a distinct
# risk/return personality so the metrics (Sharpe, VaR, correlation) produce
# meaningfully different results across the portfolio.
TICKERS = [
    ("AAPL", 180.0, 0.14, 0.28),
    ("MSFT", 340.0, 0.16, 0.24),
    ("NVDA", 480.0, 0.35, 0.55),
    ("GOOGL", 140.0, 0.13, 0.30),
    ("AMZN", 145.0, 0.15, 0.33),
    ("JPM", 155.0, 0.10, 0.22),
    ("XOM", 105.0, 0.07, 0.26),
    ("KO", 60.0, 0.06, 0.15),
    ("SPY", 460.0, 0.10, 0.16),   # broad market ETF proxy
    ("TLT", 95.0, 0.03, 0.13),    # long-term treasury bond ETF proxy
]

# Common correlation "factor" so tickers aren't independent of each other
# (real markets have systemic co-movement) — each ticker's return is a mix
# of a shared market factor and its own idiosyncratic noise.
MARKET_FACTOR_WEIGHT = 0.45


def generate_ticker_series(rng, start_price, annual_drift, annual_vol, market_factor):
    daily_drift = annual_drift / TRADING_DAYS
    daily_vol = annual_vol / np.sqrt(TRADING_DAYS)

    idio_weight = np.sqrt(max(1 - MARKET_FACTOR_WEIGHT ** 2, 0.0))
    idiosyncratic = rng.normal(0, 1, TRADING_DAYS)
    shocks = MARKET_FACTOR_WEIGHT * market_factor + idio_weight * idiosyncratic

    daily_returns = daily_drift + daily_vol * shocks
    log_prices = np.log(start_price) + np.cumsum(daily_returns)
    close = np.exp(log_prices)

    # Derive plausible open/high/low from close with small intraday noise
    intraday_noise = rng.normal(0, daily_vol * 0.3, (TRADING_DAYS, 3))
    open_ = close * np.exp(intraday_noise[:, 0])
    high = np.maximum(open_, close) * np.exp(np.abs(intraday_noise[:, 1]))
    low = np.minimum(open_, close) * np.exp(-np.abs(intraday_noise[:, 2]))
    volume = rng.integers(2_000_000, 60_000_000, TRADING_DAYS)

    return open_, high, low, close, volume


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=TRADING_DAYS)
    market_factor = rng.normal(0, 1, TRADING_DAYS)

    for ticker, start_price, drift, vol in TICKERS:
        open_, high, low, close, volume = generate_ticker_series(
            rng, start_price, drift, vol, market_factor
        )
        df = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "open": open_.round(2),
                "high": high.round(2),
                "low": low.round(2),
                "close": close.round(2),
                "volume": volume,
            }
        )
        out_path = os.path.join(OUT_DIR, f"{ticker}.csv")
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows, last close={close[-1]:.2f})")


if __name__ == "__main__":
    main()
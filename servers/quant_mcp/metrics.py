"""
Core quantitative finance calculations for quant-mcp.

These are the functions the MCP tools wrap. Keeping them separate from the
MCP server plumbing makes them independently unit-testable (see tests/) and
keeps the "non-trivial" logic the assignment requires isolated from protocol
concerns.

All calculations operate on *actual* historical price data loaded from CSV —
nothing here is estimated or guessed by an LLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TickerNotFoundError(Exception):
    pass


class InsufficientDataError(Exception):
    pass


def available_tickers() -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        f[:-4] for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")
    )


def load_prices(ticker: str) -> pd.DataFrame:
    """Load a ticker's historical OHLCV data from its CSV file."""
    path = os.path.join(DATA_DIR, f"{ticker.upper()}.csv")
    if not os.path.isfile(path):
        raise TickerNotFoundError(
            f"No price data for ticker '{ticker}'. "
            f"Available tickers: {', '.join(available_tickers())}"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def daily_returns(ticker: str, window_days: int | None = None) -> pd.Series:
    """Simple daily returns (pct change of close price) for one ticker."""
    prices = load_prices(ticker)
    if window_days:
        prices = prices.tail(window_days + 1)
    returns = prices["close"].pct_change().dropna()
    if len(returns) < 2:
        raise InsufficientDataError(f"Not enough data points for '{ticker}'.")
    returns.index = prices["date"].iloc[1:].reset_index(drop=True)
    return returns


def returns_matrix(tickers: list[str], window_days: int | None = None) -> pd.DataFrame:
    """Aligned daily returns for multiple tickers (inner-joined on date)."""
    series = {t: daily_returns(t, window_days) for t in tickers}
    df = pd.DataFrame(series).dropna(how="any")
    if df.empty:
        raise InsufficientDataError("No overlapping dates across the requested tickers.")
    return df


@dataclass
class Position:
    ticker: str
    quantity: float


def _latest_prices(tickers: list[str]) -> dict[str, float]:
    return {t: load_prices(t)["close"].iloc[-1] for t in tickers}


def portfolio_weights(positions: list[Position]) -> dict[str, float]:
    """Market-value weights for a set of (ticker, quantity) positions."""
    prices = _latest_prices([p.ticker for p in positions])
    values = {p.ticker: p.quantity * prices[p.ticker] for p in positions}
    total = sum(values.values())
    if total <= 0:
        raise InsufficientDataError("Portfolio has zero or negative total value.")
    return {t: v / total for t, v in values.items()}


def portfolio_returns(positions: list[Position], window_days: int | None = None) -> pd.Series:
    """Weighted daily return series for a portfolio of positions."""
    weights = portfolio_weights(positions)
    rets = returns_matrix(list(weights.keys()), window_days)
    weight_vector = pd.Series(weights)[rets.columns]
    return rets.mul(weight_vector, axis=1).sum(axis=1)


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized volatility (std dev of daily returns * sqrt(252))."""
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def annualized_return(returns: pd.Series) -> float:
    """Annualized return from mean daily return, compounded."""
    mean_daily = returns.mean()
    return float((1 + mean_daily) ** TRADING_DAYS_PER_YEAR - 1)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Annualized Sharpe ratio.
    risk_free_rate is an ANNUAL rate (e.g. 0.02 for 2%).
    """
    excess_daily = returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    vol = excess_daily.std(ddof=1)
    if vol == 0:
        return float("nan")
    return float((excess_daily.mean() / vol) * np.sqrt(TRADING_DAYS_PER_YEAR))


def historical_var(returns: pd.Series, confidence: float = 0.95, portfolio_value: float = 1.0) -> float:
    """
    Historical (non-parametric) 1-day Value at Risk.
    Returns a POSITIVE number representing the loss magnitude at the given
    confidence level, expressed in the same currency units as portfolio_value.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    percentile = (1 - confidence) * 100
    loss_quantile = np.percentile(returns, percentile)  # typically negative
    return float(max(-loss_quantile, 0.0) * portfolio_value)


def parametric_var(returns: pd.Series, confidence: float = 0.95, portfolio_value: float = 1.0) -> float:
    """
    Parametric (variance-covariance) 1-day VaR assuming normally
    distributed returns. Uses the inverse normal CDF (z-score) for the
    requested confidence level.
    """
    from scipy.stats import norm  # local import: only needed here

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    z = norm.ppf(1 - confidence)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    loss = -(mu + z * sigma)
    return float(max(loss, 0.0) * portfolio_value)


def correlation_matrix(tickers: list[str], window_days: int | None = None) -> pd.DataFrame:
    rets = returns_matrix(tickers, window_days)
    return rets.corr()


def top_correlated_pairs(corr: pd.DataFrame, top_n: int = 3) -> list[dict]:
    """Flatten the upper triangle of a correlation matrix, sorted descending."""
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append(
                {"ticker_a": cols[i], "ticker_b": cols[j], "correlation": float(corr.iloc[i, j])}
            )
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return pairs[:top_n]


def portfolio_summary(
    positions: list[Position],
    risk_free_rate: float = 0.02,
    window_days: int | None = None,
) -> dict:
    """One-shot bundle of the headline metrics for a portfolio."""
    rets = portfolio_returns(positions, window_days)
    weights = portfolio_weights(positions)
    return {
        "weights": weights,
        "annualized_return": annualized_return(rets),
        "annualized_volatility": annualized_volatility(rets),
        "sharpe_ratio": sharpe_ratio(rets, risk_free_rate),
        "num_observations": int(len(rets)),
    }
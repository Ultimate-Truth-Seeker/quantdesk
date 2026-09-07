"""
Unit tests for servers/quant_mcp/metrics.py

Run with:
    /home/claude/venv/bin/python3 -m pytest tests/ -v
(or just: python3 -m pytest tests/ -v  once the venv is activated)
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "servers", "quant_mcp"))
from servers.quant_mcp import metrics  # noqa: E402


def test_available_tickers_finds_generated_data():
    tickers = metrics.available_tickers()
    assert "AAPL" in tickers
    assert "TLT" in tickers
    assert len(tickers) == 10


def test_load_prices_shape_and_order():
    df = metrics.load_prices("AAPL")
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["date"].is_monotonic_increasing
    assert len(df) == 504


def test_load_prices_unknown_ticker_raises():
    with pytest.raises(metrics.TickerNotFoundError):
        metrics.load_prices("NOTATICKER")


def test_daily_returns_matches_manual_pct_change():
    df = metrics.load_prices("AAPL")
    expected = df["close"].pct_change().dropna().reset_index(drop=True)
    got = metrics.daily_returns("AAPL").reset_index(drop=True)
    assert np.allclose(got.values, expected.values)


def test_annualized_volatility_known_series():
    # Constant daily return -> zero volatility
    flat = pd.Series([0.001] * 100)
    assert metrics.annualized_volatility(flat) == pytest.approx(0.0, abs=1e-9)

    # Known std dev scales by sqrt(252)
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(0, 0.01, 5000))
    vol = metrics.annualized_volatility(daily)
    expected = daily.std(ddof=1) * np.sqrt(252)
    assert vol == pytest.approx(expected)


def test_sharpe_ratio_higher_return_lower_vol_is_higher():
    rng = np.random.default_rng(1)
    good = pd.Series(rng.normal(0.002, 0.01, 500))
    bad = pd.Series(rng.normal(0.0002, 0.03, 500))
    assert metrics.sharpe_ratio(good) > metrics.sharpe_ratio(bad)


def test_historical_var_is_nonnegative_and_scales_with_value():
    rng = np.random.default_rng(2)
    rets = pd.Series(rng.normal(-0.001, 0.02, 1000))
    var_1x = metrics.historical_var(rets, confidence=0.95, portfolio_value=1.0)
    var_10x = metrics.historical_var(rets, confidence=0.95, portfolio_value=10.0)
    assert var_1x >= 0
    assert var_10x == pytest.approx(var_1x * 10)


def test_var_increases_with_confidence():
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(0.0, 0.02, 2000))
    var_95 = metrics.historical_var(rets, confidence=0.95)
    var_99 = metrics.historical_var(rets, confidence=0.99)
    assert var_99 >= var_95


def test_parametric_var_close_to_historical_for_normal_data():
    rng = np.random.default_rng(4)
    rets = pd.Series(rng.normal(0.0005, 0.015, 5000))
    hist = metrics.historical_var(rets, confidence=0.95)
    param = metrics.parametric_var(rets, confidence=0.95)
    # For genuinely normal data, both estimators should be close.
    assert abs(hist - param) < 0.01


def test_correlation_matrix_diagonal_is_one():
    corr = metrics.correlation_matrix(["AAPL", "MSFT", "NVDA"])
    assert np.allclose(np.diag(corr.values), 1.0)
    assert corr.shape == (3, 3)


def test_top_correlated_pairs_sorted_descending():
    corr = metrics.correlation_matrix(["AAPL", "MSFT", "NVDA", "TLT", "KO"])
    pairs = metrics.top_correlated_pairs(corr, top_n=3)
    assert len(pairs) == 3
    correlations = [abs(p["correlation"]) for p in pairs]
    assert correlations == sorted(correlations, reverse=True)


def test_portfolio_weights_sum_to_one():
    positions = [
        metrics.Position("AAPL", 10),
        metrics.Position("MSFT", 5),
        metrics.Position("TLT", 20),
    ]
    weights = metrics.portfolio_weights(positions)
    assert weights.keys() == {"AAPL", "MSFT", "TLT"}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_portfolio_summary_returns_expected_keys():
    positions = [metrics.Position("AAPL", 10), metrics.Position("TLT", 30)]
    summary = metrics.portfolio_summary(positions)
    assert set(summary.keys()) == {
        "weights",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "num_observations",
    }
    assert summary["num_observations"] > 0
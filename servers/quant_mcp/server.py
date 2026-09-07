"""
quant-mcp — a local MCP server exposing quantitative portfolio analysis tools.

Transport: stdio (this process is spawned as a subprocess by the host).

Tools exposed:
    list_available_tickers
    get_portfolio_summary
    calculate_volatility
    calculate_sharpe_ratio
    calculate_var
    get_correlation_matrix
    simulate_rebalance

Run standalone for manual testing:
    python server.py
(it will wait on stdio — use the MCP Inspector or the test client in
 tests/test_quant_mcp_server.py to interact with it)
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

import metrics
from metrics import Position

mcp = FastMCP("quant-mcp")


def _positions_from_dicts(positions: list[dict]) -> list[Position]:
    try:
        return [Position(ticker=p["ticker"].upper(), quantity=float(p["quantity"])) for p in positions]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Each position must be an object with 'ticker' (string) and "
            "'quantity' (number) fields."
        ) from exc


@mcp.tool()
def list_available_tickers() -> dict:
    """List the tickers for which historical price data is available."""
    return {"tickers": metrics.available_tickers()}


@mcp.tool()
def get_portfolio_summary(
    positions: list[dict],
    risk_free_rate: float = 0.02,
    window_days: Optional[int] = None,
) -> dict:
    """
    Compute headline metrics for a portfolio: market-value weights,
    annualized return, annualized volatility, and Sharpe ratio.

    Args:
        positions: list of {"ticker": str, "quantity": number}.
        risk_free_rate: annual risk-free rate used in the Sharpe ratio (e.g. 0.02 = 2%).
        window_days: optional lookback window in trading days (default: full history).
    """
    pos = _positions_from_dicts(positions)
    return metrics.portfolio_summary(pos, risk_free_rate=risk_free_rate, window_days=window_days)


@mcp.tool()
def calculate_volatility(
    positions: list[dict],
    window_days: Optional[int] = None,
) -> dict:
    """
    Annualized volatility of the portfolio AND of each individual position,
    computed from actual historical daily returns (std dev * sqrt(252)).
    """
    pos = _positions_from_dicts(positions)
    portfolio_vol = metrics.annualized_volatility(
        metrics.portfolio_returns(pos, window_days)
    )
    per_ticker = {
        p.ticker: metrics.annualized_volatility(metrics.daily_returns(p.ticker, window_days))
        for p in pos
    }
    return {"portfolio_volatility": portfolio_vol, "per_ticker_volatility": per_ticker}


@mcp.tool()
def calculate_sharpe_ratio(
    positions: list[dict],
    risk_free_rate: float = 0.02,
    window_days: Optional[int] = None,
) -> dict:
    """
    Annualized Sharpe ratio of the portfolio: (excess return / volatility),
    computed from actual historical daily returns.
    """
    pos = _positions_from_dicts(positions)
    rets = metrics.portfolio_returns(pos, window_days)
    return {
        "sharpe_ratio": metrics.sharpe_ratio(rets, risk_free_rate),
        "risk_free_rate_used": risk_free_rate,
        "num_observations": int(len(rets)),
    }


@mcp.tool()
def calculate_var(
    positions: list[dict],
    confidence: float = 0.95,
    portfolio_value: float = 100_000.0,
    method: str = "historical",
    window_days: Optional[int] = None,
) -> dict:
    """
    1-day Value at Risk (VaR) for the portfolio.

    Args:
        positions: list of {"ticker": str, "quantity": number}.
        confidence: confidence level, e.g. 0.95 or 0.99.
        portfolio_value: total currency value the VaR amount is expressed against.
        method: "historical" (empirical percentile, no distribution assumed)
                or "parametric" (assumes normally distributed returns).
        window_days: optional lookback window in trading days.
    """
    if method not in ("historical", "parametric"):
        raise ValueError("method must be 'historical' or 'parametric'")
    pos = _positions_from_dicts(positions)
    rets = metrics.portfolio_returns(pos, window_days)
    if method == "historical":
        var = metrics.historical_var(rets, confidence, portfolio_value)
    else:
        var = metrics.parametric_var(rets, confidence, portfolio_value)
    return {
        "value_at_risk": var,
        "confidence": confidence,
        "method": method,
        "portfolio_value": portfolio_value,
        "interpretation": (
            f"With {confidence:.0%} confidence, the portfolio is not expected to "
            f"lose more than {var:,.2f} in a single day, based on {len(rets)} "
            f"historical daily observations."
        ),
    }


@mcp.tool()
def get_correlation_matrix(
    tickers: list[str],
    window_days: Optional[int] = None,
    top_n_pairs: int = 3,
) -> dict:
    """
    Pearson correlation matrix of daily returns across the given tickers,
    plus the most correlated pairs (useful for spotting concentration risk).
    """
    tickers = [t.upper() for t in tickers]
    corr = metrics.correlation_matrix(tickers, window_days)
    return {
        "correlation_matrix": corr.round(4).to_dict(),
        "most_correlated_pairs": metrics.top_correlated_pairs(corr, top_n_pairs),
    }


@mcp.tool()
def simulate_rebalance(
    current_positions: list[dict],
    proposed_positions: list[dict],
    risk_free_rate: float = 0.02,
    window_days: Optional[int] = None,
) -> dict:
    """
    Compare portfolio metrics (return, volatility, Sharpe) BEFORE and AFTER
    a proposed change in position quantities. No real trade is executed —
    this is a what-if simulation over historical data.

    Args:
        current_positions: list of {"ticker": str, "quantity": number} — today's holdings.
        proposed_positions: list of {"ticker": str, "quantity": number} — the hypothetical
            new holdings (include unchanged positions too, so the comparison is complete).
        risk_free_rate: annual risk-free rate for Sharpe ratio.
        window_days: optional lookback window in trading days.
    """
    before = metrics.portfolio_summary(
        _positions_from_dicts(current_positions), risk_free_rate, window_days
    )
    after = metrics.portfolio_summary(
        _positions_from_dicts(proposed_positions), risk_free_rate, window_days
    )
    return {
        "before": before,
        "after": after,
        "delta": {
            "annualized_return": after["annualized_return"] - before["annualized_return"],
            "annualized_volatility": after["annualized_volatility"] - before["annualized_volatility"],
            "sharpe_ratio": after["sharpe_ratio"] - before["sharpe_ratio"],
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
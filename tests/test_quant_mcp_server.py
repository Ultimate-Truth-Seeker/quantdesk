"""
Integration test: actually launches servers/quant_mcp/server.py as a
subprocess over stdio and talks MCP to it, the same way the host will.

Run with:
    /home/claude/venv/bin/python3 -m pytest tests/test_quant_mcp_server.py -v -s
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "servers", "quant_mcp")
SERVER_SCRIPT = os.path.join(SERVER_DIR, "server.py")


def _server_params():
    return StdioServerParameters(
        command=sys.executable,
        args=[SERVER_SCRIPT],
        cwd=SERVER_DIR,
    )


@pytest.mark.anyio
async def test_list_tools_returns_expected_set():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            expected = {
                "list_available_tickers",
                "get_portfolio_summary",
                "calculate_volatility",
                "calculate_sharpe_ratio",
                "calculate_var",
                "get_correlation_matrix",
                "simulate_rebalance",
            }
            assert expected.issubset(names)


@pytest.mark.anyio
async def test_list_available_tickers_call():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_available_tickers", {})
            payload = json.loads(result.content[0].text)
            assert "AAPL" in payload["tickers"]


@pytest.mark.anyio
async def test_get_portfolio_summary_call():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "get_portfolio_summary",
                {"positions": [{"ticker": "AAPL", "quantity": 10}, {"ticker": "TLT", "quantity": 30}]},
            )
            payload = json.loads(result.content[0].text)
            assert "sharpe_ratio" in payload
            assert "annualized_volatility" in payload
            assert set(payload["weights"].keys()) == {"AAPL", "TLT"}


@pytest.mark.anyio
async def test_calculate_var_call():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "calculate_var",
                {
                    "positions": [{"ticker": "NVDA", "quantity": 5}],
                    "confidence": 0.95,
                    "portfolio_value": 10000,
                    "method": "historical",
                },
            )
            payload = json.loads(result.content[0].text)
            assert payload["value_at_risk"] >= 0
            assert payload["method"] == "historical"


@pytest.mark.anyio
async def test_get_correlation_matrix_call():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "get_correlation_matrix",
                {"tickers": ["AAPL", "MSFT", "NVDA"]},
            )
            payload = json.loads(result.content[0].text)
            assert "AAPL" in payload["correlation_matrix"]
            assert len(payload["most_correlated_pairs"]) == 3


@pytest.mark.anyio
async def test_invalid_ticker_returns_error_not_crash():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "get_portfolio_summary",
                {"positions": [{"ticker": "DOESNOTEXIST", "quantity": 1}]},
            )
            # FastMCP surfaces exceptions as an error tool result rather
            # than crashing the server process.
            assert result.isError is True
"""
End-to-end integration test for the host's plumbing (config, logger,
jsonrpc_tap, mcp_client_manager) — WITHOUT calling the Anthropic API. This
proves the three week-1-3 servers (filesystem, git, quant-mcp) all connect
correctly, that tools are discovered and namespaced, and that the JSON-RPC
logger actually captures real traffic.

Run with:
    /home/claude/venv/bin/python3 -m pytest tests/test_host_integration.py -v -s
"""

import json
import os
import sys

import pytest

HOST_DIR = os.path.join(os.path.dirname(__file__), "..", "host")
sys.path.insert(0, HOST_DIR)

from host.config import SERVERS, ensure_git_initialized  # noqa: E402
from host.logger import JsonRpcLogger  # noqa: E402
from host.mcp_client_manager import MCPClientManager  # noqa: E402

TEST_LOG_PATH = os.path.join(os.path.dirname(__file__), "test_mcp_log.jsonl")


@pytest.mark.anyio
async def test_all_week1to3_servers_connect_and_expose_tools():
    ensure_git_initialized()
    jlogger = JsonRpcLogger(log_path=TEST_LOG_PATH, echo_to_console=False)

    async with MCPClientManager(SERVERS, jlogger) as manager:
        connected = set(manager.connected_servers())
        assert {"filesystem", "git", "quant-mcp"}.issubset(connected)
        assert not manager.failed_servers()

        # Namespacing: qualified tool names are "<server>__<tool>"
        assert "quant-mcp__calculate_var" in manager.tools
        assert "filesystem__write_file" in manager.tools
        assert "git__git_status" in manager.tools

        specs = manager.anthropic_tool_specs()
        names = {s["name"] for s in specs}
        assert "quant-mcp__list_available_tickers" in names
        for s in specs:
            assert "input_schema" in s and "description" in s

    # The log file should contain real JSON-RPC traffic for every server.
    with open(TEST_LOG_PATH, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    servers_logged = {rec["server"] for rec in lines}
    assert {"filesystem", "git", "quant-mcp"}.issubset(servers_logged)
    methods_logged = {
        rec["jsonrpc_message"].get("method")
        for rec in lines
        if "method" in rec["jsonrpc_message"]
    }
    assert "initialize" in methods_logged
    assert "tools/list" in methods_logged


@pytest.mark.anyio
async def test_quant_mcp_tool_call_through_manager():
    jlogger = JsonRpcLogger(log_path=TEST_LOG_PATH, echo_to_console=False)
    async with MCPClientManager(SERVERS, jlogger) as manager:
        result = await manager.call_tool(
            "quant-mcp__get_portfolio_summary",
            {"positions": [{"ticker": "AAPL", "quantity": 10}]},
        )
        payload = json.loads(result.content[0].text)
        assert "sharpe_ratio" in payload


@pytest.mark.anyio
async def test_filesystem_and_git_end_to_end_scenario():
    """
    Mirrors the assignment's suggested demo scenario: create a file with
    the Filesystem MCP server, then add + commit it with the Git MCP server.
    """
    ensure_git_initialized()
    jlogger = JsonRpcLogger(log_path=TEST_LOG_PATH, echo_to_console=False)
    async with MCPClientManager(SERVERS, jlogger) as manager:
        write_result = await manager.call_tool(
            "filesystem__write_file",
            {"path": "README.md", "content": "# QuantDesk workspace\n\nTest file.\n"},
        )
        assert write_result.isError is not True

        workspace = os.path.join(os.path.dirname(HOST_DIR), "workspace")
        add_result = await manager.call_tool(
            "git__git_add", {"repo_path": workspace, "files": ["README.md"]}
        )
        assert add_result.isError is not True

        commit_result = await manager.call_tool(
            "git__git_commit",
            {"repo_path": workspace, "message": "Add README via QuantDesk test"},
        )
        assert commit_result.isError is not True

        log_result = await manager.call_tool("git__git_log", {"repo_path": workspace, "max_count": 1})
        log_text = log_result.content[0].text
        assert "Add README via QuantDesk test" in log_text
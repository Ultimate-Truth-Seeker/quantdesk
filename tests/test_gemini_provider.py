"""
Tests for the Gemini provider and the multi-provider config plumbing.

Two kinds of tests here:
  1. Fully offline (no network, no API key needed) — config validation and
     the tool-name collision check, which is the main risk specific to
     Gemini's built-in MCP support (see providers/gemini_provider.py).
  2. A real end-to-end test against the actual Gemini API — SKIPPED
     automatically unless GEMINI_API_KEY is set in your environment, since
     this sandbox has no network access to Google's API. Run this one on
     your own machine before the final submission:

        export GEMINI_API_KEY=...
        python -m pytest tests/test_gemini_provider.py -v -s -k live

Run with:
    /home/claude/venv/bin/python3 -m pytest tests/test_gemini_provider.py -v -s
"""

import os
import sys

import pytest

HOST_DIR = os.path.join(os.path.dirname(__file__), "..", "host")
sys.path.insert(0, HOST_DIR)
sys.path.insert(0, os.path.join(HOST_DIR, "providers"))

from mcp_client_manager import MCPClientManager, ToolInfo  # noqa: E402


def _fake_manager_with_tools(tool_specs: list[tuple[str, str]]) -> MCPClientManager:
    """
    Build an MCPClientManager without actually connecting to anything, and
    inject fake tools: list of (server_name, tool_name) pairs.
    """
    manager = MCPClientManager.__new__(MCPClientManager)  # bypass __init__/connect
    manager.tools = {}
    for server_name, tool_name in tool_specs:
        qualified = f"{server_name}__{tool_name}"
        manager.tools[qualified] = ToolInfo(
            server_name=server_name, tool_name=tool_name, description="", input_schema={}
        )
    manager.sessions = {}
    return manager


def test_no_collisions_when_all_tool_names_unique():
    manager = _fake_manager_with_tools(
        [("quant-mcp", "calculate_var"), ("filesystem", "write_file"), ("git", "git_commit")]
    )
    assert manager.raw_tool_name_collisions() == {}


def test_collision_detected_across_two_servers():
    manager = _fake_manager_with_tools(
        [
            ("quant-mcp", "get_status"),
            ("classmate-server-1", "get_status"),  # collides by raw name
            ("git", "git_commit"),
        ]
    )
    collisions = manager.raw_tool_name_collisions()
    assert "get_status" in collisions
    assert set(collisions["get_status"]) == {"quant-mcp", "classmate-server-1"}


def test_collision_across_three_servers_reports_all():
    manager = _fake_manager_with_tools(
        [
            ("quant-mcp", "analyze"),
            ("classmate-server-1", "analyze"),
            ("classmate-server-2", "analyze"),
        ]
    )
    collisions = manager.raw_tool_name_collisions()
    assert set(collisions["analyze"]) == {"quant-mcp", "classmate-server-1", "classmate-server-2"}


def test_gemini_provider_raises_clear_error_on_collision():
    import gemini_provider

    manager = _fake_manager_with_tools(
        [("quant-mcp", "get_status"), ("classmate-server-1", "get_status")]
    )
    with pytest.raises(gemini_provider.ToolNameCollisionError) as exc_info:
        gemini_provider._check_no_collisions(manager)
    message = str(exc_info.value)
    assert "get_status" in message
    assert "quant-mcp" in message
    assert "classmate-server-1" in message


def test_gemini_provider_passes_with_no_collisions():
    import gemini_provider

    manager = _fake_manager_with_tools([("quant-mcp", "calculate_var"), ("filesystem", "write_file")])
    gemini_provider._check_no_collisions(manager)  # should not raise


def test_config_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    # config.py validates LLM_PROVIDER at import time, so we must force a
    # fresh import to trigger the check.
    sys.modules.pop("config", None)
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        import config  # noqa: F401
    sys.modules.pop("config", None)  # don't leak the broken import to other tests
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    import config  # noqa: F401  — re-import cleanly for any tests that follow


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — this sandbox has no network access to "
    "Google's API either way. Run this test on your own machine before "
    "the final submission.",
)
@pytest.mark.anyio
async def test_live_gemini_calls_quant_mcp_tool():
    """
    Real end-to-end test: connects quant-mcp for real, hands its session to
    Gemini, asks a question that requires a tool call, and checks the
    answer + the JSON-RPC log both reflect a real tool call happened.
    """
    import json

    from config import SERVERS
    from logger import JsonRpcLogger

    only_quant_mcp = [s for s in SERVERS if s.name == "quant-mcp"]
    log_path = os.path.join(os.path.dirname(__file__), "test_gemini_live_log.jsonl")
    jlogger = JsonRpcLogger(log_path=log_path, echo_to_console=True)

    async with MCPClientManager(only_quant_mcp, jlogger) as manager:
        import gemini_provider
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        chat = client.aio.chats.create(
            model=gemini_provider.GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=gemini_provider.SYSTEM_PROMPT,
                tools=list(manager.sessions.values()),
            ),
        )
        response = await chat.send_message(
            "What tickers do you have price data for? List them."
        )
        assert response.text  # got a real answer

    with open(log_path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    tool_calls = [
        rec
        for rec in lines
        if rec["jsonrpc_message"].get("method") == "tools/call"
    ]
    assert len(tool_calls) >= 1, "Expected Gemini to actually call an MCP tool"
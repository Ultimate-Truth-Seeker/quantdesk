"""
QuantDesk host — a console chatbot that uses either Claude (Anthropic API)
or Gemini (Google GenAI API) as the coordinating LLM, and one or more MCP
servers as its tools.

Switch providers with:
    export LLM_PROVIDER=anthropic   # default — manual tool-calling loop
    export LLM_PROVIDER=gemini      # automatic tool-calling via google-genai

Functional requirements covered here (see PDF "Proyecto 1", section 3.1):
    1) Connection with an LLM at the API level, including plain
       general-knowledge questions (no tools needed).
    2) Context maintained across a conversation (both providers keep
       history across turns — see providers/anthropic_provider.py and
       providers/gemini_provider.py for how each does it).
    3) A log of every request/response with the MCP servers (see logger.py
       + jsonrpc_tap.py — printed live and written to mcp_log.jsonl,
       regardless of which provider is driving the conversation).

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...      # if LLM_PROVIDER=anthropic
    export GEMINI_API_KEY=...                # if LLM_PROVIDER=gemini
    python main.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from config import LLM_PROVIDER, SERVERS, ensure_git_initialized
from logger import JsonRpcLogger
from mcp_client_manager import MCPClientManager


async def run_chat():
    if LLM_PROVIDER == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: set the ANTHROPIC_API_KEY environment variable before running.")
            sys.exit(1)
        from providers import anthropic_provider as provider
    else:  # "gemini" — validated in config.py, only these two values are possible
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: set the GEMINI_API_KEY environment variable before running.")
            sys.exit(1)
        from providers import gemini_provider as provider

    jsonrpc_logger = JsonRpcLogger()
    ensure_git_initialized()

    print(f"Connecting to MCP servers (provider: {LLM_PROVIDER})...")
    async with MCPClientManager(SERVERS, jsonrpc_logger) as manager:
        connected = manager.connected_servers()
        failed = manager.failed_servers()
        print(f"Connected servers: {connected}")
        if failed:
            print(f"Servers that failed to connect (continuing without them): {failed}")
        print(f"Discovered {len(manager.tools)} tool(s) total.\n")

        try:
            await provider.run(manager, api_key)
        except Exception as exc:  # noqa: BLE001 — surface provider setup errors cleanly
            print(f"ERROR: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_chat())
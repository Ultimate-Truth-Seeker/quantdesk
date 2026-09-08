"""
MCPClientManager: connects to every configured MCP server (local stdio or
remote HTTP), discovers their tools, and exposes a single unified interface
the chat loop can use without caring which server a tool lives on.

Tool names are namespaced as "<server_name>__<tool_name>" when exposed to
Claude, to avoid collisions between servers (e.g. two servers both defining
a "search" tool), and are de-namespaced again when routing a call back to
the right server.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

import anyio
import re
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from config import HttpServerConfig, StdioServerConfig
from jsonrpc_tap import JsonRpcTap
from logger import JsonRpcLogger


def _safe_tool_name(name: str) -> str:
    """Return a function name accepted by the LLM provider APIs."""
    safe_name = re.sub(r"[^A-Za-z0-9_.:-]", "_", name)
    if not safe_name or not re.match(r"[A-Za-z_]", safe_name):
        safe_name = f"_{safe_name}"
    return safe_name[:128]


@dataclass
class ToolInfo:
    server_name: str
    tool_name: str
    description: str
    input_schema: dict


class MCPClientManager:
    def __init__(self, server_configs: list, logger: JsonRpcLogger):
        self._configs = [c for c in server_configs if c.enabled]
        self._logger = logger
        self._exit_stack = AsyncExitStack()
        self._task_group: anyio.abc.TaskGroup | None = None
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, ToolInfo] = {}  # qualified_name -> ToolInfo
        self._failed_servers: dict[str, str] = {}

    async def __aenter__(self) -> "MCPClientManager":
        self._task_group = await self._exit_stack.enter_async_context(anyio.create_task_group())
        for cfg in self._configs:
            try:
                await self._connect_one(cfg)
            except Exception as exc:  # noqa: BLE001 — a single bad server shouldn't kill the host
                self._failed_servers[cfg.name] = str(exc)
                print(f"  [!] Could not connect to '{cfg.name}': {exc}")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._exit_stack.aclose()

    # -- connection -----------------------------------------------------

    async def _connect_one(self, cfg) -> None:
        def on_message(direction: str, payload: dict) -> None:
            self._logger.log(cfg.name, direction, payload)

        if isinstance(cfg, StdioServerConfig):
            params = StdioServerParameters(
                command=cfg.command, args=cfg.args, cwd=cfg.cwd, env=cfg.env
            )
            raw_read, raw_write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
        elif isinstance(cfg, HttpServerConfig):
            raw_read, raw_write, _get_session_id = await self._exit_stack.enter_async_context(
                streamablehttp_client(cfg.url)
            )
        else:
            raise TypeError(f"Unknown server config type for '{cfg.name}': {type(cfg)}")

        tap = JsonRpcTap(raw_read, raw_write, on_message)
        tap.start(self._task_group)

        session = await self._exit_stack.enter_async_context(
            ClientSession(tap.tapped_read_recv, tap.tapped_write_send)
        )
        await session.initialize()
        self.sessions[cfg.name] = session

        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            qualified = _safe_tool_name(f"{cfg.name}__{tool.name}")
            self.tools[qualified] = ToolInfo(
                server_name=cfg.name,
                tool_name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
        print(f"  [+] Connected to '{cfg.name}' — {len(tools_result.tools)} tool(s)")

    # -- usage ------------------------------------------------------------

    def connected_servers(self) -> list[str]:
        return list(self.sessions.keys())

    def failed_servers(self) -> dict[str, str]:
        return dict(self._failed_servers)

    def anthropic_tool_specs(self) -> list[dict]:
        """Tool definitions in the shape the Anthropic Messages API expects."""
        return [
            {
                "name": qualified,
                "description": f"[{info.server_name}] {info.description}",
                "input_schema": info.input_schema,
            }
            for qualified, info in self.tools.items()
        ]

    def raw_tool_name_collisions(self) -> dict[str, list[str]]:
        """
        Tool names (WITHOUT the server-name prefix) that are exposed by more
        than one connected server.

        Why this matters for Gemini: unlike the Anthropic path above (which
        always uses the namespaced "server__tool" name), Gemini's built-in
        MCP support takes raw mcp.ClientSession objects directly and merges
        their tools by RAW name across every session passed in `tools=[...]`.
        If two servers both expose a tool with the same name, the
        google-genai SDK raises `ValueError: Tool <name> is already defined
        for the request` at call time. Check this BEFORE starting a Gemini
        chat so you get a clear, actionable error instead of a crash deep
        inside the SDK.
        """
        by_name: dict[str, list[str]] = {}
        for info in self.tools.values():
            by_name.setdefault(info.tool_name, []).append(info.server_name)
        return {name: servers for name, servers in by_name.items() if len(servers) > 1}

    async def call_tool(self, qualified_name: str, arguments: dict) -> Any:
        info = self.tools.get(qualified_name)
        if info is None:
            raise KeyError(f"Unknown tool '{qualified_name}'")
        session = self.sessions[info.server_name]
        result = await session.call_tool(info.tool_name, arguments)
        return result
"""
JsonRpcTap: transparently intercepts the raw JSON-RPC message stream between
an MCP ClientSession and its transport (stdio or streamable-HTTP), so every
message can be logged without changing protocol behavior.

Why this exists: the mcp SDK's ClientSession hides the JSON-RPC framing
behind an async API (session.list_tools(), session.call_tool(), ...). To
genuinely log "all interactions ... with the MCP servers" at the JSON-RPC
level (as the assignment asks), we tap the memory-object streams that carry
mcp.shared.message.SessionMessage objects (each wrapping a JSONRPCMessage)
between the transport and the session, forwarding every message through
unchanged after logging it.
"""

from __future__ import annotations

from typing import Callable

import anyio


class JsonRpcTap:
    def __init__(self, inner_read, inner_write, on_message: Callable[[str, dict], None]):
        """
        inner_read: the transport's raw receive stream (server -> host)
        inner_write: the transport's raw send stream (host -> server)
        on_message: callback(direction, payload_dict) called for every message
        """
        self._inner_read = inner_read
        self._inner_write = inner_write
        self._on_message = on_message
        self.tapped_read_send, self.tapped_read_recv = anyio.create_memory_object_stream(100)
        self.tapped_write_send, self.tapped_write_recv = anyio.create_memory_object_stream(100)

    @staticmethod
    def _to_dict(session_message) -> dict:
        # session_message: mcp.shared.message.SessionMessage
        # .message is a JSONRPCMessage RootModel wrapping Request/Response/Notification/Error
        return session_message.message.model_dump(mode="json", exclude_none=True)

    async def _forward_read(self):
        async with self._inner_read, self.tapped_read_send:
            async for item in self._inner_read:
                if not isinstance(item, Exception):
                    self._on_message("received", self._to_dict(item))
                await self.tapped_read_send.send(item)

    async def _forward_write(self):
        async with self.tapped_write_recv, self._inner_write:
            async for item in self.tapped_write_recv:
                self._on_message("sent", self._to_dict(item))
                await self._inner_write.send(item)

    def start(self, task_group: anyio.abc.TaskGroup) -> None:
        task_group.start_soon(self._forward_read)
        task_group.start_soon(self._forward_write)
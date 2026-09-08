"""
JSON-RPC interaction logger.

Every request, response, notification, and error exchanged with every MCP
server is logged here — this satisfies the "mantener y mostrar un log sobre
todas las interacciones (solicitudes y respuestas) con los servidores MCP"
requirement, at the actual JSON-RPC message level (not just an application-
level summary).

Two outputs:
    - Console: a compact one-line-per-message summary, printed live.
    - File (mcp_log.jsonl): the FULL JSON-RPC payload, one JSON object per
      line, timestamped and tagged with which server and direction it went.
      This is what you'd use for the report in section 3.3 / point 9-10 of
      the assignment, alongside the Wireshark capture.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp_log.jsonl")


class JsonRpcLogger:
    def __init__(self, log_path: str = DEFAULT_LOG_PATH, echo_to_console: bool = True):
        self.log_path = log_path
        self.echo_to_console = echo_to_console
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)) or ".", exist_ok=True)
        # Truncate at the start of each run so the log matches this session.
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")

    def log(self, server_name: str, direction: str, payload: dict) -> None:
        """
        direction: "sent" (host -> server) or "received" (server -> host)
        payload: the raw JSON-RPC message dict (request/response/notification/error)
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": server_name,
            "direction": direction,
            "jsonrpc_message": payload,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.echo_to_console:
            print(f"  [{server_name}] {self._summarize(direction, payload)}")

    @staticmethod
    def _summarize(direction: str, payload: dict) -> str:
        arrow = "->" if direction == "sent" else "<-"
        if "method" in payload:
            kind = "request" if "id" in payload else "notification"
            method = payload.get("method")
            if method == "tools/call":
                tool_name = payload.get("params", {}).get("name", "?")
                return f"{arrow} {kind}: tools/call({tool_name}) (id={payload.get('id')})"
            return f"{arrow} {kind}: {method} (id={payload.get('id')})"
        if "result" in payload:
            return f"{arrow} response (id={payload.get('id')}): OK"
        if "error" in payload:
            err = payload["error"]
            return f"{arrow} response (id={payload.get('id')}): ERROR {err.get('code')} {err.get('message')}"
        return f"{arrow} {json.dumps(payload)[:100]}"
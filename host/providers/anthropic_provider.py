"""
Anthropic provider — manual tool-calling loop.

This is the original QuantDesk chat loop: the host receives each
`tool_use` block from Claude explicitly, calls the right MCP tool itself
via MCPClientManager, and feeds the `tool_result` back. Nothing here
changed when the Gemini provider was added — this module is a straight
extraction of what used to be in main.py directly.
"""

from __future__ import annotations

import json

import anthropic

from config import ANTHROPIC_MODEL, MAX_TOKENS, MAX_TOOL_ITERATIONS
from mcp_client_manager import MCPClientManager

SYSTEM_PROMPT = """\
You are QuantDesk, an assistant for a quantitative portfolio analyst.
You have access to tools from several MCP servers:
  - quant-mcp: real financial calculations (returns, volatility, Sharpe \
ratio, Value at Risk, correlation) computed from actual historical price \
data — never estimate these numbers yourself, always call the tool.
  - filesystem: read/write files in the analyst's workspace, e.g. to save \
a report.
  - git: version-control the workspace (init, add, commit).
You can also answer general knowledge questions directly, without tools, \
when the user isn't asking about their portfolio or files.
When you use a quant-mcp tool, briefly explain the result in plain \
language after presenting the numbers.
"""


def _extract_text(message) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


def _tool_result_to_content_block(tool_use_id: str, result) -> dict:
    """Convert an MCP CallToolResult into an Anthropic tool_result block."""
    text_parts = []
    for item in result.content:
        if getattr(item, "type", None) == "text":
            text_parts.append(item.text)
        elif hasattr(item, "model_dump"):
            text_parts.append(json.dumps(item.model_dump(mode="json", exclude_none=True)))
        else:
            text_parts.append(str(item))
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": "\n".join(text_parts) or "(empty result)",
        "is_error": bool(getattr(result, "isError", False)),
    }


async def run(manager: MCPClientManager, api_key: str) -> None:
    """Run the console chat loop against Claude, using MCPClientManager's tools."""
    client = anthropic.Anthropic(api_key=api_key)
    tool_specs = manager.anthropic_tool_specs()
    messages: list[dict] = []

    print("QuantDesk ready (provider: anthropic). Type your question ('exit' to quit).\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in ("exit", "quit", "salir"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_TOOL_ITERATIONS):
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tool_specs,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                print(f"claude> {_extract_text(response)}\n")
                break

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            result_blocks = []
            for block in tool_use_blocks:
                print(f"  [tool_use] {block.name}({json.dumps(block.input)})")
                try:
                    result = await manager.call_tool(block.name, block.input)
                    result_blocks.append(_tool_result_to_content_block(block.id, result))
                except Exception as exc:  # noqa: BLE001
                    result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error calling tool: {exc}",
                            "is_error": True,
                        }
                    )
            messages.append({"role": "user", "content": result_blocks})
        else:
            print("claude> (stopped: reached max tool-call iterations for this turn)\n")
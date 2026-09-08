"""
Gemini provider — function calling with MCP execution owned by the host.

The Gemini SDK's built-in MCP integration expects to copy/reprocess its tool
configuration on every request. Long-lived MCP ClientSession objects contain
asyncio transport state (including Futures), so passing them directly to that
integration can fail with ``cannot pickle '_asyncio.Future' object``. We pass
only JSON-compatible function declarations to Gemini and route the resulting
function calls through MCPClientManager instead.

Known limitation (see README.md "Gemini provider" for the full writeup):
Gemini's built-in MCP support merges tools from every session by their RAW
name (no "server__tool" namespacing like the Anthropic path uses). If two
connected servers expose a tool with the same name, the google-genai SDK
raises `ValueError: Tool <name> is already defined for the request` — we
check for this upfront (see MCPClientManager.raw_tool_name_collisions) and
fail with a clear message instead of letting that surface mid-conversation.

This keeps the provider independent of the SDK's experimental raw-session
adapter while preserving the existing MCP JSON-RPC logging.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from config import GEMINI_MAX_REMOTE_CALLS, GEMINI_MODEL
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


class ToolNameCollisionError(Exception):
    pass


def _check_no_collisions(manager: MCPClientManager) -> None:
    collisions = manager.raw_tool_name_collisions()
    if not collisions:
        return
    lines = [
        f"  - '{name}' is exposed by: {', '.join(servers)}"
        for name, servers in collisions.items()
    ]
    raise ToolNameCollisionError(
        "Cannot start the Gemini provider: the following tool name(s) are "
        "exposed by more than one connected MCP server. Gemini's built-in "
        "MCP support merges tools by raw name across sessions and cannot "
        "have duplicates.\n"
        + "\n".join(lines)
        + "\nFix: rename the tool in one of the servers, or disable one of "
        "the conflicting servers in host/config.py (set enabled=False), "
        "or switch back to LLM_PROVIDER=anthropic which namespaces tool "
        "names per server and doesn't have this restriction."
    )


async def run(manager: MCPClientManager, api_key: str) -> None:
    """Run the console chat loop against Gemini with host-managed tool calls."""
    _check_no_collisions(manager)

    client = genai.Client(api_key=api_key)
    tool_declarations = [
        types.FunctionDeclaration(
            name=qualified_name,
            description=f"[{info.server_name}] {info.description}",
            parameters_json_schema=info.input_schema,
        )
        for qualified_name, info in manager.tools.items()
    ]

    chat = client.aio.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=tool_declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    print("QuantDesk ready (provider: gemini). Type your question ('exit' to quit).\n")
    print(
        f"  [i] {len(tool_declarations)} MCP tool declaration(s) handed to Gemini "
        "(host-managed function calling)."
    )

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

        try:
            response = await chat.send_message(user_input)
            for _ in range(GEMINI_MAX_REMOTE_CALLS):
                function_calls = response.function_calls or []
                if not function_calls:
                    print(f"gemini> {response.text}\n")
                    break

                function_responses = []
                for function_call in function_calls:
                    name = function_call.name
                    arguments = function_call.args or {}
                    print(f"  [function_call] {name}({arguments})")
                    try:
                        result = await manager.call_tool(name, arguments)
                        payload = {
                            "result": [
                                getattr(item, "text", str(item))
                                for item in result.content
                            ]
                        }
                    except Exception as exc:  # noqa: BLE001
                        payload = {"error": str(exc)}
                    function_responses.append(
                        types.Part.from_function_response(
                            name=name,
                            response=payload,
                        )
                    )

                response = await chat.send_message(
                    types.Content(role="user", parts=function_responses)
                )
            else:
                print("gemini> (stopped: reached max tool-call iterations for this turn)\n")
        except Exception as exc:  # noqa: BLE001
            print(f"gemini> (error) {exc}\n")
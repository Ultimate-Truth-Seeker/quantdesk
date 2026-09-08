"""Optional browser UI for the QuantDesk host.

The terminal providers remain the default. This module only supplies a web
presentation layer and reuses the host's already-connected MCP sessions.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web

from config import (
    ANTHROPIC_MODEL,
    GEMINI_MODEL,
    GEMINI_MAX_REMOTE_CALLS,
    LLM_PROVIDER,
    MAX_TOKENS,
    MAX_TOOL_ITERATIONS,
)
from mcp_client_manager import MCPClientManager


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QuantDesk // MCP Console</title>
  <style>
    :root { --ink:#162321; --muted:#6e7c76; --paper:#f5f6ef; --panel:#fffef9; --line:#dce3d9; --teal:#0d716b; --teal-dark:#084d4b; --coral:#df735d; --shadow:0 18px 50px rgba(29,57,48,.10); }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:radial-gradient(circle at 85% 0%, #d5e8dc 0, transparent 30%), linear-gradient(135deg,#f5f6ef 0%,#e9f0e8 100%); font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; min-height:100vh; }
    .shell { max-width:1240px; margin:auto; padding:28px; }
    .topbar { display:flex; align-items:flex-end; justify-content:space-between; gap:24px; margin-bottom:24px; }
    .eyebrow { color:var(--teal); font:700 11px/1.2 ui-monospace,SFMono-Regular,monospace; letter-spacing:.14em; text-transform:uppercase; }
    h1 { margin:7px 0 0; font:700 clamp(30px,4vw,52px)/.98 Georgia,"Times New Roman",serif; letter-spacing:-.02em; }
    .subtitle { color:var(--muted); margin:10px 0 0; max-width:560px; }
    .live { display:flex; align-items:center; gap:9px; color:var(--teal-dark); font-size:13px; font-weight:700; white-space:nowrap; }
    .dot { width:9px; height:9px; border-radius:50%; background:#42a66d; box-shadow:0 0 0 5px #d8eedc; }
    .layout { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:20px; align-items:start; }
    .panel { background:rgba(255,254,249,.88); border:1px solid rgba(220,227,217,.95); border-radius:12px; box-shadow:var(--shadow); }
    .chat { min-height:610px; display:flex; flex-direction:column; overflow:hidden; }
    .messages { flex:1; min-height:430px; padding:28px; overflow:auto; }
    .empty { max-width:480px; padding:50px 10px; }
    .empty h2 { font:600 28px/1.1 Georgia,"Times New Roman",serif; margin:0 0 12px; }
    .empty p { color:var(--muted); }
    .message { display:flex; gap:12px; margin:0 0 22px; animation:rise .25s ease both; }
    .message.user { justify-content:flex-end; }
    .avatar { width:30px; height:30px; display:grid; place-items:center; flex:0 0 30px; border-radius:8px; background:var(--teal); color:white; font:700 11px ui-monospace,monospace; }
    .user .avatar { order:2; background:var(--coral); }
    .bubble { max-width:min(760px,82%); white-space:pre-wrap; }
    .user .bubble { background:#e3efdf; border:1px solid #cfe2d0; border-radius:12px 12px 3px 12px; padding:11px 14px; }
    .assistant .bubble { padding:3px 0; }
    .label { color:var(--muted); font:700 10px ui-monospace,monospace; letter-spacing:.1em; text-transform:uppercase; margin-bottom:4px; }
    .composer { border-top:1px solid var(--line); padding:18px; background:#fbfcf7; }
    form { display:flex; gap:10px; align-items:flex-end; }
    textarea { flex:1; resize:none; min-height:48px; max-height:140px; border:1px solid #cbd8cc; border-radius:9px; padding:13px 14px; color:var(--ink); background:white; font:inherit; outline:none; }
    textarea:focus { border-color:var(--teal); box-shadow:0 0 0 3px #d9ece5; }
    button { border:0; border-radius:8px; padding:13px 18px; min-height:48px; background:var(--teal); color:#fff; font-weight:800; cursor:pointer; }
    button:hover { background:var(--teal-dark); } button:disabled { opacity:.55; cursor:wait; }
    .side { padding:20px; }
    .side h3 { margin:0 0 15px; font-size:13px; letter-spacing:.08em; text-transform:uppercase; }
    .identity { border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:18px; }
    .model { font:600 19px Georgia,"Times New Roman",serif; overflow-wrap:anywhere; }
    .provider { display:inline-block; margin-top:8px; border:1px solid #b9d8cd; border-radius:5px; color:var(--teal-dark); background:#e7f3ec; padding:4px 7px; font:700 10px ui-monospace,monospace; text-transform:uppercase; letter-spacing:.08em; }
    .server { border-top:1px solid var(--line); padding:14px 0 2px; }
    .server:first-of-type { border-top:0; padding-top:0; }
    .server-name { display:flex; align-items:center; gap:8px; font-weight:800; }
    .server-name:before { content:""; width:7px; height:7px; border-radius:50%; background:#45a873; }
    .tool-count { color:var(--muted); font:12px ui-monospace,monospace; margin:4px 0 0 15px; }
    .status { color:var(--muted); font-size:12px; margin-top:18px; }
    .error { color:#a0483b; background:#fff0eb; border:1px solid #f1c7bd; border-radius:7px; padding:10px; margin-bottom:15px; font-size:13px; }
    @keyframes rise { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:none; } }
    @media (max-width:800px) { .shell { padding:18px 12px; } .topbar { align-items:flex-start; flex-direction:column; gap:12px; } .layout { grid-template-columns:1fr; } .side { order:-1; } .chat { min-height:560px; } .messages { padding:20px 16px; } .bubble { max-width:88%; } }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div><div class="eyebrow">QuantDesk / MCP console</div><h1>Make the numbers speak.</h1><p class="subtitle">A quiet workspace for portfolio questions, calculations, and the tools behind them.</p></div>
      <div class="live"><span class="dot"></span><span id="connection">Connecting to host</span></div>
    </header>
    <section class="layout">
      <div class="panel chat">
        <div class="messages" id="messages"><div class="empty"><h2>What are we looking at?</h2><p>Ask about portfolio risk, correlations, rebalancing, files, or anything else your connected tools can help with.</p></div></div>
        <div class="composer"><form id="chat-form"><textarea id="prompt" rows="1" placeholder="Ask QuantDesk a question..." aria-label="Message"></textarea><button id="send" type="submit">Send</button></form></div>
      </div>
      <aside class="panel side"><div class="identity"><h3>Coordinator</h3><div class="model" id="model">Loading model...</div><div class="provider" id="provider">Loading provider</div></div><h3>Connected MCPs</h3><div id="servers"></div><div class="status" id="status">Tool calls are routed through the host and logged.</div></aside>
    </section>
  </main>
  <script>
    const messages = document.querySelector('#messages'); const form = document.querySelector('#chat-form'); const prompt = document.querySelector('#prompt'); const send = document.querySelector('#send');
    function addMessage(role, text) { const empty=document.querySelector('.empty'); if(empty) empty.remove(); const row=document.createElement('div'); row.className=`message ${role}`; row.innerHTML=`<div class="avatar">${role==='user'?'YOU':'QD'}</div><div class="bubble"><div class="label">${role==='user'?'You':'QuantDesk'}</div><div></div></div>`; row.querySelector('.bubble div:last-child').textContent=text; messages.appendChild(row); messages.scrollTop=messages.scrollHeight; return row; }
    async function loadStatus() { try { const res=await fetch('/api/status'); const data=await res.json(); document.querySelector('#model').textContent=data.model; document.querySelector('#provider').textContent=data.provider; document.querySelector('#connection').textContent=`${data.connected_servers.length} MCP server${data.connected_servers.length===1?'':'s'} online`; document.querySelector('#servers').innerHTML=data.connected_servers.map(s=>`<div class="server"><div class="server-name">${s.name}</div><div class="tool-count">${s.tool_count} tool${s.tool_count===1?'':'s'} available</div></div>`).join('')||'<div class="status">No MCP servers connected.</div>'; if(data.failed_servers.length) document.querySelector('#status').textContent=`${data.failed_servers.length} server connection${data.failed_servers.length===1?'':'s'} failed.`; } catch(e) { document.querySelector('#connection').textContent='Host unavailable'; } }
    form.addEventListener('submit', async (event) => { event.preventDefault(); const text=prompt.value.trim(); if(!text || send.disabled) return; addMessage('user',text); prompt.value=''; send.disabled=true; send.textContent='Thinking...'; const pending=addMessage('assistant','Working through the connected tools...'); try { const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})}); const data=await res.json(); pending.remove(); addMessage('assistant',data.reply || data.error || 'No response returned.'); } catch(e) { pending.remove(); addMessage('assistant','The host could not complete that request.'); } finally { send.disabled=false; send.textContent='Send'; prompt.focus(); } });
    prompt.addEventListener('keydown', e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit();} }); loadStatus();
  </script>
</body>
</html>"""


def _model_name() -> str:
    return ANTHROPIC_MODEL if LLM_PROVIDER == "anthropic" else GEMINI_MODEL


def _result_text(result: Any) -> str:
    parts = []
    for item in result.content:
        if getattr(item, "type", None) == "text":
            parts.append(item.text)
        elif hasattr(item, "model_dump"):
            parts.append(json.dumps(item.model_dump(mode="json", exclude_none=True)))
        else:
            parts.append(str(item))
    return "\n".join(parts) or "(empty result)"


class WebChatSession:
    def __init__(self, manager: MCPClientManager, api_key: str):
        self.manager = manager
        self.api_key = api_key
        self.lock = asyncio.Lock()
        self.messages: list[dict] = []
        self.gemini_chat = None

    async def start(self) -> None:
        if LLM_PROVIDER == "gemini":
            from google import genai
            from google.genai import types

            declarations = [
                types.FunctionDeclaration(
                    name=name,
                    description=f"[{info.server_name}] {info.description}",
                    parameters_json_schema=info.input_schema,
                )
                for name, info in self.manager.tools.items()
            ]
            self.gemini_chat = genai.Client(api_key=self.api_key).aio.chats.create(
                model=GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt(),
                    tools=[types.Tool(function_declarations=declarations)],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )

    def _system_prompt(self) -> str:
        return """You are QuantDesk, a quantitative portfolio analyst assistant. Use the connected MCP tools for portfolio calculations and file operations. Never estimate financial metrics when a quant tool can calculate them. Explain tool results clearly and briefly. You may answer general knowledge questions without tools."""

    async def send(self, text: str) -> str:
        async with self.lock:
            if LLM_PROVIDER == "gemini":
                return await self._send_gemini(text)
            return await self._send_anthropic(text)

    async def _send_gemini(self, text: str) -> str:
        from google.genai import types

        response = await self.gemini_chat.send_message(text)
        for _ in range(GEMINI_MAX_REMOTE_CALLS):
            calls = response.function_calls or []
            if not calls:
                return response.text or "(empty response)"
            function_responses = []
            for call in calls:
                try:
                    result = await self.manager.call_tool(call.name, call.args or {})
                    payload = {"result": _result_text(result)}
                except Exception as exc:  # noqa: BLE001
                    payload = {"error": str(exc)}
                function_responses.append(types.Part.from_function_response(name=call.name, response=payload))
            response = await self.gemini_chat.send_message(function_responses)
        return "(stopped: reached the tool-call limit for this request)"

    async def _send_anthropic(self, text: str) -> str:
        import anthropic
        from providers.anthropic_provider import SYSTEM_PROMPT, _extract_text, _tool_result_to_content_block

        self.messages.append({"role": "user", "content": text})
        client = anthropic.Anthropic(api_key=self.api_key)
        for _ in range(MAX_TOOL_ITERATIONS):
            response = await asyncio.to_thread(
                client.messages.create,
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=self.messages,
                tools=self.manager.anthropic_tool_specs(),
            )
            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                return _extract_text(response)
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    result = await self.manager.call_tool(block.name, block.input)
                    results.append(_tool_result_to_content_block(block.id, result))
                except Exception as exc:  # noqa: BLE001
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(exc), "is_error": True})
            self.messages.append({"role": "user", "content": results})
        return "(stopped: reached the tool-call limit for this request)"


def _status_payload(manager: MCPClientManager) -> dict:
    servers = []
    for server_name in manager.connected_servers():
        tool_count = sum(info.server_name == server_name for info in manager.tools.values())
        servers.append({"name": server_name, "tool_count": tool_count})
    return {
        "provider": LLM_PROVIDER,
        "model": _model_name(),
        "connected_servers": servers,
        "failed_servers": [{"name": name, "error": error} for name, error in manager.failed_servers().items()],
    }


async def run(manager: MCPClientManager, api_key: str, host: str, port: int) -> None:
    """Start the optional browser UI until interrupted."""
    session = WebChatSession(manager, api_key)
    await session.start()

    async def index(_request: web.Request) -> web.Response:
        return web.Response(text=INDEX_HTML, content_type="text/html")

    async def status(_request: web.Request) -> web.Response:
        return web.json_response(_status_payload(manager))

    async def chat(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            message = str(body.get("message", "")).strip()
            if not message:
                raise ValueError("message is required")
            return web.json_response({"reply": await session.send(message)})
        except Exception as exc:  # noqa: BLE001
            return web.json_response({"error": str(exc)}, status=500)

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/api/status", status), web.post("/api/chat", chat)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"QuantDesk web UI ready at http://{host}:{port}")
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()

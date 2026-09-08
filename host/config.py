"""
Configuration for every MCP server the host connects to, plus which LLM
provider (Anthropic or Gemini) the host uses to drive the conversation.

Each server entry is either:
    - StdioServerConfig: launched as a local subprocess (Filesystem MCP,
      Git MCP, quant-mcp, and classmates' servers all fall here).
    - HttpServerConfig: connected over streamable HTTP. Not currently used
      (point 7 of the assignment, a remote MCP server, was dropped from
      the requirements) but MCPClientManager still supports it if needed.

Edit SERVERS below to match your machine (paths, npx/uvx availability, etc.)
before running the host. See README.md "Human setup steps" for details.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class StdioServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] | None = None
    enabled: bool = True


@dataclass
class HttpServerConfig:
    name: str
    url: str
    enabled: bool = True


# --- Filesystem allowlist -------------------------------------------------
# The official Filesystem MCP server only allows access inside the
# directories you pass on its command line. We scope it to a dedicated
# "workspace" folder inside this repo so the assistant can never touch
# arbitrary files on the machine.
FILESYSTEM_WORKSPACE = os.path.join(REPO_ROOT, "workspace")
os.makedirs(FILESYSTEM_WORKSPACE, exist_ok=True)

# --- Git repository used by the Git MCP server ----------------------------
GIT_REPO_PATH = os.path.join(REPO_ROOT, "workspace")


def ensure_git_initialized(path: str = GIT_REPO_PATH) -> None:
    """
    The official mcp-server-git reference server has NO 'git_init' tool —
    it only operates on a repository that already exists (it validates
    `git.Repo(path)` at startup and refuses to run otherwise). So the host
    bootstraps an empty repo here, once, before connecting. This is a plain
    local subprocess call, not an MCP interaction — the assistant still
    does every actual git operation (status/add/commit/log/branch) through
    the Git MCP server itself.
    """
    import subprocess

    if os.path.isdir(os.path.join(path, ".git")):
        return
    os.makedirs(path, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "quantdesk@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "QuantDesk Bot"], cwd=path, check=True)
    print(f"  [setup] Initialized empty git repository at {path}")


SERVERS: list[StdioServerConfig | HttpServerConfig] = [
    # 4) Official local MCP servers (Anthropic reference servers)
    StdioServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", FILESYSTEM_WORKSPACE],
        enabled=True,
    ),
    StdioServerConfig(
        name="git",
        command="uvx",
        args=["mcp-server-git", "--repository", GIT_REPO_PATH],
        enabled=True,
    ),
    # 5) Our own local MCP server (quant-mcp)
    StdioServerConfig(
        name="quant-mcp",
        command=sys.executable,
        args=[os.path.join(REPO_ROOT, "servers", "quant_mcp", "server.py")],
        cwd=os.path.join(REPO_ROOT, "servers", "quant_mcp"),
        enabled=True,
    ),
    # 6) Classmates' MCP servers — fill these in once the class publishes
    #    their repos. Left disabled for now.
    StdioServerConfig(
            name="rrhh construction",
            command="/Users/macbookproroberto/Documents/quantdesk/servers/mcp-server-rrhh-construccion/.venv/bin/python",
            args=[os.path.join(REPO_ROOT, "servers", "mcp-server-rrhh-construccion", "server.py")],
            enabled=True,
    ),
    StdioServerConfig(
            name="delivery server",
            command="/opt/homebrew/bin/uv",
            args=["run", "--directory", "/Users/macbookproroberto/Documents/quantdesk/servers/delivery-mcp-server", "python", "-m", "route_optimizer.server"],
            enabled=True,
    ),
]


# --- LLM provider selection ------------------------------------------------
# Switch providers with:  export LLM_PROVIDER=anthropic   (or gemini)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
if LLM_PROVIDER not in ("anthropic", "gemini"):
    raise ValueError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'anthropic' or 'gemini'."
    )

# -- Anthropic --
ANTHROPIC_MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 2048
MAX_TOOL_ITERATIONS = 8  # safety cap on chained tool calls per user turn (manual loop)

# -- Gemini --
# Note: Gemini's built-in MCP support (passing a ClientSession directly as
# a tool) is an experimental feature of the google-genai SDK as of this
# writing — see README.md "Gemini provider" for details and known limits.
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_MAX_REMOTE_CALLS = 8  # safety cap on chained tool calls per user turn (automatic loop)
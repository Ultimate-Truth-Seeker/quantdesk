# QuantDesk

A console chatbot for quantitative portfolio analysis, built as Project 1
("Uso de un protocolo existente") for CC3067 Networks — Universidad del
Valle de Guatemala. QuantDesk uses Claude (via the Anthropic API) as its
coordinating LLM and the **Model Context Protocol (MCP)** to give that LLM
tools: real financial calculations, file access, and version control.

> Status: this README covers **weeks 1–3** of the project — the host
> chatbot, the JSON-RPC logger, the official Filesystem/Git MCP servers,
> and the custom `quant-mcp` server. The remote MCP server, classmates'
> servers, and the Wireshark analysis (weeks 4–5) are not part of this
> delivery yet.

## What it does

The analyst describes a portfolio in plain language and asks questions
like *"what's the 95% VaR of this portfolio?"* or *"how correlated are my
tech positions?"*. Claude decides which tool to call, the host invokes it
over MCP, and the **real** calculation (not an LLM guess) comes back from
`quant-mcp`, computed with `pandas`/`numpy`/`scipy` over historical price
data.

## Architecture

```
host/main.py  (chat loop, Claude API, tool orchestration)
   │
   ├── host/mcp_client_manager.py  — connects to every MCP server below,
   │                                  discovers their tools, and routes
   │                                  tool calls to the right one
   ├── host/jsonrpc_tap.py         — transparently taps every raw
   │                                  JSON-RPC message for logging
   └── host/logger.py              — writes mcp_log.jsonl + console output

MCP servers connected:
   ├── filesystem   (official, npx @modelcontextprotocol/server-filesystem)
   ├── git          (official, uvx mcp-server-git)
   └── quant-mcp    (ours, servers/quant_mcp/server.py)
        └── metrics.py — portfolio returns, volatility, Sharpe ratio,
                          historical & parametric VaR, correlation matrix
```

All three servers run locally over **stdio** and are launched as
subprocesses by the host — see `host/config.py`.

## Requirements

- Python 3.11+
- [Node.js](https://nodejs.org/) (for `npx`, used by the Filesystem MCP server)
- [uv](https://docs.astral.sh/uv/) (for `uvx`, used by the Git MCP server)
- An Anthropic API key (the assignment's $5 free credit is enough)
- **Important:** this code targets `mcp` SDK **version 1.x** (`FastMCP`
  lives at `mcp.server.fastmcp`). MCP SDK 2.x renamed `FastMCP` to
  `MCPServer` and changed several APIs — `requirements.txt` pins
  `mcp==1.29.1` for this reason. Do not upgrade past `mcp<2` without
  porting `servers/quant_mcp/server.py` to the new API.

## Human setup steps

1. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Confirm `node`, `npx`, and `uvx` are on your PATH:**
   ```bash
   node --version
   npx --version
   uvx --version
   ```
   If `uv`/`uvx` is missing, install it: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   (see https://docs.astral.sh/uv/getting-started/installation/ for other platforms).

3. **Generate the sample historical price data** (synthetic — see
   "About the price data" below):
   ```bash
   python scripts/generate_sample_data.py
   ```
   This creates `servers/quant_mcp/data/<TICKER>.csv` for 10 tickers.

4. **Set your Anthropic API key:**
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...      # Windows: set ANTHROPIC_API_KEY=...
   ```

5. **Run the host:**
   ```bash
   cd host
   python main.py
   ```
   On first run it will:
   - initialize an empty git repository at `workspace/` (the Git MCP
     server has no `git_init` tool — it requires the repo to already
     exist, so the host bootstraps it once, as a plain local `git init`,
     *not* through MCP),
   - launch and connect to `filesystem`, `git`, and `quant-mcp`,
   - print which servers connected and how many tools each exposes.

6. **Try it.** Example prompts:
   - `Who was Alan Turing?` then `What year was he born?` (tests context — #1/#2)
   - `List the tickers you have data for.`
   - `What's the Sharpe ratio of a portfolio with 10 AAPL and 30 TLT?`
   - `What's the 95% historical VaR of that same portfolio for a $50,000 book?`
   - `How correlated are AAPL, MSFT and NVDA?`
   - `If I cut my NVDA position in half and move it to TLT, what happens to the Sharpe ratio?`
   - `Write a markdown report of that analysis to report.md, then add and commit it to git.`
     (exercises Filesystem + quant-mcp + Git in one flow — the
     "conjunto de servidores" demo scenario)

   Every MCP request/response is printed live and appended to
   `mcp_log.jsonl` at the project root (point 3 of the assignment).

## About the price data

This sandbox/environment has no network access to financial data
providers, so `scripts/generate_sample_data.py` generates **synthetic**
daily OHLCV data using Geometric Brownian Motion with a shared market
factor (so tickers are realistically correlated, not independent) and
per-ticker drift/volatility parameters. It's seeded (`SEED = 42`) for
reproducibility.

**Before the final submission, consider swapping in real historical
data** — e.g. with `yfinance`:
```python
import yfinance as yf
df = yf.download(["AAPL", "MSFT", ...], period="2y")
# reshape to the same date,open,high,low,close,volume columns per ticker
# and save as servers/quant_mcp/data/<TICKER>.csv
```
`quant_mcp/metrics.py` only cares about the CSV schema
(`date,open,high,low,close,volume`), not where the data came from.

## quant-mcp — tool specification

All tools take/return JSON. All calculations are computed with `pandas`/
`numpy`/`scipy` from the CSV data in `servers/quant_mcp/data/` — nothing
is estimated by the LLM.

| Tool | Parameters | Returns |
|---|---|---|
| `list_available_tickers` | — | `{"tickers": [...]}` |
| `get_portfolio_summary` | `positions: [{ticker, quantity}]`, `risk_free_rate?`, `window_days?` | weights, annualized return, annualized volatility, Sharpe ratio |
| `calculate_volatility` | `positions`, `window_days?` | portfolio + per-ticker annualized volatility |
| `calculate_sharpe_ratio` | `positions`, `risk_free_rate?`, `window_days?` | Sharpe ratio |
| `calculate_var` | `positions`, `confidence?` (default 0.95), `portfolio_value?`, `method?` (`historical`\|`parametric`), `window_days?` | Value at Risk amount + plain-language interpretation |
| `get_correlation_matrix` | `tickers: [string]`, `window_days?`, `top_n_pairs?` | full correlation matrix + most correlated pairs |
| `simulate_rebalance` | `current_positions`, `proposed_positions`, `risk_free_rate?`, `window_days?` | before/after metrics + deltas |

`positions` is always a list of `{"ticker": "AAPL", "quantity": 10}`
objects. `window_days` (optional, all tools) limits the calculation to
the most recent N trading days instead of the full ~2-year history.

You can run `quant-mcp` standalone (outside the host) for manual testing:
```bash
cd servers/quant_mcp
python server.py
```
It will wait on stdio — use the
[MCP Inspector](https://github.com/modelcontextprotocol/inspector)
(`npx @modelcontextprotocol/inspector python server.py`) to poke at it
interactively.

## Testing

```bash
python -m pytest tests/ -v
```

- `tests/test_metrics.py` — unit tests for the financial calculations
  (returns, volatility, Sharpe, VaR, correlation) against known/synthetic
  inputs — no MCP protocol involved.
- `tests/test_quant_mcp_server.py` — launches `quant-mcp` as a real
  subprocess and talks MCP to it (`list_tools`, `call_tool`), the same
  way the host does.
- `tests/test_host_integration.py` — launches **all three** week 1–3
  servers (filesystem, git, quant-mcp) through `MCPClientManager` and
  exercises a real create-file → add → commit flow, without calling the
  Anthropic API.

## Project layout

```
quantdesk/
├── host/
│   ├── main.py                 # chat loop entry point
│   ├── config.py                # server list + settings
│   ├── mcp_client_manager.py    # connects & routes to all MCP servers
│   ├── jsonrpc_tap.py            # raw JSON-RPC message interception
│   └── logger.py                 # JSON-RPC logging (console + file)
├── servers/
│   └── quant_mcp/
│       ├── server.py             # MCP tool definitions
│       ├── metrics.py            # financial calculations (pure functions)
│       └── data/*.csv            # historical price data
├── scripts/
│   └── generate_sample_data.py   # synthetic price data generator
├── tests/
├── workspace/                     # scratch repo for Filesystem/Git demos
├── requirements.txt
└── README.md
```

## Known limitations (weeks 1–3 scope)

- Price data is synthetic (see "About the price data" above).
- No remote MCP server yet (`market-data-mcp` — week 4).
- No classmates' MCP servers integrated yet (week 4).
- No Wireshark capture/analysis yet (week 5) — there's no remote/HTTP
  traffic to capture until the remote server exists.
# AI infrastructure: MCP server, MCP security gateway and LLM gateways

Four small, independent services for running AI agents and LLMs safely in production:

- an MCP server with strict input validation;
- a security gateway in front of MCP servers;
- a streaming LLM gateway that redacts PII in real time;
- a token-aware rate limiter that fails over between model providers.

They were built for an AI Field Developer / Solutions Engineer technical assessment; the brief itself is not
included. How the work was planned, the architecture, each service's design and the trade-offs are in
[PLANNING_AND_SOLUTION.md](PLANNING_AND_SOLUTION.md).

## The problem

| Task | What the brief asks for |
|---|---|
| 1. MCP server | An MCP server on the official Python SDK with two tools, `get_customer_record(customer_id)` and `trigger_refund(customer_id, amount, reason)`. Pydantic validates the inputs: `CUST-XXXXX` ids, a positive amount, and a reason of at least 10 characters. Invalid input gets standard JSON-RPC error codes, and over stdio, stdout carries only protocol messages. |
| 2. MCP security gateway | A proxy in front of an MCP server that reads the caller's role from the `Authorization: Bearer` token and forwards `tools/list` unchanged. Only admins may call tools whose names start with `admin_`; anyone else gets JSON-RPC error `-32001 "Unauthorized Tool Call"`, and the downstream server is never called. A mock downstream server comes with it. |
| 3. Streaming PII redaction | An LLM gateway that streams a provider's reply back while replacing emails, SSNs and credit card numbers with `[REDACTED]` in real time, without accumulating the whole response and with minimal delay to the first token. |
| 4. Rate limiting and failover | A token-aware sliding-window limit per tenant API key (for example 50,000 tokens per minute), kept in SQLite on disk. When the primary provider answers 429 or takes longer than 3,000 ms, the request fails over to a secondary provider. Errors use one standardized payload, without stack traces or internals. |

## The solution

Each task is its own Python package with its own tests, and runs on its own:

| Task | Package | What was built |
|---|---|---|
| 1 | `ai_gateway.mcp_server` | The MCP SDK's low-level server with strict Pydantic models. Invalid arguments become `-32602` errors with field-level details that never echo the input, and unexpected exceptions become `-32603`. Its own stdin reader answers malformed lines that the SDK would silently drop, and stray output can never reach stdout. |
| 2 | `ai_gateway.mcp_gateway` | A FastAPI reverse proxy that verifies HS256 JWTs and parses every JSON-RPC message strictly. It allows only known methods and answers `-32001` itself for non-admin `admin_*` calls. It forwards a message rebuilt from what it checked, never the client's token, and streams the reply back. Bypasses such as duplicate JSON keys, batches and responses carrying a method are rejected. |
| 3 | `ai_gateway.llm_gateway` | An OpenAI-compatible streaming endpoint. Its redactor holds back only the end of the text that could still become PII (at most 342 characters), so text flows about one token late and PII split across chunks is still caught. Only text the gateway can redact is relayed: tool calls, logprobs and unknown fields are refused. |
| 4 | `ai_gateway.model_router` | An OpenAI-compatible endpoint for tenants with API keys. Each request reserves its worst-case token cost in one atomic SQLite transaction (429 with an exact `Retry-After` when over budget), then settles to the provider's real usage. The primary has 3,000 ms for its whole response; on 429, a timeout or a 5xx, the secondary takes over. Every error has the same sanitized shape. |

All of it is tested: 1,349 tests, including end-to-end runs of every service as real processes, driven by the official
MCP client (Tasks 1 and 2) and the official OpenAI Python SDK (Task 3).

## How to run

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`, and uv installs it if needed.

```bash
git clone https://github.com/x-string/ai-infrastructure-mcp-llm-gateway.git
cd ai-infrastructure-mcp-llm-gateway
uv sync                                       # install the locked dependencies
uv run pytest -q                              # every test, about a minute
uv run pytest tests/llm_gateway -q            # one task: mcp_server, mcp_gateway, llm_gateway or model_router
uv run pytest -m "not e2e and not slow" -q    # the fast subset, without subprocesses or timing tests
```

### Task 1: MCP server

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"cli","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"trigger_refund","arguments":{"customer_id":"CUST-00042","amount":-5,"reason":"duplicate charge"}}}' \
  | uv run t1-mcp-server 2>/dev/null
```

The refund is refused with `-32602 Invalid params`, naming the `amount` field. To use the server from an MCP host,
run `uv --directory /path/to/repo run t1-mcp-server`, or try it with
`npx @modelcontextprotocol/inspector uv --directory "$PWD" run t1-mcp-server`.

### Task 2: MCP security gateway

```bash
cp .env.example .env    # set GATEWAY_JWT_SECRET, for example to the output of: openssl rand -hex 32
uv run t2-mock-downstream                    # terminal 1: a demo MCP server on 127.0.0.1:8001
uv run --env-file .env t2-gateway            # terminal 2: the gateway on 127.0.0.1:8000
VIEWER=$(uv run --env-file .env t2-mint-jwt --role viewer --subject alice)
curl -s http://127.0.0.1:8000/mcp -H "authorization: Bearer $VIEWER" -H 'content-type: application/json' \
  -H 'accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"key_id":"k-1"}}}'
```

The viewer gets `-32001 Unauthorized Tool Call`. A token minted with `--role admin` gets the tool's result.

### Task 3: Streaming PII redaction

```bash
uv run t3-mock-provider                      # terminal 1: a mock provider on 127.0.0.1:9100 that echoes the prompt
uv run t3-gateway                            # terminal 2: the gateway on 127.0.0.1:8200
curl -N http://127.0.0.1:8200/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "echo", "messages": [{"role": "user", "content": "Mail john.doe@example.com now"}]}'
```

The mock streams three characters at a time. The reply still arrives as `Mail `, `[REDACTED] `, `now`. Any OpenAI
SDK works as the client with `base_url="http://127.0.0.1:8200/v1"`. To use a real provider, set `T3_UPSTREAM_URL` and
`T3_UPSTREAM_API_KEY`.

### Task 4: Rate limiting and failover

```bash
KEY=$(uv run t4-tenants add acme --tokens-per-minute 3000)                # creates t4-router.db, prints the key
uv run t4-mock-provider --port 9201 --name primary --mode rate_limited    # terminal 1: a primary that answers 429
uv run t4-mock-provider --port 9202 --name secondary                      # terminal 2
uv run t4-router                                                          # terminal 3: the router on 127.0.0.1:8300
curl -si http://127.0.0.1:8300/v1/chat/completions -H "authorization: Bearer $KEY" \
  -H 'content-type: application/json' -d '{"model": "gpt-test", "messages": [{"role": "user", "content": "Hello"}]}'
```

The secondary answers (`x-gateway-provider: secondary`), and `x-ratelimit-remaining-tokens` shows what is left of
the budget. A request that could never fit, for example one with `"max_tokens": 4000`, gets `400 request_too_large`.
Concurrent requests beyond the budget get `429` with `Retry-After`.

Every setting is listed in `.env.example`. The default ports never collide, so all the services can run at once:
8000 and 8001 for Task 2, 8200 and 9100 for Task 3, and 8300, 9201 and 9202 for Task 4.

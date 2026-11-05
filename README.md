# AI infrastructure: MCP server, MCP security gateway and LLM gateways

Four independent services built for the AI Field Developer / Solutions Engineer assessment
([assessment.pdf](assessment.pdf)). Each task has its own subpackage and test folder, and runs on its own.
The subpackages never import each other.

| Task | Package | What it does | Status |
|---|---|---|---|
| 1 | `ai_gateway.mcp_server` | MCP server over stdio with strict input validation | In progress: server and error mapping done, stdio entry point next |
| 2 | `ai_gateway.mcp_gateway` | MCP security gateway: bearer-token roles, tool-level authorization | Planned |
| 3 | `ai_gateway.llm_gateway` | Streaming LLM gateway with real-time PII redaction | Planned |
| 4 | `ai_gateway.model_router` | Token-aware rate limiter on SQLite with primary-to-backup failover | Planned |

## Architecture

How the pieces fit together. In this repo, Tasks 3 and 4 run as separate services. Combining them into a single
streaming pipeline would only allow failover before the first token reaches the client.

```mermaid
flowchart TB
    agent(["AI agent / client"])
    subgraph T1["Task 1 · MCP server (local)"]
        mcpserver["MCP server<br/>strict Pydantic tools"]
    end
    agent <-->|"stdio · JSON-RPC on stdout · logs on stderr"| mcpserver
    subgraph T2["Task 2 · MCP security gateway"]
        authn["AuthN<br/>Bearer JWT → role"] --> authz["AuthZ<br/>strict JSON-RPC parse · method allowlist · admin_* rule"] --> proxy["Proxy<br/>rebuilt body · client token never forwarded"]
    end
    agent -->|"POST /mcp"| authn
    authn -.->|"401"| agent
    authz -.->|"-32001 · downstream never called"| agent
    proxy <-->|"JSON / SSE"| downstream["Downstream MCP server (mock)"]
    proxy -->|"response streamed back"| agent
    subgraph LLM["LLM gateway · Tasks 3 + 4"]
        tenant["Tenant API-key auth"] --> limiter["Rate limiter<br/>reserve → reconcile"] --> router["Model router<br/>primary deadline 3,000 ms"] --> redactor["Streaming PII redactor"]
    end
    agent -->|"POST /v1/chat/completions"| tenant
    limiter <--> db[("SQLite<br/>tenants · token_usage")]
    limiter -.->|"429 + Retry-After"| agent
    router -->|"attempt 1"| primary["Primary LLM provider"]
    router -.->|"attempt 2 on 429 / timeout"| backup["Backup LLM provider"]
    redactor -->|"SSE with [REDACTED]"| agent
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`, and uv installs it if it is missing.

```bash
uv sync                              # create .venv and install locked dependencies
uv run pytest -q                     # every test
uv run pytest tests/mcp_server -q    # one task at a time
uv run ruff check                    # lint
```

## Task 1: MCP server with strict validation

Status: the tool models (`schemas.py`) and the MCP server with its error mapping (`server.py`) are done.
The stdio entry point comes next.

Tools:

- **`get_customer_record`**: returns the customer's profile as structured content.
- **`trigger_refund`**: records a refund and returns a receipt.

Both tools publish their input and output JSON Schemas and behavior hints (read-only or not) in
`tools/list`. The input schema is generated from the same Pydantic model that validates each call, so
what clients see is exactly what the server enforces.

### Validation rules

| Argument | Rule | Rejected examples |
|---|---|---|
| `customer_id` | `^CUST-[0-9]{5}$` | `CUST-1234`, `cust-12345`, `" CUST-12345"`, `"CUST-12345\n"`, `CUST-１２３４５` (fullwidth digits), `12345` |
| `amount` | number, greater than 0, finite; JSON integers accepted | `0`, `-0.0`, `-1`, `NaN`, `Infinity`, `1e309`, `"10.5"`, `true`, `null` |
| `reason` | string of 10 to 1,000 characters, no leading or trailing whitespace | `"012345678"`, `"          "`, `" refund requested"`, `1234567890` |
| any other key | not allowed (`additionalProperties: false`) | `{"approved_by": "me"}` |

### Error mapping

| Situation | Response |
|---|---|
| Arguments fail validation (format, type, missing or unknown key) | JSON-RPC error `-32602 Invalid params` with `data: {tool, errors: [{field, message, type}]}` |
| Unknown tool | JSON-RPC error `-32602 Unknown tool: <name>` |
| Valid arguments, but no such customer | Tool result with `isError: true` (a business outcome, not a protocol error) |
| Any unexpected exception | JSON-RPC error `-32603 Internal error`; the details go to the server log only |

Example: `trigger_refund` called with `{"customer_id": "CUST-00042", "amount": -5, "reason": "duplicate charge"}`
returns

```json
{"jsonrpc": "2.0", "id": 7, "error": {"code": -32602, "message": "Invalid params", "data": {"tool": "trigger_refund",
 "errors": [{"field": "amount", "message": "Input should be greater than 0", "type": "greater_than"}]}}}
```

Error details never echo submitted values. The names of unexpected keys are shown only when they look like
ordinary identifiers. Unexpected exceptions are wrapped explicitly: without that, the SDK would answer stdio
clients with `{"code": 0, "message": "<exception text>"}`, which leaks internals. A test reproduces that leak.

## Requirements versus implementation choices

The assessment leaves some details open. These are the choices made, and why:

- **Invalid tool arguments are JSON-RPC errors (`-32602`), as the brief requires.** Recent MCP spec
  revisions (2025-11-25 onward) suggest reporting input-validation failures as `isError` tool results so
  the model can self-correct. The SDK's high-level `MCPServer` does exactly that, which is why this server
  uses the low-level `Server`. The spec itself is not consistent here: its 2026-07-28 `InvalidParamsError`
  definition still lists invalid tool arguments under `-32602`. Unknown tools are `-32602` in every
  revision.
- **`CUST-XXXXX` means five ASCII digits.** The pattern uses `[0-9]`, not `\d`, because `\d` also matches non-ASCII
  digits such as `١٢٣٤٥`, both in Python's `re` and in pydantic's regex engine.
- **No type coercion.** `"10.5"` and `true` are not accepted as amounts. JSON integers are accepted because JSON has
  a single number type.
- **`reason` may not start or end with whitespace.** This rejects whitespace-only reasons while keeping the rule
  expressible in the published JSON Schema (`minLength` plus `pattern`).
- **`amount` follows the brief literally: positive and finite.** A maximum amount or a two-decimal rule would be
  business rules the brief does not state, so they are not enforced.

## Project layout

```
src/ai_gateway/
├── mcp_server/     # Task 1
├── mcp_gateway/    # Task 2
├── llm_gateway/    # Task 3
└── model_router/   # Task 4
tests/
└── mcp_server/     # one folder per task, added as each task is built
```

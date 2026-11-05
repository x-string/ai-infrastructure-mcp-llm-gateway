# AI infrastructure: MCP server, MCP security gateway and LLM gateways

Four independent services built for an AI Field Developer / Solutions Engineer technical assessment. The
brief itself is not included in this repository. Each task has its own subpackage and test folder, and runs on
its own. The subpackages never import each other.

| Task | Package | What it does | Status |
|---|---|---|---|
| 1 | `ai_gateway.mcp_server` | MCP server over stdio with strict input validation | Done |
| 2 | `ai_gateway.mcp_gateway` | MCP security gateway: bearer-token roles, tool-level authorization | Done |
| 3 | `ai_gateway.llm_gateway` | Streaming LLM gateway with real-time PII redaction | In progress: redactor done |
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
uv run pytest -m "not e2e and not slow" -q   # skip subprocess, real-server and timing tests
uv run ruff check                    # lint
```

## Task 1: MCP server with strict validation

### Run it

```bash
uv run t1-mcp-server                 # or: uv run python -m ai_gateway.mcp_server
```

The server speaks MCP over stdin and stdout, so it is normally started by an MCP host. To try it from a shell:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"cli","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_customer_record","arguments":{"customer_id":"CUST-00042"}}}' \
  | uv run t1-mcp-server 2>/dev/null
```

- With MCP Inspector: `npx @modelcontextprotocol/inspector uv --directory "$PWD" run t1-mcp-server`.
- In an MCP host such as Claude Desktop:
  `{"mcpServers": {"customer-records": {"command": "uv", "args": ["--directory", "/absolute/path/to/repo", "run", "t1-mcp-server"]}}}`.
- Logs go to stderr. Set `MCP_SERVER_LOG_LEVEL=DEBUG` for more detail.

Demo customers: `CUST-00042`, `CUST-01815`, `CUST-12345`. The data is fictional and kept in memory.

### Tools

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

### The stdio transport

The SDK's `stdio_server()` keeps the protocol on a private copy of stdout and points fd 1 at stderr while
serving. Its stdin handling is too lenient, though, so `stdio_transport.py` reads stdin itself:

| Input line | Reply |
|---|---|
| Blank | Ignored |
| Not JSON, not UTF-8, `NaN`/`Infinity` literal, duplicate keys | `-32700 Parse error`, `"id": null` |
| Batch array; `id` that is `null`, a boolean or a float; missing `"jsonrpc": "2.0"`; line over 1 MiB | `-32600 Invalid Request`, with the `id` echoed when it is valid |
| A request whose `params` is not an object | `-32602 Invalid params` |
| `tools/call` before `initialize` | `-32602` (from the SDK) |
| Unknown method | `-32601 Method not found` (from the SDK) |

What the transport adds:

- **End-of-input drain.** When stdin closes, pending requests get up to 1.5 seconds to finish before the
  server shuts down, so a host that pipes requests and closes stdin gets every reply.
- **Stray-output guard.** While serving, `sys.stdout` is redirected to stderr, and the original stream is
  flushed before fd 1 is restored. Buffered stray output therefore cannot reach the protocol stream at exit.

Measured against the SDK's plain `stdio_server()`, same server, same piped input (20 tool calls, then two
malformed lines; the end-to-end tests cover each behavior):

| | Tool replies | Errors for malformed lines | Stray text on stdout |
|---|---|---|---|
| Plain `stdio_server()` | 11 to 12 of 20 | 0 of 2 (the client waits forever) | Yes |
| This transport | 20 of 20 | 2 of 2 | No |

## Task 2: MCP security gateway

An HTTP reverse proxy in front of an MCP server. It reads the bearer token, extracts the caller's role, and inspects
every JSON-RPC message before deciding whether to forward it.

### Run it

```bash
cp .env.example .env                       # then set GATEWAY_JWT_SECRET, for example to $(openssl rand -hex 32)
uv run t2-mock-downstream                  # demo MCP server (official SDK) on 127.0.0.1:8001
uv run --env-file .env t2-gateway          # the gateway on 127.0.0.1:8000
VIEWER=$(uv run --env-file .env t2-mint-jwt --role viewer --subject alice)
ADMIN=$(uv run --env-file .env t2-mint-jwt --role admin --subject root)

mcp() { curl -s http://127.0.0.1:8000/mcp -H "authorization: Bearer $1" \
  -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' -d "$2"; }
mcp "$VIEWER" '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
mcp "$VIEWER" '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"key_id":"k-1"}}}'
mcp "$ADMIN"  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"key_id":"k-1"}}}'
```

The viewer's admin call is answered by the gateway and never reaches the downstream server:

```json
{"jsonrpc":"2.0","id":2,"error":{"code":-32001,"message":"Unauthorized Tool Call",
 "data":{"reason":"forbidden_tool","tool":"admin_reset_key","required_role":"admin"}}}
```

The official MCP client works unchanged; the token goes on its HTTP client:

```python
async with httpx2.AsyncClient(headers={"authorization": f"Bearer {token}"}) as http:
    async with Client(streamable_http_client("http://127.0.0.1:8000/mcp", http_client=http)) as client:
        await client.call_tool("admin_reset_key", {"key_id": "k-1"})  # MCPError -32001 for a viewer
```

The demo downstream is a stateless MCP server with the tools `echo`, `get_status`, `slow_echo`, `admin_reset_key` and
`admin_delete_user`. Pass `--sse` to make it answer with event streams.

### How a request is handled

| Step | Rejected with |
|---|---|
| 1. An `Origin` header, if present, must be allow-listed (DNS-rebinding protection) | 403 |
| 2. `Authorization: Bearer <JWT>`: HS256 only; `exp`, `iat`, `sub`, `aud` and `role` required | 401 with a `WWW-Authenticate` challenge |
| 3. `Content-Type: application/json`, body at most 1 MiB | 415 / 413 |
| 4. Strict JSON-RPC parsing: exactly one request or notification | `-32700` / `-32600` (HTTP 400) |
| 5. 2026-07-28 clients: the `Mcp-Method` and `Mcp-Name` headers must match the body | `-32020` (HTTP 400) |
| 6. Policy: method allowlist and the `admin_` rule | `-32601`, `-32602` or `-32001` |
| 7. Forward: the message is rebuilt from the parsed fields, and the reply streams back as it arrives | 502 / 504 if the downstream fails |

Every decision is written to an audit log (subject, role, method, tool, decision). Tokens are never logged.

### Authorization rules

- `tools/list` is forwarded unfiltered for any authenticated role, as the brief requires.
- A `tools/call` for a tool whose name starts with `admin_` needs the `admin` role. Otherwise the gateway answers
  `-32001 Unauthorized Tool Call` itself. The prefix is compared case-insensitively, so `ADMIN_reset_key` fails closed.
- Tool names must use MCP tool-name characters (`[A-Za-z0-9_.-]`, up to 128). That rules out whitespace and Unicode
  look-alikes such as a fullwidth `ａｄｍｉｎ_`.
- Only `initialize`, `ping`, `server/discover`, `tools/list`, `tools/call` and the lifecycle notifications pass.
  Anything else is `-32601`. A `tools/call` sent without an `id` (which would run with no reply) is rejected.

### Hardening against bypasses

| Attack | Defense |
|---|---|
| Duplicate keys: `{"name": "get_status", "name": "admin_reset_key"}` | Rejected (`-32700`): JSON parsers disagree on which value wins |
| A batch with an admin call inside | Rejected (`-32600`): MCP no longer supports batching |
| A "response" that also has a method: `{"result": {}, "method": "tools/call", ...}` | Rejected (`-32600`): the SDK downstream ignores the extra keys and would run it as a request |
| Headers naming one tool while the body names another (2026-07-28) | Rejected (`-32020`), using the SDK's own checks |
| Reusing the client's token against the downstream | The `Authorization` header is never forwarded (MCP forbids token passthrough) |
| Bytes that differ from what was checked | The forwarded body is rebuilt from the parsed message |

### Tested end to end

`tests/mcp_gateway/test_e2e.py` starts the real gateway and the SDK downstream as processes, and drives them with
the official MCP client over TCP, in both protocol eras (2025-11-25 and 2026-07-28) and both downstream modes (JSON
and SSE). It covers:

- unfiltered tool listing;
- `-32001` for viewers, with the downstream's own call log confirming nothing arrived;
- admin calls;
- 401 for missing or invalid tokens;
- a tool that runs longer than 5 seconds;
- a client hang-up that cancels the downstream tool.

## Task 3: Streaming LLM gateway with real-time PII redaction

In progress. Done so far: the streaming redactor, `ai_gateway.llm_gateway.redaction`. The gateway endpoint, the
upstream client and the mock provider come next.

### What is redacted

| Kind | Shape | Example |
|---|---|---|
| Email | ASCII `local@domain.tld`; local part 1 to 64 characters, TLD 2 to 24 letters | `john.doe@example.com` |
| SSN | `ddd-dd-dddd` or `ddd dd dddd` (one separator style), and nine bare digits that can be an SSN (area not 000 or 666, group not 00, serial not 0000) | `123-45-6789`, `123456789` |
| Card | 13 to 19 digits, optionally with single spaces or hyphens between them | `4111 1111 1111 1111` |

Detection is by shape, because redacting too much is the safe direction. So `1234 5678 9012 3456` is redacted even
though it fails the Luhn check, and so is the SSA example number `987-65-4321`. Two options trade recall for
precision. `card_luhn` redacts only numbers that start with 2 to 6 and pass the Luhn check. `bare_ssn=False` leaves
bare nine-digit numbers alone.

### Redacting a stream without buffering it

PII can be split across deltas: `"Mail john.do"`, then `"e@example.com now"`. Releasing each delta as it arrives
would leak `john.do`, and buffering the whole response would defeat streaming. The redactor holds back only the
end of the text that could still become part of a match:

- Each pattern family has a closed alphabet and a maximum length. Emails use letters, digits and `._%+-@`, and are
  at most 342 characters long. SSNs and cards use digits, spaces and hyphens, and are at most 37 characters long.
- A match that later text could still change must lie in the trailing run of its family's alphabet, within its
  maximum length of the end. Everything before that point is final, so it is redacted and released at once.
- Matching is tried only where a match can start: at a digit not preceded by a digit, or at most 64 characters
  before an `@`. Each position is examined once.

For the example above, the client receives `"Mail "`, then `"[REDACTED] "`. The word `now` follows with the next
delta or at the end of the stream, because it could still be the start of an email address. Ordinary prose is
released about one token late for the same reason.

### Guarantees, and how they are tested

- **The way the text is split does not matter.** For any text split into any deltas, the streamed output equals
  redacting the whole text at once, and nothing released is ever taken back. Hypothesis checks this on random texts
  and random splits, and a unit test splits each kind of PII at every offset.
- **The rules are exactly the documented ones.** Batch output matches an independent reference implementation,
  and, without Luhn, a plain `re.sub` with the same pattern.
- **Bounded memory.** At most 342 characters are held back, whatever the size of the deltas.
- **Fast on adversarial input.** 1 MB fed in 1 to 4 character deltas takes 0.4 to 1 s for digits, email-like
  text, card-like text, JWT-like text and prose (`uv run pytest -m slow`).
- **The holdback's assumptions hold.** A test walks the parsed patterns. Every character they can consume belongs
  to their family's alphabet, every lookbehind is one character wide, and every match length is bounded.

### Known limits

- Non-ASCII, quoted and IP-literal email addresses are not detected. A local part over 64 characters leaves its
  first characters visible.
- Next to other digit groups, shapes cannot tell where a number ends. An extra group can stay visible
  (`ref 1234 4111 1111 1111 1111` becomes `ref [REDACTED] 1111`), and a card-shaped run can absorb the first digits
  of an adjacent SSN, leaving its last four visible.
- At the normal end of a stream, an unfinished fragment is released as it is, since `john.doe@exa` is not an email
  address. When a stream is cut off by an error, the held-back text is dropped instead.

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
- **Gateway tokens are HS256-signed JWTs with a `role` claim.** This is one way to "extract the user's role" from a
  bearer token. `t2-mint-jwt` mints development tokens; in production an identity provider would issue them, and
  OAuth discovery (RFC 9728 protected-resource metadata) is out of scope here.
- **`-32001` is returned with HTTP 200.** The error stays inside the JSON-RPC response, so the official MCP client
  raises `MCPError(-32001)`. An HTTP 403 would make MCP clients start an OAuth step-up flow instead. The code itself
  is what the brief asks for. Note that the 2026-07-28 spec marks `-32000` to `-32019` as a legacy range, and the
  Python SDK uses `-32001` for its own local request timeouts. The stable `data.reason = "forbidden_tool"` tells the
  two apart.
- **Methods other than `tools/list` and `tools/call` go through an allowlist.** The brief only defines those two
  methods. Lifecycle messages are allowed, and everything else is denied (fail closed).
- **Cancellation on client hang-up depends on the downstream.** The gateway always closes its downstream request
  when the client disconnects. An SDK server answering with event streams then cancels the tool. In its JSON-response
  mode, the SDK does not watch for disconnects, so the tool runs to completion.
- **PII is detected by shape, not validated.** The brief names emails, SSNs and credit card numbers without
  defining their formats. Shape matching redacts some numbers that are not real cards or SSNs, which is the safe
  direction for a privacy filter. Luhn validation is available as an option.

## Project layout

```
src/ai_gateway/
├── mcp_server/     # Task 1
├── mcp_gateway/    # Task 2
├── llm_gateway/    # Task 3
└── model_router/   # Task 4
tests/
├── mcp_server/     # one folder per task, added as each task is built
├── mcp_gateway/
└── llm_gateway/
```

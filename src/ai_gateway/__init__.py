"""Four independent services for the AI infrastructure assessment.

- ``ai_gateway.mcp_server``: Task 1, MCP server over stdio with strict input validation.
- ``ai_gateway.mcp_gateway``: Task 2, MCP security gateway (bearer roles, tool authorization).
- ``ai_gateway.llm_gateway``: Task 3, streaming LLM gateway with real-time PII redaction.
- ``ai_gateway.model_router``: Task 4, token-aware rate limiter with model failover.

The subpackages never import each other, so each task can be run and tested on its own.
"""

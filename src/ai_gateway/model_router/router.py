"""Routing one chat completion for Task 4: admission, attempts with failover, and charging.

A request is admitted once, for its full estimated cost, before any provider is tried. Then the
providers are tried in order: the primary under its 3,000 ms total deadline, then the secondary.
Whatever happens, the reservation is settled exactly once, with:

* nothing for an attempt that never reached its provider, or that it answered with 429;
* the prompt estimate for an attempt that timed out, failed with a 5xx or broke after the request
  was sent, since the provider may have done the work (otherwise a client could provoke timeouts
  to get free tokens);
* the provider's own token count for the attempt that succeeded, or the full reservation when
  that count is missing or not believable.

Settlement is queued, never awaited, and a failure to record it is logged: it never changes the
response. The client sees the outcome of the last attempt:

=========================================  ======  ============================  ===========
Last attempt                               Status  Code                          Failed over
=========================================  ======  ============================  ===========
timed out                                  504     upstream_timeout              yes
429 (429 is kept for the gateway's own)    503     upstream_unavailable          yes
5xx, unreachable, broken or invalid        502     upstream_error                yes
400, 413, 422                              400     upstream_rejected_request     no
401, 403, 404 and other 4xx                502     upstream_configuration_error  no
=========================================  ======  ============================  ===========
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import httpx2

from ai_gateway.model_router import errors
from ai_gateway.model_router.limiter import TokenLimiter
from ai_gateway.model_router.providers import (
    CHARGES_PROMPT,
    FAILS_OVER,
    Attempt,
    Outcome,
    Provider,
    attempt,
)
from ai_gateway.model_router.tenants import Tenant
from ai_gateway.model_router.tokens import request_cost

logger = logging.getLogger("ai_gateway.model_router.router")


@dataclass(frozen=True, slots=True)
class Routed:
    body: dict[str, object]
    provider: str
    limit: int
    remaining: int


class Router:
    def __init__(
        self, limiter: TokenLimiter, client: httpx2.AsyncClient, providers: Sequence[Provider]
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self._limiter = limiter
        self._client = client
        self._providers = tuple(providers)

    async def complete(
        self, *, tenant: Tenant, request_id: str, payload: dict[str, object], max_tokens: int
    ) -> Routed:
        """Admit, try the providers in order, settle once, and return or raise for the client."""
        limit = tenant.tokens_per_minute
        cost = request_cost(payload, max_tokens)
        prompt = cost - max_tokens
        if cost > limit:
            raise errors.request_too_large(cost, limit)
        admitted = await self._limiter.reserve(tenant.tenant_id, limit, request_id, cost)

        charge = 0
        last: Attempt | None = None
        try:
            for provider in self._providers:
                last = await attempt(self._client, provider, payload)
                logger.info(
                    "request %s tenant %s provider %s: %s (%s) in %d ms",
                    request_id,
                    tenant.tenant_id,
                    last.provider,
                    last.outcome,
                    last.detail,
                    last.elapsed_ms,
                )
                if last.outcome in CHARGES_PROMPT:
                    charge += prompt
                if last.outcome not in FAILS_OVER:
                    break
        except BaseException:  # cancelled mid-attempt: the provider may already be working
            self._limiter.settle(tenant.tenant_id, request_id, charge + prompt)
            raise

        assert last is not None
        completion = last.completion if last.outcome is Outcome.SUCCESS else None
        if completion is not None:
            charge += cost if completion.total_tokens is None else completion.total_tokens
        self._limiter.settle(tenant.tenant_id, request_id, charge)
        logger.info("request %s tenant %s charged %d tokens", request_id, tenant.tenant_id, charge)
        if completion is None:
            raise client_error(last.outcome)
        return Routed(completion.body, last.provider, limit, admitted.remaining + cost - charge)


def client_error(outcome: Outcome) -> errors.GatewayError:
    """The error for the client when the last attempt ended with ``outcome``."""
    if outcome is Outcome.TIMEOUT:
        return errors.upstream_timeout()
    if outcome is Outcome.RATE_LIMITED:
        return errors.upstream_unavailable()
    if outcome is Outcome.REJECTED:
        return errors.upstream_rejected()
    if outcome is Outcome.MISCONFIGURED:
        return errors.upstream_misconfigured()
    return errors.upstream_failed()

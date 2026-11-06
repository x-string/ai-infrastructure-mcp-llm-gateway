"""Task 4: one provider attempt, its total deadline, and how its outcome is classified."""

from __future__ import annotations

import json
import time

import httpx2
import pytest

from ai_gateway.model_router.providers import (
    MAX_RESPONSE_BYTES,
    USER_AGENT,
    Outcome,
    Provider,
    attempt,
    parse_completion,
)
from tests.model_router.scripted import (
    USAGE,
    Providers,
    completion,
    delayed,
    drip,
    fail,
    hang,
    ok,
    raw,
    status,
)

DEADLINE = 0.05
PRIMARY = Provider("primary", "http://primary.test/v1/chat/completions", DEADLINE, api_key="sk-p")
PAYLOAD = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}


async def run(behaviour, provider: Provider = PRIMARY):
    providers = Providers()
    providers.script(provider.name, behaviour)
    async with httpx2.AsyncClient(transport=providers.transport()) as client:
        result = await attempt(client, provider, PAYLOAD)
    return result, providers


async def test_a_good_answer_is_a_rebuilt_completion_with_the_providers_usage() -> None:
    result, providers = await run(ok(content="Hi there", extra_field="dropped"))
    assert result.outcome is Outcome.SUCCESS
    assert result.completion is not None
    assert result.completion.total_tokens == 17
    assert result.completion.body == {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_800_000_000,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop",
            }
        ],  # fmt: skip
        "usage": USAGE,
    }
    (request,) = providers.requests
    assert request.headers["authorization"] == "Bearer sk-p"
    assert request.headers["user-agent"] == USER_AGENT
    assert request.headers["accept-encoding"] == "identity"
    assert json.loads(request.content) == PAYLOAD


async def test_a_provider_model_override_is_sent_instead_of_the_clients() -> None:
    provider = Provider("primary", PRIMARY.url, DEADLINE, model="provider-model")
    _, providers = await run(ok(), provider)
    assert json.loads(providers.requests[0].content)["model"] == "provider-model"
    assert "authorization" not in providers.requests[0].headers


@pytest.mark.parametrize(
    ("behaviour", "outcome"),
    [
        (status(429), Outcome.RATE_LIMITED),
        (status(500), Outcome.SERVER_ERROR),
        (status(502), Outcome.SERVER_ERROR),
        (status(503), Outcome.SERVER_ERROR),
        (status(400), Outcome.REJECTED),
        (status(413), Outcome.REJECTED),
        (status(422), Outcome.REJECTED),
        (status(401), Outcome.MISCONFIGURED),
        (status(403), Outcome.MISCONFIGURED),
        (status(404), Outcome.MISCONFIGURED),
        (status(402), Outcome.MISCONFIGURED),
        (raw(204, b""), Outcome.BROKEN),
        (raw(302, b""), Outcome.BROKEN),
        (raw(200, b"{not json"), Outcome.BROKEN),
        (fail(httpx2.ConnectError("refused")), Outcome.UNREACHABLE),
        (fail(httpx2.ConnectTimeout("slow")), Outcome.UNREACHABLE),
        (fail(httpx2.PoolTimeout("full")), Outcome.UNREACHABLE),
        (fail(httpx2.RemoteProtocolError("reset")), Outcome.BROKEN),
        (fail(httpx2.ReadError("reset")), Outcome.BROKEN),
    ],
)
async def test_outcomes_are_classified(behaviour, outcome: Outcome) -> None:
    result, _ = await run(behaviour)
    assert result.outcome is outcome
    assert result.completion is None


async def test_errors_carry_the_status_and_never_the_body() -> None:
    result, _ = await run(status(500, {"error": {"message": "prompt was: my ssn 123-45-6789"}}))
    assert (result.status, result.detail) == (500, "HTTP 500")


@pytest.mark.parametrize(
    "behaviour",
    [hang(), delayed(DEADLINE * 4, ok()), drip(DEADLINE / 4)],
    ids=["hang", "slow-headers", "slow-drip-body"],
)
async def test_the_deadline_covers_headers_and_the_whole_body(behaviour) -> None:
    started = time.monotonic()
    result, _ = await run(behaviour)
    assert result.outcome is Outcome.TIMEOUT
    assert time.monotonic() - started < DEADLINE + 0.5


async def test_a_fast_drip_within_the_deadline_succeeds() -> None:
    provider = Provider("primary", PRIMARY.url, 2.0)
    result, _ = await run(drip(0.005), provider)
    assert result.outcome is Outcome.SUCCESS


async def test_an_oversized_response_is_broken() -> None:
    result, _ = await run(
        raw(200, b"x" * (MAX_RESPONSE_BYTES + 1)), Provider("primary", PRIMARY.url, 5)
    )
    assert (result.outcome, result.detail) == (
        Outcome.BROKEN,
        f"response over {MAX_RESPONSE_BYTES} bytes",
    )


# --- validating a completion ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        [completion()],
        {"choices": []},
        completion(choices=[]),
        completion(choices=[completion()["choices"][0]] * 2),
        completion(choices=[{"index": 0, "message": "Hi", "finish_reason": "stop"}]),
        completion(choices=[{"index": 0, "message": {"role": "user", "content": "x"},
                             "finish_reason": "stop"}]),
        completion(choices=[{"index": 0, "message": {"role": "assistant", "content": 5},
                             "finish_reason": "stop"}]),
        completion(choices=[{"index": 0, "message": {"role": "assistant", "content": None},
                             "finish_reason": "stop"}]),
        completion(choices=[{"index": 0, "message": {"role": "assistant", "content": "x"},
                             "finish_reason": "tool_calls"}]),
    ],
    ids=["array", "no-choices", "empty-choices", "two-choices", "message-not-object",
         "wrong-role", "content-not-text", "no-content", "tool-calls"],
)  # fmt: skip
def test_invalid_completions_are_refused(body: object) -> None:
    assert parse_completion(json.dumps(body).encode(), "m") is None


def test_nan_is_not_json() -> None:
    assert parse_completion(b'{"choices": [], "created": NaN}', "m") is None


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"prompt_tokens": 12, "completion_tokens": 5},
        {**USAGE, "total_tokens": -1},
        {**USAGE, "total_tokens": True},
        {**USAGE, "total_tokens": 17.0},
        {**USAGE, "total_tokens": 10**9},
        {**USAGE, "total_tokens": 16},
    ],
    ids=["missing", "no-total", "negative", "bool", "float", "huge", "less-than-parts"],
)
def test_unbelievable_usage_is_treated_as_missing(usage: object) -> None:
    parsed = parse_completion(json.dumps(completion(usage=usage)).encode(), "m")
    assert parsed is not None
    assert parsed.total_tokens is None
    assert "usage" not in parsed.body


def test_provider_specific_finish_reasons_are_mapped_and_ids_are_checked() -> None:
    body = completion(id="x" * 200, model="", created="soon")
    body["choices"][0]["finish_reason"] = "eos"
    parsed = parse_completion(json.dumps(body).encode(), "requested-model")
    assert parsed is not None
    assert parsed.body["choices"][0]["finish_reason"] == "stop"
    assert str(parsed.body["id"]).startswith("chatcmpl-") and len(str(parsed.body["id"])) < 50
    assert parsed.body["model"] == "requested-model"
    assert isinstance(parsed.body["created"], int)


def test_a_refusal_is_kept() -> None:
    body = completion()
    body["choices"][0]["message"] = {"role": "assistant", "content": None, "refusal": "No."}
    parsed = parse_completion(json.dumps(body).encode(), "m")
    assert parsed is not None
    assert parsed.body["choices"][0]["message"] == {
        "role": "assistant",
        "content": None,
        "refusal": "No.",
    }

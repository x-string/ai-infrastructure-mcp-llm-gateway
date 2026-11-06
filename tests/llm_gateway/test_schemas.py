"""Task 3: the request allowlist: only requests whose output is redactable text get through."""

from __future__ import annotations

import pytest

from ai_gateway.llm_gateway.schemas import REFUSED_PARAMETERS, RequestError, parse_request

BASE = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}


def reject(payload: object) -> RequestError:
    with pytest.raises(RequestError) as excinfo:
        parse_request(payload)
    return excinfo.value


def test_a_minimal_request_streams_and_is_rebuilt_for_the_provider() -> None:
    request = parse_request(BASE)
    assert request.stream is True
    assert request.upstream_payload() == {**BASE, "stream": True}


def test_streaming_cannot_be_turned_off() -> None:
    error = reject({**BASE, "stream": False})
    assert error.param is None
    assert "stream must be true" in error.message


def test_usage_is_requested_from_the_provider_only_when_the_client_asks() -> None:
    assert "stream_options" not in parse_request(BASE).upstream_payload()
    request = parse_request({**BASE, "stream_options": {"include_usage": True}})
    assert request.include_usage
    assert request.upstream_payload()["stream_options"] == {"include_usage": True}


def test_generation_parameters_are_passed_on() -> None:
    options = {
        "max_tokens": 50,
        "max_completion_tokens": 60,
        "temperature": 1,
        "top_p": 0.5,
        "presence_penalty": -1.5,
        "frequency_penalty": 2,
        "stop": ["\n\n", "END"],
        "seed": 7,
        "user": "user-1",
        "n": 1,
        "response_format": {"type": "json_object"},
        "reasoning_effort": "low",
        "service_tier": "default",
    }
    assert parse_request({**BASE, **options}).upstream_payload() == {
        **BASE,
        **options,
        "stream": True,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("tools", [{"type": "function", "function": {"name": "f"}}]),
        ("tool_choice", "auto"),
        ("parallel_tool_calls", True),
        ("functions", [{"name": "f"}]),
        ("function_call", "auto"),
        ("logprobs", True),
        ("top_logprobs", 3),
        ("audio", {"voice": "alloy", "format": "wav"}),
        ("modalities", ["text", "audio"]),
        ("prediction", {"type": "content", "content": "x"}),
        ("web_search_options", {}),
    ],
)
def test_parameters_that_open_another_output_channel_are_refused(name: str, value: object) -> None:
    error = reject({**BASE, name: value})
    assert error.param == name
    assert error.message == f"{name}: {REFUSED_PARAMETERS[name]}"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("logprobs", False),
        ("tools", []),
        ("tool_choice", "none"),
        ("parallel_tool_calls", False),
        ("modalities", ["text"]),
        ("audio", None),
    ],
)
def test_values_that_ask_for_nothing_are_dropped(name: str, value: object) -> None:
    assert name not in parse_request({**BASE, name: value}).upstream_payload()


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({**BASE, "n": 2}, "n"),
        ({**BASE, "foo": 1}, "foo"),
        ({**BASE, "messages": []}, "messages"),
        ({**BASE, "messages": [{"role": "tool", "content": "x"}]}, "messages.0.role"),
        ({**BASE, "messages": [{"role": "user", "content": 5}]}, "messages.0.content"),
        ({**BASE, "messages": [{"role": "user"}]}, "messages.0.content"),
        ({**BASE, "messages": [{"role": "user", "content": "x", "tool_calls": []}]},
         "messages.0.tool_calls"),
        ({**BASE, "messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
         "messages.0.content"),
        ({**BASE, "temperature": "1"}, "temperature"),
        ({**BASE, "temperature": 3}, "temperature"),
        ({**BASE, "max_tokens": 0}, "max_tokens"),
        ({**BASE, "max_tokens": True}, "max_tokens"),
        ({**BASE, "stop": ["a", "b", "c", "d", "e"]}, "stop"),
        ({**BASE, "response_format": {"type": "json_schema"}}, "response_format.type"),
        ({"messages": BASE["messages"]}, "model"),
        ({**BASE, "model": "a b"}, "model"),
        ({**BASE, "stream": "yes"}, "stream"),
    ],
)  # fmt: skip
def test_invalid_requests_name_the_first_offending_field(payload: dict, param: str) -> None:
    error = reject(payload)
    assert error.param == param
    assert error.message.startswith(f"{param}: ")


def test_errors_never_echo_what_the_client_sent() -> None:
    error = reject({**BASE, "temperature": "john.doe@example.com"})
    assert "john" not in error.message
    error = reject({**BASE, "john.doe@example.com": 1})
    assert error.param == "<redacted>"
    assert "john" not in error.message


def test_the_body_must_be_a_json_object() -> None:
    for payload in [[], "x", 1, None]:
        assert reject(payload).message == "The request body must be a JSON object."

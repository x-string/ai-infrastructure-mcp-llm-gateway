"""Task 4: the router's HTTP contract, every error in the standard payload, nothing leaked."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from ai_gateway.model_router.app import create_app
from ai_gateway.model_router.providers import Provider
from ai_gateway.model_router.settings import RouterSettings
from ai_gateway.model_router.storage import connect
from ai_gateway.model_router.tenants import add_tenant
from tests.model_router.scripted import Providers, ok, status

URL = "/v1/chat/completions"
REQUEST = {"model": "gpt-test", "messages": [{"role": "user", "content": "Hello"}]}
REQUEST_ID = re.compile(r"req_[0-9a-f]{32}")


@dataclass
class Stack:
    client: TestClient
    providers: Providers
    key: str
    db_path: Path

    def post(self, body: object = None, *, key: str | None = None, **kwargs):
        headers = {"authorization": f"Bearer {key or self.key}", **kwargs.pop("headers", {})}
        if body is not None:
            kwargs["json"] = body
        return self.client.post(URL, headers=headers, **kwargs)

    def rows(self) -> list[tuple]:
        connection = sqlite3.connect(self.db_path)
        try:
            return connection.execute(
                "SELECT tenant_id, tokens, status FROM token_usage ORDER BY ts_ms"
            ).fetchall()
        finally:
            connection.close()


def settings_for(db_path: Path, **overrides: object) -> RouterSettings:
    providers = (
        Provider("primary", "http://primary.test/v1/chat/completions", 0.2),
        Provider("secondary", "http://secondary.test/v1/chat/completions", 1.0),
    )
    return RouterSettings(db_path=str(db_path), providers=providers, **overrides)


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[Stack]:
    db_path = tmp_path / "t4.db"
    connection = connect(db_path)
    key = add_tenant(connection, "acme", 50_000)
    connection.close()
    providers = Providers()
    providers.script("primary", ok())
    providers.script("secondary", ok())
    app = create_app(settings_for(db_path), transport=providers.transport())
    with TestClient(app, raise_server_exceptions=False) as client:
        yield Stack(client, providers, key, db_path)


def error_of(response) -> dict:
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert set(error) == {"type", "code", "message", "request_id"}
    assert error["request_id"] == response.headers["x-request-id"]
    assert REQUEST_ID.fullmatch(error["request_id"])
    return error


# --- success --------------------------------------------------------------------------------------


def test_a_completion_comes_back_with_its_provider_and_rate_limit_headers(stack: Stack) -> None:
    response = stack.post(REQUEST)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello!"
    assert REQUEST_ID.fullmatch(response.headers["x-request-id"])
    assert response.headers["x-gateway-provider"] == "primary"
    assert response.headers["x-ratelimit-limit-tokens"] == "50000"
    assert response.headers["x-ratelimit-remaining-tokens"] == str(50_000 - 17)
    assert stack.rows() == [("acme", 17, "committed")]


def test_failover_is_visible_in_the_provider_header(stack: Stack) -> None:
    stack.providers.script("primary", status(429))
    response = stack.post(REQUEST)
    assert response.status_code == 200
    assert response.headers["x-gateway-provider"] == "secondary"


def test_the_enforced_cap_is_always_sent_and_nothing_else_of_the_client(stack: Stack) -> None:
    stack.post(REQUEST, headers={"x-request-id": "client-chosen", "cookie": "a=b"})
    stack.post({**REQUEST, "max_completion_tokens": 50, "temperature": 0.2})
    first, second = (json.loads(r.content) for r in stack.providers.requests)
    assert first == {**REQUEST, "max_tokens": 1024, "stream": False}
    assert second == {**REQUEST, "max_completion_tokens": 50, "temperature": 0.2, "stream": False}
    for request in stack.providers.requests:
        assert "authorization" not in request.headers  # the tenant key is never forwarded
        assert "cookie" not in request.headers and "x-request-id" not in request.headers


def test_the_request_id_is_the_servers_own(stack: Stack) -> None:
    response = stack.post(REQUEST, headers={"x-request-id": "client-chosen"})
    assert response.headers["x-request-id"] != "client-chosen"
    assert REQUEST_ID.fullmatch(response.headers["x-request-id"])


# --- authentication -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "code"),
    [
        ({}, "missing_api_key"),
        ({"authorization": "Basic YWxpY2U6c2VjcmV0"}, "missing_api_key"),
        ({"authorization": "Bearer"}, "missing_api_key"),
        ({"authorization": "Bearer sk-t4-unknown"}, "invalid_api_key"),
        ({"authorization": "Bearer has spaces inside"}, "invalid_api_key"),
        ({"authorization": "Bearer " + "x" * 300}, "invalid_api_key"),
    ],
    ids=["missing", "basic", "empty", "unknown", "malformed", "oversized"],
)
def test_requests_without_a_valid_key_get_401(stack: Stack, headers: dict, code: str) -> None:
    response = stack.client.post(URL, json=REQUEST, headers=headers)
    assert response.status_code == 401
    assert error_of(response)["code"] == code
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert stack.providers.requests == []


def test_two_authorization_headers_are_refused(stack: Stack) -> None:
    response = stack.client.post(
        URL,
        content=json.dumps(REQUEST),
        headers=[("authorization", f"Bearer {stack.key}"), ("authorization", f"Bearer {stack.key}"),
                 ("content-type", "application/json")],
    )  # fmt: skip
    assert response.status_code == 401


def test_the_scheme_is_case_insensitive(stack: Stack) -> None:
    response = stack.client.post(
        URL, json=REQUEST, headers={"authorization": f"bearer {stack.key}"}
    )
    assert response.status_code == 200


# --- the request ----------------------------------------------------------------------------------


def test_the_body_must_be_json(stack: Stack) -> None:
    response = stack.post(content=b"model=x", headers={"content-type": "text/plain"})
    assert (response.status_code, error_of(response)["code"]) == (415, "unsupported_media_type")


def test_bodies_over_the_limit_get_413(stack: Stack) -> None:
    big = {**REQUEST, "messages": [{"role": "user", "content": "x" * (1 << 20)}]}
    response = stack.post(big)
    assert (response.status_code, error_of(response)["code"]) == (413, "body_too_large")

    def chunked():  # no Content-Length: the limit applies while reading
        yield b'{"model": "m", "messages": [{"role": "user", "content": "'
        for _ in range(20):
            yield b"x" * 65536
        yield b'"}]}'

    response = stack.post(content=chunked(), headers={"content-type": "application/json"})
    assert response.status_code == 413


@pytest.mark.parametrize(
    "content",
    [b"{", b'{"model": "m", "model": "n", "messages": []}', b'{"model": NaN}', b"\xff\xfe",
     b"[" * 100_000],
    ids=["truncated", "duplicate-keys", "nan", "not-utf8", "deep"],
)  # fmt: skip
def test_invalid_json_gets_400(stack: Stack, content: bytes) -> None:
    response = stack.post(content=content, headers={"content-type": "application/json"})
    assert (response.status_code, error_of(response)["code"]) == (400, "invalid_json")


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({**REQUEST, "stream": True}, "stream (literal_error)"),
        ({**REQUEST, "n": 2}, "n (literal_error)"),
        ({**REQUEST, "tools": [{"type": "function"}]}, "tools (extra_forbidden)"),
        ({**REQUEST, "messages": []}, "messages (too_short)"),
        ({**REQUEST, "messages": [{"role": "tool", "content": "x"}]}, "messages.0.role"),
        ({**REQUEST, "messages": [{"role": "user", "content": [{"type": "text"}]}]},
         "messages.0.content (string_type)"),
        ({**REQUEST, "max_tokens": 0}, "max_tokens (greater_than_equal)"),
        ({**REQUEST, "max_tokens": "100"}, "max_tokens (int_type)"),
        ({**REQUEST, "temperature": 3}, "temperature (less_than_equal)"),
        ({"messages": REQUEST["messages"]}, "model (missing)"),
    ],
    ids=["stream", "n", "tools", "no-messages", "tool-role", "image-content", "zero-cap",
         "string-cap", "temperature", "no-model"],
)  # fmt: skip
def test_invalid_requests_name_the_field_and_never_echo_input(
    stack: Stack, body: dict, fragment: str
) -> None:
    response = stack.post(body)
    error = error_of(response)
    assert (response.status_code, error["code"]) == (400, "invalid_request")
    assert fragment in error["message"]
    assert stack.providers.requests == []


def test_validation_errors_never_echo_values_or_unsafe_field_names(stack: Stack) -> None:
    body = {**REQUEST, "max_tokens": "my ssn is 123-45-6789", "john@example.com": "x"}
    message = error_of(stack.post(body))["message"]
    assert "123-45-6789" not in message and "john" not in message
    assert "<redacted> (extra_forbidden)" in message


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({**REQUEST, "max_tokens": 10, "max_completion_tokens": 10}, "conflicting_max_tokens"),
        ({**REQUEST, "max_tokens": 4097}, "max_tokens_too_large"),
    ],
)
def test_completion_caps_are_checked(stack: Stack, body: dict, code: str) -> None:
    response = stack.post(body)
    assert (response.status_code, error_of(response)["code"]) == (400, code)


# --- rate limiting --------------------------------------------------------------------------------


def test_over_the_limit_the_tenant_gets_429_with_retry_after(tmp_path: Path) -> None:
    db_path = tmp_path / "t4.db"
    connection = connect(db_path)
    key = add_tenant(connection, "small", 3_000)
    connection.close()
    providers = Providers()
    usage = {"prompt_tokens": 400, "completion_tokens": 600, "total_tokens": 1_000}
    providers.script("primary", ok(usage=usage))
    app = create_app(settings_for(db_path), transport=providers.transport())
    body = {**REQUEST, "max_tokens": 1_000}
    with TestClient(app) as client:
        headers = {"authorization": f"Bearer {key}"}
        assert [client.post(URL, json=body, headers=headers).status_code for _ in range(2)] == [
            200,
            200,
        ]
        denied = client.post(URL, json=body, headers=headers)
    assert denied.status_code == 429
    error = error_of(denied)
    assert (error["type"], error["code"]) == ("rate_limit_error", "tenant_token_limit_exceeded")
    retry_after = int(denied.headers["retry-after"])
    assert 1 <= retry_after <= 60
    assert f"retry after {retry_after} s" in error["message"]
    assert denied.headers["x-ratelimit-limit-tokens"] == "3000"
    assert denied.headers["x-ratelimit-remaining-tokens"] == "1000"
    assert providers.calls("primary") == 2

    # The window lives in SQLite: a restarted router still enforces it.
    restarted = create_app(settings_for(db_path), transport=providers.transport())
    with TestClient(restarted) as client:
        assert (
            client.post(URL, json=body, headers={"authorization": f"Bearer {key}"}).status_code
            == 429
        )


def test_a_request_larger_than_the_whole_budget_gets_400(tmp_path: Path) -> None:
    db_path = tmp_path / "t4.db"
    connection = connect(db_path)
    key = add_tenant(connection, "tiny", 500)
    connection.close()
    app = create_app(settings_for(db_path), transport=Providers().transport())
    with TestClient(app) as client:
        response = client.post(URL, json=REQUEST, headers={"authorization": f"Bearer {key}"})
    assert (response.status_code, error_of(response)["code"]) == (400, "request_too_large")


# --- failures -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("primary", "secondary", "status_code", "code"),
    [
        (status(500), status(500), 502, "upstream_error"),
        (status(429), status(429), 503, "upstream_unavailable"),
        (status(400), ok(), 400, "upstream_rejected_request"),
        (status(401), ok(), 502, "upstream_configuration_error"),
    ],
)
def test_upstream_failures_never_pass_on_the_upstream_body(
    stack: Stack, primary, secondary, status_code: int, code: str
) -> None:
    secret = {"error": {"message": "prompt: my ssn is 123-45-6789; key sk-live-123"}}
    stack.providers.script("primary", status(primary(None).status_code, secret))
    stack.providers.script("secondary", secondary)
    response = stack.post(REQUEST)
    assert (response.status_code, error_of(response)["code"]) == (status_code, code)
    assert "123-45-6789" not in response.text and "sk-live" not in response.text


def test_an_unexpected_exception_is_a_generic_500(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explode(**_kwargs: object) -> None:
        raise RuntimeError("secret internal detail: my ssn is 123-45-6789")

    monkeypatch.setattr(stack.client.app.app.state.router, "complete", explode)
    response = stack.post(REQUEST)
    error = error_of(response)
    assert (response.status_code, error["type"], error["code"]) == (
        500,
        "server_error",
        "internal_error",
    )
    assert "secret" not in response.text and "Traceback" not in response.text


def test_a_broken_database_fails_closed_with_503(stack: Stack) -> None:
    connection = sqlite3.connect(stack.db_path)
    connection.execute("DROP TABLE token_usage")
    connection.close()
    response = stack.post(REQUEST)
    assert (response.status_code, error_of(response)["code"]) == (503, "limiter_unavailable")
    assert stack.providers.requests == []


@pytest.mark.parametrize(
    ("method", "path", "status_code", "code"),
    [
        ("GET", URL, 405, "method_not_allowed"),
        ("POST", "/v1/completions", 404, "not_found"),
        ("GET", "/nothing", 404, "not_found"),
    ],
)
def test_routing_errors_use_the_standard_payload(
    stack: Stack, method: str, path: str, status_code: int, code: str
) -> None:
    response = stack.client.request(method, path, headers={"authorization": f"Bearer {stack.key}"})
    assert (response.status_code, error_of(response)["code"]) == (status_code, code)


def test_pending_settlements_are_written_before_shutdown(tmp_path: Path) -> None:
    db_path = tmp_path / "t4.db"
    connection = connect(db_path)
    key = add_tenant(connection, "acme", 50_000)
    connection.close()
    providers = Providers()
    providers.script("primary", ok())
    app = create_app(settings_for(db_path), transport=providers.transport())
    with TestClient(app) as client:
        for _ in range(20):
            assert (
                client.post(
                    URL, json=REQUEST, headers={"authorization": f"Bearer {key}"}
                ).status_code
                == 200
            )
    connection = sqlite3.connect(db_path)
    statuses = connection.execute("SELECT status, COUNT(*) FROM token_usage GROUP BY 1").fetchall()
    connection.close()
    assert statuses == [("committed", 20)]


def test_health_needs_no_key(stack: Stack) -> None:
    assert stack.client.get("/healthz").json() == {"status": "ok"}

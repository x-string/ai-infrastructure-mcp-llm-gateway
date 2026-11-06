"""Task 3: normalizing upstream chat-completion chunks, failing closed on anything unexpected."""

from __future__ import annotations

import json

import pytest

from ai_gateway.llm_gateway.events import (
    KEEP_ALIVE,
    Channel,
    ProtocolViolation,
    ReportedError,
    StreamEnd,
    StreamError,
    StreamNormalizer,
    TextDelta,
    Usage,
)
from ai_gateway.llm_gateway.sse import Comment, Decoder, ServerSentEvent
from tests.llm_gateway.provider_streams import STREAMS, ProviderStream

CONTENT, REFUSAL = Channel.CONTENT, Channel.REFUSAL
USAGE = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def event(payload: object) -> ServerSentEvent:
    return ServerSentEvent(json.dumps(payload))


def chunk(*choices: dict, **top: object) -> ServerSentEvent:
    return event({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                  "choices": list(choices), **top})  # fmt: skip


def choice(delta: object = None, finish: object = None, **extra: object) -> dict:
    result = {"index": 0, "finish_reason": finish, **extra}
    if delta is not None:
        result["delta"] = delta
    return result


FINISH = chunk(choice({}, "stop"))


def texts(*events: ServerSentEvent) -> list[TextDelta]:
    normalizer = StreamNormalizer()
    out = [item for e in events for item in normalizer.accept(e)]
    return [item for item in out if isinstance(item, TextDelta)]


def end(*events: ServerSentEvent) -> StreamEnd:
    normalizer = StreamNormalizer()
    for e in events:
        normalizer.accept(e)
    return normalizer.end()


def failure(*events: ServerSentEvent) -> StreamError:
    normalizer = StreamNormalizer()
    with pytest.raises(StreamError) as excinfo:
        for e in events:
            normalizer.accept(e)
        normalizer.end()
    return excinfo.value


# --- text channels --------------------------------------------------------------------------------


def test_content_and_refusal_are_the_text_channels() -> None:
    assert texts(
        chunk(choice({"role": "assistant", "content": "", "refusal": None})),
        chunk(choice({"content": "Hello"})),
        chunk(choice({"refusal": "No."})),
        chunk(choice({"content": "a", "refusal": "b"})),
    ) == [
        TextDelta(CONTENT, "Hello"),
        TextDelta(REFUSAL, "No."),
        TextDelta(CONTENT, "a"),
        TextDelta(REFUSAL, "b"),
    ]


def test_content_parts_relay_text_and_drop_thinking_and_references() -> None:
    parts = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "secret 123-45-6789"}]},
        {"type": "text", "text": "See "},
        {"type": "text", "text": ""},
        {"type": "reference", "reference_ids": [1]},
        {"type": "text", "text": "[1]"},
    ]
    assert texts(chunk(choice({"content": parts}))) == [
        TextDelta(CONTENT, "See "),
        TextDelta(CONTENT, "[1]"),
    ]


@pytest.mark.parametrize(
    ("delta", "detail"),
    [
        ({"content": 123}, "content is not a string or a list"),
        ({"content": {"text": "x"}}, "content is not a string or a list"),
        ({"content": ["x"]}, "content part is not an object"),
        ({"content": [{"type": "image_url", "image_url": "data:"}]}, "unsupported content part"),
        ({"content": [{"type": "text", "text": 5}]}, "content part is not a string"),
        ({"content": [{"type": "text"}]}, "content part is not a string"),
        ({"refusal": ["no"]}, "refusal is not a string"),
        ({"content": "\ud800"}, "content is not valid Unicode"),
    ],
)
def test_malformed_text_is_refused(delta: dict, detail: str) -> None:
    error = failure(chunk(choice(delta)), FINISH)
    assert isinstance(error, ProtocolViolation)
    assert detail in error.detail


@pytest.mark.parametrize(
    "key",
    ["reasoning", "reasoning_content", "reasoning_details", "thinking_blocks", "reasoning_items"],
)
def test_reasoning_fields_are_dropped(key: str) -> None:
    delta = {key: "the user's email is john@example.com", "content": "Hi"}
    assert texts(chunk(choice(delta))) == [TextDelta(CONTENT, "Hi")]


@pytest.mark.parametrize("key", ["tool_calls", "function_call", "audio", "images"])
def test_channels_that_cannot_be_redacted_as_text_are_refused(key: str) -> None:
    error = failure(chunk(choice({key: [{"arguments": '{"e": "a\\u0040b.com"}'}]})), FINISH)
    assert isinstance(error, ProtocolViolation)
    assert error.detail == f"unsupported output channel {key}"
    for empty in (None, [], {}, ""):  # present but empty is harmless
        assert end(chunk(choice({key: empty, "content": "x"})), FINISH).finish_reason == "stop"


@pytest.mark.parametrize(
    "key",
    ["role", "annotations", "citations", "executed_tools", "index", "internal_content", "metadata",
     "provider_specific_fields", "token_id", "tool_call_id"],
)  # fmt: skip
def test_known_metadata_is_ignored(key: str) -> None:
    assert texts(chunk(choice({key: {"url": "https://e.x"}, "content": "Hi"}))) == [
        TextDelta(CONTENT, "Hi")
    ]


def test_an_unknown_delta_field_with_a_value_is_refused() -> None:
    error = failure(chunk(choice({"summary_text": "john@example.com"})), FINISH)
    assert isinstance(error, ProtocolViolation)
    assert error.detail == "unknown delta field 'summary_text'"
    assert texts(chunk(choice({"summary_text": None, "content": "x"}))) == [TextDelta(CONTENT, "x")]


def test_names_from_the_upstream_are_only_logged_when_they_cannot_carry_data() -> None:
    error = failure(chunk(choice({"john.doe@example.com": "x"})), FINISH)
    assert error.detail == "unknown delta field <redacted>"
    assert "john" not in str(error)


# --- chunk structure ------------------------------------------------------------------------------


def test_chunks_without_choices_carry_nothing_to_relay() -> None:
    for payload in [{"choices": []}, {"choices": None}, {"id": "x"}, {"usage": None}]:
        assert texts(event(payload)) == []


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"choices": {"index": 0}}, "choices is not a list"),
        ({"choices": ["x"]}, "choice is not an object"),
        ({"choices": [{"index": 1, "delta": {"content": "B"}}]}, "unexpected choice index"),
        ({"choices": [{"index": True, "delta": {}}]}, "unexpected choice index"),
        ({"choices": [{"index": "0", "delta": {}}]}, "unexpected choice index"),
        ({"choices": [{"index": 0, "delta": "Hi"}]}, "delta is not an object"),
        ({"object": "text_completion", "choices": []}, "unexpected object 'text_completion'"),
        ({"object": "chat.completion", "choices": []}, "unexpected object 'chat.completion'"),
    ],
)
def test_unexpected_structure_is_refused(payload: dict, detail: str) -> None:
    error = failure(event(payload), FINISH)
    assert isinstance(error, ProtocolViolation)
    assert error.detail == detail


def test_a_missing_index_or_delta_is_accepted() -> None:
    assert texts(event({"choices": [{"delta": {"content": "x"}}]})) == [TextDelta(CONTENT, "x")]
    assert end(event({"choices": [{"index": 0, "finish_reason": "stop"}]})).finish_reason == "stop"


@pytest.mark.parametrize("obj", [None, "", "chat.completion.chunk"])
def test_object_may_be_absent_empty_or_the_chunk_type(obj: str | None) -> None:
    assert end(event({"object": obj, "choices": [choice({}, "stop")]})).finish_reason == "stop"


@pytest.mark.parametrize(
    "data",
    ["{", "[1]", '"text"', "12", "null", "NaN", '{"a": Infinity}', "1" * 5000, "[" * 100_000],
)
def test_data_that_is_not_a_json_object_is_refused(data: str) -> None:
    error = failure(ServerSentEvent(data))
    assert isinstance(error, ProtocolViolation)
    assert error.detail in ("invalid JSON", "data is not a JSON object")


# --- errors reported by the upstream --------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"message": "boom", "type": "server_error"}},
        {"error": "Input validation error", "error_type": "validation"},
        {"error": {}},
        {"error": False},
        {"object": "error", "message": "Internal server error"},
        {"type": "error", "error": {"type": "overloaded_error"}},
        {"choices": [], "x_groq": {"error": "stream stopped"}},
        {"choices": [choice({"content": ""}, "error")], "error": {"code": "server_error"}},
    ],
)
def test_error_objects_mean_the_generation_failed(payload: dict) -> None:
    error = failure(chunk(choice({"content": "Hi"})), event(payload))
    assert isinstance(error, ReportedError)
    assert "boom" not in error.detail and "validation" not in error.detail


def test_null_error_fields_are_not_errors() -> None:
    assert end(event({"error": None, "x_groq": {"error": None}, "choices": [choice({}, "stop")]}))


def test_error_events_and_unknown_event_types() -> None:
    assert isinstance(failure(ServerSentEvent("{}", "error")), ReportedError)
    for name in ["content_block_delta", "ping", "message_start"]:
        error = failure(ServerSentEvent("{}", name))
        assert isinstance(error, ProtocolViolation)
        assert error.detail == f"unexpected event type {name!r}"
    assert failure(ServerSentEvent("{}", "a b@c")).detail == "unexpected event type <redacted>"


# --- keep-alives, [DONE] and the end of the stream ------------------------------------------------


def test_empty_events_are_keep_alives() -> None:
    normalizer = StreamNormalizer()
    for data in ["", " ", "\n"]:
        assert normalizer.accept(ServerSentEvent(data)) == [KEEP_ALIVE]


def test_done_ends_reading_but_does_not_prove_success() -> None:
    for done in ["[DONE]", " [DONE] "]:
        normalizer = StreamNormalizer()
        normalizer.accept(chunk(choice({"content": "Hi"})))
        assert normalizer.accept(ServerSentEvent(done)) == []
        assert normalizer.done
        with pytest.raises(ProtocolViolation, match="stream ended before a finish reason"):
            normalizer.end()
        with pytest.raises(ProtocolViolation, match="event after"):
            normalizer.accept(FINISH)


def test_a_stream_succeeds_once_a_finish_reason_arrived() -> None:
    assert end(chunk(choice({"content": "Hi"})), FINISH) == StreamEnd("stop", None)
    assert isinstance(failure(chunk(choice({"content": "Hi"}))), ProtocolViolation)


# --- finish reasons -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("upstream", "relayed"),
    [
        ("stop", "stop"),
        ("eos", "stop"),
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("repetition", "stop"),
        ("length", "length"),
        ("max_tokens", "length"),
        ("model_length", "length"),
        ("content_filter", "content_filter"),
    ],
)
def test_finish_reasons_map_onto_openais(upstream: str, relayed: str) -> None:
    assert end(chunk(choice({}, upstream))).finish_reason == relayed


@pytest.mark.parametrize("reason", ["error", "abort", "aborted", "insufficient_system_resource"])
def test_failure_finish_reasons_mean_the_generation_failed(reason: str) -> None:
    error = failure(chunk(choice({}, reason)))
    assert isinstance(error, ReportedError)
    assert error.detail == f"finish reason {reason}"


@pytest.mark.parametrize(
    ("reason", "detail"),
    [
        ("tool_calls", "finish reason tool_calls without tools"),
        ("function_call", "finish reason function_call without tools"),
        ("weird", "unknown finish reason 'weird'"),
        (1, "finish reason is not a string"),
        (["stop"], "finish reason is not a string"),
    ],
)
def test_unexpected_finish_reasons_are_refused(reason: object, detail: str) -> None:
    error = failure(chunk(choice({}, reason)))
    assert isinstance(error, ProtocolViolation)
    assert error.detail == detail


def test_a_later_finish_reason_can_upgrade_but_never_downgrade() -> None:
    assert end(FINISH, FINISH).finish_reason == "stop"  # OpenRouter repeats it with the usage
    assert end(FINISH, chunk(choice(None, "content_filter"))).finish_reason == "content_filter"
    assert end(chunk(choice({}, "content_filter")), FINISH).finish_reason == "content_filter"
    assert end(chunk(choice({}, "length")), FINISH).finish_reason == "length"


def test_text_after_a_finish_reason_is_refused() -> None:
    error = failure(FINISH, chunk(choice({"content": " more"})))
    assert error.detail == "text after a finish reason"
    # Text and the finish reason in the same chunk, and empty text afterwards, are fine.
    assert end(chunk(choice({"content": "last"}, "stop")), chunk(choice({"content": ""}, "stop")))


# --- usage ----------------------------------------------------------------------------------------


def test_usage_is_taken_from_wherever_providers_put_it() -> None:
    usage = Usage(5, 3, 8)
    assert end(FINISH, chunk(usage=USAGE)).usage == usage  # OpenAI: a final chunk
    assert end(chunk(choice({}, "stop"), x_groq={"usage": USAGE})).usage == usage  # Groq
    assert end(chunk(choice({}, "stop", usage=USAGE))).usage == usage  # per choice
    assert end(FINISH).usage is None


def test_the_latest_valid_usage_wins_and_invalid_counts_are_ignored() -> None:
    later = {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
    assert end(chunk(usage=USAGE), FINISH, chunk(usage=later)).usage == Usage(5, 4, 9)
    for invalid in [
        None,
        {"prompt_tokens": 5, "completion_tokens": 3},
        {**USAGE, "total_tokens": -1},
        {**USAGE, "total_tokens": True},
        {**USAGE, "total_tokens": 8.0},
        {**USAGE, "total_tokens": "8"},
        {**USAGE, "total_tokens": 10**9},
    ]:
        assert end(chunk(usage=USAGE), FINISH, chunk(usage=invalid)).usage == Usage(5, 3, 8)


# --- real provider streams ------------------------------------------------------------------------


def replay(stream: ProviderStream, piece: int) -> tuple[str, str, StreamEnd]:
    """Decode and normalize ``stream``, fed ``piece`` bytes at a time (0: all at once)."""
    decoder, normalizer = Decoder(), StreamNormalizer()
    body = stream.body
    pieces = [body] if piece == 0 else [body[i : i + piece] for i in range(0, len(body), piece)]
    out: list[TextDelta] = []
    for raw in pieces:
        for item in decoder.feed(raw):
            if isinstance(item, Comment) or normalizer.done:
                continue
            out += [x for x in normalizer.accept(item) if isinstance(x, TextDelta)]
    decoder.close()
    joined = {channel: "".join(d.text for d in out if d.channel is channel) for channel in Channel}
    return joined[CONTENT], joined[REFUSAL], normalizer.end()


@pytest.mark.parametrize("stream", STREAMS, ids=[s.name for s in STREAMS])
@pytest.mark.parametrize("piece", [0, 1, 7])
def test_provider_streams(stream: ProviderStream, piece: int) -> None:
    if stream.outcome == "success":
        content, refusal, result = replay(stream, piece)
        assert (content, refusal) == (stream.text, stream.refusal)
        assert result.finish_reason == stream.finish
        assert (result.usage is not None) is stream.usage
    else:
        expected = ReportedError if stream.outcome == "reported_error" else ProtocolViolation
        with pytest.raises(expected):
            replay(stream, piece)


def test_the_corpus_covers_every_outcome() -> None:
    assert {s.outcome for s in STREAMS} == {"success", "reported_error", "protocol_error"}
    assert len({s.name for s in STREAMS}) == len(STREAMS)

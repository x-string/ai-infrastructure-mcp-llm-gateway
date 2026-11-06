"""Task 3: the SSE decoder for upstream streams, and the encoder for client streams."""

from __future__ import annotations

import json
import re
import tracemalloc

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ai_gateway.llm_gateway.sse import (
    DONE,
    MAX_EVENT_BYTES,
    MAX_LINE_BYTES,
    Comment,
    DecodeError,
    Decoder,
    encode_comment,
    encode_json,
)
from ai_gateway.llm_gateway.sse import ServerSentEvent as E


def decode(*chunks: bytes, **limits: int) -> list[E | Comment]:
    decoder = Decoder(**limits)
    out: list[E | Comment] = []
    for chunk in chunks:
        out += decoder.feed(chunk)
    decoder.close()
    return out


def splits(raw: bytes) -> list[list[bytes]]:
    """The stream whole, cut once at every offset, and one byte at a time."""
    return (
        [[raw]]
        + [[raw[:cut], raw[cut:]] for cut in range(1, len(raw))]
        + [[raw[i : i + 1] for i in range(len(raw))]]
    )


# --- the WHATWG parsing rules ---------------------------------------------------------------------

CASES = [
    ("lf", b"data: a\ndata: b\n\n", [E("a\nb")]),
    ("crlf", b"data: a\r\ndata: b\r\n\r\n", [E("a\nb")]),
    ("cr", b"data: a\rdata: b\r\r", [E("a\nb")]),
    ("mixed-line-ends", b"data: a\r\ndata: b\rdata: c\n\n", [E("a\nb\nc")]),
    ("no-space-after-colon", b"data:a\n\n", [E("a")]),
    ("only-one-space-removed", b"data:  a\n\n", [E(" a")]),
    ("tab-is-kept", b"data:\ta\n\n", [E("\ta")]),
    ("field-without-colon", b"data\n\n", [E("")]),
    ("empty-data", b"data:\n\n", [E("")]),
    ("multi-line-data", b"data: a\ndata:\ndata:  b\ndata\n\n", [E("a\n\n b\n")]),
    ("event-type", b"event: delta\ndata: x\n\n", [E("x", "delta")]),
    ("empty-event-type", b"event:\ndata: x\n\n", [E("x")]),
    ("type-resets", b"event: a\ndata: 1\n\ndata: 2\n\n", [E("1", "a"), E("2")]),
    ("type-resets-without-data", b"event: a\n\ndata: 2\n\n", [E("2")]),
    ("blocks-without-data", b"event: foo\n\nid: 9\n\nretry: 10\n\nfoo: bar\n\n", []),
    ("id-and-retry-ignored", b"id: 1\nretry: 5\ndata: x\n\n", [E("x")]),
    ("names-are-case-sensitive", b"DATA: x\nData: y\n\n", []),
    ("space-before-colon", b"data : x\n\n", []),
    ("comments", b": ping\n:\n:  two\n", [Comment("ping"), Comment(""), Comment(" two")]),
    ("comment-inside-event", b"data: a\n: hi\ndata: b\n\n", [Comment("hi"), E("a\nb")]),
    ("bom", b"\xef\xbb\xbfdata: a\n\n", [E("a")]),
    ("only-one-bom", b"\xef\xbb\xbf\xef\xbb\xbfdata: a\n\n", []),
    ("later-bom-is-not-stripped", b"data: a\n\n\xef\xbb\xbfdata: b\n\n", [E("a")]),
    (
        "unicode-separators-are-not-line-ends",
        "data: a\u2028b\u2029c\x85d\x0be\x0cf\x1cg\x1dh\x1ei\n\n".encode(),
        [E("a\u2028b\u2029c\x85d\x0be\x0cf\x1cg\x1dh\x1ei")],
    ),
    ("nul", b"data: a\x00b\n\n", [E("a\x00b")]),
    ("multibyte", "data: café € 🙂\n\n".encode(), [E("café € 🙂")]),
    ("done-is-just-data", b"data: [DONE]\n\n", [E("[DONE]")]),
    ("unfinished-event-discarded", b"data: a\n\ndata: b\n", [E("a")]),
    ("unterminated-line-discarded", b"data: a\n\ndata: b", [E("a")]),
    ("cr-terminated-line-discarded", b"data: a\n\ndata: b\r", [E("a")]),
    ("cr-as-blank-line", b"data: b\n\r", [E("b")]),
]


@pytest.mark.parametrize(("raw", "expected"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_spec_rules_hold_however_the_stream_is_split(raw: bytes, expected: list) -> None:
    for chunks in splits(raw):
        assert decode(*chunks) == expected, chunks


def test_a_crlf_split_around_an_empty_chunk_is_one_line_end() -> None:
    assert decode(b"data: a\r", b"", b"\ndata: b\n\n") == [E("a\nb")]


def test_close_reports_whether_an_unfinished_event_was_dropped() -> None:
    for raw, unfinished in [
        (b"data: a\n\n", False),
        (b": comment\n", False),
        (b"data: a\n", True),
        (b"data: a", True),
        (b"event: x\n", True),
        (b": unterminated comment", True),
    ]:
        decoder = Decoder()
        decoder.feed(raw)
        assert decoder.close() is unfinished, raw


# --- invalid input and limits ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [b"data: a\xffb\n\n", b"data: \xc0\xaf\n\n", b"data: \xed\xa0\x80\n\n", b"data: \xe2\x82\n\n"],
    ids=["invalid-byte", "overlong", "surrogate", "truncated-sequence"],
)
def test_invalid_utf8_is_an_error_not_a_replacement_character(raw: bytes) -> None:
    for chunks in splits(raw):
        with pytest.raises(DecodeError, match="invalid UTF-8"):
            decode(*chunks)


def test_invalid_utf8_in_a_discarded_last_line_is_never_decoded() -> None:
    assert decode(b"data: ok\n\ndata: x\xe2\x82") == [E("ok")]


def test_lines_longer_than_the_limit_are_refused() -> None:
    limit = {"max_line_bytes": 16}
    assert decode(b"data: 1234567890\n\n", **limit) == [E("1234567890")]  # exactly 16 bytes
    for chunks in splits(b"data: 12345678901\n\n"):
        with pytest.raises(DecodeError, match="line too long"):
            decode(*chunks, **limit)
    with pytest.raises(DecodeError, match="line too long"):  # unterminated: refused as it grows
        decode(b"x" * 10, b"x" * 7, **limit)


def test_events_larger_than_the_limit_are_refused() -> None:
    limit = {"max_event_bytes": 20}
    assert decode(b"data: 1234567\ndata: 1\n\n", **limit) == [E("1234567\n1")]  # 13 + 7 bytes
    with pytest.raises(DecodeError, match="event too large"):
        decode(b"data: 1234567\ndata: 12\n\n", **limit)


def test_default_limits_are_one_mebibyte() -> None:
    assert MAX_LINE_BYTES == MAX_EVENT_BYTES == 1 << 20
    assert decode(b"data: " + b"x" * (MAX_LINE_BYTES - 6) + b"\n\n")[0].data == "x" * (2**20 - 6)
    with pytest.raises(DecodeError):
        decode(b"data: " + b"x" * (MAX_LINE_BYTES - 5) + b"\n\n")


def test_a_huge_chunk_without_a_line_end_is_refused_before_it_is_copied() -> None:
    chunk = b"x" * (8 << 20)
    decoder = Decoder()
    tracemalloc.start()
    try:
        with pytest.raises(DecodeError):
            decoder.feed(chunk)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1 << 20


# --- properties -----------------------------------------------------------------------------------

TEXT = st.text(st.characters(exclude_categories=("Cs",), exclude_characters="\r\n"), max_size=12)


@st.composite
def encoded_streams(draw: st.DrawFn) -> tuple[bytes, list[E | Comment]]:
    """A stream written with random line ends, comments and BOM, and what it must decode to."""
    newline = draw(st.sampled_from([b"\n", b"\r", b"\r\n"]))
    raw = b"\xef\xbb\xbf" if draw(st.booleans()) else b""
    expected: list[E | Comment] = []
    for _ in range(draw(st.integers(0, 6))):
        if draw(st.booleans()):
            comment = Comment(draw(TEXT))
            raw += b": " + comment.text.encode() + newline
            expected.append(comment)
            continue
        event = E(
            "\n".join(draw(st.lists(TEXT, min_size=1, max_size=3))),
            draw(st.sampled_from(["message", "delta", "error"])),
        )
        if event.event != "message" or draw(st.booleans()):
            raw += b"event: " + event.event.encode() + newline
        for line in event.data.split("\n"):
            raw += b"data: " + line.encode() + newline
        raw += newline
        expected.append(event)
    return raw, expected


@st.composite
def cut(draw: st.DrawFn, raw: bytes) -> list[bytes]:
    points = sorted(draw(st.sets(st.integers(0, len(raw)), max_size=8)))
    bounds = [0, *points, len(raw)]
    return [raw[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]


@settings(max_examples=400, deadline=None)
@given(st.data(), encoded_streams())
def test_round_trip_for_any_split(data: st.DataObject, case: tuple[bytes, list]) -> None:
    raw, expected = case
    assert decode(*data.draw(cut(raw))) == expected


def reference_decode(raw: bytes, max_line: int, max_event: int) -> list[E | Comment]:
    """The same rules applied to the whole stream at once, as a second implementation."""
    *lines, tail = re.split(rb"\r\n|\r|\n", raw)
    out: list[E | Comment] = []
    event, data, size = "", [], 0
    for index, raw_line in enumerate(lines):
        if len(raw_line) > max_line:
            raise DecodeError("line too long")
        try:
            line = raw_line.decode()
        except UnicodeDecodeError:
            raise DecodeError("invalid UTF-8") from None
        if index == 0 and line.startswith("\ufeff"):
            line = line[1:]
        if line == "":
            if data:
                out.append(E("\n".join(data), event or "message"))
            event, data, size = "", [], 0
        elif line.startswith(":"):
            out.append(Comment(line[2:] if line.startswith(": ") else line[1:]))
        else:
            name, colon, value = line.partition(":")
            if colon and value.startswith(" "):
                value = value[1:]
            if name == "data":
                size += len(raw_line)
                if size > max_event:
                    raise DecodeError("event too large")
                data.append(value)
            elif name == "event":
                event = value
    if len(tail) > max_line:
        raise DecodeError("line too long")
    return out


PIECES = st.sampled_from(
    [b"data", b"event", b"id", b":", b" ", b"\r", b"\n", b"\r\n", b"x", b"error", b"[DONE]",
     b"\xef\xbb\xbf", b"\xc3\xa9", b"\xe2\x80\xa8", b"\xff", b"\xe2", b"\x00"]
)  # fmt: skip


@settings(max_examples=600, deadline=None)
@given(st.data(), st.lists(PIECES, max_size=40).map(b"".join))
def test_decoder_agrees_with_the_reference_on_arbitrary_bytes(
    data: st.DataObject, raw: bytes
) -> None:
    limits = {"max_line_bytes": 24, "max_event_bytes": 40}
    try:
        expected: list | None = reference_decode(raw, 24, 40)
    except DecodeError:
        expected = None
    chunks = data.draw(cut(raw))
    if expected is None:
        with pytest.raises(DecodeError):
            decode(*chunks, **limits)
    else:
        assert decode(*chunks, **limits) == expected


# --- the encoder ----------------------------------------------------------------------------------


def test_json_events_are_compact_and_keep_utf8() -> None:
    assert (
        encode_json({"a": 1, "b": [1, 2], "c": "é"})
        == 'data: {"a":1,"b":[1,2],"c":"é"}\n\n'.encode()
    )


def test_json_events_never_contain_characters_that_line_based_clients_split_on() -> None:
    payload = {"text": "a\u2028b\u2029c\x85d\re\nf\x0bg\x0ch\x1ci"}
    encoded = encode_json(payload)
    (line,) = encoded.decode().removesuffix("\n\n").splitlines()
    assert json.loads(line.removeprefix("data: ")) == payload
    assert decode(encoded) == [E(line.removeprefix("data: "))]


JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(st.characters(exclude_categories=("Cs",))),
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(st.text(st.characters(exclude_categories=("Cs",)), max_size=5), children,
                      max_size=3),
    max_leaves=10,
)  # fmt: skip


@settings(max_examples=300, deadline=None)
@given(st.data(), st.dictionaries(st.text(max_size=5), JSON_VALUES, max_size=4))
def test_any_json_payload_round_trips_as_exactly_one_line(
    data: st.DataObject, payload: dict
) -> None:
    encoded = encode_json(payload)
    (event,) = decode(*data.draw(cut(encoded)))
    assert json.loads(event.data) == payload
    assert event.data.splitlines() == [event.data]


def test_json_events_refuse_nan() -> None:
    with pytest.raises(ValueError):
        encode_json({"x": float("nan")})


def test_comments_and_done() -> None:
    assert encode_comment() == b":\n\n"
    assert encode_comment("keep-alive") == b": keep-alive\n\n"
    assert decode(encode_comment("keep-alive")) == [Comment("keep-alive")]
    assert DONE == b"data: [DONE]\n\n"
    for bad in ["a\nb", "a\rb"]:
        with pytest.raises(ValueError, match="single line"):
            encode_comment(bad)

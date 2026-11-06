"""Task 3: relaying upstream items to the client as redacted OpenAI chunks."""

from __future__ import annotations

import json
from contextlib import aclosing

import anyio
import pytest
from openai.types.chat import ChatCompletionChunk

from ai_gateway.llm_gateway.events import KEEP_ALIVE, Channel, StreamEnd, TextDelta, Usage
from ai_gateway.llm_gateway.redaction import REDACTED, RedactionRules
from ai_gateway.llm_gateway.sse import Comment, Decoder
from ai_gateway.llm_gateway.streaming import StreamRecord, relay
from ai_gateway.llm_gateway.upstream import UpstreamError

R = REDACTED
HANG = object()
END = StreamEnd("stop", None)


def text(value: str) -> TextDelta:
    return TextDelta(Channel.CONTENT, value)


class Source:
    """Plays a script of stream items; records how often it was closed."""

    def __init__(self, *script: object) -> None:
        self.script = script
        self.closed = 0

    async def items(self):
        for item in self.script:
            if item is HANG:
                await anyio.sleep_forever()
            if isinstance(item, BaseException):
                raise item
            yield item

    async def aclose(self) -> None:
        self.closed += 1


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def frames(
    source: Source,
    *,
    include_usage: bool = False,
    max_seconds: float = 60,
    keepalive_seconds: float = 15,
    clock: Clock | None = None,
    record: StreamRecord | None = None,
) -> list[dict | str]:
    """Everything the client receives: chunk payloads, "[DONE]", and "comment: ..." strings."""
    stream = relay(
        source,
        model="gpt-test",
        rules=RedactionRules(),
        include_usage=include_usage,
        max_seconds=max_seconds,
        keepalive_seconds=keepalive_seconds,
        record=record or StreamRecord("req-1"),
        clock=clock or Clock(),
    )
    decoder, out = Decoder(), []
    async with aclosing(stream) as chunks:
        async for raw in chunks:
            for item in decoder.feed(raw):
                if isinstance(item, Comment):
                    out.append(f"comment: {item.text}")
                elif item.data == "[DONE]":
                    out.append("[DONE]")
                else:
                    out.append(json.loads(item.data))
    assert not decoder.close()
    return out


def deltas(out: list) -> list[dict]:
    return [f["choices"][0]["delta"] for f in out if isinstance(f, dict) and f.get("choices")]


def joined(out: list, channel: str = "content") -> str:
    return "".join(d.get(channel, "") for d in deltas(out))


async def test_a_stream_is_a_role_chunk_redacted_text_one_finish_chunk_and_done() -> None:
    source = Source(text("Mail john.do"), text("e@example.com now"), END)
    out = await frames(source)
    assert deltas(out) == [
        {"role": "assistant", "content": ""},
        {"content": "Mail "},
        {"content": f"{R} "},
        {"content": "now"},
        {},
    ]
    assert out[-2]["choices"][0]["finish_reason"] == "stop"
    assert out[-1] == "[DONE]"
    assert source.closed == 1


async def test_no_frame_ever_carries_part_of_the_pii() -> None:
    pieces = ["Card 41", "11 1111 ", "1111 11", "11 and jo", "hn.doe@ex", "ample.com", " ok"]
    out = await frames(Source(*map(text, pieces), END))
    assert joined(out) == f"Card {R} and {R} ok"
    for delta in deltas(out):
        assert not any(fragment in delta.get("content", "") for fragment in ["41", "john", "ex"])


async def test_every_chunk_is_a_valid_openai_chunk_with_the_same_id_and_model() -> None:
    usage = Usage(5, 3, 8)
    out = await frames(Source(text("Hi"), StreamEnd("length", usage)), include_usage=True)
    chunks = [frame for frame in out if isinstance(frame, dict)]
    for frame in chunks:
        ChatCompletionChunk.model_validate(frame, strict=True)
    assert len({(f["id"], f["created"], f["model"]) for f in chunks}) == 1
    assert chunks[0]["model"] == "gpt-test" and chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


async def test_usage_is_only_sent_when_requested_and_known() -> None:
    usage = Usage(5, 3, 8)
    assert not any(
        "usage" in f for f in await frames(Source(StreamEnd("stop", usage))) if isinstance(f, dict)
    )
    out = await frames(Source(END), include_usage=True)
    assert not any("usage" in f for f in out if isinstance(f, dict))


async def test_refusals_are_redacted_by_their_own_redactor() -> None:
    source = Source(
        text("mail a@b"),
        TextDelta(Channel.REFUSAL, "I won't share x@y.io"),
        text(".io ok"),
        END,
    )
    out = await frames(source)
    assert joined(out) == f"mail {R} ok"
    assert joined(out, "refusal") == f"I won't share {R}"


async def test_an_error_mid_stream_discards_held_back_text_and_sends_one_error_frame() -> None:
    error = UpstreamError(502, "upstream_error", retryable=False, detail="error object")
    record = StreamRecord("req-1")
    source = Source(text("Mail john.do"), error)
    out = await frames(source, record=record)
    assert joined(out) == "Mail "
    assert out[-1] == {
        "error": {
            "message": "The model provider reported an error.",
            "type": "server_error",
            "code": "upstream_error",
            "param": None,
        }
    }
    assert "[DONE]" not in out
    assert "john" not in json.dumps(out)
    assert (record.outcome, record.error_code) == ("upstream_error", "upstream_error")
    assert source.closed == 1


async def test_an_unexpected_failure_ends_the_stream_closed() -> None:
    source = Source(text("Mail john.do"), RuntimeError("bug"))
    out = await frames(source)
    assert out[-1]["error"]["code"] == "internal_error"
    assert "john" not in json.dumps(out)
    assert source.closed == 1


async def test_a_stream_longer_than_its_maximum_duration_ends_with_an_error() -> None:
    source = Source(text("Hello"), HANG)
    with anyio.fail_after(5):
        out = await frames(source, max_seconds=0.2, clock=None)
    assert out[-1]["error"]["code"] == "upstream_timeout"
    assert source.closed == 1


async def test_keep_alives_are_relayed_at_most_once_per_interval() -> None:
    clock = Clock()

    class Ticking(Source):
        async def items(self):
            for now in (5, 16, 20, 40):
                clock.now = now
                yield KEEP_ALIVE
            yield END

    out = await frames(Ticking(), keepalive_seconds=15, clock=clock)
    assert [f for f in out if isinstance(f, str)] == [
        "comment: keep-alive",
        "comment: keep-alive",
        "[DONE]",
    ]


async def test_the_record_counts_redactions_and_the_outcome() -> None:
    record = StreamRecord("req-1")
    await frames(Source(text("a@b.io and c@d.io"), END), record=record)
    assert (record.outcome, record.finish_reason, record.redactions) == ("completed", "stop", 2)


async def test_closing_the_relay_early_closes_the_source() -> None:
    source = Source(text("Hello"), HANG)
    stream = relay(
        source,
        model="m",
        rules=RedactionRules(),
        include_usage=False,
        max_seconds=60,
        keepalive_seconds=15,
        record=StreamRecord("req-1"),
    )
    async with aclosing(stream) as chunks:
        async for _ in chunks:
            break
    assert source.closed == 1


@pytest.mark.parametrize("reason", ["stop", "length", "content_filter"])
async def test_the_finish_reason_is_relayed(reason: str) -> None:
    out = await frames(Source(StreamEnd(reason, None)))
    assert out[-2]["choices"][0]["finish_reason"] == reason

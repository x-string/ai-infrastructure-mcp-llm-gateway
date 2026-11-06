"""Server-Sent Events: an incremental decoder for the upstream stream and the encoder for clients.

The decoder follows the WHATWG "Parsing an event stream" rules, with stricter input handling:

* lines end at CR, LF or CRLF only (never at U+2028, U+0085 and the other separators that
  ``str.splitlines`` and httpx2's ``aiter_lines`` split on), including a CRLF split across chunks;
* every line is decoded as strict UTF-8, whatever charset the response declares, and invalid bytes
  are an error rather than U+FFFD;
* a line or an event larger than its limit (1 MiB each) is an error, checked before it is buffered;
* comments are returned, because they show that the upstream is still alive;
* at the end of the stream an unfinished event is discarded, as the spec requires (the current
  OpenAI Python SDK does the same), so a stream cut off mid-event cannot pass for a complete one.

``id`` and ``retry`` are ignored: the gateway never reconnects to the upstream.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

MAX_LINE_BYTES = 1 << 20
MAX_EVENT_BYTES = 1 << 20

_LINE_BREAK = re.compile(rb"\r\n?|\n")
# JSON leaves these raw inside strings, but line-based clients would split events at them.
_LINE_SEPARATOR_ESCAPES = str.maketrans({" ": "\\u2028", " ": "\\u2029", "\x85": "\\u0085"})

DONE = b"data: [DONE]\n\n"


class DecodeError(Exception):
    """The bytes are not a valid event stream within the decoder's limits."""


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    data: str
    event: str = "message"


@dataclass(frozen=True, slots=True)
class Comment:
    text: str


class Decoder:
    """Turns chunks of bytes, split anywhere, into events and comments."""

    __slots__ = (
        "_max_line",
        "_max_event",
        "_partial",
        "_after_cr",
        "_started",
        "_event",
        "_data",
        "_data_size",
    )

    def __init__(
        self, *, max_line_bytes: int = MAX_LINE_BYTES, max_event_bytes: int = MAX_EVENT_BYTES
    ) -> None:
        self._max_line = max_line_bytes
        self._max_event = max_event_bytes
        self._partial = bytearray()  # an unterminated line
        self._after_cr = False  # the last chunk ended with CR: a leading LF belongs to it
        self._started = False  # a first line has been seen (only it may start with a BOM)
        self._event = ""
        self._data: list[str] = []
        self._data_size = 0

    def feed(self, chunk: bytes) -> list[ServerSentEvent | Comment]:
        """Decode the next chunk; return the events and comments it completes."""
        out: list[ServerSentEvent | Comment] = []
        if not chunk:  # must not lose the pending CR of a CRLF split around it
            return out
        start = 0
        if self._after_cr:
            self._after_cr = False
            if chunk.startswith(b"\n"):
                start = 1
        for match in _LINE_BREAK.finditer(chunk, start):
            line = chunk[start : match.start()]
            if self._partial:
                if len(self._partial) + len(line) > self._max_line:
                    raise DecodeError("line too long")
                line = bytes(self._partial) + line
                self._partial.clear()
            elif len(line) > self._max_line:
                raise DecodeError("line too long")
            self._process_line(line, out)
            start = match.end()
            self._after_cr = start == len(chunk) and match.group() == b"\r"
        if start < len(chunk):
            if len(self._partial) + len(chunk) - start > self._max_line:
                raise DecodeError("line too long")
            self._partial += chunk[start:]
        return out

    def close(self) -> bool:
        """End of the stream. Returns whether an unfinished event (or line) was discarded."""
        unfinished = bool(self._partial or self._data or self._event)
        self._partial.clear()
        self._reset_event()
        return unfinished

    def _process_line(self, raw: bytes, out: list[ServerSentEvent | Comment]) -> None:
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise DecodeError("invalid UTF-8") from None
        if not self._started:
            self._started = True
            line = line.removeprefix("﻿")
        if not line:
            if self._data:
                out.append(ServerSentEvent("\n".join(self._data), self._event or "message"))
            self._reset_event()
        elif line[0] == ":":
            out.append(Comment(line[1:].removeprefix(" ")))
        else:
            name, colon, value = line.partition(":")
            if colon:
                value = value.removeprefix(" ")
            if name == "data":
                self._data_size += len(raw)
                if self._data_size > self._max_event:
                    raise DecodeError("event too large")
                self._data.append(value)
            elif name == "event":
                self._event = value

    def _reset_event(self) -> None:
        self._event = ""
        self._data = []
        self._data_size = 0


def encode_json(payload: object) -> bytes:
    """One ``data:`` event carrying ``payload`` as compact JSON."""
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return b"data: " + text.translate(_LINE_SEPARATOR_ESCAPES).encode() + b"\n\n"


def encode_comment(text: str = "") -> bytes:
    """A comment, which clients ignore: used as a keep-alive."""
    if "\r" in text or "\n" in text:
        raise ValueError("a comment must be a single line")
    return f": {text}\n\n".encode() if text else b":\n\n"

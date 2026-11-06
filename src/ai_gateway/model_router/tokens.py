"""Conservative token estimates, for reserving a request's cost before it is sent (Task 4).

The limiter has to decide before the provider says how many tokens a request used, so it reserves
an estimate that errs high: every character of the payload actually sent counts, ASCII at three
characters per token (English averages about four), anything else at one token per character, plus
a fixed overhead per message, plus the full completion cap. When the response arrives, the
provider's own count replaces the reservation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Final

ASCII_CHARS_PER_TOKEN: Final = 3
MESSAGE_OVERHEAD_TOKENS: Final = 4
REQUEST_OVERHEAD_TOKENS: Final = 3


def text_tokens(text: str) -> int:
    ascii_chars = len(text.encode("ascii", "ignore"))
    return math.ceil(ascii_chars / ASCII_CHARS_PER_TOKEN) + (len(text) - ascii_chars)


def prompt_tokens(payload: Mapping[str, object]) -> int:
    """An upper estimate of the prompt tokens of ``payload``, the exact body sent upstream."""
    messages = payload.get("messages")
    count = len(messages) if isinstance(messages, list) else 0
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text_tokens(canonical) + MESSAGE_OVERHEAD_TOKENS * count + REQUEST_OVERHEAD_TOKENS


def request_cost(payload: Mapping[str, object], max_tokens: int) -> int:
    """What admission reserves: the prompt estimate plus the completion cap."""
    return prompt_tokens(payload) + max_tokens

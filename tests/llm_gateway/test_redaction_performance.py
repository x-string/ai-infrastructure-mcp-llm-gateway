"""Task 3: the redactor stays fast and bounded on adversarial streams.

1 MB of text is fed in 1 to 4 character deltas, the size of LLM tokens. Each input takes about a
second on a laptop; the bound leaves room for slower machines.
"""

from __future__ import annotations

import random
import time

import pytest

from ai_gateway.llm_gateway.redaction import MAX_EMAIL_LENGTH, StreamRedactor

pytestmark = pytest.mark.slow

SIZE = 1_000_000
WORDS = "the quick report shows revenue of $1,234 in Q3, details. \n".split(" ")

INPUTS = {
    "digits": ("0123456789" * (SIZE // 10))[:SIZE],
    "email-like": ("a@a." * (SIZE // 4))[:SIZE],
    "card-like": ("4111 1111 1111 1111 " * (SIZE // 20))[:SIZE],
    "jwt-like": (("eyJhbGciOi." + "x" * 50 + "-_") * (SIZE // 63 + 1))[:SIZE],
    "prose": " ".join(random.Random(1).choice(WORDS) for _ in range(SIZE // 4))[:SIZE],
}


@pytest.mark.parametrize("name", list(INPUTS))
def test_one_megabyte_in_token_sized_deltas_is_fast_and_bounded(name: str) -> None:
    text = INPUTS[name]
    sizes = random.Random(2)
    redactor = StreamRedactor()
    started = time.perf_counter()
    position = 0
    while position < len(text):
        step = sizes.randint(1, 4)
        redactor.feed(text[position : position + step])
        assert redactor.pending <= MAX_EMAIL_LENGTH
        position += step
    redactor.flush()
    assert time.perf_counter() - started < 3.0

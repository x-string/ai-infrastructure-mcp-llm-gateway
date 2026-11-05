"""Task 3: property-based tests of the streaming redactor.

The central claim: for any text and any way of splitting it into deltas, streaming redaction
releases exactly what redacting the whole text at once produces, never takes anything back, and
holds back at most MAX_EMAIL_LENGTH characters.
"""

from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from ai_gateway.llm_gateway.redaction import (
    CARD,
    EMAIL,
    MAX_EMAIL_LENGTH,
    REDACTED,
    SSN_BARE,
    SSN_SEPARATED,
    RedactionRules,
    StreamRedactor,
    compiled_pattern,
    is_valid_card,
    luhn_valid,
    redact_text,
)

RULES = st.sampled_from(
    [RedactionRules(), RedactionRules(bare_ssn=False), RedactionRules(card_luhn=True)]
)


@st.composite
def valid_cards(draw: st.DrawFn) -> str:
    """A card number that passes the Luhn check, in groups of four or not."""
    body = draw(st.from_regex(r"[2-6][0-9]{11,17}", fullmatch=True))
    number = body + next(digit for digit in "0123456789" if luhn_valid(body + digit))
    separator = draw(st.sampled_from(["", " ", "-"]))
    return separator.join(number[i : i + 4] for i in range(0, len(number), 4))


PIECES = st.one_of(
    st.sampled_from(
        [
            "john.doe@example.com",
            "A.B+c@mail.co.uk",
            "x@b.io",
            "bad@host",
            "user@ex-ample.org",
            "q" * 70 + "@long.example.com",
            "@",
            "@@",
            ".",
            "-",
            " ",
            "\n",
            ",",
            "hello",
            "card",
            "SSN",
        ]
    ),
    st.from_regex(r"[0-9]{3}[- ][0-9]{2}[- ][0-9]{4}", fullmatch=True),
    st.from_regex(r"[0-9]{9}", fullmatch=True),
    st.from_regex(r"(?:[0-9]{4}[ -]?){3,4}[0-9]{1,4}", fullmatch=True),
    valid_cards(),
    st.tuples(valid_cards(), st.from_regex(r"[ -][0-9]{1,4}", fullmatch=True)).map("".join),
    st.text(alphabet="abcxyz0123._-@ ", max_size=60),
    st.text(max_size=8),
)
TEXTS = st.lists(PIECES, max_size=25).map("".join)


@st.composite
def split_texts(draw: st.DrawFn) -> tuple[str, list[str]]:
    text = draw(TEXTS)
    if len(text) < 2:
        return text, [text]
    cuts = sorted(draw(st.sets(st.integers(1, len(text) - 1), max_size=len(text) - 1)))
    bounds = [0, *cuts, len(text)]
    return text, [text[a:b] for a, b in zip(bounds, bounds[1:], strict=False)]


FAMILY = {
    "email": re.compile(EMAIL, re.ASCII),
    "ssn": re.compile(SSN_SEPARATED, re.ASCII),
    "bare_ssn": re.compile(SSN_BARE, re.ASCII),
    "card": re.compile(CARD, re.ASCII),
}


def reference_match_end(text: str, start: int, rules: RedactionRules) -> int | None:
    """Where a redaction starting at ``start`` ends: each family on its own, in the documented
    order, and in Luhn mode every card length tried by brute force."""
    for name in ["email", "ssn", "bare_ssn"] if rules.bare_ssn else ["email", "ssn"]:
        if match := FAMILY[name].match(text, start):
            return match.end()
    longest = FAMILY["card"].match(text, start)
    if longest is None or not rules.card_luhn:
        return None if longest is None else longest.end()
    for end in range(longest.end(), start, -1):
        followed_by_digit = end < len(text) and text[end] in "0123456789"
        if (
            not followed_by_digit
            and FAMILY["card"].fullmatch(text, start, end)
            and is_valid_card(text[start:end])
        ):
            return end
    return None


def independent_reference(text: str, rules: RedactionRules) -> str:
    """Batch redaction written another way: try every position, left to right."""
    pieces: list[str] = []
    position = copied = 0
    while position < len(text):
        end = reference_match_end(text, position, rules)
        if end is None:
            position += 1
        else:
            pieces += [text[copied:position], REDACTED]
            copied = position = end
    return "".join(pieces) + text[copied:]


@settings(max_examples=500, deadline=None)
@given(split_texts(), RULES)
def test_streaming_matches_batch_redaction_for_any_split(
    case: tuple[str, list[str]], rules: RedactionRules
) -> None:
    text, deltas = case
    expected = redact_text(text, rules)
    redactor = StreamRedactor(rules)
    released = ""
    for delta in deltas:
        released += redactor.feed(delta)
        assert expected.startswith(released)  # nothing released is ever taken back
        assert redactor.pending <= MAX_EMAIL_LENGTH
    assert released + redactor.flush() == expected


@settings(max_examples=500, deadline=None)
@given(TEXTS, RULES)
def test_batch_redaction_matches_an_independent_reference(text: str, rules: RedactionRules) -> None:
    expected = independent_reference(text, rules)
    assert redact_text(text, rules) == expected
    if not rules.card_luhn:  # without Luhn, the rules are exactly a regular-expression substitution
        assert compiled_pattern(rules).sub(REDACTED, text) == expected


DIGIT_GROUPS = st.lists(st.from_regex(r"[0-9]{1,4}", fullmatch=True), max_size=3)


@settings(max_examples=300, deadline=None)
@given(
    st.one_of(st.sampled_from(["123-45-6789", "123 45 6789", "4111111111111111"]), valid_cards()),
    DIGIT_GROUPS,
    DIGIT_GROUPS,
    st.sampled_from([" ", "-"]),
    st.sampled_from([RedactionRules(), RedactionRules(card_luhn=True)]),
)
def test_pii_next_to_other_digit_groups_is_never_released_whole(
    pii: str, before: list[str], after: list[str], joiner: str, rules: RedactionRules
) -> None:
    text = joiner.join([*before, pii, *after])
    assert pii not in redact_text(text, rules)

"""Task 3: what the PII patterns catch, and how the streaming redactor releases text."""

from __future__ import annotations

import re
from re import _constants as sre  # private; used only to inspect the patterns
from re import _parser as sre_parse

import pytest

from ai_gateway.llm_gateway.redaction import (
    CARD,
    EMAIL,
    EMAIL_ALPHABET,
    MAX_EMAIL_LENGTH,
    MAX_NUMBER_LENGTH,
    NUMBER_ALPHABET,
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

R = REDACTED
LUHN = RedactionRules(card_luhn=True)


# --- emails ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail john.doe@example.com now", f"mail {R} now"),
        ("John.Doe+billing@Mail.Example.co.uk!", f"{R}!"),
        ("<a@b.io>", f"<{R}>"),
        ("ends with john@example.com.", f"ends with {R}."),
        ("x_y-z%1@sub-domain.example.museum", R),
        ("two: a@b.io, c@d.io", f"two: {R}, {R}"),
        ("123-45-6789@example.com", R),  # one email, not an SSN followed by a visible domain
    ],
)
def test_emails_are_redacted(text: str, expected: str) -> None:
    assert redact_text(text) == expected


@pytest.mark.parametrize(
    "text",
    ["user@", "@example.com", "user@example", "user@example.c", "john at example dot com", "a@b"],
)
def test_text_that_is_not_an_email_is_left_alone(text: str) -> None:
    assert redact_text(text) == text


def test_an_overlong_local_part_is_redacted_except_its_first_characters() -> None:
    # RFC 5321 caps the local part at 64 characters; a documented limit of the pattern.
    assert redact_text("a" * 70 + "@example.com") == "a" * 6 + R


# --- US Social Security numbers -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SSN 123-45-6789.", f"SSN {R}."),
        ("123 45 6789", R),
        ("987-65-4321", R),  # SSA advertising range: still SSN-shaped, so redacted
        ("999-99-9999", R),
        ("000-12-3456", R),
        ("ssn:123456789,", f"ssn:{R},"),
    ],
)
def test_ssns_are_redacted_by_shape(text: str, expected: str) -> None:
    assert redact_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "123-45 6789",  # inconsistent separators
        "1234-56-7890",
        "123-45-67890",
        "12-345-6789",
        "000123456",  # bare nine digits that cannot be an SSN
        "666123456",
        "123006789",
        "123450000",
    ],
)
def test_ssn_lookalikes_are_left_alone(text: str) -> None:
    assert redact_text(text) == text


def test_bare_nine_digit_numbers_can_be_excluded() -> None:
    rules = RedactionRules(bare_ssn=False)
    assert redact_text("order 123456789", rules) == "order 123456789"
    assert redact_text("SSN 123-45-6789", rules) == f"SSN {R}"


# --- payment cards --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4111 1111 1111 1111", R),
        ("4111-1111-1111-1111", R),
        ("4111111111111111", R),
        ("1234 5678 9012 3456", R),  # not Luhn-valid, but card-shaped
        ("3782 822463 10005", R),  # 15-digit Amex grouping
        ("4222222222222", R),  # 13 digits
        ("6011000000000000004", R),  # 19 digits
        ("Card 4111 1111 1111 1111 exp 12/26", f"Card {R} exp 12/26"),
        ("SSN 123-45-6789 4111 1111 1111 1111 ok", f"SSN {R} {R} ok"),
    ],
)
def test_card_shaped_numbers_are_redacted(text: str, expected: str) -> None:
    assert redact_text(text) == expected


@pytest.mark.parametrize(
    "text", ["call 555-0100", "+1 (415) 555-0100", "order 12345", "year 2024", "123456789012"]
)
def test_ordinary_numbers_are_left_alone(text: str) -> None:
    assert redact_text(text) == text


def test_a_card_next_to_another_digit_group_keeps_that_group() -> None:
    # A documented limit of shape matching: the trailing group falls outside the 19-digit match.
    assert redact_text("ref 1234 4111 1111 1111 1111 ok") == f"ref {R} 1111 ok"


def test_luhn_mode_redacts_only_valid_cards() -> None:
    assert redact_text("4111 1111 1111 1111", LUHN) == R
    assert redact_text("1234 5678 9012 3456", LUHN) == "1234 5678 9012 3456"
    assert redact_text("4111 1111 1111 1112", LUHN) == "4111 1111 1111 1112"
    assert redact_text("id 1730800000000", LUHN) == "id 1730800000000"  # a millisecond timestamp


def test_luhn_mode_finds_a_valid_card_inside_a_rejected_candidate() -> None:
    # The candidate starts before the card: the next digit group is tried.
    assert redact_text("ref 1234 4111 1111 1111 1111 ok", LUHN) == f"ref 1234 {R} ok"
    # The candidate runs past the card: shorter numbers at the same start are tried.
    assert redact_text("4111 1111 1111 1111 22", LUHN) == f"{R} 22"


@pytest.mark.parametrize(
    ("number", "valid"),
    [
        ("4111111111111111", True),
        ("4111111111111112", False),
        ("378282246310005", True),
        ("5555555555554444", True),
    ],
)
def test_luhn_checksum(number: str, valid: bool) -> None:
    assert luhn_valid(number) is valid


def test_luhn_checksum_takes_digits_only() -> None:
    with pytest.raises(ValueError, match="ASCII digits"):
        luhn_valid("4111 1111")


def test_card_check_also_requires_a_major_industry_identifier_from_2_to_6() -> None:
    assert is_valid_card("4111 1111 1111 1111") is True
    assert luhn_valid("9111111111111110") is True
    assert is_valid_card("9111111111111110") is False  # no card network starts with 9
    assert is_valid_card("4111 1111 1111") is False  # too short


# --- the invariants the streaming algorithm relies on ---------------------------------------------


def inspect_pattern(pattern: str) -> tuple[set[str], int, int]:
    """Every character the pattern can consume, its widest lookbehind, and its longest match."""
    consumed: set[str] = set()
    lookbehind = 0

    def walk(items: sre_parse.SubPattern) -> None:
        nonlocal lookbehind
        for op, value in items:
            if op is sre.LITERAL:
                consumed.add(chr(value))
            elif op is sre.IN:
                for item, bounds in value:
                    assert item in (sre.LITERAL, sre.RANGE), item  # no \d, \w or negated sets
                    low, high = (bounds, bounds) if item is sre.LITERAL else bounds
                    consumed.update(map(chr, range(low, high + 1)))
            elif op in (sre.MAX_REPEAT, sre.MIN_REPEAT, sre.POSSESSIVE_REPEAT):
                walk(value[2])
            elif op is sre.SUBPATTERN:
                walk(value[3])
            elif op is sre.BRANCH:
                for branch in value[1]:
                    walk(branch)
            elif op in (sre.ASSERT, sre.ASSERT_NOT):  # lookarounds consume nothing
                direction, body = value
                if direction < 0:
                    lookbehind = max(lookbehind, body.getwidth()[1])
            else:
                raise AssertionError(f"unexpected regex construct {op}")

    parsed = sre_parse.parse(pattern, re.ASCII)
    walk(parsed)
    return consumed, lookbehind, parsed.getwidth()[1]


@pytest.mark.parametrize(
    ("pattern", "alphabet", "longest"),
    [
        (EMAIL, EMAIL_ALPHABET, MAX_EMAIL_LENGTH),
        (SSN_SEPARATED, NUMBER_ALPHABET, MAX_NUMBER_LENGTH),
        (SSN_BARE, NUMBER_ALPHABET, MAX_NUMBER_LENGTH),
        (CARD, NUMBER_ALPHABET, MAX_NUMBER_LENGTH),
    ],
    ids=["email", "ssn", "bare-ssn", "card"],
)
def test_each_pattern_stays_within_its_family_alphabet_and_length(
    pattern: str, alphabet: frozenset[str], longest: int
) -> None:
    consumed, lookbehind, width = inspect_pattern(pattern)
    assert consumed <= alphabet
    assert lookbehind <= 1  # the redactor keeps one released character as left context
    assert width <= longest


@pytest.mark.parametrize("rules", [RedactionRules(), RedactionRules(bare_ssn=False), LUHN])
def test_the_combined_pattern_is_ascii_and_case_sensitive(rules: RedactionRules) -> None:
    # Under IGNORECASE without ASCII, [a-z] also matches U+212A KELVIN SIGN, which is outside the
    # email alphabet. The classes list both cases instead.
    flags = compiled_pattern(rules).flags
    assert flags & re.ASCII
    assert not flags & re.IGNORECASE


# --- streaming ------------------------------------------------------------------------------------

SAMPLES = [
    "john.doe@example.com",
    "123-45-6789",
    "123 45 6789",
    "123456789",
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "4111111111111111",
    "3782 822463 10005",
]


@pytest.mark.parametrize("pii", SAMPLES)
def test_pii_split_at_every_offset_is_never_released(pii: str) -> None:
    text = f"Contact: {pii}, thanks."
    expected = redact_text(text)
    assert pii not in expected
    for cut in range(1, len(text)):
        redactor = StreamRedactor()
        first = redactor.feed(text[:cut])
        rest = redactor.feed(text[cut:]) + redactor.flush()
        assert expected.startswith(first), cut  # nothing released early is ever taken back
        assert first + rest == expected, cut


@pytest.mark.parametrize("pii", SAMPLES)
def test_one_character_at_a_time_gives_the_same_result(pii: str) -> None:
    text = f"a {pii} b {pii}."
    redactor = StreamRedactor()
    released = "".join(redactor.feed(char) for char in text) + redactor.flush()
    assert released == redact_text(text) == f"a {R} b {R}."


def test_prose_is_released_one_token_behind() -> None:
    tokens = [
        "Sure", ",", " email", " john", ".do", "e@exa", "mple", ".com",
        " or", " call", " 4111", " 1111", " 1111", " 1111", " now", ".",
    ]  # fmt: skip
    redactor = StreamRedactor()
    released = [redactor.feed(token) for token in tokens] + [redactor.flush()]
    assert released == [
        "", "Sure,", " ", "email ", "", "", "", "", f"{R} ",
        "or ", "call ", "", "", "", f"{R} ", "", "now.",
    ]  # fmt: skip


@pytest.mark.parametrize("unit", ["a", "7", "a@a.", "-", "x.y", "1 ", "4111 1111 1111 1111 "])
def test_held_back_text_never_exceeds_the_longest_email(unit: str) -> None:
    redactor = StreamRedactor()
    for _ in range(3_000):
        redactor.feed(unit * 3)
        assert redactor.pending <= MAX_EMAIL_LENGTH


def test_a_large_delta_is_released_up_to_its_last_342_characters() -> None:
    redactor = StreamRedactor()
    released = redactor.feed("x" * 10_000)
    assert released == "x" * (10_000 - MAX_EMAIL_LENGTH)
    assert redactor.pending == MAX_EMAIL_LENGTH


def test_empty_deltas_change_nothing() -> None:
    redactor = StreamRedactor()
    assert redactor.feed("") == ""
    assert redactor.feed("john@") == ""
    assert redactor.feed("") == ""
    assert redactor.pending == len("john@")


def test_flush_releases_an_unfinished_candidate_unchanged() -> None:
    redactor = StreamRedactor()
    released = redactor.feed("write to john.doe@exa") + redactor.flush()
    assert released == "write to john.doe@exa"  # not an email address, so nothing to redact


def test_discard_drops_held_back_text() -> None:
    redactor = StreamRedactor()
    assert redactor.feed("card 4111 1111 1111") == "card "
    assert redactor.discard() == len("4111 1111 1111")
    assert redactor.pending == 0
    assert redactor.flush() == ""


def test_a_redactor_starts_over_after_flush() -> None:
    redactor = StreamRedactor()
    assert redactor.feed("1234") + redactor.flush() == "1234"
    # Without the reset, "1234" would still be left context and 5678... would not start a number.
    assert redactor.feed("5678 1234 5678 9012") + redactor.flush() == R

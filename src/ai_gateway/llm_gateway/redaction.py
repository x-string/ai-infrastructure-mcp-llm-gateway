"""Real-time PII redaction for streamed text (Task 3).

Email addresses, US Social Security numbers and payment card numbers become ``[REDACTED]`` while
the text streams, without waiting for the end of the response.

What is redacted. Detection is by shape, because redacting too much is the safe direction:

* email: ``local@domain.tld`` in ASCII, with a local part of 1-64 characters and a TLD of 2-24
  letters;
* SSN: ``123-45-6789`` or ``123 45 6789`` (one separator style), and, unless ``bare_ssn=False``,
  nine bare digits that can be an SSN (area not 000 or 666, group not 00, serial not 0000);
* card: 13-19 digits, optionally with single spaces or hyphens between them. With
  ``card_luhn=True``, only numbers that start with 2-6 and pass the Luhn check count: at each
  position the longest such number wins.

How streaming works. PII can arrive split across deltas (``"john.do"`` then ``"e@example.com"``),
so the redactor holds back the end of the text that could still become part of a match and
releases the rest. The patterns fall into two families, each with a closed alphabet and a maximum
match length:

* emails use ASCII letters, digits and ``._%+-@``, and are at most 342 characters long;
* SSNs and cards use digits, space and hyphen, start at a digit, and are at most 37 long.

A match attempt reads forward only while characters belong to its family's alphabet, never further
than the maximum length, plus one character on either side for the boundary checks. So when the
text so far ends at ``end``, everything before this point is final::

    min(max(start of the trailing run of email characters, end - 342),
        max(first digit in the trailing run of SSN/card characters, end - 37))

No later delta can change whether or where a match occurs before that point. The redactor tries
the pattern only at positions where a match can start (a digit not preceded by a digit, or a
position at most 64 characters before an ``@``), redacts what matches, and releases the text up to
that point or to the end of the last match.

Guarantees (checked by the property tests):

* for any split of a text into deltas, the output equals ``redact_text`` of the whole text, and
  nothing released is ever taken back;
* at most 342 characters are held back, and each position is examined once;
* ordinary prose is released about one token late, because ``john`` could still become an email.

When a stream ends normally, ``flush`` releases the rest: its last characters are final. A stream
cut off by an error should be ``discard``-ed instead, because its held-back text may be the start
of PII that never finished arriving (``john.doe@exa``).
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from functools import cache

REDACTED = "[REDACTED]"

# The local part is possessive ("+"): "@" is outside its class, so this only removes backtracking.
EMAIL = r"[A-Za-z0-9._%+\-]{1,64}+@[A-Za-z0-9.\-]{1,252}\.[A-Za-z]{2,24}"
SSN_SEPARATED = r"(?<![0-9])[0-9]{3}(?:-[0-9]{2}-|[ ][0-9]{2}[ ])[0-9]{4}(?![0-9])"
SSN_BARE = r"(?<![0-9])(?!000|666)[0-9]{3}(?!00)[0-9]{2}(?!0000)[0-9]{4}(?![0-9])"
CARD = r"(?<![0-9])[0-9](?:[ \-]?[0-9]){12,18}(?![0-9])"

EMAIL_ALPHABET = frozenset(string.ascii_letters + string.digits + "._%+-@")
NUMBER_ALPHABET = frozenset(string.digits + " -")
MAX_LOCAL_PART = 64  # RFC 5321; an email starts at most this many characters before its "@"
MAX_EMAIL_LENGTH = 342  # 64 + "@" + 252 + "." + 24
MAX_NUMBER_LENGTH = 37  # 19 digits with a separator between each pair

_EMAIL_CHARS = "".join(sorted(EMAIL_ALPHABET))
_NUMBER_CHARS = "".join(sorted(NUMBER_ALPHABET))
_DIGIT = re.compile("[0-9]")
_DIGIT_RUN_START = re.compile("(?<![0-9])[0-9]")
_CARD_MIN_DIGITS = 13


@dataclass(frozen=True, slots=True)
class RedactionRules:
    """Optional rules; the defaults favour redacting too much over leaking."""

    bare_ssn: bool = True  # also redact nine bare digits that can be an SSN, e.g. 123456789
    card_luhn: bool = False  # redact only card numbers that start with 2-6 and pass Luhn


DEFAULT_RULES = RedactionRules()


@cache
def compiled_pattern(rules: RedactionRules = DEFAULT_RULES) -> re.Pattern[str]:
    """All patterns as one alternation, tried in this order at each position: email, SSN, card.

    Email comes first so that ``123-45-6789@example.com`` is one email, not an SSN followed by a
    visible domain. SSN comes before card so that, in Luhn mode, a rejected card candidate cannot
    hide an SSN at the same position.
    """
    ssn = f"{SSN_SEPARATED}|{SSN_BARE}" if rules.bare_ssn else SSN_SEPARATED
    return re.compile(f"(?P<email>{EMAIL})|(?P<ssn>{ssn})|(?P<card>{CARD})", re.ASCII)


def luhn_valid(digits: str) -> bool:
    """Whether a string of ASCII digits passes the Luhn checksum that card numbers carry."""
    if not (digits.isascii() and digits.isdigit()):
        raise ValueError("expected ASCII digits")
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def is_valid_card(number: str) -> bool:
    """Whether a card number, separators allowed, can be real: 13-19 digits, starts with 2-6
    (the major industry identifiers of the card networks) and passes the Luhn check."""
    digits = number.replace(" ", "").replace("-", "")
    return (
        13 <= len(digits) <= 19
        and digits.isascii()
        and digits.isdigit()
        and digits[0] in "23456"
        and luhn_valid(digits)
    )


class StreamRedactor:
    """Redacts one stream of text, delta by delta; see the module docstring for how.

    Use one instance per stream and per text channel (for example content and refusal).
    """

    __slots__ = ("_pattern", "_luhn", "_buffer", "_start", "_email_run", "_number_digit")

    def __init__(self, rules: RedactionRules = DEFAULT_RULES) -> None:
        self._pattern = compiled_pattern(rules)
        self._luhn = rules.card_luhn
        self._reset()

    @property
    def pending(self) -> int:
        """How many characters are held back."""
        return len(self._buffer) - self._start

    def feed(self, text: str) -> str:
        """Add the next delta; return the text that is now safe to release (possibly empty)."""
        if not text:
            return ""
        offset = len(self._buffer)
        self._buffer += text
        end = len(self._buffer)
        # Update where each family's trailing run starts, looking only at the new text.
        if head := text.rstrip(_EMAIL_CHARS):
            self._email_run = offset + len(head)
        if head := text.rstrip(_NUMBER_CHARS):
            self._number_digit = _first_digit(self._buffer, offset + len(head))
        elif self._number_digit is None:
            self._number_digit = _first_digit(self._buffer, offset)
        number_run = end if self._number_digit is None else self._number_digit
        safe = min(
            max(self._email_run, end - MAX_EMAIL_LENGTH),
            max(number_run, end - MAX_NUMBER_LENGTH),
        )
        return self._release(safe)

    def flush(self) -> str:
        """End of the stream: release everything held back, redacted, and start over."""
        released = self._release(len(self._buffer))
        self._reset()
        return released

    def discard(self) -> int:
        """Drop the held-back text, for a stream cut off by an error; return its length."""
        dropped = self.pending
        self._reset()
        return dropped

    def _reset(self) -> None:
        # The buffer holds the last released character, the left context for the "not after a
        # digit" checks, followed by the held-back text from index _start.
        self._buffer = ""
        self._start = 0
        self._email_run = 0  # start of the trailing run of email characters
        self._number_digit: int | None = None  # first digit in the trailing run of SSN/card ones

    def _release(self, safe: int) -> str:
        """Redact the held-back text before ``safe`` and release it, with any match that starts
        there; keep the rest."""
        buffer, start = self._buffer, self._start
        if safe <= start:
            return ""
        pieces: list[str] = []
        copied = position = start
        next_digit = next_at = -1
        while position < safe:
            # A match can start only at a digit not preceded by a digit (SSN, card) or at most
            # MAX_LOCAL_PART characters before an "@" (email).
            if next_digit < position:
                found = _DIGIT_RUN_START.search(buffer, position, safe)
                next_digit = safe if found is None else found.start()
            if next_at <= position:
                next_at = buffer.find("@", position + 1)
                if next_at < 0:
                    next_at = len(buffer) + MAX_LOCAL_PART
            candidate = min(next_digit, max(position, next_at - MAX_LOCAL_PART))
            if candidate >= safe:
                break
            match_end = self._match_end(buffer, candidate)
            if match_end is None:
                position = candidate + 1
            else:
                pieces += (buffer[copied:candidate], REDACTED)
                copied = position = match_end
        released_to = max(safe, copied)
        pieces.append(buffer[copied:released_to])
        shift = released_to - 1  # keep the last released character as left context
        self._buffer = buffer[shift:]
        self._start = 1
        self._email_run -= shift
        if self._number_digit is not None:
            self._number_digit -= shift
        return "".join(pieces)

    def _match_end(self, text: str, position: int) -> int | None:
        match = self._pattern.match(text, position)
        if match is None:
            return None
        if self._luhn and match.lastgroup == "card":
            return _longest_valid_card(text, position, match.end())
        return match.end()


def redact_text(text: str, rules: RedactionRules = DEFAULT_RULES) -> str:
    """Redact a complete text. Streaming it in pieces of any size gives the same result."""
    redactor = StreamRedactor(rules)
    return redactor.feed(text) + redactor.flush()


def _first_digit(text: str, start: int) -> int | None:
    found = _DIGIT.search(text, start)
    return None if found is None else found.start()


def _longest_valid_card(text: str, start: int, end: int) -> int | None:
    """Where the longest valid card number at ``start`` ends, given that ``text[start:end]`` is
    the longest card-shaped one. Shorter card-shaped numbers end just before one of its
    separators, so ``4111 1111 1111 1111 22`` still yields the valid card before `` 22``."""
    while end - start >= _CARD_MIN_DIGITS:
        if is_valid_card(text[start:end]):
            return end
        end = max(text.rfind(" ", start, end), text.rfind("-", start, end))
    return None

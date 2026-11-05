"""Shared edge cases for the Task 1 tool arguments.

Each value is stored as JSON *text*. The model-level tests decode it with ``json.loads``. The
SDK-level and raw-stdio tests added in later steps can embed the same text verbatim in a JSON-RPC
line, including non-standard tokens such as ``NaN`` that a JSON-RPC client library could never send.
"""

from __future__ import annotations

import json
from typing import NamedTuple


class Case(NamedTuple):
    id: str
    json: str


VALID_CUSTOMER_IDS = [
    Case("lowest", '"CUST-00000"'),
    Case("typical", '"CUST-12345"'),
    Case("highest", '"CUST-99999"'),
]

INVALID_CUSTOMER_IDS = [
    Case("four-digits", '"CUST-1234"'),
    Case("six-digits", '"CUST-123456"'),
    Case("lowercase-prefix", '"cust-12345"'),
    Case("underscore-separator", '"CUST_12345"'),
    Case("missing-separator", '"CUST12345"'),
    Case("letter-in-digits", '"CUST-1234A"'),
    Case("leading-space", '" CUST-12345"'),
    Case("trailing-space", '"CUST-12345 "'),
    Case("trailing-newline", '"CUST-12345\\n"'),
    Case("fullwidth-digits", '"CUST-\\uff11\\uff12\\uff13\\uff14\\uff15"'),
    Case("arabic-indic-digits", '"CUST-\\u0661\\u0662\\u0663\\u0664\\u0665"'),
    Case("unicode-hyphen", '"CUST\\u201012345"'),
    Case("empty-string", '""'),
    Case("integer", "12345"),
    Case("null", "null"),
    Case("list", '["CUST-12345"]'),
]

VALID_AMOUNTS = [
    Case("one-cent", "0.01"),
    Case("json-integer", "100"),
    Case("decimal", "100.5"),
    Case("large", "999999.99"),
]

INVALID_AMOUNTS = [
    Case("zero", "0"),
    Case("zero-float", "0.0"),
    Case("negative-zero", "-0.0"),
    Case("negative", "-1"),
    Case("nan-literal", "NaN"),
    Case("infinity-literal", "Infinity"),
    Case("negative-infinity-literal", "-Infinity"),
    Case("overflows-to-infinity", "1e309"),
    Case("huge-integer", "1" + "0" * 400),
    Case("numeric-string", '"10.5"'),
    Case("boolean", "true"),
    Case("null", "null"),
    Case("list", "[10]"),
    Case("object", '{"value": 10}'),
]

VALID_REASONS = [
    Case("exactly-ten-chars", '"0123456789"'),
    Case("typical", '"Duplicate charge on invoice INV-2041"'),
    Case("inner-newline", '"Charged twice\\nfor order 42"'),
    Case("non-ascii", '"Remboursement demand\\u00e9"'),
    Case("max-length", json.dumps("x" * 1000)),
]

INVALID_REASONS = [
    Case("nine-chars", '"012345678"'),
    Case("empty-string", '""'),
    Case("ten-spaces", '"          "'),
    Case("leading-space", '" refund requested"'),
    Case("trailing-space", '"refund requested "'),
    Case("trailing-newline", '"refund requested\\n"'),
    Case("too-long", json.dumps("x" * 1001)),
    Case("integer", "1234567890"),
    Case("null", "null"),
    Case("list", '["refund requested"]'),
]

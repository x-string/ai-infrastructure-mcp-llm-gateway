r"""Strict input models for the Task 1 MCP tools.

These models are the only validation logic. The server publishes ``Model.model_json_schema()``
as each tool's ``inputSchema`` and validates incoming arguments with the same model, so the schema
clients see is exactly what the server enforces.

Strictness choices:

* ``extra="forbid"``: unknown argument keys are rejected, and the schema says
  ``additionalProperties: false``.
* ``strict=True``: no type coercion, so ``"10.5"`` and ``true`` are not numbers. JSON integers are
  still accepted for ``amount`` because JSON has only one number type.
* ``customer_id`` uses ``[0-9]`` rather than ``\d``. ``\d`` also matches non-ASCII digits such as
  ``"١٢٣٤٥"`` in both Python's ``re`` and pydantic's Rust regex engine.
* ``reason`` may not start or end with whitespace. This rejects whitespace-only reasons and, unlike
  strip-then-count, can still be expressed in the published JSON Schema.
"""

from __future__ import annotations

import re
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

CUSTOMER_ID_PATTERN: Final = r"^CUST-[0-9]{5}$"
REASON_PATTERN: Final = r"^\S(?:[\s\S]*\S)?$"
REASON_MIN_LENGTH: Final = 10
REASON_MAX_LENGTH: Final = 1_000

CustomerId = Annotated[
    str,
    Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer identifier: 'CUST-' followed by exactly five ASCII digits.",
        examples=["CUST-00042"],
    ),
]


class StrictModel(BaseModel):
    """Base class for tool arguments: reject unknown keys and never coerce types."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class GetCustomerRecordInput(StrictModel):
    """Arguments of the ``get_customer_record`` tool."""

    customer_id: CustomerId


class TriggerRefundInput(StrictModel):
    """Arguments of the ``trigger_refund`` tool."""

    customer_id: CustomerId
    amount: Annotated[
        float,
        Field(
            gt=0,
            allow_inf_nan=False,
            description="Refund amount. Must be a positive, finite number.",
            examples=[25.0],
        ),
    ]
    reason: Annotated[
        str,
        Field(
            min_length=REASON_MIN_LENGTH,
            max_length=REASON_MAX_LENGTH,
            pattern=REASON_PATTERN,
            description=(
                "Why the refund is issued: 10 to 1,000 characters, "
                "with no leading or trailing whitespace."
            ),
            examples=["Duplicate charge on invoice INV-2041"],
        ),
    ]


_SAFE_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_REDACTED_FIELD: Final = "<redacted>"
_ROOT_FIELD: Final = "<root>"


def validation_error_details(exc: ValidationError) -> list[dict[str, str]]:
    """Summarize a ``ValidationError`` without echoing client input.

    Returns one ``{"field", "message", "type"}`` entry per error. Input values, documentation URLs
    and error context are dropped. Unknown keys are client-controlled, so their names are kept only
    when they look like ordinary identifiers (at most 64 characters). Every value is a ``str``, so
    the result is always JSON-serializable.
    """
    details: list[dict[str, str]] = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        parts = [str(part) for part in error["loc"]]
        if error["type"] == "extra_forbidden":
            parts = [
                part if _SAFE_FIELD_NAME.fullmatch(part) else _REDACTED_FIELD for part in parts
            ]
        details.append(
            {
                "field": ".".join(parts) or _ROOT_FIELD,
                "message": error["msg"],
                "type": error["type"],
            }
        )
    return details

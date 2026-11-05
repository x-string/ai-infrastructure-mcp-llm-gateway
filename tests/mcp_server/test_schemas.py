"""Task 1: strict input schemas, tested at the model level (the MCP SDK is wired in later steps)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ai_gateway.mcp_server.schemas import (
    CUSTOMER_ID_PATTERN,
    REASON_MAX_LENGTH,
    REASON_MIN_LENGTH,
    REASON_PATTERN,
    GetCustomerRecordInput,
    TriggerRefundInput,
    validation_error_details,
)
from tests.mcp_server.edge_cases import (
    INVALID_AMOUNTS,
    INVALID_CUSTOMER_IDS,
    INVALID_REASONS,
    VALID_AMOUNTS,
    VALID_CUSTOMER_IDS,
    VALID_REASONS,
    Case,
)

VALID_REFUND: dict[str, Any] = {
    "customer_id": "CUST-00042",
    "amount": 25.0,
    "reason": "Duplicate charge on invoice INV-2041",
}


def decoded(cases: list[Case]) -> list[Any]:
    """Parametrize with the decoded JSON value of each case, keeping the case id."""
    return [pytest.param(json.loads(case.json), id=case.id) for case in cases]


def rejected_fields(exc: ValidationError) -> list[str]:
    return [detail["field"] for detail in validation_error_details(exc)]


# --- customer_id -------------------------------------------------------------------------------


@pytest.mark.parametrize("customer_id", decoded(VALID_CUSTOMER_IDS))
def test_customer_id_accepts_cust_plus_five_ascii_digits(customer_id: str) -> None:
    assert (
        GetCustomerRecordInput.model_validate({"customer_id": customer_id}).customer_id
        == customer_id
    )
    refund = TriggerRefundInput.model_validate({**VALID_REFUND, "customer_id": customer_id})
    assert refund.customer_id == customer_id


@pytest.mark.parametrize("customer_id", decoded(INVALID_CUSTOMER_IDS))
def test_customer_id_rejects_everything_else(customer_id: Any) -> None:
    with pytest.raises(ValidationError) as excinfo:
        GetCustomerRecordInput.model_validate({"customer_id": customer_id})
    assert rejected_fields(excinfo.value) == ["customer_id"]

    with pytest.raises(ValidationError) as excinfo:
        TriggerRefundInput.model_validate({**VALID_REFUND, "customer_id": customer_id})
    assert rejected_fields(excinfo.value) == ["customer_id"]


# --- amount ------------------------------------------------------------------------------------


@pytest.mark.parametrize("amount", decoded(VALID_AMOUNTS))
def test_amount_accepts_positive_finite_numbers(amount: float) -> None:
    refund = TriggerRefundInput.model_validate({**VALID_REFUND, "amount": amount})
    assert isinstance(refund.amount, float)
    assert refund.amount == amount


@pytest.mark.parametrize("amount", decoded(INVALID_AMOUNTS))
def test_amount_rejects_non_positive_non_finite_and_non_numeric_values(amount: Any) -> None:
    with pytest.raises(ValidationError) as excinfo:
        TriggerRefundInput.model_validate({**VALID_REFUND, "amount": amount})
    assert rejected_fields(excinfo.value) == ["amount"]


# --- reason ------------------------------------------------------------------------------------


@pytest.mark.parametrize("reason", decoded(VALID_REASONS))
def test_reason_accepts_ten_to_one_thousand_characters_without_outer_whitespace(
    reason: str,
) -> None:
    assert TriggerRefundInput.model_validate({**VALID_REFUND, "reason": reason}).reason == reason


@pytest.mark.parametrize("reason", decoded(INVALID_REASONS))
def test_reason_rejects_short_blank_padded_oversized_and_non_string_values(reason: Any) -> None:
    with pytest.raises(ValidationError) as excinfo:
        TriggerRefundInput.model_validate({**VALID_REFUND, "reason": reason})
    assert rejected_fields(excinfo.value) == ["reason"]


# --- object shape ------------------------------------------------------------------------------


def test_unknown_arguments_are_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        TriggerRefundInput.model_validate({**VALID_REFUND, "approved_by": "me"})
    assert validation_error_details(excinfo.value) == [
        {
            "field": "approved_by",
            "message": "Extra inputs are not permitted",
            "type": "extra_forbidden",
        }
    ]


@pytest.mark.parametrize("missing", ["customer_id", "amount", "reason"])
def test_every_argument_is_required(missing: str) -> None:
    arguments = {key: value for key, value in VALID_REFUND.items() if key != missing}
    with pytest.raises(ValidationError) as excinfo:
        TriggerRefundInput.model_validate(arguments)
    assert validation_error_details(excinfo.value) == [
        {"field": missing, "message": "Field required", "type": "missing"}
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(None, id="null"),
        pytest.param([], id="list"),
        pytest.param("CUST-00042", id="string"),
        pytest.param(42, id="integer"),
    ],
)
def test_arguments_must_be_an_object(arguments: Any) -> None:
    with pytest.raises(ValidationError) as excinfo:
        GetCustomerRecordInput.model_validate(arguments)
    assert rejected_fields(excinfo.value) == ["<root>"]


# --- error details never echo client input -----------------------------------------------------


def test_error_details_never_contain_submitted_values() -> None:
    secret = "123-45-6789"
    with pytest.raises(ValidationError) as excinfo:
        TriggerRefundInput.model_validate(
            {"customer_id": secret, "amount": secret, "reason": f" {secret} "}
        )
    details = validation_error_details(excinfo.value)
    rendered = json.dumps(details)  # also proves the details are JSON-serializable
    assert "6789" not in rendered
    assert {detail["field"] for detail in details} == {"customer_id", "amount", "reason"}
    assert all(isinstance(value, str) for detail in details for value in detail.values())


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("123-45-6789", id="ssn-like"),
        pytest.param("x" * 10_000, id="ten-kilobytes"),
        pytest.param("has space", id="space"),
        pytest.param("<script>", id="markup"),
        pytest.param("x" * 65, id="sixty-five-chars"),
    ],
)
def test_unsafe_unknown_key_names_are_redacted_in_error_details(key: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        TriggerRefundInput.model_validate({**VALID_REFUND, key: "x"})
    assert validation_error_details(excinfo.value) == [
        {
            "field": "<redacted>",
            "message": "Extra inputs are not permitted",
            "type": "extra_forbidden",
        }
    ]


# --- the published JSON Schema matches what is enforced ----------------------------------------


def test_refund_schema_publishes_every_constraint() -> None:
    schema = TriggerRefundInput.model_json_schema()
    properties = schema["properties"]

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["amount", "customer_id", "reason"]
    assert properties["customer_id"]["pattern"] == CUSTOMER_ID_PATTERN
    assert properties["amount"]["type"] == "number"
    assert properties["amount"]["exclusiveMinimum"] == 0
    assert properties["reason"]["minLength"] == REASON_MIN_LENGTH == 10
    assert properties["reason"]["maxLength"] == REASON_MAX_LENGTH == 1000
    assert properties["reason"]["pattern"] == REASON_PATTERN


def test_customer_record_schema_is_closed() -> None:
    schema = GetCustomerRecordInput.model_json_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["customer_id"]
    assert schema["properties"]["customer_id"]["pattern"] == CUSTOMER_ID_PATTERN

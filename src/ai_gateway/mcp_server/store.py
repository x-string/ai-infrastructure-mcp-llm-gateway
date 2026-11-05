"""In-memory customer data for the Task 1 MCP server. Every record is fictional."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime

from ai_gateway.mcp_server.schemas import CustomerRecord, RefundReceipt


class CustomerNotFoundError(LookupError):
    """A well-formed customer id that has no record."""


class CustomerStore:
    """Customer lookups and an append-only list of issued refunds."""

    def __init__(self, customers: Iterable[CustomerRecord]) -> None:
        self._customers = {customer.customer_id: customer for customer in customers}
        self.refunds: list[RefundReceipt] = []

    @classmethod
    def with_demo_data(cls) -> CustomerStore:
        return cls(DEMO_CUSTOMERS)

    def get(self, customer_id: str) -> CustomerRecord:
        try:
            return self._customers[customer_id]
        except KeyError:
            raise CustomerNotFoundError(f"No customer with id {customer_id}.") from None

    def record_refund(self, customer_id: str, amount: float, reason: str) -> RefundReceipt:
        self.get(customer_id)  # refunds are only issued to known customers
        receipt = RefundReceipt(
            refund_id=f"rf_{uuid.uuid4().hex[:16]}",
            customer_id=customer_id,
            amount=amount,
            reason=reason,
            status="accepted",
            created_at=datetime.now(UTC),
        )
        self.refunds.append(receipt)
        return receipt


DEMO_CUSTOMERS = (
    CustomerRecord(
        customer_id="CUST-00042",
        name="Ada Lovelace",
        email="ada.lovelace@example.com",
        tier="premium",
        status="active",
    ),
    CustomerRecord(
        customer_id="CUST-01815",
        name="Alan Turing",
        email="alan.turing@example.com",
        tier="standard",
        status="active",
    ),
    CustomerRecord(
        customer_id="CUST-12345",
        name="Grace Hopper",
        email="grace.hopper@example.com",
        tier="standard",
        status="suspended",
    ),
)

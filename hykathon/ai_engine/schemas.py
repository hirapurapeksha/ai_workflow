"""Validated data models and deterministic plan-safety checks."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

INVOICE_ID_PATTERN = re.compile(r"^INV\d+$")


class Intent(str, Enum):
    PAY_INVOICE = "PAY_INVOICE"
    CHECK_INVOICE_STATUS = "CHECK_INVOICE_STATUS"
    LIST_INVOICES = "LIST_INVOICES"
    UNKNOWN = "UNKNOWN"


class ExtractedEntities(BaseModel):
    """Entities confidently extracted from a user request."""

    invoice_id: str | None = None
    amount: float | None = None
    currency: str | None = None

    @field_validator("invoice_id")
    @classmethod
    def validate_invoice_id(cls, value: str | None) -> str | None:
        if value is not None and not INVOICE_ID_PATTERN.fullmatch(value):
            raise ValueError("invoice_id must match INV followed by digits")
        return value

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("amount must be greater than zero")
        return value


class RouterOutput(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)


class InvoiceAction(BaseModel):
    invoice_id: str
    amount: float = Field(gt=0)
    action: str

    @field_validator("invoice_id")
    @classmethod
    def validate_invoice_id(cls, value: str) -> str:
        if not INVOICE_ID_PATTERN.fullmatch(value):
            raise ValueError("invoice_id must match INV followed by digits")
        return value


class WorkerAction(str, Enum):
    FIND_INVOICE = "FIND_INVOICE"
    CHECK_STATUS = "CHECK_STATUS"
    VERIFY_ELIGIBILITY = "VERIFY_ELIGIBILITY"
    REQUEST_HUMAN_APPROVAL = "REQUEST_HUMAN_APPROVAL"
    EXECUTE_PAYMENT = "EXECUTE_PAYMENT"


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_number: int = Field(ge=1)
    action: WorkerAction
    description: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def validate_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        invoice_id = value.get("invoice_id")
        if invoice_id is not None and (
            not isinstance(invoice_id, str) or not INVOICE_ID_PATTERN.fullmatch(invoice_id)
        ):
            raise ValueError("params.invoice_id must match INV followed by digits")
        amount = value.get("amount")
        if amount is not None:
            try:
                parsed_amount = float(amount)
            except (TypeError, ValueError) as exc:
                raise ValueError("params.amount must be a positive number") from exc
            if parsed_amount <= 0:
                raise ValueError("params.amount must be greater than zero")
            value["amount"] = parsed_amount
        return value


class TaskPlan(BaseModel):
    request_summary: str
    steps: list[PlanStep]

    @model_validator(mode="after")
    def validate_sequence(self) -> "TaskPlan":
        numbers = [step.step_number for step in self.steps]
        if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
            raise ValueError("plan steps must have unique, ascending step_number values")

        actions = [step.action for step in self.steps]
        if WorkerAction.EXECUTE_PAYMENT in actions:
            required = [
                WorkerAction.FIND_INVOICE,
                WorkerAction.CHECK_STATUS,
                WorkerAction.VERIFY_ELIGIBILITY,
                WorkerAction.REQUEST_HUMAN_APPROVAL,
                WorkerAction.EXECUTE_PAYMENT,
            ]
            positions: list[int] = []
            for action in required:
                try:
                    positions.append(actions.index(action))
                except ValueError as exc:
                    raise ValueError(f"payment plan is missing required action {action.value}") from exc
            if positions != sorted(positions):
                raise ValueError(
                    "payment steps must follow the required safe order: "
                    "find, check, verify, approval, execute"
                )
        elif WorkerAction.REQUEST_HUMAN_APPROVAL in actions and (
            WorkerAction.VERIFY_ELIGIBILITY not in actions
            or actions.index(WorkerAction.VERIFY_ELIGIBILITY)
            > actions.index(WorkerAction.REQUEST_HUMAN_APPROVAL)
        ):
            raise ValueError("human approval must follow eligibility verification")
        return self


class CriticVerdict(BaseModel):
    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class BackendInvoice(BaseModel):
    invoice_id: str
    amount: float = Field(gt=0)
    currency: str
    status: str

    @field_validator("invoice_id")
    @classmethod
    def validate_invoice_id(cls, value: str) -> str:
        if not INVOICE_ID_PATTERN.fullmatch(value):
            raise ValueError("invoice_id must match INV followed by digits")
        return value


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


class ExecutionResult(BaseModel):
    status: ExecutionStatus
    message: str
    completed_steps: list[int] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)


def plan_rule_issues(plan: TaskPlan) -> list[str]:
    """Return deterministic business-rule issues, including payment sequence checks."""
    issues: list[str] = []
    actions = [step.action for step in plan.steps]
    if WorkerAction.EXECUTE_PAYMENT in actions:
        expected = [
            WorkerAction.FIND_INVOICE,
            WorkerAction.CHECK_STATUS,
            WorkerAction.VERIFY_ELIGIBILITY,
            WorkerAction.REQUEST_HUMAN_APPROVAL,
            WorkerAction.EXECUTE_PAYMENT,
        ]
        missing = [action.value for action in expected if action not in actions]
        if missing:
            issues.append(f"Payment plan is missing required actions: {', '.join(missing)}")
        elif [actions.index(action) for action in expected] != sorted(
            actions.index(action) for action in expected
        ):
            issues.append("Payment actions are not in the required safe order.")
    return issues


def cross_check_plan_amount(plan_amount: float, invoice: BackendInvoice) -> bool:
    """Check that the requested amount equals the amount recorded on the invoice."""
    return abs(plan_amount - invoice.amount) < 0.005

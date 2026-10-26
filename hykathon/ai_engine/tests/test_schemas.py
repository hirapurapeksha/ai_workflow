import pytest
from pydantic import ValidationError

from ai_engine.schemas import (
    PlanStep,
    TaskPlan,
    WorkerAction,
)


def test_invoice_action_rejects_negative_amount_and_invalid_id():
    from ai_engine.schemas import InvoiceAction

    with pytest.raises(ValidationError):
        InvoiceAction(invoice_id="INV101", amount=-1, action="pay")
    with pytest.raises(ValidationError):
        InvoiceAction(invoice_id="bad-id", amount=1, action="pay")


def test_payment_plan_requires_human_approval_before_execute():
    steps = [
        PlanStep(step_number=1, action=WorkerAction.FIND_INVOICE, description="Find"),
        PlanStep(step_number=2, action=WorkerAction.CHECK_STATUS, description="Check"),
        PlanStep(step_number=3, action=WorkerAction.VERIFY_ELIGIBILITY, description="Verify"),
        PlanStep(step_number=4, action=WorkerAction.EXECUTE_PAYMENT, description="Pay"),
    ]
    with pytest.raises(ValidationError, match="missing required action REQUEST_HUMAN_APPROVAL"):
        TaskPlan(request_summary="Pay", steps=steps)


def test_plan_step_rejects_unknown_action():
    with pytest.raises(ValidationError):
        PlanStep(step_number=1, action="UNKNOWN_ACTION", description="invalid")


def test_router_output_rejects_negative_amount():
    from ai_engine.schemas import Intent, RouterOutput

    with pytest.raises(ValidationError):
        RouterOutput(
            intent=Intent.PAY_INVOICE,
            confidence=1,
            entities={"invoice_id": "INV101", "amount": -1, "currency": "INR"},
        )


def test_payment_steps_must_be_in_safe_order():
    steps = [
        PlanStep(step_number=1, action=WorkerAction.REQUEST_HUMAN_APPROVAL, description="Approve"),
        PlanStep(step_number=2, action=WorkerAction.FIND_INVOICE, description="Find"),
        PlanStep(step_number=3, action=WorkerAction.CHECK_STATUS, description="Check"),
        PlanStep(step_number=4, action=WorkerAction.VERIFY_ELIGIBILITY, description="Verify"),
        PlanStep(step_number=5, action=WorkerAction.EXECUTE_PAYMENT, description="Pay"),
    ]
    with pytest.raises(ValidationError, match="required safe order"):
        TaskPlan(request_summary="Pay", steps=steps)

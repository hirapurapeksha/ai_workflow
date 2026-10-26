from ai_engine.schemas import Intent, RouterOutput
from ai_engine.planner import create_plan
from ai_engine.workers import MockBackendClient, execute_payment, run_plan


def _plan(amount=5000):
    return create_plan(
        f"Pay INV101 for {amount}",
        RouterOutput(
            intent=Intent.PAY_INVOICE,
            confidence=1,
            entities={"invoice_id": "INV101", "amount": amount, "currency": "INR"},
        ),
    )


def test_payment_is_blocked_without_human_approval():
    backend = MockBackendClient()
    state = {"invoice_id": "INV101", "amount": 5000}
    result = execute_payment({"invoice_id": "INV101", "amount": 5000}, state, backend)
    assert result["waiting"] is True
    assert backend.invoices["INV101"]["status"] == "unpaid"


def test_run_plan_waits_for_human_approval():
    backend = MockBackendClient()
    result = run_plan(_plan(), backend=backend)
    assert result.status.value == "awaiting_approval"
    assert result.state["approval_status"] == "PENDING"
    backend.approve(result.state["approval_id"])
    resumed = run_plan(_plan(), backend=backend, state=result.state)
    assert resumed.status.value == "completed"
    assert backend.invoices["INV101"]["status"] == "paid"


def test_worker_blocks_amount_mismatch():
    result = run_plan(_plan(4000), backend=MockBackendClient())
    assert result.status.value == "failed"
    assert "Amount mismatch" in result.message


def test_worker_blocks_already_paid_invoice():
    plan = create_plan(
        "Pay INV102 for 2500",
        RouterOutput(
            intent=Intent.PAY_INVOICE,
            confidence=1,
            entities={"invoice_id": "INV102", "amount": 2500, "currency": "INR"},
        ),
    )
    result = run_plan(plan, backend=MockBackendClient())
    assert result.status.value == "failed"
    assert "not unpaid" in result.message

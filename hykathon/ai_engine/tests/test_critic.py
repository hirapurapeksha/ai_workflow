import pytest
from pydantic import ValidationError

from ai_engine import critic
from ai_engine.schemas import (
    CriticVerdict,
    Intent,
    RouterOutput,
    TaskPlan,
    WorkerAction,
)


def _router() -> RouterOutput:
    return RouterOutput(
        intent=Intent.PAY_INVOICE,
        confidence=0.99,
        entities={"invoice_id": "INV101", "amount": 5000, "currency": "INR"},
    )


def test_critic_loop_retries_then_accepts(monkeypatch):
    verdicts = iter(
        [
            CriticVerdict(is_valid=False, issues=["Add verification."], suggestions=[]),
            CriticVerdict(is_valid=True, issues=[], suggestions=[]),
        ]
    )
    monkeypatch.setattr(critic, "critique", lambda request, plan: next(verdicts))
    plans = []

    def planner(request, output, feedback=None):
        plans.append(feedback)
        return TaskPlan(
            request_summary="Pay INV101",
            steps=[
                {"step_number": 1, "action": WorkerAction.FIND_INVOICE, "description": "find"},
                {"step_number": 2, "action": WorkerAction.CHECK_STATUS, "description": "check"},
                {"step_number": 3, "action": WorkerAction.VERIFY_ELIGIBILITY, "description": "verify"},
                {"step_number": 4, "action": WorkerAction.REQUEST_HUMAN_APPROVAL, "description": "approve"},
                {"step_number": 5, "action": WorkerAction.EXECUTE_PAYMENT, "description": "execute"},
            ],
        )

    monkeypatch.setattr(critic, "create_plan", planner)
    plan, history = critic.plan_with_critic("Pay INV101 for 5000", _router())
    assert plan.request_summary == "Pay INV101"
    assert len(history) == 2
    assert plans[1] == ["Add verification."]


def test_critic_loop_stops_at_retry_limit(monkeypatch):
    monkeypatch.setattr(
        critic,
        "create_plan",
        lambda *args, **kwargs: TaskPlan(request_summary="no steps", steps=[]),
    )
    monkeypatch.setattr(
        critic,
        "critique",
        lambda *args: CriticVerdict(is_valid=False, issues=["still invalid"], suggestions=[]),
    )
    monkeypatch.setattr(critic, "MAX_CRITIC_RETRIES", 1)
    with pytest.raises(critic.PlanValidationError, match="after 2 iteration"):
        critic.plan_with_critic("request", _router())


def test_critic_rule_checks_payment_steps():
    # A TaskPlan with an unsafe payment order is rejected by Pydantic before use.
    with pytest.raises(
        ValidationError, match="missing required action REQUEST_HUMAN_APPROVAL"
    ):
        TaskPlan(
            request_summary="invalid",
            steps=[
                {"step_number": 1, "action": WorkerAction.FIND_INVOICE, "description": "find"},
                {"step_number": 2, "action": WorkerAction.CHECK_STATUS, "description": "check"},
                {
                    "step_number": 3,
                    "action": WorkerAction.VERIFY_ELIGIBILITY,
                    "description": "verify",
                },
                {
                    "step_number": 4,
                    "action": WorkerAction.EXECUTE_PAYMENT,
                    "description": "pay",
                },
            ],
        )

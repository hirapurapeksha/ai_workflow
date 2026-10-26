"""Workers that execute approved plan steps through the backend API."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

try:
    from .schemas import (
        BackendInvoice,
        ExecutionResult,
        ExecutionStatus,
        PlanStep,
        TaskPlan,
        WorkerAction,
        cross_check_plan_amount,
    )
except ImportError:
    from schemas import (
        BackendInvoice,
        ExecutionResult,
        ExecutionStatus,
        PlanStep,
        TaskPlan,
        WorkerAction,
        cross_check_plan_amount,
    )

load_dotenv()
logger = logging.getLogger(__name__)
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")


class BackendError(RuntimeError):
    """Raised when the payment backend returns an invalid or unsuccessful response."""


class BackendClient:
    """HTTP wrapper for Member 1's invoice and payment endpoints."""

    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self.base_url = (base_url or BACKEND_BASE_URL).rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.http.request(method, path, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendError(f"Backend request {method} {path} failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise BackendError(f"Backend response for {path} must be a JSON object.")
        return payload

    def get_invoice(self, invoice_id: str) -> BackendInvoice:
        payload = self._request("GET", f"/invoices/{invoice_id}")
        try:
            return BackendInvoice.model_validate(payload)
        except ValidationError as exc:
            raise BackendError(f"Invalid invoice response: {exc}") from exc

    def request_approval(self, invoice_id: str, amount: float) -> dict[str, Any]:
        return self._request(
            "POST",
            "/payments/approval-request",
            json={"invoice_id": invoice_id, "amount": amount},
        )

    def get_approval_status(self, approval_id: str) -> dict[str, Any]:
        return self._request("GET", f"/payments/approval/{quote(approval_id, safe='')}")

    def execute_payment(self, invoice_id: str, amount: float, approval_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/payments/execute",
            json={
                "invoice_id": invoice_id,
                "amount": amount,
                "approval_id": approval_id,
            },
        )

    def close(self) -> None:
        self.http.close()


class MockBackendClient:
    """In-memory backend for local demos and tests; approval starts as pending."""

    def __init__(self) -> None:
        self.invoices: dict[str, dict[str, Any]] = {
            "INV101": {
                "invoice_id": "INV101",
                "amount": 5000.0,
                "currency": "INR",
                "status": "unpaid",
            },
            "INV102": {
                "invoice_id": "INV102",
                "amount": 2500.0,
                "currency": "INR",
                "status": "paid",
            },
            "INV103": {
                "invoice_id": "INV103",
                "amount": 1250.0,
                "currency": "INR",
                "status": "unpaid",
            },
        }
        self.approvals: dict[str, dict[str, Any]] = {}
        self._next_approval_id = 1

    def get_invoice(self, invoice_id: str) -> BackendInvoice:
        raw = self.invoices.get(invoice_id)
        if raw is None:
            raise BackendError(f"Invoice {invoice_id} was not found.")
        return BackendInvoice.model_validate(raw)

    def request_approval(self, invoice_id: str, amount: float) -> dict[str, Any]:
        approval_id = f"APR{self._next_approval_id}"
        self._next_approval_id += 1
        approval = {
            "approval_id": approval_id,
            "invoice_id": invoice_id,
            "amount": amount,
            "status": "PENDING",
        }
        self.approvals[approval_id] = approval
        return approval.copy()

    def get_approval_status(self, approval_id: str) -> dict[str, Any]:
        try:
            return self.approvals[approval_id].copy()
        except KeyError as exc:
            raise BackendError(f"Approval {approval_id} was not found.") from exc

    def approve(self, approval_id: str) -> None:
        """Simulate a human approval action; application workers never call this."""
        try:
            self.approvals[approval_id]["status"] = "APPROVED"
        except KeyError as exc:
            raise BackendError(f"Approval {approval_id} was not found.") from exc

    def execute_payment(self, invoice_id: str, amount: float, approval_id: str) -> dict[str, Any]:
        approval = self.get_approval_status(approval_id)
        if approval.get("status") != "APPROVED":
            raise BackendError("Backend refused payment: approval is not APPROVED.")
        invoice = self.get_invoice(invoice_id)
        if invoice.status.lower() != "unpaid" or not cross_check_plan_amount(amount, invoice):
            raise BackendError("Backend refused payment: invoice status or amount is invalid.")
        self.invoices[invoice_id]["status"] = "paid"
        return {"status": "COMPLETED", "invoice_id": invoice_id, "amount": amount}

    def close(self) -> None:
        return None


def _make_backend() -> BackendClient | MockBackendClient:
    if os.getenv("USE_MOCK_BACKEND", "false").lower() == "true":
        return MockBackendClient()
    return BackendClient()


def _invoice_id(params: dict[str, Any], state: dict[str, Any]) -> str:
    invoice_id = params.get("invoice_id") or state.get("invoice_id")
    if not invoice_id:
        raise BackendError("Plan step is missing invoice_id.")
    return str(invoice_id)


def find_invoice(
    params: dict[str, Any], state: dict[str, Any], backend: BackendClient | MockBackendClient
) -> dict[str, Any]:
    invoice = backend.get_invoice(_invoice_id(params, state))
    state["invoice_id"] = invoice.invoice_id
    state["invoice"] = invoice.model_dump()
    return {"message": f"Found invoice {invoice.invoice_id}.", "data": invoice.model_dump()}


def check_status(
    params: dict[str, Any], state: dict[str, Any], backend: BackendClient | MockBackendClient
) -> dict[str, Any]:
    invoice_id = _invoice_id(params, state)
    invoice_data = state.get("invoice")
    invoice = (
        BackendInvoice.model_validate(invoice_data)
        if invoice_data
        else backend.get_invoice(invoice_id)
    )
    state["invoice"] = invoice.model_dump()
    state["invoice_id"] = invoice.invoice_id
    return {
        "message": f"Invoice status is {invoice.status}; recorded amount is {invoice.amount:g} {invoice.currency}.",
        "data": {"status": invoice.status, "amount": invoice.amount, "currency": invoice.currency},
    }


def verify_eligibility(
    params: dict[str, Any], state: dict[str, Any], backend: BackendClient | MockBackendClient
) -> dict[str, Any]:
    invoice_id = _invoice_id(params, state)
    invoice_data = state.get("invoice")
    invoice = (
        BackendInvoice.model_validate(invoice_data)
        if invoice_data
        else backend.get_invoice(invoice_id)
    )
    state["invoice"] = invoice.model_dump()
    requested_amount = params.get("amount")
    if requested_amount is None:
        raise BackendError("Cannot verify eligibility: request amount is missing.")
    if invoice.status.lower() != "unpaid":
        raise BackendError(f"Invoice {invoice_id} is not unpaid (status: {invoice.status}).")
    if not cross_check_plan_amount(float(requested_amount), invoice):
        raise BackendError(
            f"Amount mismatch: requested {float(requested_amount):g}, "
            f"invoice amount is {invoice.amount:g}."
        )
    state["eligible"] = True
    state["amount"] = float(requested_amount)
    return {"message": "Invoice is unpaid and the amount matches.", "data": {"eligible": True}}


def request_human_approval(
    params: dict[str, Any], state: dict[str, Any], backend: BackendClient | MockBackendClient
) -> dict[str, Any]:
    if not state.get("eligible"):
        raise BackendError("Approval cannot be requested before eligibility is verified.")
    invoice_id = _invoice_id(params, state)
    amount = float(params.get("amount", state.get("amount", 0)))
    approval_id = state.get("approval_id")
    approval = (
        backend.get_approval_status(approval_id)
        if approval_id
        else backend.request_approval(invoice_id, amount)
    )
    approval_id = approval.get("approval_id") or approval_id
    if not approval_id:
        raise BackendError("Backend approval response is missing approval_id.")
    state["approval_id"] = str(approval_id)
    state["approval_status"] = str(approval.get("status", "PENDING")).upper()
    if state["approval_status"] != "APPROVED":
        return {
            "message": f"Awaiting human approval (approval ID: {approval_id}).",
            "waiting": True,
            "data": {"approval_id": approval_id, "status": state["approval_status"]},
        }
    return {
        "message": "Human approval is confirmed.",
        "data": {"approval_id": approval_id, "status": "APPROVED"},
    }


def execute_payment(
    params: dict[str, Any], state: dict[str, Any], backend: BackendClient | MockBackendClient
) -> dict[str, Any]:
    approval_id = state.get("approval_id")
    if not approval_id:
        return {"message": "Awaiting human approval; payment was not executed.", "waiting": True}
    approval = backend.get_approval_status(str(approval_id))
    status = str(approval.get("status", "")).upper()
    state["approval_status"] = status
    if status != "APPROVED":
        return {"message": "Awaiting human approval; payment was not executed.", "waiting": True}
    invoice_id = _invoice_id(params, state)
    amount = float(params.get("amount", state.get("amount", 0)))
    result = backend.execute_payment(invoice_id, amount, str(approval_id))
    state["payment"] = result
    return {"message": "Payment executed by the backend.", "data": result}


_WORKERS = {
    WorkerAction.FIND_INVOICE: find_invoice,
    WorkerAction.CHECK_STATUS: check_status,
    WorkerAction.VERIFY_ELIGIBILITY: verify_eligibility,
    WorkerAction.REQUEST_HUMAN_APPROVAL: request_human_approval,
    WorkerAction.EXECUTE_PAYMENT: execute_payment,
}


def run_plan(
    plan: TaskPlan,
    backend: BackendClient | MockBackendClient | None = None,
    state: dict[str, Any] | None = None,
) -> ExecutionResult:
    """Execute plan steps in order and stop on failure or pending human approval."""
    owns_client = backend is None
    client = backend or _make_backend()
    shared_state = state if state is not None else {}
    completed_steps: list[int] = []
    try:
        for step in plan.steps:
            worker = _WORKERS[step.action]
            outcome = worker(step.params, shared_state, client)
            if outcome.get("waiting"):
                return ExecutionResult(
                    status=ExecutionStatus.AWAITING_APPROVAL,
                    message=str(outcome["message"]),
                    completed_steps=completed_steps,
                    state=shared_state.copy(),
                )
            completed_steps.append(step.step_number)
            logger.info("Completed plan step %s (%s)", step.step_number, step.action.value)
        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            message="Plan completed.",
            completed_steps=completed_steps,
            state=shared_state.copy(),
        )
    except (BackendError, ValidationError, ValueError, KeyError) as exc:
        logger.error("Plan stopped at step %d: %s", len(completed_steps) + 1, exc)
        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            message=str(exc),
            completed_steps=completed_steps,
            state=shared_state.copy(),
        )
    finally:
        if owns_client:
            client.close()

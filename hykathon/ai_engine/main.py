"""Command-line entry point for the invoice AI workflow."""

from __future__ import annotations

import argparse
import logging
from typing import Iterable

try:
    from .critic import PlanValidationError, plan_with_critic
    from .router import route
    from .schemas import Intent, RouterOutput
    from .workers import MockBackendClient, run_plan
except ImportError:
    from critic import PlanValidationError, plan_with_critic
    from router import route
    from schemas import Intent, RouterOutput
    from workers import MockBackendClient, run_plan

logger = logging.getLogger(__name__)


def run_request(user_request: str, backend: MockBackendClient | None = None) -> str:
    """Run the complete request workflow and format its result for the CLI."""
    intent = route(user_request)
    lines = [f"Intent: {intent.intent.value} (confidence {intent.confidence:.2f})"]
    if intent.intent == Intent.UNKNOWN:
        lines.append("Execution: not run; the request is unknown, unrelated, or unsafe.")
        return "\n".join(lines)

    try:
        plan, verdicts = plan_with_critic(user_request, intent)
    except PlanValidationError as exc:
        lines.append(f"Planning failed: {exc}")
        return "\n".join(lines)
    lines.append("Plan:")
    lines.extend(
        f"  {step.step_number}. {step.action.value}: {step.description}" for step in plan.steps
    )
    lines.append(f"Critic iterations: {len(verdicts)}")
    for index, verdict in enumerate(verdicts, start=1):
        lines.append(
            f"  {index}. {'accepted' if verdict.is_valid else 'rejected'}"
            + (f" — {'; '.join(verdict.issues)}" if verdict.issues else "")
        )
    result = run_plan(plan, backend=backend)
    lines.append(f"Execution: {result.status.value} — {result.message}")
    return "\n".join(lines)


def _demo(backend: MockBackendClient) -> Iterable[str]:
    return (
        "Pay invoice INV101 for ₹5,000.",
        "Pay invoice INV101 for -₹5,000.",
        "Pay invoice INV999 for ₹500.",
        "Pay invoice INV101 for ₹4,000.",
        "Pay invoice INV102 for ₹2,500.",
        "Tell me a joke about the weather.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoice payment AI workflow")
    parser.add_argument("request", nargs="?", help="Natural-language invoice request")
    parser.add_argument("--demo", action="store_true", help="Run several sample requests")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.demo:
        backend = MockBackendClient()
        for request in _demo(backend):
            print(f"\nRequest: {request}")
            try:
                print(run_request(request, backend))
            except Exception as exc:
                logger.error("Request could not be processed: %s", exc)
                print(f"Execution: failed — {exc}")
    elif args.request:
        try:
            print(run_request(args.request))
        except Exception as exc:
            logger.error("Request could not be processed: %s", exc)
            raise SystemExit(1) from exc
    else:
        parser.error("provide a request or use --demo")


if __name__ == "__main__":
    main()

# Invoice AI Engine

`ai_engine/` contains the natural-language routing, safe plan creation, critic loop,
and backend worker layer for the invoice payment workflow. The AI can prepare a
payment request, but it cannot approve one. The payment backend must independently
enforce approval before execution.

## Setup

1. Create an API key in [Google AI Studio](https://aistudio.google.com/app/apikey).
2. From this directory, create `.env` by copying `.env.example`, then set
   `GEMINI_API_KEY`. Never commit `.env`.
3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Use the mock backend for local runs by setting `USE_MOCK_BACKEND=true` in `.env`.
   The CLI demo also uses an in-memory mock backend. Run one request:

   ```powershell
   python main.py "Pay invoice INV101 for ₹5,000."
   ```

   Run sample requests:

   ```powershell
   python main.py --demo
   ```

   The demo calls Gemini for routing, planning review, and critique, so a valid API
   key is required even when the mock backend is enabled.

5. Run tests from the repository root:

   ```powershell
   python -m pytest ai_engine/tests
   ```

## Architecture

```text
User request
    |
    v
Router (Gemini JSON -> Pydantic) -- unsafe/unknown --> Stop
    |
    v
Planner (payment sequence enforced in code)
    |
    v
Critic <---- revise with feedback (up to MAX_CRITIC_RETRIES + 1 reviews)
    |
    v
Workers -- invoice/status/eligibility --> human approval request
    |                                        |
    |                                  pending: stop
    |                                  approved: resume
    v
Backend payment endpoint (must enforce approval itself)
```

The critic only evaluates plans. `REQUEST_HUMAN_APPROVAL` creates or checks an
approval request and stops while it is pending. Resume `run_plan` with the same
state and backend after a human has approved; only then can the execution worker
call the backend payment endpoint.

## Backend contract to confirm with Member 1

The HTTP client assumes these placeholder routes and response formats:

- `GET /invoices/{invoice_id}` returns an object with `invoice_id`, positive
  numeric `amount`, `currency`, and `status` (for example `"unpaid"` / `"paid"`).
- `POST /payments/approval-request` accepts `invoice_id` and `amount`, and
  returns `approval_id` plus `status` (`"PENDING"` or `"APPROVED"`).
- `GET /payments/approval/{approval_id}` returns at least `status`.
- `POST /payments/execute` accepts `invoice_id`, `amount`, and `approval_id`;
  the backend must verify approval and invoice eligibility atomically and return
  a JSON object describing the result.

Confirm authentication, error response shapes, status spelling/casing, amount
precision/currency rules, and whether the execute endpoint requires an idempotency
key. No API credentials are stored in source code.

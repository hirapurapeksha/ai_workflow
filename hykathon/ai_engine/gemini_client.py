"""Small Gemini client wrapper with validated structured responses."""

from __future__ import annotations

import logging
import os
import time
from typing import TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
MODEL = "gemini-2.5-flash"
T = TypeVar("T", bound=BaseModel)


class GeminiClientError(RuntimeError):
    """Raised when Gemini cannot return a usable response."""


load_dotenv()


def _client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiClientError(
            "GEMINI_API_KEY is missing. Add it to ai_engine/.env before using Gemini."
        )
    try:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30_000),
        )
    except Exception as exc:
        raise GeminiClientError(f"Could not initialize the Gemini client: {exc}") from exc


def _generate(prompt: str, system: str | None = None, schema: type[BaseModel] | None = None) -> str:
    config_values: dict[str, object] = {}
    if system:
        config_values["system_instruction"] = system
    if schema is not None:
        config_values["response_mime_type"] = "application/json"
        config_values["response_schema"] = schema
    try:
        response = _client().models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(**config_values),
        )
    except Exception as exc:
        raise GeminiClientError(f"Gemini request failed: {exc}") from exc
    text = response.text
    if not text:
        if schema is not None:
            return ""
        raise GeminiClientError("Gemini returned an empty response.")
    return text


def generate_text(prompt: str, system: str | None = None) -> str:
    """Generate plain text using the configured Gemini model."""
    return _generate(prompt, system=system)


def generate_structured(
    prompt: str,
    schema: type[T],
    system: str | None = None,
) -> T:
    """Generate JSON and validate it; retry malformed or invalid output twice."""
    last_error: Exception | None = None
    retry_prompt = prompt
    for attempt in range(3):
        try:
            return schema.model_validate_json(
                _generate(retry_prompt, system=system, schema=schema)
            )
        except GeminiClientError as exc:
            # API failures are not repaired by asking the model to repeat itself.
            raise
        except (ValidationError, ValueError, TypeError) as exc:
            last_error = exc
            logger.warning("Invalid Gemini JSON response (attempt %d/3): %s", attempt + 1, exc)
            if attempt < 2:
                retry_prompt = (
                    f"{prompt}\n\nYour previous response failed validation: {exc}. "
                    "Return corrected JSON that strictly matches the requested schema."
                )
                time.sleep(0.2 * (attempt + 1))
    raise GeminiClientError(
        f"Gemini did not return valid {schema.__name__} JSON after 3 attempts: {last_error}"
    ) from last_error

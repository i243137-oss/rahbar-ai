"""Safe fallback service for validation and provider failures (section 3.4 & 3.7).

Guarantees the user receives a safe, schema-compliant response rather than an uncaught HTTP 500.
"""

from __future__ import annotations

import logging

from ..schemas import Answer

log = logging.getLogger("rahbar.services.fallback")

SAFE_FALLBACK_MESSAGE = (
    "An error occurred while generating a response from official documents. "
    "Please try rephrasing your question or try again shortly."
)


def build_fallback_answer(error_context: str) -> Answer:
    """Return a canned safe fallback Answer when validation/retry fails."""
    log.warning("safe fallback executed", extra={"context": error_context})
    return Answer(
        reasoning=f"FALLBACK executed due to error: {error_context}",
        answer=SAFE_FALLBACK_MESSAGE,
        citations=[],
        answered_from_context=False,
        confidence="low",
    )

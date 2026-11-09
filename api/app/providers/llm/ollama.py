"""Ollama provider: ollama-host or ollama-incluster (section 3.5).

Connects to an Ollama instance, uses structured output enforcement,
handles timeouts and connection errors, and validates response with Pydantic.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from pydantic import ValidationError

from ...config import Settings
from ...schemas import Answer, Passage
from .prompt import SYSTEM_PROMPT, build_user_prompt

log = logging.getLogger("rahbar.provider.ollama")


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ollama_url = settings.ollama_url.rstrip("/")
        self.model = settings.llm_model
        self.temperature = settings.temperature
        self.timeout = settings.llm_timeout

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        """Call Ollama /api/chat with structured format and return validated Answer."""
        user_prompt = build_user_prompt(question, passages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": Answer.model_json_schema(),
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.ollama_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")

                if not content:
                    raise ValueError("Ollama returned empty response content")

                answer_obj = Answer.model_validate_json(content)
                return answer_obj

        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            ValidationError,
            ValueError,
            json.JSONDecodeError,
        ) as e:
            log.warning(
                "ollama generation failure",
                extra={
                    "error": str(e),
                    "model": self.model,
                    "url": self.ollama_url,
                },
            )
            raise

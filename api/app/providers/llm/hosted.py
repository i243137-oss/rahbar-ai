"""Hosted free-tier provider (Groq, Google AI Studio, OpenRouter, OpenAI-compatible).

The API key comes from LLM_API_KEY (from a Kubernetes Secret created from CLI).
It is NEVER logged and NEVER committed.
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

log = logging.getLogger("rahbar.provider.hosted")


class HostedProvider:
    name = "hosted"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.hosted_base_url.rstrip("/")
        self.model = settings.llm_model
        self.api_key = settings.llm_api_key
        self.temperature = settings.temperature
        self.timeout = settings.llm_timeout

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        """Call OpenAI-compatible /chat/completions and return validated Answer."""
        if not self.base_url:
            raise ValueError("HOSTED_BASE_URL is not configured for hosted provider")
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured for hosted provider")

        user_prompt = build_user_prompt(question, passages)
        endpoint = (
            f"{self.base_url}/chat/completions"
            if not self.base_url.endswith("/chat/completions")
            else self.base_url
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

                choices = data.get("choices", [])
                if not choices:
                    raise ValueError("Hosted provider returned no choices")

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError("Hosted provider returned empty content")

                answer_obj = Answer.model_validate_json(content)
                return answer_obj

        except (
            httpx.RequestError,
            httpx.HTTPStatusError,
            ValidationError,
            ValueError,
            json.JSONDecodeError,
        ) as e:
            # Sensitive: Never log the API key or authorization header
            log.warning(
                "hosted generation failure",
                extra={
                    "error": str(e),
                    "model": self.model,
                    "endpoint": endpoint,
                },
            )
            raise

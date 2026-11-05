"""Tests for LLM providers (Phase 1): Ollama, Hosted, and Stub providers.

All tests use mocked HTTP responses; no network or live model is ever required.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.providers.llm import get_provider
from app.providers.llm.hosted import HostedProvider
from app.providers.llm.ollama import OllamaProvider
from app.providers.llm.stub import StubProvider
from app.schemas import Answer, Citation, Passage


@pytest.fixture
def sample_passages() -> list[Passage]:
    return [
        Passage(
            chunk_id="chunk-001",
            text="Students on academic warning have CGPA below 2.0.",
            source="handbook_2026.pdf",
            section="4.2 Academic Standing",
            page=12,
            score=0.92,
        ),
        Passage(
            chunk_id="chunk-002",
            text="A student on warning may register for at most 12 credit hours.",
            source="handbook_2026.pdf",
            section="4.3 Course Load",
            page=13,
            score=0.88,
        ),
    ]


@pytest.fixture
def valid_answer_dict() -> dict[str, object]:
    return {
        "reasoning": "Checked academic standing clause 4.2 in handbook.",
        "answer": "A student with a CGPA below 2.0 is placed on academic warning.",
        "citations": [
            {"source": "handbook_2026.pdf", "section": "4.2 Academic Standing", "page": 12}
        ],
        "answered_from_context": True,
        "confidence": "high",
    }


# ==============================================================================
# Provider Factory Tests
# ==============================================================================


def test_provider_factory_selection() -> None:
    s_stub = Settings(llm_provider="stub")
    assert isinstance(get_provider(s_stub), StubProvider)

    s_host = Settings(llm_provider="ollama-host")
    assert isinstance(get_provider(s_host), OllamaProvider)

    s_incluster = Settings(llm_provider="ollama-incluster")
    assert isinstance(get_provider(s_incluster), OllamaProvider)

    s_hosted = Settings(llm_provider="hosted", hosted_base_url="https://api.example.com", llm_api_key="key")
    assert isinstance(get_provider(s_hosted), HostedProvider)

    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        get_provider(Settings(llm_provider="invalid-provider"))


# ==============================================================================
# Ollama Provider Tests
# ==============================================================================


def test_ollama_generate_valid(sample_passages: list[Passage], valid_answer_dict: dict[str, object]) -> None:
    settings = Settings(
        llm_provider="ollama-host",
        ollama_url="http://localhost:11434",
        llm_model="llama3.2:3b",
        temperature=0.2,
    )
    provider = OllamaProvider(settings)

    mock_resp = httpx.Response(
        200,
        json={"message": {"content": json.dumps(valid_answer_dict)}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp) as mock_post:
        ans = provider.generate("What is academic warning?", sample_passages)
        assert isinstance(ans, Answer)
        assert ans.answered_from_context is True
        assert ans.confidence == "high"
        assert len(ans.citations) == 1
        assert ans.citations[0].source == "handbook_2026.pdf"
        assert ans.citations[0].page == 12

        # Verify call payload
        mock_post.assert_called_once()
        sent_payload = mock_post.call_args[1]["json"]
        assert sent_payload["model"] == "llama3.2:3b"
        assert "format" in sent_payload
        assert sent_payload["options"]["temperature"] == 0.2


def test_ollama_generate_empty_response(sample_passages: list[Passage]) -> None:
    settings = Settings(ollama_url="http://localhost:11434")
    provider = OllamaProvider(settings)

    mock_resp = httpx.Response(
        200,
        json={"message": {"content": ""}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(ValueError, match="empty response content"):
            provider.generate("test question", sample_passages)


def test_ollama_generate_invalid_json(sample_passages: list[Passage]) -> None:
    settings = Settings(ollama_url="http://localhost:11434")
    provider = OllamaProvider(settings)

    mock_resp = httpx.Response(
        200,
        json={"message": {"content": "This is plain text, not valid JSON."}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(ValidationError):
            provider.generate("test question", sample_passages)


def test_ollama_generate_missing_fields(sample_passages: list[Passage]) -> None:
    settings = Settings(ollama_url="http://localhost:11434")
    provider = OllamaProvider(settings)

    # Missing confidence and answered_from_context
    incomplete = {"answer": "Some answer", "citations": []}
    mock_resp = httpx.Response(
        200,
        json={"message": {"content": json.dumps(incomplete)}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(ValidationError):
            provider.generate("test question", sample_passages)


def test_ollama_generate_http_error(sample_passages: list[Passage]) -> None:
    settings = Settings(ollama_url="http://localhost:11434")
    provider = OllamaProvider(settings)

    mock_resp = httpx.Response(
        500,
        text="Internal Server Error",
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(httpx.HTTPStatusError):
            provider.generate("test question", sample_passages)


def test_ollama_generate_timeout(sample_passages: list[Passage]) -> None:
    settings = Settings(ollama_url="http://localhost:11434")
    provider = OllamaProvider(settings)

    with patch.object(httpx.Client, "post", side_effect=httpx.TimeoutException("Timeout")):
        with pytest.raises(httpx.TimeoutException):
            provider.generate("test question", sample_passages)


def test_ollama_generate_connect_error(sample_passages: list[Passage]) -> None:
    settings = Settings(ollama_url="http://localhost:11434")
    provider = OllamaProvider(settings)

    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("Failed to connect")):
        with pytest.raises(httpx.ConnectError):
            provider.generate("test question", sample_passages)


# ==============================================================================
# Hosted Provider Tests
# ==============================================================================


def test_hosted_missing_config() -> None:
    with pytest.raises(ValueError, match="HOSTED_BASE_URL"):
        HostedProvider(Settings(hosted_base_url="", llm_api_key="key")).generate("q", [])

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        HostedProvider(Settings(hosted_base_url="https://api.openai.com/v1", llm_api_key="")).generate("q", [])


def test_hosted_generate_valid(sample_passages: list[Passage], valid_answer_dict: dict[str, object]) -> None:
    settings = Settings(
        llm_provider="hosted",
        hosted_base_url="https://api.groq.com/openai/v1",
        llm_model="llama-3.3-70b-versatile",
        llm_api_key="test-secret-key-12345",
        temperature=0.1,
    )
    provider = HostedProvider(settings)

    mock_resp = httpx.Response(
        200,
        json={"choices": [{"message": {"content": json.dumps(valid_answer_dict)}}]},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp) as mock_post:
        ans = provider.generate("What is academic warning?", sample_passages)
        assert isinstance(ans, Answer)
        assert ans.answered_from_context is True
        assert ans.confidence == "high"

        mock_post.assert_called_once()
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-secret-key-12345"
        payload = mock_post.call_args[1]["json"]
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["temperature"] == 0.1


def test_hosted_generate_http_401(sample_passages: list[Passage]) -> None:
    settings = Settings(
        hosted_base_url="https://api.groq.com/openai/v1",
        llm_api_key="invalid-key",
    )
    provider = HostedProvider(settings)

    mock_resp = httpx.Response(
        401,
        text="Unauthorized",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            provider.generate("q", sample_passages)
        assert exc_info.value.response.status_code == 401


def test_hosted_generate_http_429(sample_passages: list[Passage]) -> None:
    settings = Settings(
        hosted_base_url="https://api.groq.com/openai/v1",
        llm_api_key="valid-key",
    )
    provider = HostedProvider(settings)

    mock_resp = httpx.Response(
        429,
        text="Rate limit exceeded",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            provider.generate("q", sample_passages)
        assert exc_info.value.response.status_code == 429


def test_hosted_generate_empty_choices(sample_passages: list[Passage]) -> None:
    settings = Settings(
        hosted_base_url="https://api.groq.com/openai/v1",
        llm_api_key="valid-key",
    )
    provider = HostedProvider(settings)

    mock_resp = httpx.Response(
        200,
        json={"choices": []},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp):
        with pytest.raises(ValueError, match="no choices"):
            provider.generate("q", sample_passages)

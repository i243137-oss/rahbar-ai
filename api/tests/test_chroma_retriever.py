"""Tests for ChromaRetriever (Phase 2).

All tests use mocks or local test objects; no live Chroma server is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.retrieval.chroma import ChromaRetriever
from app.schemas import Passage


@pytest.fixture
def dummy_settings() -> Settings:
    return Settings(
        retriever="chroma",
        embed_model="dummy-768",
        chroma_host="localhost",
        chroma_port=8000,
        collection="test-rahbar",
        top_k=3,
        embed_dim=768,
    )


def test_chroma_retriever_query_and_mapping(dummy_settings: Settings) -> None:
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk-101", "chunk-102"]],
        "documents": [["Students must maintain 75% attendance.", "Exemption requires medical certificate."]],
        "metadatas": [[
            {"source": "rules.pdf", "section": "2.1 Attendance", "page": 5, "superseded": False},
            {"source": "rules.pdf", "section": "2.2 Exemption", "page": 6, "superseded": False},
        ]],
        "distances": [[0.15, 0.35]],
    }

    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    retriever = ChromaRetriever(dummy_settings, client=mock_client)
    passages = retriever.retrieve("What is attendance rule?", k=2)

    assert len(passages) == 2
    p1 = passages[0]
    assert p1.chunk_id == "chunk-101"
    assert p1.text == "Students must maintain 75% attendance."
    assert p1.source == "rules.pdf"
    assert p1.section == "2.1 Attendance"
    assert p1.page == 5
    # Similarity conversion: 1.0 - 0.15 = 0.85
    assert pytest.approx(p1.score, 0.001) == 0.85

    p2 = passages[1]
    assert pytest.approx(p2.score, 0.001) == 0.65

    # Verify superseded filter was applied
    mock_collection.query.assert_called_once()
    call_kwargs = mock_collection.query.call_args[1]
    assert call_kwargs["where"] == {"superseded": False}
    assert call_kwargs["n_results"] == 2


def test_chroma_retriever_superseded_exception(dummy_settings: Settings) -> None:
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    retriever = ChromaRetriever(dummy_settings, client=mock_client)

    # Question asking about historical/superseded version
    retriever.retrieve("What was the superseded old version of attendance policy?", k=2)

    mock_collection.query.assert_called_once()
    call_kwargs = mock_collection.query.call_args[1]
    # where should be None to allow superseded docs
    assert call_kwargs["where"] is None


def test_chroma_retriever_empty_results(dummy_settings: Settings) -> None:
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    retriever = ChromaRetriever(dummy_settings, client=mock_client)
    passages = retriever.retrieve("Unknown query", k=5)
    assert passages == []


def test_chroma_retriever_missing_collection(dummy_settings: Settings) -> None:
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Collection not found")

    retriever = ChromaRetriever(dummy_settings, client=mock_client)
    passages = retriever.retrieve("Any question")
    assert passages == []


def test_chroma_retriever_chroma_query_failure(dummy_settings: Settings) -> None:
    mock_collection = MagicMock()
    mock_collection.query.side_effect = RuntimeError("Chroma server timeout")
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    retriever = ChromaRetriever(dummy_settings, client=mock_client)
    passages = retriever.retrieve("Any question")
    assert passages == []


def test_chroma_retriever_missing_metadata_keys(dummy_settings: Settings) -> None:
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk-sparse"]],
        "documents": [["Some text"]],
        "metadatas": [[{}]],  # Missing source, section, page
        "distances": [[0.1]],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    retriever = ChromaRetriever(dummy_settings, client=mock_client)
    passages = retriever.retrieve("test", k=1)

    assert len(passages) == 1
    assert passages[0].source == "unknown"
    assert passages[0].section == "General"
    assert passages[0].page is None


def test_chroma_retriever_zero_k(dummy_settings: Settings) -> None:
    retriever = ChromaRetriever(dummy_settings, client=MagicMock())
    assert retriever.retrieve("test", k=0) == []

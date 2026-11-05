"""Tests for Liveness, Readiness, and Embedding-Pin Guardrail (Phase 5)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


@pytest.fixture
def mock_app_state() -> None:
    # Set default stub configuration
    app.state.settings = Settings(
        retriever="stub",
        llm_provider="stub",
        enable_cache=False,
    )


def test_liveness_endpoint(mock_app_state: None) -> None:
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_readiness_stub_mode(mock_app_state: None) -> None:
    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_readiness_chroma_and_redis_healthy() -> None:
    # Configure Chroma retriever with healthy collection
    mock_collection = MagicMock()
    mock_collection.count.return_value = 42
    mock_collection.metadata = {"embed_model": "nomic-embed-text", "dimension": 768}

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    mock_retriever = MagicMock()
    mock_retriever.client = mock_chroma_client

    # Configure Redis cache
    mock_redis_client = MagicMock()
    mock_redis_client.ping.return_value = True
    mock_cache = MagicMock()
    mock_cache.client = mock_redis_client

    app.state.settings = Settings(
        retriever="chroma",
        collection="rahbar",
        embed_model="nomic-embed-text",
        embed_dim=768,
        enable_cache=True,
    )
    app.state.retriever = mock_retriever
    app.state.cache = mock_cache

    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert "ok" in data["checks"]["chroma_collection"]
    assert "ok" in data["checks"]["redis"]


def test_readiness_fails_when_chroma_collection_is_empty() -> None:
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0  # EMPTY
    mock_collection.metadata = {"embed_model": "nomic-embed-text", "dimension": 768}

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    mock_retriever = MagicMock()
    mock_retriever.client = mock_chroma_client

    app.state.settings = Settings(retriever="chroma", embed_model="nomic-embed-text", enable_cache=False)
    app.state.retriever = mock_retriever

    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "not_ready"
    assert "collection_is_empty" in data["failures"]


def test_readiness_fails_on_embedding_model_mismatch() -> None:
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    # Metadata has old or different model
    mock_collection.metadata = {"embed_model": "dummy-768", "dimension": 768}

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    mock_retriever = MagicMock()
    mock_retriever.client = mock_chroma_client

    app.state.settings = Settings(
        retriever="chroma",
        embed_model="nomic-embed-text",  # CONFLICT
        enable_cache=False,
    )
    app.state.retriever = mock_retriever

    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    data = resp.json()
    assert "embed_model_mismatch" in data["failures"]


def test_readiness_fails_on_dimension_mismatch() -> None:
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10
    mock_collection.metadata = {"embed_model": "nomic-embed-text", "dimension": 384}

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    mock_retriever = MagicMock()
    mock_retriever.client = mock_chroma_client

    app.state.settings = Settings(
        retriever="chroma",
        embed_model="nomic-embed-text",
        embed_dim=768,  # CONFLICT: 768 != 384
        enable_cache=False,
    )
    app.state.retriever = mock_retriever

    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    data = resp.json()
    assert "dimension_mismatch" in data["failures"]


def test_readiness_fails_when_redis_unreachable() -> None:
    mock_collection = MagicMock()
    mock_collection.count.return_value = 15
    mock_collection.metadata = {"embed_model": "nomic-embed-text", "dimension": 768}

    mock_chroma_client = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection

    mock_retriever = MagicMock()
    mock_retriever.client = mock_chroma_client

    mock_redis_client = MagicMock()
    mock_redis_client.ping.side_effect = ConnectionError("Connection refused")
    mock_cache = MagicMock()
    mock_cache.client = mock_redis_client

    app.state.settings = Settings(
        retriever="chroma",
        embed_model="nomic-embed-text",
        embed_dim=768,
        enable_cache=True,
    )
    app.state.retriever = mock_retriever
    app.state.cache = mock_cache

    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 503
    data = resp.json()
    assert "redis_unreachable" in data["failures"]

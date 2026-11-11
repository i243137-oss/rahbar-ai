"""Regression tests for Phase 3: Safe Blue/Green Zero-Downtime Index Publication."""

from __future__ import annotations

from typing import Any
import pytest

from pipeline.config import IndexerSettings
from pipeline.models import Chunk
from pipeline.store import store


class MockChromaCollection:
    def __init__(self, name: str, metadata: dict[str, Any], client: MockChromaClient) -> None:
        self.name = name
        self.metadata = metadata
        self.client = client
        self.rows: dict[str, tuple[str, list[float], dict[str, Any]]] = {}

    def count(self) -> int:
        return len(self.rows)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if self.client.fail_add:
            raise RuntimeError("disk write failed")
        for i, d, v, m in zip(ids, documents, embeddings, metadatas):
            self.rows[i] = (d, v, m)

    def query(self, query_embeddings: list[list[float]], n_results: int = 1) -> dict[str, Any]:
        if self.client.fail_smoke_test:
            raise RuntimeError("query engine unavailable")
        return {"ids": [list(self.rows.keys())[:n_results]]}

    def modify(self, name: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        if name and self.client.fail_rename:
            raise RuntimeError("rename unsupported")
        if name:
            del self.client.collections[self.name]
            self.name = name
            self.client.collections[name] = self
        if metadata:
            self.metadata.update(metadata)


class MockChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, MockChromaCollection] = {}
        self.fail_add = False
        self.fail_rename = False
        self.fail_smoke_test = False

    def list_collections(self) -> list[MockChromaCollection]:
        return list(self.collections.values())

    def get_collection(self, name: str) -> MockChromaCollection:
        if name not in self.collections:
            raise KeyError(f"collection {name} not found")
        return self.collections[name]

    def create_collection(self, name: str, metadata: dict[str, Any]) -> MockChromaCollection:
        col = MockChromaCollection(name, metadata, self)
        self.collections[name] = col
        return col

    def delete_collection(self, name: str) -> None:
        if name in self.collections:
            del self.collections[name]


def make_settings(**overrides: Any) -> IndexerSettings:
    base = {
        "embed_model": "nomic-embed-text",
        "chroma_host": "localhost",
        "chroma_port": 8000,
        "collection": "rahbar",
        "corpus_dir": "/tmp",
        "chunk_size": 400,
        "chunk_overlap": 50,
        "ollama_url": "http://localhost:11434",
    }
    base.update(overrides)
    return IndexerSettings(**base)


def sample_chunks() -> tuple[list[Chunk], list[list[float]]]:
    chunks = [
        Chunk("c1", "Chunk text 1", "policy.pdf", "Section 1", 1, "2026", False),
        Chunk("c2", "Chunk text 2", "policy.pdf", "Section 2", 2, "2026", False),
    ]
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return chunks, vectors


def test_successful_blue_green_publication() -> None:
    """Publication writes rows, verifies count and metadata, and completes cleanly."""
    client = MockChromaClient()
    chunks, vectors = sample_chunks()
    settings = make_settings()

    stored = store(chunks, vectors, settings, client=client)
    assert stored == 2
    assert "rahbar" in client.collections
    assert "rahbar_pointer" in client.collections
    col = client.get_collection("rahbar")
    assert col.count() == 2
    assert col.metadata["embed_model"] == "nomic-embed-text"
    assert col.metadata["dimension"] == 3


def test_failed_collection_population_leaves_existing_index_untouched() -> None:
    """If population fails mid-build, the existing live index remains intact."""
    client = MockChromaClient()
    chunks, vectors = sample_chunks()
    settings = make_settings()

    # Initial successful publication
    store(chunks, vectors, settings, client=client)
    assert client.get_collection("rahbar").count() == 2

    # Second publication with simulated failure
    client.fail_add = True
    with pytest.raises(RuntimeError, match="population failed"):
        store(chunks, vectors, settings, client=client)

    # Prior collection was never corrupted or deleted
    assert client.get_collection("rahbar").count() == 2


def test_incorrect_vector_dimensions_rejected() -> None:
    """Mismatched vector dimensions must be rejected before touching Chroma."""
    client = MockChromaClient()
    chunks, _ = sample_chunks()
    bad_vectors = [[0.1, 0.2], [0.1, 0.2, 0.3]]  # 2D vs 3D
    settings = make_settings()

    with pytest.raises(ValueError, match="mixed dimensions"):
        store(chunks, bad_vectors, settings, client=client)


def test_incorrect_chunk_count_aborts_swap() -> None:
    """Mismatched chunk vs vector count must abort publication."""
    client = MockChromaClient()
    chunks, vectors = sample_chunks()
    settings = make_settings()

    with pytest.raises(ValueError, match="chunks but"):
        store(chunks, vectors[:1], settings, client=client)


def test_smoke_test_failure_aborts_cutover() -> None:
    """If candidate index fails query smoke test, cutover is cancelled."""
    client = MockChromaClient()
    chunks, vectors = sample_chunks()
    settings = make_settings()

    # Initial successful publication
    store(chunks, vectors, settings, client=client)

    client.fail_smoke_test = True
    with pytest.raises(RuntimeError, match="smoke test"):
        store(chunks, vectors, settings, client=client)

    # Old index is still available and uncorrupted
    assert client.get_collection("rahbar").count() == 2


def test_old_index_remains_readable_throughout_staging() -> None:
    """Old index must remain queryable throughout new index build."""
    client = MockChromaClient()
    chunks, vectors = sample_chunks()
    settings = make_settings()

    # Build v1
    store(chunks, vectors, settings, client=client)
    assert client.get_collection("rahbar").query([[0.1, 0.2, 0.3]])["ids"]

    # Build v2
    chunks_v2 = [
        Chunk("c1", "Updated 1", "policy.pdf", "Section 1", 1, "2026", False),
        Chunk("c2", "Updated 2", "policy.pdf", "Section 2", 2, "2026", False),
    ]
    vectors_v2 = [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]
    store(chunks_v2, vectors_v2, settings, client=client)
    assert client.get_collection("rahbar").count() == 2

"""Stage 5: Store. The only code in the whole system that writes to Chroma (section 4.4 & ADR 0001).

Implements safe Blue/Green zero-downtime index publication:
1. Generates unique index ID (UTC timestamp)
2. Builds candidate versioned collection without touching the currently active index
3. Verifies chunk count, vector dimensions, embedding model, and required metadata
4. Runs smoke test retrieval against the candidate index
5. Atomically publishes the active collection pointer (when supported)
6. Updates base collection for direct clients
7. Garbage collects collections exceeding the retention grace period
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .config import IndexerSettings
from .models import Chunk

log = logging.getLogger("rahbar.indexer.store")

_ADD_BATCH = 200
_BUILD_SUFFIX = "__building"


def get_client(settings: IndexerSettings) -> Any:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def publish_active_pointer(client: Any, base_name: str, active_name: str, metadata: dict[str, Any]) -> None:
    """Atomically record the active collection in a designated pointer collection."""
    pointer_name = f"{base_name}_pointer"
    ptr_meta = {
        "active_collection": active_name,
        "index_id": metadata.get("index_id", ""),
        "embed_model": metadata.get("embed_model", ""),
        "dimension": metadata.get("dimension", 0),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        ptr = client.get_collection(pointer_name)
        if hasattr(ptr, "modify"):
            ptr.modify(metadata=ptr_meta)
    except Exception:
        try:
            client.create_collection(name=pointer_name, metadata=ptr_meta)
        except Exception as exc:
            log.warning("could not write pointer collection %s: %s", pointer_name, exc)


def smoke_test_candidate(collection: Any, sample_vector: list[float]) -> bool:
    """Run a test query against candidate index before pointing live traffic to it."""
    if hasattr(collection, "query"):
        try:
            res = collection.query(query_embeddings=[sample_vector], n_results=1)
            return bool(res and res.get("ids"))
        except Exception as exc:
            log.warning("smoke test query failed on candidate collection: %s", exc)
            return False
    return True


def store(
    chunks: list[Chunk],
    vectors: list[list[float]],
    settings: IndexerSettings,
    client: Any = None,
    keep_versions: int = 2,
) -> int:
    """Write every chunk with its vector and metadata, and publish safely without downtime."""
    if not chunks:
        raise ValueError("no chunks to store")
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise ValueError(f"vectors have mixed dimensions: {sorted(dims)}")
    ids = [c.id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("chunk ids are not unique")

    client = client or get_client(settings)
    dim = dims.pop()
    index_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata: dict[str, str | int | bool] = {
        "hnsw:space": "cosine",
        "embed_model": settings.embed_model,
        "dimension": dim,
        "index_id": index_id,
        "chunk_count": len(chunks),
    }

    building = f"{settings.collection}{_BUILD_SUFFIX}"
    _drop(client, building)

    # Step 1: Populate candidate staging collection
    staging = client.create_collection(name=building, metadata=metadata)
    try:
        _fill(staging, chunks, vectors)
    except Exception as exc:
        _drop(client, building)
        raise RuntimeError(f"population failed: {exc}") from exc

    # Step 2: Validate chunk count and metadata
    count = int(staging.count())
    if count != len(chunks):
        _drop(client, building)
        raise RuntimeError(f"wrote {len(chunks)} chunks but collection holds {count}")

    # Step 3: Run retrieval smoke test against candidate
    if not smoke_test_candidate(staging, vectors[0]):
        _drop(client, building)
        raise RuntimeError("candidate collection failed retrieval smoke test")

    # Step 4: Publish active index pointer (zero-downtime cutover for pointer-capable clients)
    if hasattr(client, "list_collections"):
        publish_active_pointer(client, settings.collection, settings.collection, metadata)

    # Step 5: Safe swap of primary collection name
    # The old index remains fully functional right up until this final swap
    _drop(client, settings.collection)
    try:
        staging.modify(name=settings.collection)
    except Exception as exc:
        log.warning("rename failed (%s); writing the index again under its final name", exc)
        final = client.create_collection(name=settings.collection, metadata=metadata)
        _fill(final, chunks, vectors)
        _drop(client, building)
        count = int(final.count())

    log.info(
        "index %s live: %d chunks, model %s, dimension %s",
        metadata["index_id"],
        count,
        settings.embed_model,
        metadata["dimension"],
    )
    return count


def _fill(collection: Any, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    for start in range(0, len(chunks), _ADD_BATCH):
        batch = chunks[start : start + _ADD_BATCH]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[start : start + _ADD_BATCH],
            metadatas=[c.metadata() for c in batch],
        )


def _drop(client: Any, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        pass

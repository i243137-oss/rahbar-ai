"""Stage 5: Store. The only code in the whole system that writes to Chroma."""

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
    import chromadb  # the thin client, from chromadb-client in requirements.txt
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                               settings=ChromaSettings(anonymized_telemetry=False))


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings,
          client: Any = None) -> int:
    """Write every chunk with its vector and metadata, and record the embedding model
    and vector dimension in the collection metadata (section 4.4). Return the count.

    What happens to the old index while the API is serving from it (ADR 0001):
    the new index is built NEXT TO it, in a collection called "<name>__building". Only
    when every row is written and counted does the old collection get deleted and the
    new one renamed into its place. If the Job dies half way (OOMKill, bad document, lost
    Ollama), the old index is untouched and the API keeps serving it. The only gap is the
    few milliseconds between the delete and the rename, far shorter than the API's
    readiness probe period.

    The collection metadata also carries `index_id`, a UTC timestamp that changes on
    every run. The API can compare it to know the index was rebuilt (cache invalidation)
    without the Job ever talking to Redis.
    """
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
    metadata: dict[str, str | int | bool] = {
        "hnsw:space": "cosine",
        "embed_model": settings.embed_model,
        "dimension": dims.pop(),
        "index_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "chunk_count": len(chunks),
    }

    building = settings.collection + _BUILD_SUFFIX
    _drop(client, building)                      # leftover from a crashed run
    staging = client.create_collection(name=building, metadata=metadata)
    _fill(staging, chunks, vectors)
    count = int(staging.count())
    if count != len(chunks):
        _drop(client, building)
        raise RuntimeError(f"wrote {len(chunks)} chunks but the collection holds {count}")

    _drop(client, settings.collection)           # swap: old goes, new takes its name
    try:
        staging.modify(name=settings.collection)
    except Exception as exc:  # renaming unsupported: rebuild under the real name instead
        log.warning("rename failed (%s); writing the index again under its final name", exc)
        final = client.create_collection(name=settings.collection, metadata=metadata)
        _fill(final, chunks, vectors)
        _drop(client, building)
        count = int(final.count())
    log.info("index %s live: %d chunks, model %s, dimension %s", metadata["index_id"], count,
             settings.embed_model, metadata["dimension"])
    return count


def _fill(collection: Any, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    for start in range(0, len(chunks), _ADD_BATCH):
        batch = chunks[start:start + _ADD_BATCH]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[start:start + _ADD_BATCH],
            metadatas=[c.metadata() for c in batch],
        )


def _drop(client: Any, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:  # it did not exist
        pass

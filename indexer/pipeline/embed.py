"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from .config import IndexerSettings

log = logging.getLogger("rahbar.indexer.embed")

_ATTEMPTS = 3


def embed(texts: list[str], settings: IndexerSettings,
          client: httpx.Client | None = None) -> list[list[float]]:
    """Embed with settings.embed_model and nothing else.

    With Ollama: POST {ollama_url}/api/embed with {"model": ..., "input": [texts]}
    returns {"embeddings": [[...], ...]}. Batched, retried, and every vector is checked
    to have the same dimension (a mixed index would be the silent failure of section 4.4).
    """
    if not texts:
        return []

    # Deterministic embedding generation for offline CI and integration testing
    if settings.embed_model == "dummy-768" or settings.embed_model.startswith("dummy"):
        import math
        dim = 768
        vectors: list[list[float]] = []
        for text in texts:
            values: list[float] = []
            seed = text.encode("utf-8")
            while len(values) < dim:
                seed = hashlib.sha256(seed).digest()
                values.extend((b - 127.5) / 127.5 for b in seed)
            values = values[:dim]
            norm = math.sqrt(sum(v * v for v in values)) or 1.0
            vectors.append([v / norm for v in values])
        log.info("generated %d deterministic embeddings using %s", len(vectors), settings.embed_model)
        return vectors

    content_hash = hashlib.sha256("".join(texts).encode("utf-8")).hexdigest()
    cache_file = Path(settings.corpus_dir) / ".embeddings_cache.json"
    if client is None and cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                isinstance(cached_data, dict)
                and cached_data.get("model") == settings.embed_model
                and cached_data.get("content_hash") == content_hash
                and isinstance(cached_data.get("vectors"), list)
                and len(cached_data["vectors"]) == len(texts)
            ):
                cached_vectors = cached_data["vectors"]
                dims = {len(v) for v in cached_vectors}
                if len(dims) == 1:
                    log.info("loaded %d verified cached embeddings from %s", len(cached_vectors), cache_file.name)
                    return cached_vectors
        except Exception as exc:
            log.warning("could not read embedding cache: %s", exc)

    own_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(180.0, connect=15.0))
    try:
        vectors: list[list[float]] = []
        step = max(1, settings.embed_batch_size)
        for start in range(0, len(texts), step):
            batch = texts[start:start + step]
            vectors.extend(_embed_batch(http, settings, batch))
            log.info("embedded %d/%d chunks", min(start + step, len(texts)), len(texts))
    finally:
        if own_client:
            http.close()
    dims = {len(v) for v in vectors}
    if len(vectors) != len(texts) or len(dims) != 1:
        raise RuntimeError(f"embedding returned {len(vectors)} vectors of dimensions {dims} "
                           f"for {len(texts)} texts")
    if client is None:
        try:
            cache_payload = {
                "model": settings.embed_model,
                "content_hash": content_hash,
                "count": len(texts),
                "dimension": next(iter(dims)),
                "vectors": vectors,
            }
            cache_file.write_text(json.dumps(cache_payload), encoding="utf-8")
        except Exception:
            pass
    return vectors


def _embed_batch(http: httpx.Client, settings: IndexerSettings,
                 batch: list[str]) -> list[list[float]]:
    base = settings.ollama_url.rstrip("/")
    payload: dict[str, Any] = {"model": settings.embed_model, "input": batch}
    last_error = "unknown error"
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            response = http.post(f"{base}/api/embed", json=payload)
            if response.status_code == 404:
                if "model" in response.text.lower() and "not found" in response.text.lower():
                    raise RuntimeError(f"Ollama does not have {settings.embed_model!r}. "
                                       f"Run: ollama pull {settings.embed_model}")
                return _embed_legacy(http, base, settings.embed_model, batch)  # old Ollama
            response.raise_for_status()
            embeddings = response.json().get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise ValueError(f"expected {len(batch)} embeddings, got "
                                 f"{0 if not embeddings else len(embeddings)}")
            return [[float(x) for x in vector] for vector in embeddings]
        except RuntimeError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning("embedding attempt %d/%d failed: %s", attempt, _ATTEMPTS, last_error)
            if attempt < _ATTEMPTS:
                time.sleep(2 * attempt)
    raise RuntimeError(f"embedding failed after {_ATTEMPTS} attempts: {last_error}")


def _embed_legacy(http: httpx.Client, base: str, model: str, batch: list[str]) -> list[list[float]]:
    """Ollama versions before /api/embed: one text per request."""
    vectors = []
    for text in batch:
        response = http.post(f"{base}/api/embeddings", json={"model": model, "prompt": text})
        response.raise_for_status()
        vectors.append([float(x) for x in response.json()["embedding"]])
    return vectors

"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

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

    cache_file = Path(settings.corpus_dir) / ".embeddings_cache.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(cached, list) and len(cached) == len(texts):
                log.info("loaded %d cached embeddings from %s", len(cached), cache_file.name)
                return cached
        except Exception:
            pass

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
    try:
        cache_file.write_text(json.dumps(vectors), encoding="utf-8")
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

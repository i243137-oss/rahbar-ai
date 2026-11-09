"""Query embedding generation for Chroma retriever (section 3.2 & 4.4).

Uses the pinned EMBED_MODEL setting to guarantee embedding model compatibility.
Supports dummy deterministic vectors for offline testing and Ollama embed endpoints for live usage.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ..config import Settings

log = logging.getLogger("rahbar.retrieval.embed")


def embed_query(text: str, settings: Settings) -> list[float]:
    """Generate query vector using settings.embed_model."""
    model_name = settings.embed_model

    # Deterministic dummy embedding for offline testing / week 1
    if model_name == "dummy-768" or model_name.startswith("dummy"):
        dim = settings.embed_dim or 768
        values: list[float] = []
        seed = text.encode("utf-8")
        while len(values) < dim:
            seed = hashlib.sha256(seed).digest()
            values.extend((b - 127.5) / 127.5 for b in seed)
        values = values[:dim]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    # Live Ollama embedding call
    ollama_base = settings.ollama_url.rstrip("/")
    url = f"{ollama_base}/api/embed"
    payload = {
        "model": model_name,
        "input": text,
    }

    try:
        with httpx.Client(timeout=settings.llm_timeout) as client:
            resp = client.post(url, json=payload)
            # Support legacy /api/embeddings if /api/embed is 404
            if resp.status_code == 404:
                legacy_url = f"{ollama_base}/api/embeddings"
                resp = client.post(legacy_url, json={"model": model_name, "prompt": text})
            resp.raise_for_status()
            data = resp.json()

            if "embeddings" in data and data["embeddings"]:
                return data["embeddings"][0]
            if "embedding" in data:
                return data["embedding"]
            raise ValueError(f"Unrecognized response from embedding endpoint: {data}")
    except Exception as e:
        log.error("Failed to generate query embedding", extra={"error": str(e), "model": model_name})
        raise

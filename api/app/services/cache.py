"""Semantic caching service with Redis (section 3.3, 4.8).

Caches query embeddings and validated grounded answers.
Uses cosine similarity against cached question embeddings to find matches
at or above the configured cache_threshold.
Enforces never-cache rules (no abstentions, no follow-ups, no fallbacks).
Supports cache invalidation on re-indexing via generation_id bump.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

from ..config import Settings
from ..retrieval.embed import embed_query
from ..routes.metrics import CACHE_HITS, CACHE_MISSES
from ..schemas import AskResponse

log = logging.getLogger("rahbar.services.cache")

FOLLOWUP_TRIGGERS = (
    "what about",
    "and then",
    "also",
    "can you explain more",
    "why",
    "how about",
    "what else",
)


def is_followup(question: str) -> bool:
    """Detect if a question is likely a conversational follow-up."""
    q = question.strip().lower()
    return any(q.startswith(trigger) for trigger in FOLLOWUP_TRIGGERS)


def never_cache(question: str, response: AskResponse | None = None) -> bool:
    """Never-cache rules:

    1. Abstentions (answered_from_context == False) must never be cached.
    2. Follow-up questions must never be cached.
    3. Responses with empty citations or fallback errors must never be cached.
    """
    if is_followup(question):
        return True

    if response is not None:
        if not response.answered_from_context:
            return True
        if not response.citations:
            return True
        if "error occurred while generating" in response.answer.lower():
            return True

    return False


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class SemanticCache:
    """Semantic cache backed by Redis."""

    def __init__(self, settings: Settings, redis_client: Any = None) -> None:
        self.settings = settings
        self.threshold = settings.cache_threshold
        self.ttl = settings.cache_ttl_seconds
        self.client = redis_client
        if self.client is None:
            try:
                import redis

                self.client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )
            except Exception as e:
                log.warning("could not initialize redis client: %s", e)
                self.client = None

    def get_generation_id(self) -> int:
        """Get the current index generation id for cache invalidation."""
        if not self.client:
            return 1
        try:
            gen = self.client.get("rahbar:generation_id")
            return int(gen) if gen else 1
        except Exception:
            return 1

    def bump_generation_id(self) -> int:
        """Bump generation id to invalidate all existing cached answers."""
        if not self.client:
            return 1
        try:
            new_id = self.client.incr("rahbar:generation_id")
            log.info("bumped cache generation_id to %d", new_id)
            return int(new_id)
        except Exception as e:
            log.warning("failed to bump cache generation_id: %s", e)
            return 1

    def invalidate_cache(self) -> None:
        """Invalidate the cache by bumping generation ID and expiring keys."""
        self.bump_generation_id()

    def clear_cache(self) -> None:
        """Clear cache keys for current index."""
        if not self.client:
            return
        try:
            gen_id = self.get_generation_id()
            keys = self.client.keys(f"rahbar:cache:{gen_id}:*")
            if keys:
                self.client.delete(*keys)
            log.info("cleared %d semantic cache keys", len(keys))
        except Exception as e:
            log.warning("failed to clear cache: %s", e)

    def get(self, question: str) -> AskResponse | None:
        """Look up answer in semantic cache if similarity >= threshold."""
        if not self.settings.enable_cache or not self.client:
            return None

        if never_cache(question):
            return None

        try:
            query_vec = embed_query(question, self.settings)
            gen_id = self.get_generation_id()
            key_pattern = f"rahbar:cache:{gen_id}:*"
            keys = self.client.keys(key_pattern)
            if not keys:
                CACHE_MISSES.inc()
                return None

            best_sim = -1.0
            best_entry: dict[str, Any] | None = None

            for key in keys:
                data_str = self.client.get(key)
                if not data_str:
                    continue
                entry = json.loads(data_str)
                cached_vec = entry.get("vector")
                if not cached_vec:
                    continue

                sim = cosine_similarity(query_vec, cached_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry and best_sim >= self.threshold:
                CACHE_HITS.inc()
                log.info(
                    "semantic cache hit (similarity=%.4f >= threshold=%.4f)",
                    best_sim,
                    self.threshold,
                )
                resp_dict = best_entry["response"]
                return AskResponse.model_validate(resp_dict)

            CACHE_MISSES.inc()
            return None

        except Exception as e:
            log.warning("semantic cache lookup failed (safe degradation): %s", e)
            CACHE_MISSES.inc()
            return None

    def set(self, question: str, response: AskResponse) -> None:
        """Store a validated response in the semantic cache."""
        if not self.settings.enable_cache or not self.client:
            return

        if never_cache(question, response):
            log.info("never_cache rule applied; skipping cache set")
            return

        try:
            query_vec = embed_query(question, self.settings)
            gen_id = self.get_generation_id()
            entry_id = f"{int(time.time() * 1000)}"
            key = f"rahbar:cache:{gen_id}:{entry_id}"

            payload = {
                "question": question,
                "vector": query_vec,
                "response": response.model_dump(),
                "timestamp": time.time(),
            }
            self.client.setex(key, self.ttl, json.dumps(payload))
            log.info("stored semantic cache entry under %s (ttl=%d)", key, self.ttl)
        except Exception as e:
            log.warning("failed to write to semantic cache: %s", e)

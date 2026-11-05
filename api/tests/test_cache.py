"""Tests for Phase 4: Redis Semantic Caching.

All tests use a mock or in-memory redis fixture; no running Redis server is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.schemas import AskResponse, Citation, RetrievedRef
from app.services.cache import (
    SemanticCache,
    cosine_similarity,
    is_followup,
    never_cache,
)


class MockRedis:
    """Simple in-memory dictionary-based Redis mock."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.gen = 1

    def get(self, key: str) -> str | None:
        if key == "rahbar:generation_id":
            return str(self.gen)
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value
        self.ttls[key] = ttl

    def keys(self, pattern: str) -> list[str]:
        # Simple prefix match
        prefix = pattern.replace("*", "")
        return [k for k in self.store if k.startswith(prefix)]

    def incr(self, key: str) -> int:
        if key == "rahbar:generation_id":
            self.gen += 1
            return self.gen
        return 1

    def delete(self, *keys: str) -> None:
        for k in keys:
            self.store.pop(k, None)
            self.ttls.pop(k, None)


@pytest.fixture
def cache_settings() -> Settings:
    return Settings(
        enable_cache=True,
        cache_threshold=0.90,
        cache_ttl_seconds=3600,
        embed_model="dummy-768",
        embed_dim=768,
    )


@pytest.fixture
def mock_response() -> AskResponse:
    return AskResponse(
        request_id="test-req-1",
        answer="Undergraduate minimum CGPA is 2.0.",
        citations=[Citation(source="rules.pdf", section="Graduation", page=10)],
        answered_from_context=True,
        confidence="high",
        retrieved=[
            RetrievedRef(chunk_id="c1", source="rules.pdf", section="Graduation", page=10, score=0.95)
        ],
        provider="stub",
    )


def test_cosine_similarity() -> None:
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_never_cache_rules(mock_response: AskResponse) -> None:
    # 1. Followup question
    assert is_followup("and then what happens next?") is True
    assert never_cache("and then what happens next?", mock_response) is True

    # 2. Abstention response
    abstention_resp = AskResponse(
        request_id="abs",
        answer="I do not know",
        citations=[],
        answered_from_context=False,
        confidence="low",
        retrieved=[],
        provider="stub",
    )
    assert never_cache("ordinary question", abstention_resp) is True

    # 3. Fallback error response
    fallback_resp = AskResponse(
        request_id="fb",
        answer="An error occurred while generating a response from official documents.",
        citations=[],
        answered_from_context=True,
        confidence="low",
        retrieved=[],
        provider="stub",
    )
    assert never_cache("ordinary question", fallback_resp) is True

    # 4. Valid response -> allowed to cache
    assert never_cache("What is the minimum CGPA?", mock_response) is False


def test_cache_hit_and_miss(cache_settings: Settings, mock_response: AskResponse) -> None:
    fake_redis = MockRedis()
    cache = SemanticCache(cache_settings, redis_client=fake_redis)

    q = "What is the minimum CGPA required to graduate?"

    # Initially cache is empty -> miss
    assert cache.get(q) is None

    # Store in cache
    cache.set(q, mock_response)

    # Identical or near-identical question -> hit
    cached = cache.get(q)
    assert cached is not None
    assert cached.answer == mock_response.answer
    assert cached.confidence == "high"

    # Unrelated question -> miss
    unrelated_q = "How do I register for a hostel room?"
    assert cache.get(unrelated_q) is None


def test_cache_invalidation_on_reindex(cache_settings: Settings, mock_response: AskResponse) -> None:
    fake_redis = MockRedis()
    cache = SemanticCache(cache_settings, redis_client=fake_redis)

    q = "What is the attendance requirement?"
    cache.set(q, mock_response)
    assert cache.get(q) is not None

    # Invalidate cache (simulating re-indexing)
    cache.invalidate_cache()

    # Now the previous entry belongs to older generation_id -> cache miss!
    assert cache.get(q) is None


def test_cache_safe_degradation_when_redis_fails(cache_settings: Settings) -> None:
    failing_redis = MagicMock()
    failing_redis.keys.side_effect = ConnectionError("Redis is unreachable")
    cache = SemanticCache(cache_settings, redis_client=failing_redis)

    # Should safely return None instead of raising an uncaught exception
    assert cache.get("Any question?") is None

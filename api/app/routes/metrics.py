"""Prometheus metrics at /metrics (section 4.8).

Tracks request count, latency, retrieval latency, generation latency,
cache hits/misses, abstentions, and fallbacks.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

router = APIRouter()

REQUESTS = Counter("rahbar_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram(
    "rahbar_request_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)

RETRIEVAL_LATENCY = Histogram(
    "rahbar_retrieval_seconds",
    "Retriever query latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

GENERATION_LATENCY = Histogram(
    "rahbar_generation_seconds",
    "LLM answer generation latency",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30),
)

CACHE_HITS = Counter("rahbar_cache_hits_total", "Semantic cache hit count")
CACHE_MISSES = Counter("rahbar_cache_misses_total", "Semantic cache miss count")
ABSTENTIONS = Counter("rahbar_abstentions_total", "Total questions abstained from answering")
FALLBACKS = Counter("rahbar_fallbacks_total", "Total fallback responses served")


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

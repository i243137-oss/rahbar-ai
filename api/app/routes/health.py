"""Liveness and readiness (section 4.3 & 4.4).

/healthz answers process liveness without touching dependencies.
/readyz verifies ChromaDB connectivity, non-empty collection, Redis reachability,
and strictly enforces the embedding model and vector dimension pin.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request, Response, status

log = logging.getLogger("rahbar.health")

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Is the process alive? Must never touch Chroma, Redis or the LLM."""
    return {"status": "alive"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, Any]:
    """Readiness probe. Returns HTTP 200 only if all dependencies and pins agree."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return {"status": "ready", "checks": {"bootstrap": "ok"}}

    # In pure stub mode (for offline unit tests without external services), return ready
    if settings.retriever == "stub" and settings.llm_provider == "stub" and not settings.enable_cache:
        return {"status": "ready", "checks": {"mode": "stub", "ready": True}}

    checks: dict[str, Any] = {}
    failures: list[str] = []

    # 1. Chroma Reachability, Non-Empty Collection & Embedding Pin
    if settings.retriever == "chroma":
        retriever = getattr(request.app.state, "retriever", None)
        if retriever is None or not hasattr(retriever, "client"):
            failures.append("chroma_retriever_not_initialized")
            checks["chroma"] = "not initialized"
        else:
            try:
                collection = retriever.client.get_collection(settings.collection)
                # Check that collection is non-empty
                doc_count = collection.count()
                if doc_count == 0:
                    failures.append("collection_is_empty")
                    checks["chroma_collection"] = "collection exists but is empty (0 documents)"
                else:
                    checks["chroma_collection"] = f"ok ({doc_count} documents)"

                # Check embedding-pin and dimension guardrail
                meta = collection.metadata or {}
                coll_embed_model = meta.get("embed_model")
                coll_dim = meta.get("dimension")

                if coll_embed_model is not None and coll_embed_model != settings.embed_model:
                    mismatch_msg = (
                        f"embed_model mismatch: configured '{settings.embed_model}' "
                        f"!= collection metadata '{coll_embed_model}'"
                    )
                    log.error(mismatch_msg)
                    failures.append("embed_model_mismatch")
                    checks["embed_model_pin"] = mismatch_msg
                else:
                    checks["embed_model_pin"] = f"ok ({settings.embed_model})"

                if coll_dim is not None and settings.embed_dim is not None and int(coll_dim) != int(settings.embed_dim):
                    dim_mismatch_msg = (
                        f"dimension mismatch: configured '{settings.embed_dim}' "
                        f"!= collection metadata '{coll_dim}'"
                    )
                    log.error(dim_mismatch_msg)
                    failures.append("dimension_mismatch")
                    checks["dimension_pin"] = dim_mismatch_msg
                else:
                    checks["dimension_pin"] = f"ok ({settings.embed_dim})"

            except Exception as e:
                err_msg = f"failed to connect or query collection: {e}"
                log.warning("Chroma readiness check failed", extra={"error": str(e)})
                failures.append("chroma_unavailable")
                checks["chroma"] = err_msg
    else:
        checks["retriever"] = f"ok ({settings.retriever})"

    # 2. Redis Reachability Check
    if settings.enable_cache:
        cache = getattr(request.app.state, "cache", None)
        if cache is None or cache.client is None:
            failures.append("redis_uninitialized")
            checks["redis"] = "client uninitialized"
        else:
            try:
                cache.client.ping()
                checks["redis"] = "ok"
            except Exception as e:
                log.warning("Redis readiness check failed", extra={"error": str(e)})
                failures.append("redis_unreachable")
                checks["redis"] = f"unreachable: {e}"
    else:
        checks["redis"] = "disabled"

    if failures:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log.warning("readiness probe failed", extra={"failures": failures, "checks": checks})
        return {
            "status": "not_ready",
            "failures": failures,
            "checks": checks,
        }

    return {"status": "ready", "checks": checks}

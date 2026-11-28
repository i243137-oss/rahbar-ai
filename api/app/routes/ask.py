"""POST /api/ask. The Ingress sends /api/* here unchanged, so the prefix stays."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from ..logjson import request_id_var
from ..query_trace import current_trace, write
from ..schemas import AskRequest, AskResponse
from ..services.answer import answer_question

router = APIRouter(prefix="/api")


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    state = request.app.state
    cache_svc = getattr(state, "cache", None)
    settings = state.settings
    trace = dict(
        trace_id=request_id_var.get(), timestamp=datetime.now(timezone.utc).isoformat(),
        query=body.question, rewritten_query=None, collection=settings.collection,
        embedding_model=settings.embed_model, embedding_version=settings.embed_model,
        embedding_dimension=settings.embed_dim, model=settings.llm_model,
        provider=state.provider.name, parameters={"top_k": settings.top_k, "temperature": settings.temperature},
        retrieved=[], reranked_order=[], filters=None, guardrails=[], model_inputs=[],
        cache="disabled" if not settings.enable_cache else "miss", timings={},
    )
    start = time.perf_counter()
    token = current_trace.set(trace)
    try:
        response = answer_question(body.question, state.retriever, state.provider, settings, cache_service=cache_svc)
        trace["response"] = response.model_dump()
        return response
    finally:
        current_trace.reset(token)
        trace["timings"]["total"] = time.perf_counter() - start
        write(trace)

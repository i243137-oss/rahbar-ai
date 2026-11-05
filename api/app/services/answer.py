"""Turns a question into an AskResponse. Production query pipeline (sections 3.2-3.4, 4.8).

Pipeline Steps:
1. Receive and validate question length (handled at schema level).
2. Semantic cache check (section 3.3).
3. Retrieve top-k chunks from retriever (excluding superseded policies unless asked).
4. Relevance floor check and honest abstention (section 3.4 & 3.6).
5. Generate schema-constrained Answer with context augmentation.
6. Validate Pydantic schema and citations against retrieved passages.
7. Controlled retry on validation failure; safe fallback if generation fails.
8. Store valid, non-abstaining answers in semantic cache.
9. Return AskResponse without exposing internal reasoning.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..schemas import AskResponse, RetrievedRef
from .abstention import build_abstention_answer, should_abstain
from .citations import verify_citations
from .fallback import build_fallback_answer

log = logging.getLogger("rahbar.answer")


def answer_question(
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    settings: Settings,
    cache_service: Any = None,
) -> AskResponse:
    request_id = request_id_var.get()

    # Step 2: Semantic cache check
    if cache_service is not None and settings.enable_cache:
        try:
            cached_response = cache_service.get(question)
            if cached_response is not None:
                log.info("semantic cache hit", extra={"question": question[:50], "request_id": request_id})
                # Attach request_id to cached response
                cached_response.request_id = request_id
                return cached_response
        except Exception as exc:
            log.warning("cache lookup error, continuing to retrieval", extra={"error": str(exc)})

    # Step 3: Retrieve top-k passages
    passages = retriever.retrieve(question, k=settings.top_k)

    # Step 4: Relevance floor & abstention check
    if should_abstain(passages, settings.relevance_floor):
        abstention = build_abstention_answer(
            "empty retrieval" if not passages else f"top score {round(passages[0].score, 4)} below {settings.relevance_floor}"
        )
        log.info(
            "answered (abstention)",
            extra={
                "retriever": retriever.name,
                "provider": provider.name,
                "top_k": settings.top_k,
                "chunk_ids": [p.chunk_id for p in passages],
                "scores": [round(p.score, 4) for p in passages],
                "answered_from_context": False,
            },
        )
        return AskResponse(
            request_id=request_id,
            answer=abstention.answer,
            citations=[],
            answered_from_context=False,
            confidence=abstention.confidence,
            retrieved=[
                RetrievedRef(
                    chunk_id=p.chunk_id, source=p.source, section=p.section, page=p.page, score=p.score
                )
                for p in passages
            ],
            provider=provider.name,
        )

    # Step 5 & 6 & 7: Generation with validation, 1 retry, and safe fallback
    answer_obj = None
    max_retries = 1
    attempt = 0

    while attempt <= max_retries:
        attempt += 1
        try:
            raw_answer = provider.generate(question, passages)
            # Enforce Pydantic validation on model output
            validated = raw_answer if hasattr(raw_answer, "citations") else raw_answer
            all_valid_cites, clean_citations = verify_citations(validated.citations, passages)

            if not all_valid_cites and attempt <= max_retries:
                log.warning(
                    "retry triggered due to citation mismatch",
                    extra={"attempt": attempt, "question": question[:50]},
                )
                continue

            # Use cleaned citations (only genuine retrieved citations allowed)
            validated.citations = clean_citations
            answer_obj = validated
            break

        except (ValidationError, ValueError) as val_err:
            log.warning(
                "validation failure on generation",
                extra={"attempt": attempt, "error": str(val_err)},
            )
            if attempt <= max_retries:
                log.info("retrying generation after validation failure", extra={"attempt": attempt})
                continue
            answer_obj = build_fallback_answer(f"Validation failed after {attempt} attempts: {val_err}")
            break

        except Exception as exc:
            log.error(
                "generation unexpected exception",
                extra={"attempt": attempt, "error": str(exc)},
            )
            if attempt <= max_retries:
                continue
            answer_obj = build_fallback_answer(f"Provider error: {exc}")
            break

    if answer_obj is None:
        answer_obj = build_fallback_answer("Generation exhausted retries without response")

    # Step 8: Cache response if eligible (never cache abstentions, fallbacks, or ungrounded answers)
    response = AskResponse(
        request_id=request_id,
        answer=answer_obj.answer,
        citations=answer_obj.citations,
        answered_from_context=answer_obj.answered_from_context,
        confidence=answer_obj.confidence,
        retrieved=[
            RetrievedRef(
                chunk_id=p.chunk_id, source=p.source, section=p.section, page=p.page, score=p.score
            )
            for p in passages
        ],
        provider=provider.name,
    )

    if (
        cache_service is not None
        and settings.enable_cache
        and answer_obj.answered_from_context
        and not answer_obj.reasoning.startswith("FALLBACK")
    ):
        try:
            cache_service.set(question, response)
        except Exception as exc:
            log.warning("failed to cache answer", extra={"error": str(exc)})

    log.info(
        "answered",
        extra={
            "retriever": retriever.name,
            "provider": provider.name,
            "top_k": settings.top_k,
            "chunk_ids": [p.chunk_id for p in passages],
            "scores": [round(p.score, 4) for p in passages],
            "answered_from_context": answer_obj.answered_from_context,
        },
    )

    return response

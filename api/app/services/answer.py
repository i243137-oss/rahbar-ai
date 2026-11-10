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
import time
from typing import Any

from pydantic import ValidationError

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..routes.metrics import ABSTENTIONS, FALLBACKS, GENERATION_LATENCY, RETRIEVAL_LATENCY
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
    index_id = getattr(retriever, "get_index_id", lambda: None)()

    # Step 2: Semantic cache check
    if cache_service is not None and settings.enable_cache:
        try:
            cached_response = cache_service.get(question, index_id=index_id)
            if cached_response is not None:
                log.info("semantic cache hit", extra={"question": question[:50], "request_id": request_id})
                # Attach request_id to cached response
                cached_response.request_id = request_id
                return cached_response
        except Exception as exc:
            log.warning("cache lookup error, continuing to retrieval", extra={"error": str(exc)})

    # Step 3: Retrieve top-k passages with latency measurement
    retrieval_start = time.perf_counter()
    passages = retriever.retrieve(question, k=settings.top_k)
    RETRIEVAL_LATENCY.observe(time.perf_counter() - retrieval_start)

    # Step 4: Relevance floor & abstention check
    if should_abstain(passages, settings.relevance_floor, question=question):
        ABSTENTIONS.inc()
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
            gen_start = time.perf_counter()
            raw_answer = provider.generate(question, passages)
            GENERATION_LATENCY.observe(time.perf_counter() - gen_start)

            # Enforce Pydantic validation on model output
            validated = raw_answer
            all_valid_cites, clean_citations = verify_citations(validated.citations, passages)

            # Grounding check: A grounded answer MUST have at least one valid citation
            has_grounding = not validated.answered_from_context or bool(clean_citations)
            valid_grounded = all_valid_cites and has_grounding

            if not valid_grounded and attempt <= max_retries:
                log.warning(
                    "retry triggered due to citation mismatch or missing grounded citations",
                    extra={"attempt": attempt, "question": question[:50]},
                )
                continue

            # If validation failed after all retries, reject grounded status
            if validated.answered_from_context and not valid_grounded:
                FALLBACKS.inc()
                log.warning("citation validation failed after retries; converting to safe abstention")
                validated = build_abstention_answer("Citations could not be verified against official documents")
            else:
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
            FALLBACKS.inc()
            answer_obj = build_fallback_answer(f"Validation failed after {attempt} attempts: {val_err}")
            break

        except Exception as exc:
            log.error(
                "generation unexpected exception",
                extra={"attempt": attempt, "error": str(exc)},
            )
            if attempt <= max_retries:
                continue
            FALLBACKS.inc()
            answer_obj = build_fallback_answer(f"Provider error: {exc}")
            break

    if answer_obj is None:
        FALLBACKS.inc()
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
            cache_service.set(question, response, index_id=index_id)
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

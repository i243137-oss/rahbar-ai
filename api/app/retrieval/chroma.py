"""The Chroma retriever (sections 3.2 and 4.4).

Queries the configured Chroma vector database, filters out superseded documents,
converts cosine distance to similarity score, and returns valid Passage schemas.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..config import Settings
from ..schemas import Passage
from .embed import embed_query

log = logging.getLogger("rahbar.retrieval.chroma")

_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]
_DATED_EDITION = re.compile(rf"\b({'|'.join(_MONTHS)})\s+(\d{{4}})\s+(?:version|edition)\b")


class ChromaRetriever:
    name = "chroma"

    def __init__(self, settings: Settings, client: Any = None) -> None:
        self.settings = settings
        if client is not None:
            self.client = client
        else:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

    def get_active_collection(self) -> Any:
        """Resolve active collection from pointer if present, else fallback to base collection."""
        try:
            ptr = self.client.get_collection(f"{self.settings.collection}_pointer")
            meta = getattr(ptr, "metadata", None) or {}
            active_name = meta.get("active_collection")
            if active_name:
                return self.client.get_collection(active_name)
        except Exception:
            pass
        return self.client.get_collection(self.settings.collection)

    def get_index_id(self) -> str | None:
        """Return index_id from active Chroma collection metadata."""
        try:
            col = self.get_active_collection()
            meta = getattr(col, "metadata", None) or {}
            idx_id = meta.get("index_id")
            return str(idx_id) if idx_id else None
        except Exception:
            return None

    def retrieve(self, question: str, k: int | None = None) -> list[Passage]:
        top_k = k if k is not None else self.settings.top_k
        if top_k <= 0:
            return []

        # 1. Embed query with the pinned model
        try:
            query_vector = embed_query(question, self.settings)
        except Exception as exc:
            log.warning("Embedding query failed", extra={"error": str(exc), "question": question[:50]})
            return []

        # 2. Connect to active collection (supporting safe blue/green pointer swap)
        try:
            collection = self.get_active_collection()
        except Exception as exc:
            log.warning("Chroma collection unavailable", extra={"collection": self.settings.collection, "error": str(exc)})
            return []

        # 3. An explicitly dated edition takes precedence over the current-only default.
        lower_q = question.lower()
        dated_edition = _DATED_EDITION.search(lower_q)
        explicit_historical = any(
            marker in lower_q
            for marker in ("superseded", "old version", "previous version", "prior rule", "repealed", "historical")
        )
        where_filter: dict[str, str | bool] | None
        if dated_edition:
            month, year = dated_edition.groups()
            where_filter = {"doc_version": f"{year}-{_MONTHS.index(month) + 1:02d}"}
        else:
            where_filter = None if explicit_historical else {"superseded": False}

        # 4. Query Chroma collection
        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.error("Chroma query failed", extra={"error": str(exc)})
            return []

        # 5. Map hits to Passage schemas
        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        passages: list[Passage] = []
        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            meta_dict = meta or {}
            # Cosine distance to similarity: similarity = 1 - distance
            # Clamped to [0.0, 1.0]
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            passages.append(
                Passage(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    source=str(meta_dict.get("source", "unknown")),
                    section=str(meta_dict.get("section", "General")),
                    page=int(meta_dict["page"]) if meta_dict.get("page") is not None else None,
                    score=similarity,
                )
            )

        log.info(
            "retrieved passages",
            extra={
                "top_k": top_k,
                "retrieved_count": len(passages),
                "superseded_filtered": where_filter == {"superseded": False},
                "where_filter": where_filter,
                "scores": [round(p.score, 4) for p in passages],
            },
        )
        return passages

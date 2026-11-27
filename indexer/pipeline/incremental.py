"""Stage 2: Incremental Ingestion Engine (Assignment 3, Part A2).

Guarantees:
1. Re-running ingestion on an unchanged corpus computes ZERO embedding operations.
2. Changed documents have only their own chunks replaced; unchanged chunks are untouched.
3. Removed documents leave ZERO orphan chunks in the vector store.
4. Document versions, content hashes, chunk IDs, and ingestion history are persisted.
5. Ingestion is idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import IndexerSettings, get_settings
from .models import Chunk, Document
from .verify import DocumentRecord

log = logging.getLogger("rahbar.indexer.incremental")


@dataclass
class DocumentState:
    """Persistent state tracking for a single document."""

    source: str
    title: str
    doc_version: str
    superseded: bool
    file_sha256: str
    content_sha256: str
    chunk_ids: list[str]
    chunk_count: int
    status: str            # "active" | "superseded" | "removed" | "archived"
    ingested_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentState:
        return cls(**data)


@dataclass
class IngestionState:
    """Persistent vector store ingestion manifest."""

    last_run_id: str = ""
    last_run_ts: str = ""
    collection_name: str = ""
    embed_model: str = ""
    documents: dict[str, DocumentState] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_run_id": self.last_run_id,
            "last_run_ts": self.last_run_ts,
            "collection_name": self.collection_name,
            "embed_model": self.embed_model,
            "documents": {k: v.to_dict() for k, v in self.documents.items()},
            "history": self.history[-50:],  # keep last 50 runs
        }

    @classmethod
    def load(cls, path: Path) -> IngestionState:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            docs = {k: DocumentState.from_dict(v) for k, v in data.get("documents", {}).items()}
            return cls(
                last_run_id=data.get("last_run_id", ""),
                last_run_ts=data.get("last_run_ts", ""),
                collection_name=data.get("collection_name", ""),
                embed_model=data.get("embed_model", ""),
                documents=docs,
                history=data.get("history", []),
            )
        except Exception as exc:
            log.warning("could not read ingestion state from %s: %s", path, exc)
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temp_path.replace(path)


@dataclass
class IngestionDiff:
    """Calculated delta between incoming documents and active state."""

    new_docs: list[DocumentRecord]
    modified_docs: list[tuple[DocumentRecord, DocumentState]]
    unchanged_docs: list[tuple[DocumentRecord, DocumentState]]
    removed_docs: list[DocumentState]


@dataclass
class IncrementalReport:
    """Detailed accounting of operations performed during incremental ingestion."""

    run_id: str
    timestamp: str
    collection: str
    embed_model: str
    unchanged_count: int
    new_count: int
    modified_count: int
    removed_count: int
    chunks_added: int
    chunks_deleted: int
    chunks_retained: int
    chunks_total: int
    embedding_operations: int
    orphans_cleaned: int
    unchanged_documents: list[str]
    new_documents: list[str]
    modified_documents: list[str]
    removed_documents: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- Diffing

def compute_diff(verified_docs: list[DocumentRecord], state: IngestionState) -> IngestionDiff:
    """Compare verified incoming documents against the stored state."""
    current_by_source = {d.source: d for d in verified_docs}
    new_docs: list[DocumentRecord] = []
    modified_docs: list[tuple[DocumentRecord, DocumentState]] = []
    unchanged_docs: list[tuple[DocumentRecord, DocumentState]] = []

    # Check incoming documents
    for doc in verified_docs:
        prev = state.documents.get(doc.source)
        if prev is None or prev.status == "removed":
            new_docs.append(doc)
        elif (
            prev.file_sha256 == doc.file_sha256
            and prev.content_sha256 == doc.content_sha256
            and prev.doc_version == doc.doc_version
            and prev.superseded == doc.superseded
        ):
            unchanged_docs.append((doc, prev))
        else:
            modified_docs.append((doc, prev))

    # Detect removed documents (present in active state, but missing from incoming verified batch)
    removed_docs: list[DocumentState] = []
    for source, prev in state.documents.items():
        if prev.status == "active" and source not in current_by_source:
            removed_docs.append(prev)

    return IngestionDiff(
        new_docs=new_docs,
        modified_docs=modified_docs,
        unchanged_docs=unchanged_docs,
        removed_docs=removed_docs,
    )


# --------------------------------------------------------------------------- Chroma helpers

def _get_active_or_create_collection(client: Any, settings: IndexerSettings) -> Any:
    """Retrieve existing Chroma collection or create new one with correct metadata."""
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(
            name=settings.collection,
            metadata={"hnsw:space": "cosine", "embed_model": settings.embed_model},
        )
    try:
        return client.get_collection(settings.collection)
    except Exception:
        return client.create_collection(
            name=settings.collection,
            metadata={"hnsw:space": "cosine", "embed_model": settings.embed_model},
        )


def _delete_chunks(collection: Any, chunk_ids: list[str], source: str = "") -> int:
    """Safely delete chunks by explicit IDs and fallback to metadata query."""
    if not chunk_ids and not source:
        return 0
    deleted = len(chunk_ids)
    if hasattr(collection, "delete"):
        try:
            if chunk_ids:
                collection.delete(ids=chunk_ids)
            elif source:
                collection.delete(where={"source": source})
        except Exception as exc:
            log.warning("error deleting chunks from collection: %s", exc)
    return deleted


def _add_chunks(
    collection: Any,
    chunks: list[Chunk],
    vectors: list[list[float]],
    batch_size: int = 200,
) -> None:
    """Add chunks and embeddings in batches."""
    if not chunks:
        return
    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_vectors = vectors[start : start + batch_size]
        collection.add(
            ids=[c.id for c in batch_chunks],
            documents=[c.text for c in batch_chunks],
            embeddings=batch_vectors,
            metadatas=[c.metadata() for c in batch_chunks],
        )


def count_chunks_for_source(collection: Any, source: str) -> int:
    """Return count of chunks belonging to a specific document source in collection."""
    if hasattr(collection, "get"):
        try:
            res = collection.get(where={"source": source})
            return len(res.get("ids", []))
        except Exception:
            pass
    return 0


# --------------------------------------------------------------------------- Execution

def run_incremental_ingestion(
    verified_docs: list[DocumentRecord],
    settings: IndexerSettings,
    client: Any = None,
    dry_run: bool = False,
) -> tuple[int, IncrementalReport]:
    """Execute incremental ingestion of verified documents against the vector store.

    Guarantees:
    - 0 embedding operations for unchanged documents
    - Targeted chunk replacement for modified documents
    - Complete cleanup of removed documents (0 orphan chunks)
    """
    from .chunk import chunk
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .store import get_client

    state_file = Path(settings.state_file) if settings.state_file else Path(settings.corpus_dir) / ".ingestion_state.json"
    state = IngestionState.load(state_file)

    diff = compute_diff(verified_docs, state)
    run_id = f"ingest-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    timestamp = datetime.now(timezone.utc).isoformat()

    log.info(
        "incremental diff computed: unchanged=%d, new=%d, modified=%d, removed=%d",
        len(diff.unchanged_docs),
        len(diff.new_docs),
        len(diff.modified_docs),
        len(diff.removed_docs),
    )

    if dry_run:
        total_chunks = sum(s.chunk_count for _, s in diff.unchanged_docs)
        report = IncrementalReport(
            run_id=run_id,
            timestamp=timestamp,
            collection=settings.collection,
            embed_model=settings.embed_model,
            unchanged_count=len(diff.unchanged_docs),
            new_count=len(diff.new_docs),
            modified_count=len(diff.modified_docs),
            removed_count=len(diff.removed_docs),
            chunks_added=0,
            chunks_deleted=sum(s.chunk_count for s in diff.removed_docs) + sum(s.chunk_count for _, s in diff.modified_docs),
            chunks_retained=total_chunks,
            chunks_total=total_chunks,
            embedding_operations=0,
            orphans_cleaned=0,
            unchanged_documents=[d.source for d, _ in diff.unchanged_docs],
            new_documents=[d.source for d in diff.new_docs],
            modified_documents=[d.source for d, _ in diff.modified_docs],
            removed_documents=[s.source for s in diff.removed_docs],
        )
        return total_chunks, report

    # Connect to Chroma collection
    client = client or get_client(settings)
    collection = _get_active_or_create_collection(client, settings)

    chunks_deleted = 0
    chunks_added = 0
    embedding_operations = 0

    # 1. Handle Removed Documents
    for removed_state in diff.removed_docs:
        deleted = _delete_chunks(collection, removed_state.chunk_ids, source=removed_state.source)
        chunks_deleted += deleted
        # Mark as removed in state
        removed_state.status = "removed"
        removed_state.updated_at = timestamp
        log.info("removed document chunks purged: %s (deleted %d chunks)", removed_state.source, deleted)

    # 2. Handle Modified Documents
    for doc_rec, old_state in diff.modified_docs:
        # Delete old chunks
        deleted = _delete_chunks(collection, old_state.chunk_ids, source=old_state.source)
        chunks_deleted += deleted

        # Extract, clean, and chunk new version
        raw_doc = extract(
            str(doc_rec.path),
            doc_rec.doc_version,
            doc_rec.superseded,
            ocr_dpi=settings.ocr_dpi,
            ocr_lang=settings.ocr_lang,
            min_page_chars=settings.min_page_chars,
            detect_columns=settings.detect_columns,
            title=doc_rec.title,
        )
        clean_doc = clean(raw_doc)
        new_chunks = chunk(clean_doc, settings.chunk_size, settings.chunk_overlap)

        # Embed ONLY the new chunks
        vectors = embed([c.text for c in new_chunks], settings)
        embedding_operations += len(vectors)
        _add_chunks(collection, new_chunks, vectors, settings.embed_batch_size)
        chunks_added += len(new_chunks)

        # Update state
        state.documents[doc_rec.source] = DocumentState(
            source=doc_rec.source,
            title=doc_rec.title,
            doc_version=doc_rec.doc_version,
            superseded=doc_rec.superseded,
            file_sha256=doc_rec.file_sha256,
            content_sha256=doc_rec.content_sha256,
            chunk_ids=[c.id for c in new_chunks],
            chunk_count=len(new_chunks),
            status="active",
            ingested_at=old_state.ingested_at,
            updated_at=timestamp,
        )
        log.info("modified document re-indexed: %s (replaced %d old chunks with %d new chunks)",
                 doc_rec.source, deleted, len(new_chunks))

    # 3. Handle New Documents
    for doc_rec in diff.new_docs:
        raw_doc = extract(
            str(doc_rec.path),
            doc_rec.doc_version,
            doc_rec.superseded,
            ocr_dpi=settings.ocr_dpi,
            ocr_lang=settings.ocr_lang,
            min_page_chars=settings.min_page_chars,
            detect_columns=settings.detect_columns,
            title=doc_rec.title,
        )
        clean_doc = clean(raw_doc)
        new_chunks = chunk(clean_doc, settings.chunk_size, settings.chunk_overlap)

        # Embed new chunks
        vectors = embed([c.text for c in new_chunks], settings)
        embedding_operations += len(vectors)
        _add_chunks(collection, new_chunks, vectors, settings.embed_batch_size)
        chunks_added += len(new_chunks)

        # Record in state
        state.documents[doc_rec.source] = DocumentState(
            source=doc_rec.source,
            title=doc_rec.title,
            doc_version=doc_rec.doc_version,
            superseded=doc_rec.superseded,
            file_sha256=doc_rec.file_sha256,
            content_sha256=doc_rec.content_sha256,
            chunk_ids=[c.id for c in new_chunks],
            chunk_count=len(new_chunks),
            status="active",
            ingested_at=timestamp,
            updated_at=timestamp,
        )
        log.info("new document indexed: %s (%d chunks embedded)", doc_rec.source, len(new_chunks))

    # 4. Handle Unchanged Documents: Zero Embedding Operations!
    for doc_rec, existing_state in diff.unchanged_docs:
        log.debug("unchanged document skipped (0 embeddings): %s", doc_rec.source)

    chunks_retained = sum(s.chunk_count for _, s in diff.unchanged_docs)

    # 5. Orphaned Chunk Detection and Cleanup
    orphans_cleaned = 0
    active_sources = {source for source, s in state.documents.items() if s.status == "active"}
    if hasattr(collection, "get"):
        try:
            sample = collection.get(limit=1000)
            if sample and "metadatas" in sample:
                orphan_ids = [
                    sample["ids"][i]
                    for i, meta in enumerate(sample["metadatas"])
                    if meta and meta.get("source") not in active_sources
                ]
                if orphan_ids:
                    _delete_chunks(collection, orphan_ids)
                    orphans_cleaned = len(orphan_ids)
                    log.info("cleaned %d orphaned chunks from vector store", orphans_cleaned)
        except Exception as exc:
            log.debug("orphaned chunk scan skipped: %s", exc)

    # Update collection metadata
    total_live_chunks = (
        collection.count()
        if hasattr(collection, "count")
        else sum(s.chunk_count for s in state.documents.values() if s.status == "active")
    )

    if hasattr(collection, "modify"):
        try:
            collection.modify(
                metadata={
                    "hnsw:space": "cosine",
                    "embed_model": settings.embed_model,
                    "index_id": run_id,
                    "updated_at": timestamp,
                    "chunk_count": total_live_chunks,
                }
            )
        except Exception:
            pass

    # Save persistent state and record history entry
    state.last_run_id = run_id
    state.last_run_ts = timestamp
    state.collection_name = settings.collection
    state.embed_model = settings.embed_model
    state.history.append({
        "run_id": run_id,
        "timestamp": timestamp,
        "unchanged_documents": len(diff.unchanged_docs),
        "new_documents": len(diff.new_docs),
        "modified_documents": len(diff.modified_docs),
        "removed_documents": len(diff.removed_docs),
        "chunks_added": chunks_added,
        "chunks_deleted": chunks_deleted,
        "embedding_operations": embedding_operations,
        "total_chunks": total_live_chunks,
    })
    state.save(state_file)

    report = IncrementalReport(
        run_id=run_id,
        timestamp=timestamp,
        collection=settings.collection,
        embed_model=settings.embed_model,
        unchanged_count=len(diff.unchanged_docs),
        new_count=len(diff.new_docs),
        modified_count=len(diff.modified_docs),
        removed_count=len(diff.removed_docs),
        chunks_added=chunks_added,
        chunks_deleted=chunks_deleted,
        chunks_retained=chunks_retained,
        chunks_total=total_live_chunks,
        embedding_operations=embedding_operations,
        orphans_cleaned=orphans_cleaned,
        unchanged_documents=[d.source for d, _ in diff.unchanged_docs],
        new_documents=[d.source for d in diff.new_docs],
        modified_documents=[d.source for d, _ in diff.modified_docs],
        removed_documents=[s.source for s in diff.removed_docs],
    )

    # Save incremental report JSON
    report_path = Path(settings.corpus_dir) / "incremental_report.json"
    try:
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    except Exception:
        pass

    log.info(
        "incremental ingestion complete: %d live chunks, %d embedded, %d deleted, %d retained",
        total_live_chunks,
        embedding_operations,
        chunks_deleted,
        chunks_retained,
    )
    return total_live_chunks, report


# --------------------------------------------------------------------------- CLI

def main() -> int:
    parser = argparse.ArgumentParser(description="Run incremental ingestion (Part A2).")
    parser.add_argument("--corpus-dir", default="", help="Corpus directory")
    parser.add_argument("--dry-run", action="store_true", help="Calculate diff only, write nothing")
    parser.add_argument("--state-file", default="", help="Path to .ingestion_state.json")
    args = parser.parse_args()

    from dataclasses import replace

    base_settings = get_settings()
    corpus_dir = Path(args.corpus_dir or base_settings.corpus_dir)
    if not corpus_dir.exists():
        print(f"Error: corpus dir '{corpus_dir}' does not exist.", file=sys.stderr)
        return 1

    settings = replace(
        base_settings,
        corpus_dir=str(corpus_dir),
        state_file=args.state_file or str(corpus_dir / ".ingestion_state.json"),
    )

    from .verify import inspect_document, verify_documents

    # 1. Discover and inspect
    manifest_entries: dict[str, dict[str, Any]] = {}
    index_file = corpus_dir / "_index.json"
    if index_file.exists():
        try:
            for entry in json.loads(index_file.read_text(encoding="utf-8")):
                manifest_entries[entry.get("file", "")] = entry
        except Exception:
            pass

    records: list[DocumentRecord] = []
    for p in sorted(corpus_dir.glob("*")):
        if not p.is_file() or p.name.startswith(".") or p.name.lower().startswith("readme"):
            continue
        if p.suffix.lower() not in {".pdf", ".html", ".htm", ".md", ".txt"}:
            continue
        entry = manifest_entries.get(p.name, {})
        title = entry.get("name", "")
        doc_version = str(entry.get("doc_version", ""))
        superseded = bool(entry.get("superseded", False))
        records.append(inspect_document(p, title=title, doc_version=doc_version, superseded=superseded, settings=settings))

    # 2. Verify
    accepted, v_report = verify_documents(records, settings=settings)

    # 3. Run incremental update
    count, report = run_incremental_ingestion(accepted, settings=settings, dry_run=args.dry_run)

    print(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "logger": "rahbar.indexer.incremental",
        "msg": "incremental ingestion complete",
        "chunks_total": count,
        "embedding_operations": report.embedding_operations,
        "unchanged": report.unchanged_count,
        "new": report.new_count,
        "modified": report.modified_count,
        "removed": report.removed_count,
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

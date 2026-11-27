"""Stage 2.5: Document Lifecycle Archive & Restoration (Assignment 3, Part A3).

Provides:
1. Version archival: preserves older/superseded document versions in an archive directory.
2. Metadata tagging: sets superseded=True in Chroma chunk metadata, excluding them from
   default retrieval while preserving historical traceability.
3. Restoration / Rollback: restores a previously archived document back to the active corpus,
   updating metadata (superseded=False) without creating duplicate chunks.
4. Audit tracking: logs every archive and restoration event in archive_manifest.json and
   .ingestion_state.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import IndexerSettings, get_settings
from .incremental import IngestionState
from .verify import compute_file_sha256, compute_text_sha256

log = logging.getLogger("rahbar.indexer.archive")


@dataclass
class ArchivedDocumentRecord:
    """Entry in archive_manifest.json."""

    source: str
    archive_file: str
    doc_version: str
    archived_at: str
    reason: str
    superseded_by: str = ""
    file_sha256: str = ""
    content_sha256: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    chunk_count: int = 0
    restored_at: str | None = None
    restoration_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchivedDocumentRecord:
        return cls(**data)


class ArchiveManager:
    """Manages document version preservation, supersede tagging, and rollback restoration."""

    def __init__(self, settings: IndexerSettings | None = None, archive_dir: Path | None = None) -> None:
        self.settings = settings or get_settings()
        corpus_path = Path(self.settings.corpus_dir)
        self.archive_dir = archive_dir or (
            Path(self.settings.archive_dir) if self.settings.archive_dir else corpus_path / "archive"
        )
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.archive_dir / "archive_manifest.json"
        self.state_file = (
            Path(self.settings.state_file) if self.settings.state_file else corpus_path / ".ingestion_state.json"
        )

    def load_manifest(self) -> dict[str, ArchivedDocumentRecord]:
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return {k: ArchivedDocumentRecord.from_dict(v) for k, v in data.items()}
        except Exception as exc:
            log.warning("could not read archive manifest: %s", exc)
            return {}

    def save_manifest(self, entries: dict[str, ArchivedDocumentRecord]) -> None:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.manifest_path.with_suffix(".tmp")
        payload = {k: v.to_dict() for k, v in entries.items()}
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self.manifest_path)

    def archive_document(
        self,
        doc_path: Path,
        doc_version: str,
        reason: str = "superseded_by_newer_version",
        superseded_by: str = "",
        client: Any = None,
    ) -> ArchivedDocumentRecord:
        """Archive a document version into archive_dir and mark its chunks as superseded."""
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        iso_ts = datetime.now(timezone.utc).isoformat()
        stem = doc_path.stem
        suffix = doc_path.suffix
        archive_name = f"{stem}_v{doc_version}_{timestamp_str}{suffix}"
        archive_dest = self.archive_dir / archive_name

        shutil.copy2(doc_path, archive_dest)
        file_hash = compute_file_sha256(doc_path)
        content_hash = ""
        try:
            content_hash = compute_text_sha256(doc_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            content_hash = file_hash

        # Update IngestionState
        state = IngestionState.load(self.state_file)
        chunk_ids: list[str] = []
        chunk_count = 0
        if doc_path.name in state.documents:
            doc_state = state.documents[doc_path.name]
            doc_state.status = "superseded"
            doc_state.superseded = True
            doc_state.updated_at = iso_ts
            chunk_ids = list(doc_state.chunk_ids)
            chunk_count = doc_state.chunk_count

        # Mark chunks as superseded=True in Chroma vector collection
        if client is not None and hasattr(client, "get_collection"):
            try:
                collection = client.get_collection(self.settings.collection)
                self._mark_collection_superseded(collection, doc_path.name, superseded=True)
            except Exception as exc:
                log.debug("could not update collection metadata to superseded: %s", exc)

        record = ArchivedDocumentRecord(
            source=doc_path.name,
            archive_file=archive_name,
            doc_version=doc_version,
            archived_at=iso_ts,
            reason=reason,
            superseded_by=superseded_by,
            file_sha256=file_hash,
            content_sha256=content_hash,
            chunk_ids=chunk_ids,
            chunk_count=chunk_count,
        )

        manifest = self.load_manifest()
        manifest[archive_name] = record
        self.save_manifest(manifest)

        state.history.append({
            "event": "archive_document",
            "timestamp": iso_ts,
            "source": doc_path.name,
            "archive_file": archive_name,
            "doc_version": doc_version,
            "reason": reason,
        })
        state.save(self.state_file)

        log.info("archived document %s as %s (version %s)", doc_path.name, archive_name, doc_version)
        return record

    def list_archived(self) -> list[ArchivedDocumentRecord]:
        """List all archived document versions."""
        manifest = self.load_manifest()
        return list(manifest.values())

    def restore_document(
        self,
        identifier: str,
        target_dir: Path | None = None,
        make_active: bool = True,
        client: Any = None,
    ) -> tuple[Path, int]:
        """Restore an archived document back to the active corpus and un-supersede its chunks.

        Guarantees:
        - Does NOT create duplicate chunks (updates existing chunk metadata or re-indexes cleanly).
        - Records restoration operation in state and archive audit history.
        """
        manifest = self.load_manifest()
        target_record: ArchivedDocumentRecord | None = None
        record_key: str | None = None

        # 1. Locate record by archive_file or by source
        if identifier in manifest:
            record_key = identifier
            target_record = manifest[identifier]
        else:
            for k, rec in manifest.items():
                if rec.source == identifier or rec.archive_file == identifier:
                    record_key = k
                    target_record = rec
                    break

        if not target_record:
            raise FileNotFoundError(f"Archived document '{identifier}' not found in archive manifest.")

        archive_file_path = self.archive_dir / target_record.archive_file
        if not archive_file_path.exists():
            raise FileNotFoundError(f"Archive file '{archive_file_path}' does not exist on disk.")

        dest_dir = target_dir or Path(self.settings.corpus_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        restored_path = dest_dir / target_record.source

        # Copy back to active corpus
        shutil.copy2(archive_file_path, restored_path)

        iso_ts = datetime.now(timezone.utc).isoformat()
        target_record.restored_at = iso_ts
        target_record.restoration_count += 1
        manifest[record_key or target_record.archive_file] = target_record
        self.save_manifest(manifest)

        # Update IngestionState
        state = IngestionState.load(self.state_file)
        if target_record.source in state.documents:
            doc_state = state.documents[target_record.source]
            doc_state.status = "active" if make_active else "superseded"
            doc_state.superseded = not make_active
            doc_state.updated_at = iso_ts
        else:
            # Reconstruct document state from archive record
            from .incremental import DocumentState
            state.documents[target_record.source] = DocumentState(
                source=target_record.source,
                title=target_record.source,
                doc_version=target_record.doc_version,
                superseded=not make_active,
                file_sha256=target_record.file_sha256,
                content_sha256=target_record.content_sha256,
                chunk_ids=target_record.chunk_ids,
                chunk_count=target_record.chunk_count,
                status="active" if make_active else "superseded",
                ingested_at=target_record.archived_at,
                updated_at=iso_ts,
            )

        # Update Chroma collection chunks: mark superseded=False
        chunks_restored = target_record.chunk_count
        if client is not None and hasattr(client, "get_collection"):
            try:
                collection = client.get_collection(self.settings.collection)
                self._mark_collection_superseded(collection, target_record.source, superseded=not make_active)
                if hasattr(collection, "count"):
                    chunks_restored = collection.count()
            except Exception as exc:
                log.debug("could not update collection metadata on restore: %s", exc)

        state.history.append({
            "event": "restore_document",
            "timestamp": iso_ts,
            "source": target_record.source,
            "archive_file": target_record.archive_file,
            "doc_version": target_record.doc_version,
            "chunks_restored": chunks_restored,
            "make_active": make_active,
        })
        state.save(self.state_file)

        log.info(
            "restored document %s from %s (version %s, %d chunks)",
            target_record.source,
            target_record.archive_file,
            target_record.doc_version,
            chunks_restored,
        )
        return restored_path, chunks_restored

    def _mark_collection_superseded(self, collection: Any, source: str, superseded: bool) -> None:
        """Update superseded metadata flag for all chunks of a source without duplicating vectors."""
        if hasattr(collection, "get"):
            res = collection.get(where={"source": source})
            chunk_ids = res.get("ids", [])
            metadatas = res.get("metadatas", [])
            if chunk_ids and metadatas:
                updated_metas = []
                for meta in metadatas:
                    m = dict(meta) if meta else {}
                    m["superseded"] = superseded
                    updated_metas.append(m)
                if hasattr(collection, "update"):
                    collection.update(ids=chunk_ids, metadatas=updated_metas)
                elif hasattr(collection, "modify"):
                    # fallback if in-memory test collection
                    for cid, m in zip(chunk_ids, updated_metas):
                        if hasattr(collection, "rows") and cid in collection.rows:
                            doc, vec, _ = collection.rows[cid]
                            collection.rows[cid] = (doc, vec, m)


# --------------------------------------------------------------------------- CLI

def main() -> int:
    parser = argparse.ArgumentParser(description="Manage document archival and restoration (Part A3).")
    parser.add_argument("--list", action="store_true", help="List all archived document versions")
    parser.add_argument("--archive", metavar="FILE", help="Archive a document file")
    parser.add_argument("--doc-version", default="unknown", help="Document version string")
    parser.add_argument("--reason", default="superseded_by_newer_version", help="Reason for archival")
    parser.add_argument("--restore", metavar="IDENTIFIER", help="Restore an archived document by filename or archive name")
    parser.add_argument("--corpus-dir", default="", help="Corpus directory")
    parser.add_argument("--archive-dir", default="", help="Archive directory")
    args = parser.parse_args()

    settings = get_settings()
    corpus_dir = Path(args.corpus_dir or settings.corpus_dir)
    archive_dir = Path(args.archive_dir or (settings.archive_dir if settings.archive_dir else corpus_dir / "archive"))

    from dataclasses import replace
    configured_settings = replace(
        settings,
        corpus_dir=str(corpus_dir),
        archive_dir=str(archive_dir),
        state_file=str(corpus_dir / ".ingestion_state.json"),
    )

    manager = ArchiveManager(settings=configured_settings, archive_dir=archive_dir)

    if args.list:
        archived_list = manager.list_archived()
        print(json.dumps([r.to_dict() for r in archived_list], indent=2))
        return 0

    if args.archive:
        doc_path = Path(args.archive)
        if not doc_path.exists():
            doc_path = corpus_dir / args.archive
        if not doc_path.exists():
            print(f"Error: Document '{args.archive}' not found.", file=sys.stderr)
            return 1
        rec = manager.archive_document(doc_path, doc_version=args.doc_version, reason=args.reason)
        print(json.dumps({
            "status": "archived",
            "source": rec.source,
            "archive_file": rec.archive_file,
            "doc_version": rec.doc_version,
            "archived_at": rec.archived_at,
        }, indent=2))
        return 0

    if args.restore:
        try:
            path, chunks = manager.restore_document(args.restore)
            print(json.dumps({
                "status": "restored",
                "source": path.name,
                "restored_path": str(path),
                "chunks_restored": chunks,
            }, indent=2))
            return 0
        except Exception as exc:
            print(f"Error restoring document: {exc}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

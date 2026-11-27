"""Unit and integration tests for Incremental Ingestion (Assignment 3, Part A2).

Tests cover:
- Idempotent repeated ingestion
- Zero embedding operations on unchanged corpus
- Selective chunk replacement on changed documents
- Complete chunk cleanup on removed documents (zero orphan chunks)
- New document detection and embedding
- State persistence and ingestion history tracking
- Orphan chunk detection and purging
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pipeline.config import IndexerSettings
from pipeline.incremental import (
    DocumentState,
    IngestionState,
    compute_diff,
    count_chunks_for_source,
    run_incremental_ingestion,
)
from pipeline.models import Page
from pipeline.verify import DocumentRecord, compute_file_sha256, compute_text_sha256


class MockCollection:
    """In-memory Chroma collection supporting get, add, delete, and count."""

    def __init__(self, name: str = "rahbar") -> None:
        self.name = name
        self.metadata: dict[str, Any] = {}
        # chunk_id -> (text, embedding, metadata)
        self.rows: dict[str, tuple[str, list[float], dict[str, Any]]] = {}

    def count(self) -> int:
        return len(self.rows)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for cid, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            self.rows[cid] = (doc, emb, meta)

    def get(self, ids: list[str] | None = None, where: dict[str, Any] | None = None, limit: int = 1000) -> dict[str, Any]:
        matched_ids = []
        matched_metas = []
        matched_docs = []
        for cid, (doc, emb, meta) in self.rows.items():
            if ids is not None and cid not in ids:
                continue
            if where:
                match = True
                for k, v in where.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            matched_ids.append(cid)
            matched_metas.append(meta)
            matched_docs.append(doc)
            if len(matched_ids) >= limit:
                break
        return {"ids": matched_ids, "metadatas": matched_metas, "documents": matched_docs}

    def delete(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> None:
        if ids:
            for cid in ids:
                self.rows.pop(cid, None)
        elif where:
            to_del = [cid for cid, (_, _, meta) in self.rows.items()
                      if all(meta.get(k) == v for k, v in where.items())]
            for cid in to_del:
                self.rows.pop(cid, None)

    def modify(self, metadata: dict[str, Any] | None = None) -> None:
        if metadata:
            self.metadata.update(metadata)


class MockClient:
    def __init__(self) -> None:
        self.collections: dict[str, MockCollection] = {}

    def get_or_create_collection(self, name: str, metadata: dict[str, Any] | None = None) -> MockCollection:
        if name not in self.collections:
            col = MockCollection(name)
            if metadata:
                col.metadata = metadata
            self.collections[name] = col
        return self.collections[name]

    def get_collection(self, name: str) -> MockCollection:
        return self.collections[name]


def make_doc_record(path: Path, source: str, title: str, version: str, content: str) -> DocumentRecord:
    path.write_text(content, encoding="utf-8")
    file_hash = compute_file_sha256(path)
    content_hash = compute_text_sha256(content)
    page = Page(source, 1, content)
    return DocumentRecord(
        path=path,
        source=source,
        title=title,
        doc_version=version,
        superseded=False,
        file_sha256=file_hash,
        content_sha256=content_hash,
        pages=[page],
        raw_text=content,
        char_count=len(content),
        word_count=len(content.split()),
        page_count=1,
        cleaned_text=content,
        lineage_key=title.lower().replace(" ", "_"),
    )


class TestIncrementalIngestion(unittest.TestCase):
    """Test the complete incremental update lifecycle."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.corpus_dir = Path(self.tmp_dir) / "corpus"
        self.corpus_dir.mkdir(parents=True)
        self.client = MockClient()
        self.settings = IndexerSettings(
            embed_model="dummy-768",
            chroma_host="localhost",
            chroma_port=8000,
            collection="rahbar",
            corpus_dir=str(self.corpus_dir),
            chunk_size=100,
            chunk_overlap=20,
            ollama_url="http://dummy",
            state_file=str(self.corpus_dir / ".ingestion_state.json"),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_first_run_indexes_all_documents(self) -> None:
        doc1 = make_doc_record(
            self.corpus_dir / "doc1.txt",
            "doc1.txt",
            "Policy One",
            "2026-06",
            "# Policy One\n\nStudents must register before the first week of classes.",
        )
        doc2 = make_doc_record(
            self.corpus_dir / "doc2.txt",
            "doc2.txt",
            "Policy Two",
            "2026-06",
            "# Policy Two\n\nTuition fees must be paid in full to confirm registration.",
        )

        count, report = run_incremental_ingestion([doc1, doc2], self.settings, client=self.client)
        self.assertGreater(count, 0)
        self.assertEqual(report.new_count, 2)
        self.assertEqual(report.unchanged_count, 0)
        self.assertEqual(report.modified_count, 0)
        self.assertEqual(report.removed_count, 0)
        self.assertEqual(report.embedding_operations, count)

        # Vector store holds chunks for both docs
        self.assertGreater(count_chunks_for_source(self.client.get_collection("rahbar"), "doc1.txt"), 0)
        self.assertGreater(count_chunks_for_source(self.client.get_collection("rahbar"), "doc2.txt"), 0)

    def test_unchanged_corpus_produces_zero_embeddings(self) -> None:
        """Crucial Part A2 requirement: unchanged corpus produces zero embedding operations."""
        doc1 = make_doc_record(
            self.corpus_dir / "doc1.txt",
            "doc1.txt",
            "Policy One",
            "2026-06",
            "# Policy One\n\nStudents must register before the first week of classes.",
        )

        # Run 1: Initial ingestion
        count1, report1 = run_incremental_ingestion([doc1], self.settings, client=self.client)
        self.assertGreater(count1, 0)
        self.assertEqual(report1.embedding_operations, count1)

        # Run 2: Ingest the exact same document without changes
        count2, report2 = run_incremental_ingestion([doc1], self.settings, client=self.client)

        # MUST have zero embedding operations and zero added/deleted chunks
        self.assertEqual(count2, count1)
        self.assertEqual(report2.embedding_operations, 0, "Unchanged document produced non-zero embeddings!")
        self.assertEqual(report2.unchanged_count, 1)
        self.assertEqual(report2.new_count, 0)
        self.assertEqual(report2.modified_count, 0)
        self.assertEqual(report2.chunks_added, 0)
        self.assertEqual(report2.chunks_deleted, 0)

    def test_changed_document_replaces_only_its_own_chunks(self) -> None:
        """When doc1 changes and doc2 is unchanged, only doc1 is re-embedded."""
        doc1 = make_doc_record(
            self.corpus_dir / "doc1.txt",
            "doc1.txt",
            "Policy One",
            "2026-06",
            "# Policy One\n\nVersion 1 text for policy one.",
        )
        doc2 = make_doc_record(
            self.corpus_dir / "doc2.txt",
            "doc2.txt",
            "Policy Two",
            "2026-06",
            "# Policy Two\n\nVersion 1 text for policy two which stays completely unchanged.",
        )

        # Initial run
        count_init, rep_init = run_incremental_ingestion([doc1, doc2], self.settings, client=self.client)
        doc2_chunks_before = count_chunks_for_source(self.client.get_collection("rahbar"), "doc2.txt")

        # Now modify doc1
        doc1_v2 = make_doc_record(
            self.corpus_dir / "doc1.txt",
            "doc1.txt",
            "Policy One",
            "2026-06",
            "# Policy One\n\nVersion 2 updated text with substantial modifications and revisions.",
        )

        # Run incremental update
        count_after, rep_update = run_incremental_ingestion([doc1_v2, doc2], self.settings, client=self.client)

        self.assertEqual(rep_update.unchanged_count, 1)  # doc2 unchanged
        self.assertEqual(rep_update.modified_count, 1)   # doc1 modified
        self.assertEqual(rep_update.new_count, 0)
        self.assertEqual(rep_update.removed_count, 0)

        # Embeddings computed ONLY for doc1's chunks
        self.assertEqual(rep_update.embedding_operations, rep_update.chunks_added)
        # doc2 chunks in vector store were untouched
        doc2_chunks_after = count_chunks_for_source(self.client.get_collection("rahbar"), "doc2.txt")
        self.assertEqual(doc2_chunks_before, doc2_chunks_after)

    def test_removed_document_leaves_zero_orphan_chunks(self) -> None:
        """When doc1 is deleted from the corpus, all of its chunks are purged from vector store."""
        doc1 = make_doc_record(
            self.corpus_dir / "doc1.txt",
            "doc1.txt",
            "Policy One",
            "2026-06",
            "# Policy One\n\nText of policy one to be removed later.",
        )
        doc2 = make_doc_record(
            self.corpus_dir / "doc2.txt",
            "doc2.txt",
            "Policy Two",
            "2026-06",
            "# Policy Two\n\nText of policy two which remains active.",
        )

        # Ingest both
        run_incremental_ingestion([doc1, doc2], self.settings, client=self.client)
        self.assertGreater(count_chunks_for_source(self.client.get_collection("rahbar"), "doc1.txt"), 0)

        # Remove doc1: batch now only contains doc2
        (self.corpus_dir / "doc1.txt").unlink()
        count_after, rep_del = run_incremental_ingestion([doc2], self.settings, client=self.client)

        self.assertEqual(rep_del.removed_count, 1)
        self.assertEqual(rep_del.removed_documents, ["doc1.txt"])
        self.assertEqual(rep_del.unchanged_count, 1)
        self.assertEqual(rep_del.embedding_operations, 0)

        # Verify PROOF of zero orphan chunks for doc1.txt in collection
        doc1_remaining = count_chunks_for_source(self.client.get_collection("rahbar"), "doc1.txt")
        self.assertEqual(doc1_remaining, 0, "Orphan chunks for doc1.txt were left in vector store!")

    def test_ingestion_history_and_state_tracking(self) -> None:
        """State records run metadata, document checksums, chunk IDs, and run history."""
        doc1 = make_doc_record(
            self.corpus_dir / "doc1.txt",
            "doc1.txt",
            "Policy One",
            "2026-06",
            "# Policy One\n\nContent for tracking test.",
        )

        # Run 1
        run_incremental_ingestion([doc1], self.settings, client=self.client)
        state_file = Path(self.settings.state_file)
        self.assertTrue(state_file.exists())

        state = IngestionState.load(state_file)
        self.assertEqual(len(state.documents), 1)
        doc_state = state.documents["doc1.txt"]
        self.assertEqual(doc_state.source, "doc1.txt")
        self.assertEqual(doc_state.status, "active")
        self.assertGreater(len(doc_state.chunk_ids), 0)
        self.assertEqual(len(state.history), 1)

        # Run 2 (unchanged)
        run_incremental_ingestion([doc1], self.settings, client=self.client)
        state = IngestionState.load(state_file)
        self.assertEqual(len(state.history), 2)
        self.assertEqual(state.history[-1]["embedding_operations"], 0)


if __name__ == "__main__":
    unittest.main()

"""Unit and integration tests for Document Archival and Restoration (Assignment 3, Part A3).

Tests cover:
- Preserving previous versions in archive_dir
- Marking superseded documents in state and collection metadata
- Tracking archive manifests and history logs
- Restoring archived documents without creating duplicate chunks
- Rollback demonstration: superseded=False un-marking
- Non-existent document error handling
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pipeline.archive import ArchiveManager, ArchivedDocumentRecord
from pipeline.config import IndexerSettings
from pipeline.incremental import DocumentState, IngestionState


class MockChromaCollection:
    def __init__(self, name: str = "rahbar") -> None:
        self.name = name
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

    def get(self, ids: list[str] | None = None, where: dict[str, Any] | None = None) -> dict[str, Any]:
        res_ids = []
        res_metas = []
        for cid, (_, _, meta) in self.rows.items():
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            res_ids.append(cid)
            res_metas.append(dict(meta))
        return {"ids": res_ids, "metadatas": res_metas}

    def update(self, ids: list[str], metadatas: list[dict[str, Any]]) -> None:
        for cid, meta in zip(ids, metadatas):
            if cid in self.rows:
                doc, vec, _ = self.rows[cid]
                self.rows[cid] = (doc, vec, meta)


class MockChromaClient:
    def __init__(self) -> None:
        self.collection = MockChromaCollection("rahbar")

    def get_collection(self, name: str) -> MockChromaCollection:
        return self.collection


class TestArchiveAndRestoration(unittest.TestCase):
    """Test document version lifecycle: archive, supersede tagging, and rollback restoration."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.corpus_dir = Path(self.tmp_dir) / "corpus"
        self.archive_dir = Path(self.tmp_dir) / "archive"
        self.corpus_dir.mkdir(parents=True)
        self.archive_dir.mkdir(parents=True)
        self.client = MockChromaClient()

        self.settings = IndexerSettings(
            embed_model="dummy-768",
            chroma_host="localhost",
            chroma_port=8000,
            collection="rahbar",
            corpus_dir=str(self.corpus_dir),
            chunk_size=100,
            chunk_overlap=20,
            ollama_url="http://dummy",
            archive_dir=str(self.archive_dir),
            state_file=str(self.corpus_dir / ".ingestion_state.json"),
        )
        self.manager = ArchiveManager(settings=self.settings, archive_dir=self.archive_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_archive_preserves_version_and_updates_metadata(self) -> None:
        # Create doc in corpus
        doc_file = self.corpus_dir / "academic_rules_2023.txt"
        doc_file.write_text("Old 2023 academic rules content.", encoding="utf-8")

        # Simulate 2 chunks in collection for this doc
        chunk_ids = ["academic_rules_2023-0000", "academic_rules_2023-0001"]
        self.client.collection.add(
            ids=chunk_ids,
            documents=["Old 2023 academic rules content part 1", "part 2"],
            embeddings=[[0.1] * 768, [0.2] * 768],
            metadatas=[
                {"source": "academic_rules_2023.txt", "doc_version": "2023-08", "superseded": False},
                {"source": "academic_rules_2023.txt", "doc_version": "2023-08", "superseded": False},
            ],
        )

        # Set up state
        state = IngestionState(
            documents={
                "academic_rules_2023.txt": DocumentState(
                    source="academic_rules_2023.txt",
                    title="Academic Rules 2023",
                    doc_version="2023-08",
                    superseded=False,
                    file_sha256="hash23",
                    content_sha256="hash23",
                    chunk_ids=chunk_ids,
                    chunk_count=2,
                    status="active",
                    ingested_at="2026-11-01T00:00:00Z",
                    updated_at="2026-11-01T00:00:00Z",
                )
            }
        )
        state.save(Path(self.settings.state_file))

        # Archive document
        rec = self.manager.archive_document(
            doc_file,
            doc_version="2023-08",
            reason="superseded_by_2026_edition",
            superseded_by="academic_rules_2026.txt",
            client=self.client,
        )

        # 1. File exists in archive_dir
        archived_file = self.archive_dir / rec.archive_file
        self.assertTrue(archived_file.exists())
        self.assertEqual(archived_file.read_text(encoding="utf-8"), "Old 2023 academic rules content.")

        # 2. Manifest record created
        manifest = self.manager.load_manifest()
        self.assertIn(rec.archive_file, manifest)
        self.assertEqual(manifest[rec.archive_file].doc_version, "2023-08")
        self.assertEqual(manifest[rec.archive_file].reason, "superseded_by_2026_edition")

        # 3. Chunks in collection tagged superseded=True (excluded from default retrieval)
        res = self.client.collection.get(where={"source": "academic_rules_2023.txt"})
        for meta in res["metadatas"]:
            self.assertTrue(meta["superseded"], "Chunk metadata superseded was not set to True!")

        # 4. IngestionState updated
        updated_state = IngestionState.load(Path(self.settings.state_file))
        self.assertEqual(updated_state.documents["academic_rules_2023.txt"].status, "superseded")
        self.assertTrue(updated_state.documents["academic_rules_2023.txt"].superseded)

    def test_restore_document_unmarks_superseded_without_creating_duplicates(self) -> None:
        """Rollback test: Restoring a previously archived document must NOT duplicate chunks."""
        doc_file = self.corpus_dir / "academic_rules_2023.txt"
        doc_file.write_text("Old 2023 academic rules content.", encoding="utf-8")

        chunk_ids = ["academic_rules_2023-0000", "academic_rules_2023-0001"]
        self.client.collection.add(
            ids=chunk_ids,
            documents=["chunk 0", "chunk 1"],
            embeddings=[[0.1] * 768, [0.2] * 768],
            metadatas=[
                {"source": "academic_rules_2023.txt", "doc_version": "2023-08", "superseded": False},
                {"source": "academic_rules_2023.txt", "doc_version": "2023-08", "superseded": False},
            ],
        )

        # Archive it first
        rec = self.manager.archive_document(doc_file, doc_version="2023-08", client=self.client)
        # Delete original from corpus to simulate replacement
        doc_file.unlink()
        self.assertFalse(doc_file.exists())

        count_before_restore = self.client.collection.count()
        self.assertEqual(count_before_restore, 2)

        # Restore from archive
        restored_path, chunks_restored = self.manager.restore_document(
            rec.archive_file,
            make_active=True,
            client=self.client,
        )

        # 1. Document restored back to corpus
        self.assertTrue(restored_path.exists())
        self.assertEqual(restored_path.read_text(encoding="utf-8"), "Old 2023 academic rules content.")

        # 2. Chunk count must NOT have doubled (zero duplicate chunks)
        count_after_restore = self.client.collection.count()
        self.assertEqual(
            count_after_restore,
            count_before_restore,
            f"Restoration created duplicate chunks: had {count_before_restore}, now {count_after_restore}!",
        )

        # 3. Chunks are active again (superseded=False)
        res = self.client.collection.get(where={"source": "academic_rules_2023.txt"})
        for meta in res["metadatas"]:
            self.assertFalse(meta["superseded"], "Chunk metadata superseded was not reset to False on restore!")

        # 4. Audit trail recorded in state history
        state = IngestionState.load(Path(self.settings.state_file))
        restore_events = [ev for ev in state.history if ev.get("event") == "restore_document"]
        self.assertEqual(len(restore_events), 1)
        self.assertEqual(restore_events[0]["source"], "academic_rules_2023.txt")

    def test_restore_nonexistent_raises_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.manager.restore_document("nonexistent_policy_v99.pdf")


if __name__ == "__main__":
    unittest.main()

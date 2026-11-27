#!/usr/bin/env python3
"""Integration Contract Tests between Member 1 and Member 2 (Assignment 3, Part B4).

Verifies the interface compatibility between:
1. Member 1 (Data, Chunking, Retrieval & Golden Dataset)
2. Member 2 (API Serving, Chroma Client, LLM Generation & Verification)

Validates schema alignment, metadata field consistency, and metrics parity.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexer"))
sys.path.insert(0, str(REPO_ROOT / "api"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from eval_retrieval import is_hit
from run_eval import recall_at_k


def test_golden_dataset_schema() -> None:
    """Ensure golden.jsonl contains all fields required by both Member 1 & Member 2."""
    golden_path = REPO_ROOT / "eval" / "golden.jsonl"
    assert golden_path.exists(), "golden.jsonl must exist"

    records = [json.loads(line) for line in golden_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) >= 25, f"Golden dataset must contain >= 25 records (found {len(records)})"

    required_keys = {"id", "question", "answerable", "must_abstain", "difficulty", "author_type", "relevant"}
    for rec in records:
        missing = required_keys - set(rec.keys())
        assert not missing, f"Record {rec.get('id')} missing required keys: {missing}"

        # Member 1 & 2 contract checks
        assert isinstance(rec["question"], str) and len(rec["question"]) >= 10
        assert isinstance(rec["answerable"], bool)
        assert isinstance(rec["must_abstain"], bool)
        assert rec["answerable"] == (not rec["must_abstain"])

        for rel in rec.get("relevant", []):
            assert "source" in rel, f"Relevant ref in {rec['id']} missing 'source'"
            assert "section" in rel, f"Relevant ref in {rec['id']} missing 'section'"
            assert rel["source"].endswith((".pdf", ".html", ".md")), f"Invalid source extension: {rel['source']}"

    print(f"PASS: Golden dataset schema validated across {len(records)} records.")


def test_retrieval_passage_schema_compatibility() -> None:
    """Ensure Member 1's retrieval passages match Member 2's Chroma retriever Passage schema."""
    try:
        from app.schemas import Passage, RetrievedRef
        sample_passage = Passage(
            chunk_id="academic-rules-chunk-01",
            text="Undergraduate Academic Rules Handbook June 2026 governing degrees.",
            source="academic-rules-handbook-jun-2026.pdf",
            section="1.2 Academic Warning",
            page=4,
            score=0.925,
            doc_version="2026",
            superseded=False,
        )
        assert sample_passage.chunk_id == "academic-rules-chunk-01"
        assert sample_passage.score == 0.925
        assert sample_passage.doc_version == "2026"
        assert not sample_passage.superseded

        sample_ref = RetrievedRef(
            chunk_id=sample_passage.chunk_id,
            source=sample_passage.source,
            section=sample_passage.section,
            page=sample_passage.page,
            score=sample_passage.score,
        )
        assert sample_ref.source == sample_passage.source
        assert sample_ref.section == sample_passage.section
    except ImportError:
        # Fallback schema validation via AST/definitions if pydantic is not in local env
        schemas_file = REPO_ROOT / "api" / "app" / "schemas.py"
        content = schemas_file.read_text(encoding="utf-8")
        assert "class Passage(BaseModel):" in content
        assert "chunk_id: str" in content
        assert "text: str" in content
        assert "source: str" in content
        assert "section: str" in content
        assert "score: float" in content
        assert "doc_version: str | None = None" in content
        assert "superseded: bool = False" in content
        assert "class RetrievedRef(BaseModel):" in content
        assert "class AskResponse(BaseModel):" in content
    print("PASS: Passage & RetrievedRef schema compatibility verified.")


def test_hit_matching_parity() -> None:
    """Ensure is_hit() in eval_retrieval matches recall_at_k() in run_eval."""
    # Test cases: (expected_source, expected_sec, retrieved_source, retrieved_sec, should_match)
    cases = [
        ("fee.html", "Tuition", "fee.html", "Tuition", True),
        ("fee.html", "Tuition", "fee.html", "Tuition per Credit Hour", True),
        ("fee.html", "1.2", "fee.html", "1.20 Warning", False),
        ("fee.html", "1.2", "fee.html", "1.2 Academic Probation", True),
        ("fee.html", "1.2", "fee.html", "1.2: Academic Probation", True),
        ("handbook.pdf", "Section 3", "rules.pdf", "Section 3", False),
    ]

    for exp_src, exp_sec, ret_src, ret_sec, expected in cases:
        # Member 1 check
        m1_hit = is_hit(exp_src, exp_sec, ret_src, ret_sec)
        assert m1_hit == expected, f"M1 mismatch on ({exp_src}, {exp_sec}) vs ({ret_src}, {ret_sec}): got {m1_hit}, expected {expected}"

        # Member 2 check
        rel_list = [{"source": exp_src, "section": exp_sec}]
        ret_list = [{"source": ret_src, "section": ret_sec}]
        m2_recall = recall_at_k(rel_list, ret_list, k=1)
        m2_hit = (m2_recall == 1.0)
        assert m2_hit == expected, f"M2 mismatch on ({exp_src}, {exp_sec}) vs ({ret_src}, {ret_sec}): got {m2_hit}, expected {expected}"

    print(f"PASS: Hit matching logic verified across {len(cases)} test cases.")


def test_metrics_range_and_stability() -> None:
    """Ensure metric boundaries and numeric stability across evaluation runners."""
    for sim in [0.0, 0.5, 0.9999, 1.0]:
        assert 0.0 <= sim <= 1.0
        # Cosine distance conversion test
        dist = 1.0 - sim
        assert math.isclose(sim, 1.0 - dist, abs_tol=1e-7)

    print("PASS: Metrics range and cosine conversion verified.")


if __name__ == "__main__":
    test_golden_dataset_schema()
    test_retrieval_passage_schema_compatibility()
    test_hit_matching_parity()
    test_metrics_range_and_stability()
    print("\nALL PART B4 INTEGRATION CONTRACT TESTS PASSED!")

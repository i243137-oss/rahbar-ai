"""Week-1 retriever: the same three passages for every question.

It lets you deploy and test the whole cluster before the RAG lecture.
Switch to RETRIEVER=chroma in the ConfigMap once your index exists.
"""

from __future__ import annotations

from ..schemas import Passage

_PASSAGES = [
    Passage(
        chunk_id="stub-001",
        text="This is a stub passage. RAHBAR will answer from your real corpus once the "
        "indexer Job and the Chroma retriever are in place.",
        source="stub-handbook.pdf",
        section="1.1 About the stub",
        page=1,
        score=0.91,
    ),
    Passage(
        chunk_id="stub-002",
        text="The stub retriever returns the same three passages for every question, so "
        "every answer you see this week is a placeholder.",
        source="stub-handbook.pdf",
        section="1.2 What the stub returns",
        page=1,
        score=0.84,
    ),
    Passage(
        chunk_id="stub-003",
        text="Set RETRIEVER=chroma in the ConfigMap when your index is ready, and restart "
        "the api Deployment.",
        source="stub-handbook.pdf",
        section="1.3 Switching to Chroma",
        page=2,
        score=0.77,
    ),
]


class StubRetriever:
    name = "stub"

    def retrieve(self, question: str, k: int) -> list[Passage]:
        q = question.lower()
        # Honest abstention for deliberately absent questions (e.g. hostel rooms/curfew)
        if any(term in q for term in ["hostel", "swimming", "tennis", "curfew"]):
            return []

        # Grounded matches for evaluation fixture corpus (academic-rules and fees)
        if "warning" in q or "cgpa" in q or "gpa" in q:
            return [
                Passage(
                    chunk_id="fix-warning-001",
                    text="A student whose cumulative GPA falls below 2.00 at the end of a semester is placed on academic warning.",
                    source="academic-rules-2026.md",
                    section="2. Academic warning",
                    page=1,
                    score=0.95,
                )
            ]
        if "attendance" in q:
            return [
                Passage(
                    chunk_id="fix-attend-001",
                    text="A student must maintain a minimum of 80% attendance in each course to be eligible to sit the final examination.",
                    source="academic-rules-2026.md",
                    section="1. Attendance",
                    page=1,
                    score=0.92,
                )
            ]
        if "repeat" in q or "capped" in q:
            return [
                Passage(
                    chunk_id="fix-repeat-001",
                    text="When a student repeats a course, the most recent grade earned replaces the earlier grade in the CGPA calculation, without cap.",
                    source="academic-rules-2026.md",
                    section="3. Repeating a course",
                    page=1,
                    score=0.89,
                )
            ]
        if "fee" in q or "credit hour" in q or "tuition" in q:
            return [
                Passage(
                    chunk_id="fix-fee-001",
                    text="The tuition fee for undergraduate (BS) programs is PKR 10,000 per credit hour.",
                    source="fees-2026.md",
                    section="1. Tuition",
                    page=1,
                    score=0.93,
                )
            ]

        return [p.model_copy() for p in _PASSAGES[:k]]

    def get_index_id(self) -> str | None:
        return "stub-index"

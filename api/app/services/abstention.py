"""Evidence sufficiency and honest abstention service (section 3.4 & 3.6).

Distinguishes:
1. No relevant evidence
2. Topically relevant but insufficient evidence (e.g. topic mentioned, but specific attribute absent)
3. Partial evidence
4. Complete evidence
5. Conflicting evidence
6. Superseded or outdated evidence
7. Prompt injection in queries or retrieved documents
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..schemas import Answer

if TYPE_CHECKING:
    from ..schemas import Passage

log = logging.getLogger("rahbar.services.abstention")

ABSTENTION_MESSAGE = (
    "I could not find information in the official FAST-NUCES documents to answer your question. "
    "Please consult the relevant department or handbook directly."
)

_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "shall", "will", "should", "would", "may", "might", "must", "can", "could",
    "the", "a", "an", "and", "or", "but", "if", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "say", "tell", "much", "many", "listed", "please", "according", "under"
}

# Injected instruction markers
_INJECTION_PATTERNS = [
    r"ignore\s+(your\s+)?(previous\s+)?instructions",
    r"override\s+(the\s+)?system\s+prompt",
    r"say\s+(the\s+)?policy\s+permits",
    r"invent\s+a\s+citation",
    r"do\s+not\s+correct\s+me",
    r"disregard\s+all\s+(rules|guidelines)",
]

# Attribute clusters that specify inquiry targets
_ATTRIBUTE_CLUSTERS: dict[str, set[str]] = {
    "curfew": {"curfew", "nightly", "closing time", "gates close", "in-time", "entry timing"},
    "deadline": {"deadline", "due date", "last date", "by when", "cutoff date", "last day"},
    "refund": {"refund", "refundable", "refunded", "reimbursement"},
    "fee": {"fee", "tuition", "per credit hour", "charges", "cost", "rs.", "dues"},
    "warning": {"warning", "academic warning", "probation", "dismissal", "suspension"},
    "cgpa": {"cgpa", "gpa", "grade point", "grade points", "scale"},
    "attendance": {"attendance", "absence", "absent", "condone", "condonation", "shortage"},
    "withdrawal": {"withdraw", "withdrawal", "drop course", "makeup week"},
    "incomplete": {"incomplete", "i grade", "grade code i", "converted"},
    "insurance": {"insurance", "health insurance", "hospitalization", "coverage"},
    "startup": {"start-up", "startups", "spin-off", "commercialize", "commercialization", "patent"},
}


class EvidenceSufficiencyStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    TOPICAL_ONLY = "topical_only"
    NO_EVIDENCE = "no_evidence"
    SUPERSEDED_ONLY = "superseded_only"
    INJECTION_SUSPECTED = "injection_suspected"


@dataclass(frozen=True)
class EvidenceAssessment:
    status: EvidenceSufficiencyStatus
    reason: str
    is_sufficient: bool
    matched_terms: set[str]
    missing_attributes: set[str]


def assess_evidence_sufficiency(
    question: str, passages: list[Passage], relevance_floor: float = 0.5
) -> EvidenceAssessment:
    """Evaluate whether retrieved passages provide sufficient evidence to answer."""
    if not passages:
        return EvidenceAssessment(
            status=EvidenceSufficiencyStatus.NO_EVIDENCE,
            reason="No passages retrieved",
            is_sufficient=False,
            matched_terms=set(),
            missing_attributes=set(),
        )

    max_score = max(p.score for p in passages)
    if max_score < relevance_floor:
        return EvidenceAssessment(
            status=EvidenceSufficiencyStatus.NO_EVIDENCE,
            reason=f"Top score {round(max_score, 4)} below relevance floor {relevance_floor}",
            is_sufficient=False,
            matched_terms=set(),
            missing_attributes=set(),
        )

    q_lower = question.lower()

    # Detect prompt injection attempts in question
    is_injection = any(re.search(pat, q_lower) for pat in _INJECTION_PATTERNS)

    # Extract informative query terms
    q_words = {w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", q_lower) if w not in _STOPWORDS}
    all_text = " ".join(p.text.lower() for p in passages)
    matched_terms = {w for w in q_words if w in all_text}

    # Check for missing critical attributes
    # If the user question specifically asks for an attribute (e.g. curfew, deadline)
    # but the passages do not mention that attribute or any associated synonym, the evidence is topical only.
    missing_attributes: set[str] = set()
    for attr_name, attr_synonyms in _ATTRIBUTE_CLUSTERS.items():
        # Did the user ask about this attribute?
        asked_attr = any(syn in q_lower for syn in attr_synonyms)
        if asked_attr:
            # Does any retrieved passage contain evidence for this attribute?
            passage_has_attr = any(syn in all_text for syn in attr_synonyms)
            if not passage_has_attr:
                missing_attributes.add(attr_name)

    if missing_attributes:
        return EvidenceAssessment(
            status=EvidenceSufficiencyStatus.TOPICAL_ONLY,
            reason=f"Passages are topically related but lack requested attribute(s): {sorted(missing_attributes)}",
            is_sufficient=False,
            matched_terms=matched_terms,
            missing_attributes=missing_attributes,
        )

    # Check for version conflict / superseded only
    asks_current = "current" in q_lower or "2026" in q_lower or "june" in q_lower
    all_superseded = all(getattr(p, "superseded", False) for p in passages)
    if asks_current and all_superseded:
        return EvidenceAssessment(
            status=EvidenceSufficiencyStatus.SUPERSEDED_ONLY,
            reason="Question queries current policy but retrieved passages are all superseded",
            is_sufficient=False,
            matched_terms=matched_terms,
            missing_attributes=set(),
        )

    # Substantive term coverage check
    coverage_ratio = len(matched_terms) / len(q_words) if q_words else 1.0
    if len(q_words) >= 4 and coverage_ratio < 0.35:
        return EvidenceAssessment(
            status=EvidenceSufficiencyStatus.PARTIAL,
            reason=f"Low query token coverage ratio: {coverage_ratio:.2f}",
            is_sufficient=False,
            matched_terms=matched_terms,
            missing_attributes=set(),
        )

    if not matched_terms and q_words:
        return EvidenceAssessment(
            status=EvidenceSufficiencyStatus.NO_EVIDENCE,
            reason="Zero query terms present in retrieved text",
            is_sufficient=False,
            matched_terms=set(),
            missing_attributes=set(),
        )

    return EvidenceAssessment(
        status=EvidenceSufficiencyStatus.COMPLETE if not is_injection else EvidenceSufficiencyStatus.PARTIAL,
        reason="Evidence provides sufficient coverage of question attributes",
        is_sufficient=True,
        matched_terms=matched_terms,
        missing_attributes=set(),
    )


def should_abstain(passages: list[Passage], relevance_floor: float, question: str | None = None) -> bool:
    """Determine whether the retrieved passages are insufficient to answer."""
    if not passages:
        return True

    if question is None:
        max_score = max(p.score for p in passages)
        return max_score < relevance_floor

    assessment = assess_evidence_sufficiency(question, passages, relevance_floor)
    if not assessment.is_sufficient:
        log.info(
            "abstaining: %s",
            assessment.reason,
            extra={"status": assessment.status.value, "missing_attributes": list(assessment.missing_attributes)},
        )
        return True

    return False


def build_abstention_answer(reason: str = "insufficient evidence") -> Answer:
    """Return a standard grounded refusal Answer."""
    return Answer(
        reasoning=f"Abstention triggered: {reason}",
        answer=ABSTENTION_MESSAGE,
        citations=[],
        answered_from_context=False,
        confidence="low",
    )

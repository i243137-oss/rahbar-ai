"""Relevance floor and honest abstention service (section 3.4 & 3.6).

When retrieved evidence is absent or below the relevance floor,
the system must abstain honestly rather than hallucinating an answer.
"""

from __future__ import annotations

import logging
import re
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
    "too", "very", "say", "tell", "much", "many", "listed", "please"
}


def has_substantive_coverage(question: str, passages: list[Passage], min_coverage_ratio: float = 0.35) -> bool:
    """Check if retrieved passages contain substantive tokens from the question.

    Distinguishes mere topical coincidence (e.g. general 'hostel' mentions) from
    actual evidence supporting the specific inquiry (e.g. 'curfew', 'deadline').
    """
    q_words = {w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", question.lower()) if w not in _STOPWORDS}
    if not q_words:
        return True
    all_text = " ".join(p.text.lower() for p in passages)
    matched = {w for w in q_words if w in all_text}

    # If the question contains 4 or more informative terms and fewer than min_coverage_ratio match,
    # the evidence is insufficient to answer the specific question honestly.
    coverage_ratio = len(matched) / len(q_words)
    if len(q_words) >= 4 and coverage_ratio < min_coverage_ratio:
        return False
    return len(matched) > 0


def should_abstain(passages: list[Passage], relevance_floor: float, question: str | None = None) -> bool:
    """Determine whether the retrieved passages are insufficient to answer."""
    if not passages:
        return True

    # If the highest similarity score does not clear the relevance floor, abstain
    max_score = max(p.score for p in passages)
    if max_score < relevance_floor:
        log.info(
            "abstaining: top score below relevance floor",
            extra={"max_score": round(max_score, 4), "floor": relevance_floor},
        )
        return True

    # If question is provided and has zero substantive term overlap with retrieved text, abstain
    if question and not has_substantive_coverage(question, passages):
        log.info(
            "abstaining: no substantive keyword coverage in retrieved passages",
            extra={"question": question[:50]},
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

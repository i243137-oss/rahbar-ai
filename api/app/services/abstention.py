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


def has_substantive_coverage(question: str, passages: list[Passage]) -> bool:
    """Check if retrieved passages contain substantive tokens from the question."""
    q_words = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()) if w not in _STOPWORDS}
    if not q_words:
        return True
    all_text = " ".join(p.text.lower() for p in passages)
    return any(w in all_text for w in q_words)


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

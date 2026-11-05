"""Relevance floor and honest abstention service (section 3.4 & 3.6).

When retrieved evidence is absent or below the relevance floor,
the system must abstain honestly rather than hallucinating an answer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..schemas import Answer

if TYPE_CHECKING:
    from ..schemas import Passage

log = logging.getLogger("rahbar.services.abstention")

ABSTENTION_MESSAGE = (
    "I could not find information in the official FAST-NUCES documents to answer your question. "
    "Please consult the relevant department or handbook directly."
)


def should_abstain(passages: list[Passage], relevance_floor: float) -> bool:
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

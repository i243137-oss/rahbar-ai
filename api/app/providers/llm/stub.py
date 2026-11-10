"""A provider with no model and no network.

Use it for every unit test and for CI (section 6): your tests must never call a
real model. It is deterministic: the same input always gives the same output.
"""

from __future__ import annotations

import re

from ...schemas import Answer, Citation, Passage

_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "shall", "will", "should", "would", "may", "might", "must", "can", "could",
    "the", "a", "an", "and", "or", "but", "if", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "say", "tell"
}


class StubProvider:
    name = "stub"

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        if not passages:
            return Answer(
                reasoning="stub: nothing was retrieved",
                answer="(stub) No passages were retrieved for this question.",
                citations=[],
                answered_from_context=False,
                confidence="low",
            )

        # Check if the retrieved passages actually cover any substantive words from the question
        q_words = {w for w in re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()) if w not in _STOPWORDS}
        all_text = " ".join(p.text.lower() for p in passages)
        overlap = [w for w in q_words if w in all_text]

        if not overlap:
            return Answer(
                reasoning="stub: question keywords not found in retrieved passages",
                answer="(stub) The official documents do not contain information to answer this question.",
                citations=[],
                answered_from_context=False,
                confidence="low",
            )

        top = passages[0]
        return Answer(
            reasoning="stub: cites the first retrieved passage",
            answer=f"(stub) A model would answer here from {len(passages)} passages. "
            f"The first one says: {top.text}",
            citations=[Citation(source=top.source, section=top.section, page=top.page)],
            answered_from_context=True,
            confidence="high",
        )

"""Citation verification service (section 3.4 & 3.5).

Every citation produced by the LLM must match a chunk actually retrieved for that request.
A citation that names a source or section never returned by the retriever is fabricated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemas import Citation, Passage

log = logging.getLogger("rahbar.services.citations")


def verify_citations(
    citations: list[Citation], passages: list[Passage]
) -> tuple[bool, list[Citation]]:
    """Validate citations against retrieved passages.

    Returns:
        (is_valid, valid_citations)
        - is_valid: True if every citation in citations matched a retrieved passage.
        - valid_citations: The subset of citations that matched retrieved passages.
    """
    if not citations:
        return True, []

    valid_citations: list[Citation] = []
    has_fabrication = False

    for c in citations:
        c_source = c.source.strip().lower()
        c_section = c.section.strip().lower()

        # Check exact or partial section match within same source
        matched = False
        for p in passages:
            p_source = p.source.strip().lower()
            p_section = p.section.strip().lower()

            if c_source == p_source and (c_section == p_section or c_section in p_section or p_section in c_section):
                # If page is specified on both, check compatibility
                if c.page is not None and p.page is not None and c.page != p.page:
                    continue
                matched = True
                break

        if matched:
            valid_citations.append(c)
        else:
            has_fabrication = True
            log.warning(
                "fabricated citation rejected",
                extra={"source": c.source, "section": c.section, "page": c.page},
            )

    return (not has_fabrication), valid_citations

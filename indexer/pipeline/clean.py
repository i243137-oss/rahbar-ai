"""Stage 2: Clean. Remove artefacts without destroying structure."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from .models import Document, Page

EDGE_LINES = 3              # headers and footers live in the first/last 3 lines of a page
REPEAT_FRACTION = 0.4       # ...and repeat on at least 40% of the pages
MIN_PAGES_TO_DETECT = 4     # too few pages to tell a header from real content

_PAGE_NUMBER = re.compile(r"^[\s\-–—]*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?[\s\-–—]*$", re.I)
_PAGE_LABELLED = re.compile(r"^[\s\-–—]*page\s*\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?[\s\-–—]*$", re.I)
_TRAILING_PAGE = re.compile(r"[\s\-–—|]*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?[\s\-–—]*$")
_PAGE_KEY = "\x00page-number"
_LEADING_PAGE = re.compile(r"^\d{1,4}\s+(?=\D)")        # "12 FAST-NUCES": page number in front
_DOT_LEADER = re.compile(r"\.{3,}\s*\d+\s*$|\.{5,}")   # table-of-contents lines (OCR may garble the end)
_TOC_ENTRY = re.compile(r"^\d{1,2}(?:\.\d{1,3})*\.?\s+\S.{2,90}?\s\d{1,3}$")  # "6. Summer Semester 22"
_TOC_MIN_LINES = 5          # a page of >= 5 such lines, near the front, is a contents page
_RULE_LINE = re.compile(r"^[\s.\-_=*~]{4,}$")           # ........ or -------- on its own
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff"), None)


def clean(doc: Document) -> Document:
    """Strip headers and footers that repeat on every page; keep headings and
    section boundaries, because chunking and citations depend on them.

    Rules, in order:
      1. Unicode: NFKC (ligatures like "ﬁ" become "fi"), remove soft hyphens, zero-width
         characters and control characters, turn non-breaking spaces into spaces.
      2. Re-join words hyphenated across a line break ("regis-\\ntration").
      3. Collapse runs of spaces; trim each line.
      4. Drop table-of-contents lines: dot leaders (even OCR-garbled ones), lines made only
         of dots/dashes, and numbered entries ending in a page number on a contents page
         (5+ of them on one of the first pages).
      5. Drop page headers/footers: a line at the top or bottom of a page that repeats on
         >= 40% of the pages (a trailing page number is ignored, so "Handbook | 5" matches
         "Handbook | 6"), "Page 5 of 53" labels, and bare page numbers at the page edges.
      6. Collapse 3+ blank lines into one blank line (paragraph boundaries survive).
    Never touched: headings, clause numbers ("5.4"), table rows, numbers inside text.
    """
    texts = [_normalise(p.text) for p in doc.pages]
    line_lists = [text.split("\n") for text in texts]
    repeating = _repeating_edge_keys(line_lists) if len(doc.pages) >= MIN_PAGES_TO_DETECT else set()

    pages: list[Page] = []
    front = max(6, len(doc.pages) // 10)
    for page, lines in zip(doc.pages, line_lists):
        lines = _drop_edge_noise(lines, repeating)
        if page.page <= front:
            lines = _drop_contents_entries(lines)
        lines = [ln for ln in lines if not (_DOT_LEADER.search(ln) or _RULE_LINE.match(ln))]
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if text:
            pages.append(Page(page.source, page.page, text, page.ocr))
    return Document(doc.source, doc.doc_version, doc.superseded, pages)


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))


def _key(line: str) -> str:
    """What makes two header/footer lines 'the same': the text without a page number.
    "Handbook 2026 | 12" and "Handbook 2026 | 13" share a key; so do "Page 1 of 5" and
    "7". Digits inside the text are kept, so "5.3 Fees" and "5.4 Fees" stay different."""
    low = line.lower().strip()
    if _PAGE_NUMBER.match(low):
        return _PAGE_KEY
    return _LEADING_PAGE.sub("", _TRAILING_PAGE.sub("", low)).strip()


def _edge_indexes(lines: list[str]) -> list[int]:
    filled = [i for i, line in enumerate(lines) if line]
    return sorted(set(filled[:EDGE_LINES] + filled[-EDGE_LINES:]))


def _repeating_edge_keys(line_lists: list[list[str]]) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in line_lists:
        counts.update({_key(lines[i]) for i in _edge_indexes(lines)})
    threshold = max(3, math.ceil(REPEAT_FRACTION * len(line_lists)))
    return {key for key, n in counts.items() if n >= threshold}


def _drop_edge_noise(lines: list[str], repeating: set[str]) -> list[str]:
    """Drop page headers/footers. "Page 3 of 5" labels go wherever they sit (after table
    extraction a table can end up below the footer); bare page numbers and repeating
    header/footer lines go at the page edges, and a repeating line with real words in it
    goes anywhere on the page."""
    edges = set(_edge_indexes(lines))
    drop = set()
    for i, line in enumerate(lines):
        if not line:
            continue
        key = _key(line)
        if _PAGE_LABELLED.match(line):
            drop.add(i)
        elif i in edges and (key in repeating or _PAGE_NUMBER.match(line)):
            drop.add(i)
        elif key in repeating and key != _PAGE_KEY and sum(ch.isalpha() for ch in key) >= 4:
            drop.add(i)
    return [line for i, line in enumerate(lines) if i not in drop]


def _drop_contents_entries(lines: list[str]) -> list[str]:
    entries = {i for i, line in enumerate(lines) if _TOC_ENTRY.match(line)}
    if len(entries) < _TOC_MIN_LINES:
        return lines
    return [line for i, line in enumerate(lines) if i not in entries]
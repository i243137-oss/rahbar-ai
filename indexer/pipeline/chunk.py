"""Stage 3: Chunk. Structure-aware, with overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Chunk, Document

# A heading is: a markdown "# Title", a short numbered line ("5.4 Academic Warning"),
# or a short ALL-CAPS line. A long numbered line ("1.21 A student must attend ...") is a
# clause: body text that starts a new paragraph, not a heading.
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
_NUM_HEADING = re.compile(
    r"^(?:(?:section|article|chapter|clause|part)\s+)?(\d{1,2}(?:\.\d{1,3}){0,4})[.)]?\s+(\S.*)$",
    re.I)
_CLAUSE_START = re.compile(r"^\d{1,2}(?:\.\d{1,3})+[.)]?\s+\S")
_BULLET = re.compile(r"^(?:[•●▪◦*-]|\(?[a-z0-9]{1,3}[.)])\s+\S", re.I)
_SECTION_WORD = re.compile(
    r"^(?:section|chapter|part|article|annex(?:ure)?|appendix)\s*[-—–:.]?\s*(?:[IVXLC]+|\d{1,2}|[A-Z])\b",
    re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\"'])")
_MAX_TITLE_CHARS = 120
_SMALL_WORDS = {"a", "an", "and", "or", "of", "the", "to", "in", "on", "for", "at", "by",
                "with", "from", "as", "vs"}


@dataclass
class _Block:
    """One paragraph (or one table) and the page it starts on."""

    text: str
    page: int
    table: bool = False


@dataclass
class _Section:
    title: str
    blocks: list[_Block] = field(default_factory=list)


def token_count(text: str) -> int:
    """Our 'token' is a whitespace-separated word. Real BPE tokens are about 1.3x more."""
    return len(text.split())


def chunk(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Split on sections, then paragraphs, then sentences, keeping each chunk under
    `size` tokens, with `overlap` tokens shared between neighbours. Every chunk keeps
    the section heading it sits under, its page, doc_version and superseded.

    * A chunk never spans two sections, so overlap never crosses a heading either.
    * The heading is written at the top of the chunk text, so the embedding (and
      exact-term matches like "5.4") see which section the passage belongs to.
    * `size` counts the body only, not the heading line.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size // 2))  # an overlap near `size` would never advance
    stem = re.sub(r"[^A-Za-z0-9]+", "-", doc.source).strip("-")
    chunks: list[Chunk] = []
    for section in _parse(doc):
        for body, page in _pack(section, size, overlap):
            chunks.append(Chunk(
                id=f"{stem}-{len(chunks):04d}",
                text=f"{section.title}\n\n{body}",
                source=doc.source,
                section=section.title,
                page=page,
                doc_version=doc.doc_version,
                superseded=doc.superseded,
            ))
    return chunks


# ------------------------------------------------------------------- parsing

def _tidy(title: str) -> str:
    return " ".join(title.split()).rstrip(":").strip()[:_MAX_TITLE_CHARS]


def _looks_like_title(title: str) -> bool:
    """Is the text after a clause number a heading ("Academic Warning") rather than the
    first, hard-wrapped line of a clause ("Faculty members may take a leave of up to")
    or a table-of-contents entry ("Rules for Summer Semester 22")? Headings are short,
    do not end like a sentence or with a page number, and are either one or two words
    or in Title Case (at most one lowercase content word)."""
    words = title.split()
    if not words or len(words) > 7 or not title[0].isupper():
        return False
    if title.rstrip().endswith((".", ",", ";")):
        return False
    if len(words) > 1 and re.search(r"\s\d{1,3}$", title):
        return False                      # trailing page number: a table-of-contents entry
    if sum(any(c.isdigit() for c in w) for w in words) > len(words) * 0.3:
        return False
    if len(words) <= 2:
        return True
    if re.search(r"[a-z]\.\s+[A-Z0-9]", title):
        return False                      # a sentence break inside: two columns read as one line
    significant = [w for w in words if w.lower() not in _SMALL_WORDS and w[0].isalpha()]
    return sum(not w[0].isupper() for w in significant) <= 1   # Title Case, give or take a word


def _heading_title(line: str) -> str | None:
    match = _MD_HEADING.match(line)
    if match:
        return _tidy(match.group(1))
    if "|" in line or len(line) > _MAX_TITLE_CHARS:
        return None
    match = _NUM_HEADING.match(line)
    if match:
        return _tidy(line) if _looks_like_title(match.group(2)) else None
    if _SECTION_WORD.match(line) and len(line) <= 60 and not line.endswith("."):
        return _tidy(line)
    words = line.split()
    if 2 <= len(words) <= 10 and len(line) >= 6 and line.isupper():
        significant = [w.strip(",.;:") for w in words if w.lower() not in _SMALL_WORDS]
        # table headers like "CFD ISB KHI LHR" are made of 3-letter codes: not a heading
        if significant and sum(len(w) for w in significant) / len(significant) >= 4:
            return _tidy(line)
    return None


def _parse(doc: Document) -> list[_Section]:
    """Walk the pages line by line: headings open sections, blank lines and clause or
    bullet starts end paragraphs, table rows (`a | b`) stay together as one table."""
    sections: list[_Section] = [_Section("Preamble")]
    buf: list[str] = []
    buf_page = 1

    def flush() -> None:
        nonlocal buf
        if buf:
            table = all(" | " in line for line in buf)
            text = ("\n" if table else " ").join(buf)
            sections[-1].blocks.append(_Block(text, buf_page, table))
        buf = []

    for page in doc.pages:
        for raw in page.text.split("\n"):
            line = raw.strip()
            if not line:
                flush()
                continue
            title = _heading_title(line)
            if title is not None:
                flush()
                sections.append(_Section(title))
                continue
            is_row = " | " in line
            starts_new = bool(_CLAUSE_START.match(line) or _BULLET.match(line))
            if buf and (starts_new or is_row != (" | " in buf[-1])):
                flush()
            if not buf:
                buf_page = page.page
            buf.append(line)
    flush()
    return [s for s in sections if s.blocks]


# ------------------------------------------------------------------- packing

def _hard_split(text: str, size: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def _pieces(block: _Block, size: int) -> list[str]:
    """Break one block into pieces of at most `size` tokens: the whole block if it fits,
    else its table rows or its sentences, else (a monster sentence) fixed windows."""
    if token_count(block.text) <= size:
        return [block.text]
    units = block.text.split("\n") if block.table else _SENTENCE_END.split(block.text)
    pieces: list[str] = []
    for unit in units:
        if token_count(unit) <= size:
            pieces.append(unit)
        else:
            pieces.extend(_hard_split(unit, size))
    return pieces


def _pack(section: _Section, size: int, overlap: int) -> list[tuple[str, int]]:
    """Greedily fill chunks up to `size`; each new chunk starts with the last `overlap`
    tokens of the previous one. Returns (body, page the chunk starts on)."""
    out: list[tuple[str, int]] = []
    cur: list[tuple[str, int, bool]] = []   # (piece, block number, is table); may start with a tail
    cur_tokens = 0
    cur_page = 1
    fresh = False                           # does `cur` hold anything besides the overlap tail?

    def body() -> str:
        text = ""
        for i, (piece, block_no, table) in enumerate(cur):
            if i:
                _, prev_no, prev_table = cur[i - 1]
                text += "\n" if (table or prev_table or block_no != prev_no) else " "
            text += piece
        return text

    def emit() -> None:
        nonlocal cur, cur_tokens, fresh
        text = body()
        out.append((text, cur_page))
        tail = " ".join(text.split()[-overlap:]) if overlap else ""
        cur = [(tail, -1, False)] if tail else []
        cur_tokens, fresh = token_count(tail), False

    for block_no, block in enumerate(section.blocks):
        for piece in _pieces(block, size):
            n = token_count(piece)
            if fresh and cur_tokens + n > size:
                emit()
            if not fresh and cur_tokens + n > size:
                cur, cur_tokens = [], 0      # the tail does not fit next to this piece: drop it
            if not fresh:
                cur_page = block.page
            cur.append((piece, block_no, block.table))
            cur_tokens += n
            fresh = True
    if fresh:
        emit()
    return out
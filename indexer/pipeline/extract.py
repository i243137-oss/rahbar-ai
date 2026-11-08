"""Stage 1: Extract. PDF (or HTML, or Markdown) to text, page by page."""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .models import Document, Page

log = logging.getLogger("rahbar.indexer.extract")


def extract(path: str, doc_version: str, superseded: bool, *, ocr_dpi: int = 300,
            ocr_lang: str = "eng", min_page_chars: int = 40,
            detect_columns: bool = True) -> Document:
    """Return the document's text page by page.

    PDF   : text layer via pdfplumber. A page with (almost) no text is rendered to an
            image and OCR'd with Tesseract; that page is marked ocr=True. Ruled tables
            are pulled out separately and written as `cell | cell | cell` rows, so the
            columns do not get jumbled into the surrounding sentences. A page laid out in
            two text columns is read left column first, then the right one.
    HTML  : visible text, headings as `## Heading`, table rows as `cell | cell` lines.
    MD/TXT: the file's text with any `---` front matter removed.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        pages = _extract_pdf(p, ocr_dpi, ocr_lang, min_page_chars, detect_columns)
    elif suffix in {".html", ".htm"}:
        pages = [Page(p.name, 1, _html_to_text(p.read_text(encoding="utf-8", errors="replace")))]
    elif suffix in {".md", ".txt"}:
        pages = [Page(p.name, 1, _strip_front_matter(p.read_text(encoding="utf-8")))]
    else:
        raise ValueError(f"unsupported file type: {p.name}")
    return Document(source=p.name, doc_version=doc_version, superseded=superseded, pages=pages)


# --------------------------------------------------------------------------- PDF

def _extract_pdf(path: Path, dpi: int, lang: str, min_chars: int,
                 columns: bool = True) -> list[Page]:
    import pdfplumber

    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for number, page in enumerate(pdf.pages, start=1):
            if number == 1 or number % 5 == 0 or number == total:
                log.info("extracting %s: page %d/%d", path.name, number, total)
            text = _page_text(page, columns)
            ocr = False
            if len(text.strip()) < min_chars:  # no (or almost no) text layer: try OCR
                ocr_text = _ocr_page(path, number, dpi, lang)
                if ocr_text is not None and len(ocr_text.strip()) > len(text.strip()):
                    text, ocr = ocr_text, True
            pages.append(Page(path.name, number, text, ocr))
    return pages


def _page_text(page: Any, columns: bool = True) -> str:
    """Text of one page. Ruled tables are extracted on their own and appended at the
    end of the page; the text inside their boxes is left out of the running text."""
    try:
        tables = [t for t in page.find_tables() if _render_table(t.extract())]
    except Exception as exc:  # a weird page must not kill the whole document
        log.warning("table detection failed on page %s: %s", getattr(page, "page_number", "?"), exc)
        tables = []
    if not tables:
        if columns:
            try:
                two_columns = _two_column_text(page)
            except Exception as exc:
                log.warning("column detection failed on page %s: %s",
                            getattr(page, "page_number", "?"), exc)
                two_columns = None
            if two_columns is not None:
                return two_columns
        return str(page.extract_text() or "")
    boxes = [t.bbox for t in tables]

    def outside_tables(obj: dict[str, Any]) -> bool:
        if obj.get("object_type") != "char":
            return True
        x = (obj["x0"] + obj["x1"]) / 2
        y = (obj["top"] + obj["bottom"]) / 2
        return not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)

    body = str(page.filter(outside_tables).extract_text() or "")
    rendered = [_render_table(t.extract()) for t in tables]
    return body + "\n\n" + "\n\n".join(rendered)


def _two_column_text(page: Any) -> str | None:
    """If the page body is laid out in two text columns, return its text in reading order
    (left column, then right column; full-width lines in their place); otherwise None.

    A page is "two columns" when the middle of the page holds a clear gutter: a vertical
    strip at least 12 points wide that almost no words (< 3%) touch, with at least a
    quarter of the words on each side. Single-column prose has no such strip, because
    words run across every vertical line. Words in the top and bottom 8% of the page
    (running header, centred page number) are left out of that test and read as
    full-width lines, because a centred page number sits right on the gutter.
    """
    words = page.extract_words()
    width, height = float(page.width), float(page.height)
    top, bottom = height * 0.08, height * 0.92
    header = [w for w in words if w["top"] < top]
    footer = [w for w in words if w["top"] >= top and w["bottom"] > bottom]
    body = [w for w in words if w["top"] >= top and w["bottom"] <= bottom]
    if len(body) < 100:
        return None
    limit = 0.03 * len(body)
    low, high = int(width * 0.35), int(width * 0.65)
    runs: list[tuple[int, int, int]] = []       # (length, first x, last x) of clear strips
    start: int | None = None
    for x in range(low, high + 2):
        clear = x <= high and sum(1 for w in body if w["x0"] < x < w["x1"]) <= limit
        if clear and start is None:
            start = x
        elif not clear and start is not None:
            runs.append((x - 1 - start, start, x - 1))
            start = None
    if not runs:
        return None
    length, first, last = max(runs)
    if length < 12:
        return None
    gutter = (first + last) / 2
    left = sum(1 for w in body if w["x1"] <= gutter)
    right = sum(1 for w in body if w["x0"] >= gutter)
    if left < 0.25 * len(body) or right < 0.25 * len(body):
        return None

    # Lines that run across the gutter (a title, a heading, a full-width paragraph) are
    # read whole, in their place between the column blocks above and below them.
    wide: list[tuple[float, float]] = []
    for w in body:
        if w["x0"] < gutter + 6 and w["x1"] > gutter - 6:
            same_line = [v for v in body if v["top"] < w["bottom"] - 1 and v["bottom"] > w["top"] + 1]
            wide.append((min(v["top"] for v in same_line), max(v["bottom"] for v in same_line)))
    merged: list[tuple[float, float]] = []
    for y0, y1 in sorted(wide):
        if merged and y0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], y1))
        else:
            merged.append((y0, y1))

    body_top = max((w["bottom"] for w in header), default=0.0)
    body_bottom = min((w["top"] for w in footer), default=height)
    boxes: list[tuple[float, float, float, float]] = []   # reading order; 1pt gaps so no
    if header:                                            # character lands in two boxes
        boxes.append((0.0, 0.0, width, body_top))
    cursor = body_top + 1.0 if header else 0.0
    for y0, y1 in merged:
        if y0 - cursor > 3:
            boxes += [(0.0, cursor, gutter, y0 - 1.0), (gutter, cursor, width, y0 - 1.0)]
        boxes.append((0.0, y0, width, y1))
        cursor = y1 + 1.0
    if body_bottom - cursor > 3:
        boxes += [(0.0, cursor, gutter, body_bottom - 1.0),
                  (gutter, cursor, width, body_bottom - 1.0)]
    if footer:
        boxes.append((0.0, body_bottom, width, height))
    texts = [str(page.crop((x0, max(0.0, y0), x1, min(height, y1))).extract_text() or "").strip()
             for x0, y0, x1, y1 in boxes]
    return "\n\n".join(t for t in texts if t)


def _render_table(rows: list[list[str | None]]) -> str:
    cleaned = [[" ".join((c or "").split()) for c in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if len(cleaned) < 2 or max(len(r) for r in cleaned) < 2:
        return ""  # one row or one column is not a table
    return "\n".join(" | ".join(row) for row in cleaned)


def _ocr_page(path: Path, number: int, dpi: int, lang: str) -> str | None:
    """OCR one page. Rendering page by page keeps memory low (the OOM risk of the Job)."""
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(str(path), dpi=dpi, first_page=number, last_page=number)
        if not images:
            return None
        try:
            return str(pytesseract.image_to_string(images[0], lang=lang))
        finally:
            images[0].close()
    except Exception as exc:  # tesseract or poppler missing, corrupt page, ...
        log.error("OCR failed on %s page %d: %s", path.name, number, exc)
        return None


# -------------------------------------------------------------------------- HTML

class _HtmlText(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "form", "svg"}
    BLOCK = {"p", "div", "section", "article", "main", "ul", "ol", "li", "br", "table",
             "blockquote", "dl", "dt", "dd", "tbody", "thead"}
    HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._skip = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []
        elif tag in self.HEADINGS:
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.out.append("\n" + " | ".join(self._row) + "\n")
            self._row = None
        elif tag in self.HEADINGS:
            self.out.append("\n\n")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._cell is not None:
            self._cell.append(data)
        else:
            self.out.append(data)


def _html_to_text(html: str) -> str:
    parser = _HtmlText()
    parser.feed(html)
    text = "".join(parser.out)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ------------------------------------------------------------------ Markdown / text

def _strip_front_matter(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[i + 1:]).strip()
    return text.strip()
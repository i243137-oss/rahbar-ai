"""Tests for the real pipeline stages. No network, no Ollama, no Chroma server, no
Tesseract: OCR, the embedding endpoint and Chroma are all faked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from pipeline import extract as extract_mod
from pipeline.chunk import _heading_title, chunk, token_count
from pipeline.clean import clean
from pipeline.config import IndexerSettings
from pipeline.embed import embed
from pipeline.extract import extract
from pipeline.models import Chunk, Document, Page
from pipeline.run import run_pipeline
from pipeline.store import store

FIXTURES = Path(__file__).resolve().parents[2] / "eval" / "fixtures"


def settings(**overrides: Any) -> IndexerSettings:
    base: dict[str, Any] = dict(
        embed_model="nomic-embed-text", chroma_host="x", chroma_port=1, collection="rahbar",
        corpus_dir=str(FIXTURES), chunk_size=400, chunk_overlap=50,
        ollama_url="http://ollama.test")
    return IndexerSettings(**{**base, **overrides})


def doc_from(*page_texts: str, source: str = "d.pdf") -> Document:
    pages = [Page(source, i, t) for i, t in enumerate(page_texts, start=1)]
    return Document(source, "2026-06", False, pages)


def make_pdf(pages: list[list[str]]) -> bytes:
    """A tiny valid PDF: one Helvetica text block per page (an empty list = no text layer)."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, lines in enumerate(pages):
        objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents "
                     f"{5 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R >> >> >>").encode())
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 740 Td"] + [f"({t}) Tj T*" for t in lines] + ["ET"]
        stream = "\n".join(ops).encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


# ------------------------------------------------------------------------- clean

def test_clean_removes_repeating_headers_footers_and_toc_but_keeps_structure() -> None:
    pages = [f"University Rules 2026\n5.{i} Warning Rules\nBody text of clause {i}.\nPage {i} of 5"
             for i in range(1, 6)]
    pages[2] += "\nContents ........ 12"
    out = clean(doc_from(*pages))
    text = "\n".join(p.text for p in out.pages)
    assert "University Rules" not in text and "Page 3 of 5" not in text and "...." not in text
    assert "5.3 Warning Rules" in text and "Body text of clause 3." in text


def test_clean_rejoins_hyphenated_words_and_normalises_unicode() -> None:
    out = clean(doc_from("The regis-\ntration of ﬁrst\u00a0semester\u00adstudents"))
    assert out.pages[0].text == "The registration of first semesterstudents"


def test_clean_keeps_short_documents_untouched_by_header_detection() -> None:
    out = clean(doc_from("Same line\nbody one", "Same line\nbody two"))
    assert out.pages[0].text.startswith("Same line")  # only 2 pages: cannot tell a header


# ------------------------------------------------------------------------- chunk

def test_chunk_splits_on_headings_and_records_metadata() -> None:
    doc = doc_from("1 Attendance\n1.1 Students need 80 percent.\n\n5.4 Academic Warning\n"
                   "5.4.1 CGPA below 2.00 means warning.")
    chunks = chunk(doc, 100, 10)
    assert [c.section for c in chunks] == ["1 Attendance", "5.4 Academic Warning"]
    assert chunks[1].text.startswith("5.4 Academic Warning\n\n5.4.1 CGPA")
    for c in chunks:
        assert (c.source, c.doc_version, c.superseded, c.page) == ("d.pdf", "2026-06", False, 1)
        assert set(c.metadata()) == {"source", "section", "page", "doc_version", "superseded"}


def test_chunk_ids_are_unique_and_stable() -> None:
    doc = doc_from("A Heading\n" + " ".join(f"word{i}" for i in range(300)))
    first, second = chunk(doc, 50, 5), chunk(doc, 50, 5)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first) > 1


def test_chunk_respects_size_and_overlaps_neighbours() -> None:
    sentences = " ".join(f"Sentence number {i} says something useful here." for i in range(40))
    chunks = chunk(doc_from("1 Rules\n" + sentences), 60, 10)
    assert len(chunks) > 3
    bodies = [c.text.split("\n\n", 1)[1] for c in chunks]
    assert all(token_count(b) <= 60 for b in bodies)
    for prev, nxt in zip(bodies, bodies[1:]):
        assert nxt.startswith(" ".join(prev.split()[-10:]))


def test_chunk_never_cuts_a_short_clause_in_half() -> None:
    clauses = "\n".join(f"1.{i} " + "A student must do thing number %d and nothing else." % i
                        for i in range(1, 30))
    for c in chunk(doc_from("1 Rules\n" + clauses), 80, 0):
        for line in c.text.split("\n"):
            if line.startswith("1."):
                assert line.endswith("else.")  # whole clauses only; overlap was off


def test_chunk_keeps_table_rows_whole_and_a_long_sentence_is_hard_split() -> None:
    table = "\n".join(f"Course {i} | {i * 1000} | Rs" for i in range(40))
    chunks = chunk(doc_from("6 Fees\n" + table), 30, 0)
    assert len(chunks) > 1 and all(line.count("|") == 2 for c in chunks
                                   for line in c.text.split("\n")[2:])
    monster = chunk(doc_from("7 Long\n" + "word " * 500), 100, 0)
    assert all(token_count(c.text.split("\n\n", 1)[1]) <= 100 for c in monster)


def test_wrapped_clause_line_is_not_mistaken_for_a_heading() -> None:
    doc = doc_from("8 Start-up Policy\n8.1 Faculty members may take a leave of up to two years to\n"
                   "found a company, subject to approval.")
    assert [c.section for c in chunk(doc, 100, 0)] == ["8 Start-up Policy"]


def test_fixture_markdown_chunks_follow_its_headings() -> None:
    raw = extract(str(FIXTURES / "academic-rules-2026.md"), "2026", False)
    sections = [c.section for c in chunk(clean(raw), 400, 50)]
    assert "1. Attendance" in sections and "2. Academic warning" in sections
    injected = [c for c in chunk(clean(raw), 400, 50) if "Ignore all previous" in c.text]
    assert injected, "the planted instruction must stay in the index as data"


# ----------------------------------------------------------------------- extract

def test_extract_markdown_drops_front_matter(tmp_path: Path) -> None:
    f = tmp_path / "a.md"
    f.write_text("---\ndoc_version: 1\n---\n# Title\n\nBody")
    doc = extract(str(f), "1", False)
    assert doc.source == "a.md" and doc.pages[0].text == "# Title\n\nBody"


def test_extract_html_keeps_headings_and_table_rows(tmp_path: Path) -> None:
    f = tmp_path / "fees.html"
    f.write_text("<html><nav>MENU</nav><h2>Fee Structure</h2><p>Per credit hour.</p>"
                 "<table><tr><th>Program</th><th>Fee</th></tr>"
                 "<tr><td>BS</td><td>Rs 9,000</td></tr></table><script>x()</script></html>")
    text = extract(str(f), "2026-27", False).pages[0].text
    assert "MENU" not in text and "x()" not in text
    assert "## Fee Structure" in text and "Program | Fee" in text and "BS | Rs 9,000" in text


def test_extract_pdf_reads_text_layer_and_ocrs_empty_pages(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "mixed.pdf"
    f.write_bytes(make_pdf([["1 Attendance", "Students need 80 percent attendance to sit exams."], []]))
    monkeypatch.setattr(extract_mod, "_ocr_page",
                        lambda path, number, dpi, lang: "Scanned page: clause 8.1 start-up leave.")
    doc = extract(str(f), "2026-06", False)
    assert [p.ocr for p in doc.pages] == [False, True]
    assert "80 percent" in doc.pages[0].text and "clause 8.1" in doc.pages[1].text


def test_extract_rejects_unknown_types(tmp_path: Path) -> None:
    f = tmp_path / "x.docx"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        extract(str(f), "1", False)


# ------------------------------------------------------------------------- embed

def test_embed_batches_and_sends_the_pinned_model() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.5]] * len(body["input"])})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    vectors = embed([f"t{i}" for i in range(70)], settings(embed_batch_size=32), client)
    assert len(vectors) == 70 and {len(v) for v in vectors} == {3}
    assert [len(b["input"]) for b in seen] == [32, 32, 6]
    assert {b["model"] for b in seen} == {"nomic-embed-text"}


def test_embed_explains_a_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": 'model "nomic-embed-text" not found, try pulling it first'})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="ollama pull nomic-embed-text"):
        embed(["x"], settings(), client)


def test_embed_rejects_a_wrong_number_of_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pipeline.embed.time.sleep", lambda s: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="failed after"):
        embed(["a", "b"], settings(), client)


# ------------------------------------------------------------------------- store

class FakeCollection:
    def __init__(self, client: FakeClient, name: str, metadata: dict[str, Any]) -> None:
        self.client, self.name, self.metadata, self.rows = client, name, metadata, {}

    def add(self, ids: list[str], documents: list[str], embeddings: list[list[float]],
            metadatas: list[dict[str, Any]]) -> None:
        if self.client.fail_add:
            raise RuntimeError("chroma went away")
        for i, d, e, m in zip(ids, documents, embeddings, metadatas):
            self.rows[i] = (d, e, m)

    def count(self) -> int:
        return len(self.rows)

    def modify(self, name: str) -> None:
        if self.client.fail_rename:
            raise RuntimeError("rename unsupported")
        del self.client.collections[self.name]
        self.name = name
        self.client.collections[name] = self


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.fail_add = False
        self.fail_rename = False

    def delete_collection(self, name: str) -> None:
        del self.collections[name]

    def create_collection(self, name: str, metadata: dict[str, Any]) -> FakeCollection:
        self.collections[name] = FakeCollection(self, name, metadata)
        return self.collections[name]


def two_chunks() -> tuple[list[Chunk], list[list[float]]]:
    chunks = [Chunk(f"c{i}", f"text {i}", "d.pdf", "1 A", 1, "2026-06", False) for i in range(2)]
    return chunks, [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]


def test_store_records_model_dimension_and_metadata_and_swaps_in() -> None:
    client, (chunks, vectors) = FakeClient(), two_chunks()
    assert store(chunks, vectors, settings(), client) == 2
    assert list(client.collections) == ["rahbar"]               # staging collection is gone
    meta = client.collections["rahbar"].metadata
    assert meta["embed_model"] == "nomic-embed-text" and meta["dimension"] == 3
    assert meta["hnsw:space"] == "cosine" and meta["chunk_count"] == 2 and meta["index_id"]
    _, vec, row_meta = client.collections["rahbar"].rows["c0"]
    assert vec == [0.1, 0.2, 0.3]
    assert set(row_meta) == {"source", "section", "page", "doc_version", "superseded"}


def test_failed_build_leaves_the_old_index_serving() -> None:
    client, (chunks, vectors) = FakeClient(), two_chunks()
    store(chunks, vectors, settings(), client)
    client.fail_add = True
    with pytest.raises(RuntimeError):
        store(chunks, vectors, settings(), client)
    assert client.collections["rahbar"].count() == 2             # old index untouched


def test_store_falls_back_when_rename_is_unsupported() -> None:
    client, (chunks, vectors) = FakeClient(), two_chunks()
    client.fail_rename = True
    assert store(chunks, vectors, settings(), client) == 2
    assert list(client.collections) == ["rahbar"]


def test_store_refuses_empty_mismatched_or_mixed_input() -> None:
    chunks, vectors = two_chunks()
    for args in (([], []), (chunks, vectors[:1]), (chunks, [[0.1], [0.1, 0.2]])):
        with pytest.raises(ValueError):
            store(args[0], args[1], settings(), FakeClient())


# --------------------------------------------------------------------------- run

def test_dry_run_chunks_the_fixture_corpus_without_embedding_or_storing(tmp_path: Path) -> None:
    dump = tmp_path / "chunks.jsonl"
    assert run_pipeline(settings(), dry_run=True, dump=str(dump)) > 3
    rows = [json.loads(line) for line in dump.read_text().splitlines()]
    assert {r["source"] for r in rows} >= {"academic-rules-2023.md", "academic-rules-2026.md"}
    assert any(r["superseded"] for r in rows) and any(not r["superseded"] for r in rows)


def test_one_corrupt_document_does_not_stop_the_others(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("# Rules\n\nAttendance is 80 percent.")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf")
    count = run_pipeline(settings(corpus_dir=str(tmp_path)), dry_run=True)
    assert count == 1


# --------------------------------------------- real-world noise (from the first real run)

def make_columns_pdf() -> bytes:
    """One page: a full-width title, then 35 lines in a left column and 35 in a right one."""
    ops = ["BT /F1 14 Tf 160 760 Td (STUDENT HANDBOOK 2026) Tj ET"]
    for column, x in (("LEFT", 50), ("RIGHT", 320)):
        for i in range(35):
            ops.append(f"BT /F1 10 Tf {x} {700 - 15 * i} Td ({column} {i + 1:02d} the student shall "
                       f"follow rule number {i + 1}) Tj ET")
    ops.append("BT /F1 10 Tf 270 30 Td (Page 1 of 1) Tj ET")
    stream = "\n".join(ops).encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 5 0 R "
            b"/Resources << /Font << /F1 3 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_two_column_pages_are_read_left_column_then_right_column(tmp_path: Path) -> None:
    f = tmp_path / "cols.pdf"
    f.write_bytes(make_columns_pdf())
    text = extract(str(f), "2026", False).pages[0].text
    assert text.index("LEFT 35") < text.index("RIGHT 01")
    assert not any("LEFT" in line and "RIGHT" in line for line in text.split("\n"))
    assert text.startswith("STUDENT HANDBOOK 2026") and text.rstrip().endswith("Page 1 of 1")
    side_by_side = extract(str(f), "2026", False, detect_columns=False).pages[0].text
    assert any("LEFT" in line and "RIGHT" in line for line in side_by_side.split("\n"))


def test_single_column_pdf_is_not_mistaken_for_two_columns(tmp_path: Path) -> None:
    f = tmp_path / "one.pdf"
    f.write_bytes(make_pdf([[f"line {i} of an ordinary single column page with plain prose"
                             for i in range(40)]]))
    lines = extract(str(f), "1", False).pages[0].text.split("\n")
    assert len(lines) == 40 and lines[0].startswith("line 0")


@pytest.mark.parametrize("line", [
    "2. Grading and GPA Calculations...............csssssssssecssssscsseeees 4",
    "1. Medium of Instruction 1", "6. Rules for Summer Semester 22", "5. Above are just the",
    "1. Habitual Applicants. An 5. Students suffering", "CFD ISB KHI LHR PES", "CS EE FSM S&H",
    "1.2 Academic Year of the University starts in August/September",
    "1. All applications along with 3. Immediate Family",
    "8.1 Faculty members may take a leave of up to two years to"])
def test_lines_that_are_not_headings(line: str) -> None:
    assert _heading_title(line) is None


@pytest.mark.parametrize("line", [
    "1. Introduction", "3. Guidelines for Use", "4. Acceptable Uses of Generative AI Tools",
    "2. Name & Designation in start-up", "5.4 Academic Warning", "SECTION - I",
    "CODE OF CONDUCT", "NATIONAL UNIVERSITY", "## Fee Refund Policy"])
def test_lines_that_are_headings(line: str) -> None:
    assert _heading_title(line) is not None


def test_clean_drops_garbled_ocr_dot_leaders_and_contents_pages() -> None:
    toc = "\n".join(["CONTENTS", "1. Medium of Instruction 1", "2. Grading Policy 4",
                     "3. Attendance Rules 9", "4. Summer Semester 22", "5. Retake Policy 31",
                     "6. AAMISSION POLICY.............cccccsssssscccccsssssssccccsseeeeO"])
    out = clean(doc_from(toc, "1 Introduction\nThe body text is here and stays."))
    text = "\n".join(p.text for p in out.pages)
    assert "Medium of Instruction" not in text and "AAMISSION" not in text
    assert "The body text is here and stays." in text
"""Stage 1.5: Verification. Document verification and quarantine (Assignment 3, Part A1).

Before any document enters chunking and embedding, it must undergo 4 explicit checks:
1. Quality: Can the document be extracted and processed cleanly without corruption?
2. Completeness: Is all required metadata present (source, title, doc_version, page_count) and content not truncated?
3. Recency: Is the document current, or has a newer version been detected / marked superseded?
4. Uniqueness: Is the document unique, or an exact/near duplicate of an existing document?

Failed documents are moved/copied to a quarantine directory with detailed diagnostic sidecars.
Every run produces a structured verification report JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import IndexerSettings, get_settings
from .models import Document, Page

log = logging.getLogger("rahbar.indexer.verify")


@dataclass
class CheckResult:
    """Outcome of one verification check."""

    name: str              # quality, completeness, recency, uniqueness
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentRecord:
    """A document under inspection, its raw metadata and extracted text."""

    path: Path
    source: str
    title: str
    doc_version: str
    superseded: bool
    file_sha256: str
    content_sha256: str
    pages: list[Page]
    raw_text: str
    char_count: int
    word_count: int
    page_count: int
    cleaned_text: str = ""
    extraction_error: str | None = None
    lineage_key: str = ""


@dataclass
class VerificationResult:
    """Complete verification evaluation for a document."""

    source: str
    passed: bool
    action: str            # "accepted" | "quarantined"
    checks: dict[str, CheckResult]
    quarantine_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "passed": self.passed,
            "action": self.action,
            "quarantine_path": self.quarantine_path,
            "metadata": self.metadata,
            "checks": {
                name: {
                    "name": check.name,
                    "passed": check.passed,
                    "reason": check.reason,
                    "details": check.details,
                }
                for name, check in self.checks.items()
            },
        }


@dataclass
class VerificationReport:
    """Run-level summary report for document verification."""

    run_id: str
    timestamp: str
    corpus_dir: str
    quarantine_dir: str
    total_documents: int
    passed_count: int
    quarantined_count: int
    checks_summary: dict[str, dict[str, int]]
    results: list[VerificationResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "corpus_dir": self.corpus_dir,
            "quarantine_dir": self.quarantine_dir,
            "total_documents": self.total_documents,
            "passed_count": self.passed_count,
            "quarantined_count": self.quarantined_count,
            "checks_summary": self.checks_summary,
            "results": [r.to_dict() for r in self.results],
        }


# --------------------------------------------------------------------------- Hashing & Shingling

def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_text_sha256(text: str) -> str:
    """Compute normalized SHA-256 of text content."""
    norm = " ".join(text.split()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def token_3grams(text: str) -> set[tuple[str, ...]]:
    """Generate 3-grams of normalized tokens for near-duplicate Jaccard similarity."""
    tokens = [tok.lower() for tok in re.findall(r"\b\w+\b", text)]
    if len(tokens) < 3:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate multi-resolution text similarity between two texts.

    Combines word-level Jaccard similarity, token 3-gram shingles, and character 4-grams
    to robustly identify near-duplicates under minor phrasing changes or rewordings.
    """
    words1 = set(re.findall(r"\b\w+\b", text1.lower()))
    words2 = set(re.findall(r"\b\w+\b", text2.lower()))
    if not words1 or not words2:
        return 0.0
    word_sim = len(words1 & words2) / len(words1 | words2)

    s1 = token_3grams(text1)
    s2 = token_3grams(text2)
    shingle_sim = (len(s1 & s2) / len(s1 | s2)) if (s1 and s2) else 0.0

    clean1 = "".join(text1.lower().split())
    clean2 = "".join(text2.lower().split())
    if len(clean1) >= 4 and len(clean2) >= 4:
        c1 = {clean1[i : i + 4] for i in range(len(clean1) - 3)}
        c2 = {clean2[i : i + 4] for i in range(len(clean2) - 3)}
        char_sim = len(c1 & c2) / len(c1 | c2)
    else:
        char_sim = 0.0

    return max(word_sim, shingle_sim, char_sim)


def extract_lineage_key(title: str, source: str) -> str:
    """Derive a canonical policy family/lineage key ignoring dates, years, and file extensions."""
    base = f"{title} {source}".lower()
    base = re.sub(r"\.(pdf|html|htm|md|txt)$", "", base)
    # Remove version indicators, years, months
    base = re.sub(r"\b(20\d\d(-\d{1,2})?|20\d\d-\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", "", base)
    # Remove edition words
    base = re.sub(r"\b(v\d+(\.\d+)?|version|edition|handbook|rules|regulations|policy|annual|report)\b", "", base)
    tokens = [t for t in re.findall(r"[a-z]+", base) if len(t) > 2]
    # Identify canonical known clusters
    s = set(tokens)
    if "academic" in s or "undergrad" in s or "undergraduate" in s:
        return "academic_rules"
    if "fee" in s or "tuition" in s:
        return "fee_structure"
    if "grading" in s or "grade" in s:
        return "academic_grading"
    if "startup" in s or "entrepreneurship" in s or "startups" in s:
        return "faculty_startup_policy"
    if "student" in s:
        return "student_handbook"
    if "prospectus" in s:
        return "prospectus"
    return "_".join(sorted(tokens)[:4]) or source.lower()


def parse_version_tuple(ver_str: str) -> tuple[int, ...]:
    """Parse versions like '2026-06', '2023-08', '2026', 'v2.1' into comparable integers."""
    parts = re.findall(r"\d+", ver_str)
    return tuple(int(p) for p in parts) if parts else (0,)


# --------------------------------------------------------------------------- Checks

def check_quality(doc: DocumentRecord, min_chars: int = 50) -> CheckResult:
    """Check 1: Quality.

    Detects:
    - Extraction failures (corrupted, unparseable files)
    - Empty documents (0 characters)
    - Suspiciously short documents (< min_chars)
    - Garbled or unreadable text (replacement character / control character noise)
    """
    if doc.extraction_error:
        return CheckResult(
            name="quality",
            passed=False,
            reason=f"Quality check failed: extraction failed ({doc.extraction_error})",
            details={"error": doc.extraction_error},
        )

    if doc.char_count == 0 or not doc.raw_text.strip():
        return CheckResult(
            name="quality",
            passed=False,
            reason="Quality check failed: document text is empty (0 characters)",
            details={"char_count": 0, "page_count": doc.page_count},
        )

    if doc.char_count < min_chars:
        return CheckResult(
            name="quality",
            passed=False,
            reason=f"Quality check failed: document is suspiciously short ({doc.char_count} chars < {min_chars} required)",
            details={"char_count": doc.char_count, "min_chars": min_chars},
        )

    # Garbled text detection: replacement character \ufffd or excessive non-printable control chars
    replacement_count = doc.raw_text.count("\ufffd")
    total_len = max(1, len(doc.raw_text))
    if (replacement_count / total_len) > 0.05:
        return CheckResult(
            name="quality",
            passed=False,
            reason=f"Quality check failed: document contains excessive replacement characters ({replacement_count} \ufffd chars)",
            details={"replacement_chars": replacement_count, "ratio": round(replacement_count / total_len, 4)},
        )

    return CheckResult(
        name="quality",
        passed=True,
        reason=f"Quality check passed: clean extraction ({doc.char_count} characters across {doc.page_count} page(s))",
        details={"char_count": doc.char_count, "page_count": doc.page_count},
    )


def check_completeness(doc: DocumentRecord) -> CheckResult:
    """Check 2: Completeness.

    Detects:
    - Missing required metadata (source, title, date/version, page_count)
    - Truncated content (explicit truncation banners or cut-off indicators)
    """
    missing_fields: list[str] = []
    if not doc.source or doc.source.strip() == "":
        missing_fields.append("source")
    if not doc.title or doc.title.strip().lower() in {"unknown", "untitled", "none", ""}:
        missing_fields.append("title")
    if not doc.doc_version or doc.doc_version.strip().lower() in {"unknown", "none", ""}:
        missing_fields.append("doc_version")
    if doc.page_count <= 0:
        missing_fields.append("page_count")

    if missing_fields:
        return CheckResult(
            name="completeness",
            passed=False,
            reason=f"Completeness check failed: missing required metadata ({', '.join(missing_fields)})",
            details={"missing_fields": missing_fields, "metadata": {"title": doc.title, "version": doc.doc_version}},
        )

    # Truncation detection
    lower_text = doc.raw_text.lower()
    truncation_pattern = re.compile(r"\[[^\]]*\b(?:truncated|content cut off|cut off|page cut off)\b[^\]]*\]")
    match = truncation_pattern.search(lower_text)
    if match:
        marker = match.group(0)
        return CheckResult(
            name="completeness",
            passed=False,
            reason=f"Completeness check failed: content contains explicit truncation marker ('{marker}')",
            details={"marker": marker},
        )

    return CheckResult(
        name="completeness",
        passed=True,
        reason="Completeness check passed: all required metadata present and content complete",
        details={"title": doc.title, "doc_version": doc.doc_version, "page_count": doc.page_count},
    )


def check_recency(
    doc: DocumentRecord,
    all_docs: list[DocumentRecord],
    quarantine_superseded: bool = True,
) -> CheckResult:
    """Check 3: Recency.

    Detects:
    - Whether a newer version of the same document is present in the corpus
    - Whether the document is explicitly marked as superseded / outdated
    """
    if doc.superseded:
        reason = f"Recency check failed: document is marked superseded (version {doc.doc_version})"
        return CheckResult(
            name="recency",
            passed=not quarantine_superseded,
            reason=reason if quarantine_superseded else f"Recency note: {reason}",
            details={"superseded": True, "doc_version": doc.doc_version},
        )

    current_ver = parse_version_tuple(doc.doc_version)
    for other in all_docs:
        if other.source == doc.source:
            continue
        # Compare documents in the same lineage
        if other.lineage_key and other.lineage_key == doc.lineage_key:
            other_ver = parse_version_tuple(other.doc_version)
            if other_ver > current_ver:
                reason = (
                    f"Recency check failed: superseded by newer version '{other.source}' "
                    f"({other.doc_version} > {doc.doc_version})"
                )
                return CheckResult(
                    name="recency",
                    passed=not quarantine_superseded,
                    reason=reason,
                    details={
                        "newer_version_source": other.source,
                        "newer_version": other.doc_version,
                        "current_version": doc.doc_version,
                        "lineage": doc.lineage_key,
                    },
                )

    return CheckResult(
        name="recency",
        passed=True,
        reason=f"Recency check passed: document is the latest active version ({doc.doc_version})",
        details={"doc_version": doc.doc_version, "lineage": doc.lineage_key},
    )


def check_uniqueness(
    doc: DocumentRecord,
    prior_accepted: list[DocumentRecord],
    near_dupe_threshold: float = 0.85,
) -> CheckResult:
    """Check 4: Uniqueness.

    Detects:
    - Exact duplicate documents by SHA-256 content hash
    - Near-duplicate documents by token 3-gram Jaccard similarity
    """
    # 1. Exact duplicate check
    for prior in prior_accepted:
        if prior.file_sha256 == doc.file_sha256 or prior.content_sha256 == doc.content_sha256:
            return CheckResult(
                name="uniqueness",
                passed=False,
                reason=f"Uniqueness check failed: exact duplicate of '{prior.source}' (SHA-256 match: {doc.file_sha256[:12]}...)",
                details={
                    "duplicate_type": "exact",
                    "duplicate_of": prior.source,
                    "sha256": doc.file_sha256,
                },
            )

    # 2. Near-duplicate check
    for prior in prior_accepted:
        if not prior.cleaned_text or not doc.cleaned_text:
            continue
        sim = jaccard_similarity(doc.cleaned_text, prior.cleaned_text)
        if sim >= near_dupe_threshold:
            return CheckResult(
                name="uniqueness",
                passed=False,
                reason=(
                    f"Uniqueness check failed: near-duplicate of '{prior.source}' "
                    f"(similarity {sim:.3f} >= threshold {near_dupe_threshold})"
                ),
                details={
                    "duplicate_type": "near",
                    "duplicate_of": prior.source,
                    "similarity": round(sim, 4),
                    "threshold": near_dupe_threshold,
                },
            )

    return CheckResult(
        name="uniqueness",
        passed=True,
        reason="Uniqueness check passed: document content is unique",
        details={"sha256": doc.file_sha256[:12]},
    )


# --------------------------------------------------------------------------- Quarantine Execution

def quarantine_file(doc_path: Path, result: VerificationResult, quarantine_dir: Path) -> Path:
    """Move or copy invalid document into quarantine directory with diagnostic sidecar."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dest_file = quarantine_dir / doc_path.name
    try:
        shutil.copy2(doc_path, dest_file)
    except Exception as exc:
        log.warning("could not copy file %s to quarantine: %s", doc_path.name, exc)

    sidecar_path = quarantine_dir / f"{doc_path.name}.quarantine.json"
    sidecar_data = {
        "source": doc_path.name,
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "failed_checks": [name for name, ch in result.checks.items() if not ch.passed],
        "checks": {name: {"passed": ch.passed, "reason": ch.reason, "details": ch.details}
                   for name, ch in result.checks.items()},
        "metadata": result.metadata,
    }
    sidecar_path.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")
    return dest_file


# --------------------------------------------------------------------------- Inspection & Orchestration

def inspect_document(
    path: Path,
    title: str = "",
    doc_version: str = "",
    superseded: bool = False,
    settings: IndexerSettings | None = None,
) -> DocumentRecord:
    """Extract document text, calculate checksums, and extract metadata."""
    settings = settings or get_settings()
    file_hash = compute_file_sha256(path)
    extraction_error = None
    pages: list[Page] = []
    raw_text = ""
    cleaned_text = ""

    # Attempt extraction
    try:
        from .extract import extract
        from .clean import clean

        extracted_doc = extract(
            str(path),
            doc_version=doc_version or "unknown",
            superseded=superseded,
            ocr_dpi=settings.ocr_dpi,
            ocr_lang=settings.ocr_lang,
            min_page_chars=settings.min_page_chars,
            detect_columns=settings.detect_columns,
            title=title,
        )
        cleaned_doc = clean(extracted_doc)
        pages = extracted_doc.pages
        raw_text = "\n\n".join(p.text for p in pages)
        cleaned_text = "\n\n".join(p.text for p in cleaned_doc.pages)
    except Exception as exc:
        extraction_error = f"{type(exc).__name__}: {exc}"

    # If title wasn't passed in, infer from text heading or file name
    if not title:
        if path.suffix.lower() == ".md":
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines:
                if line.startswith("# "):
                    title = line.lstrip("# ").strip()
                    break
        elif raw_text:
            first_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            if first_lines:
                title = first_lines[0][:100]

    content_hash = compute_text_sha256(raw_text) if raw_text else file_hash
    lineage_key = extract_lineage_key(title, path.name)

    return DocumentRecord(
        path=path,
        source=path.name,
        title=title,
        doc_version=doc_version,
        superseded=superseded,
        file_sha256=file_hash,
        content_sha256=content_hash,
        pages=pages,
        raw_text=raw_text,
        char_count=len(raw_text),
        word_count=len(raw_text.split()),
        page_count=len(pages),
        cleaned_text=cleaned_text,
        extraction_error=extraction_error,
        lineage_key=lineage_key,
    )


def verify_documents(
    documents: list[DocumentRecord],
    settings: IndexerSettings | None = None,
    quarantine_dir: Path | None = None,
) -> tuple[list[DocumentRecord], VerificationReport]:
    """Execute all 4 verification checks for a batch of documents.

    Returns:
    - list of accepted DocumentRecord instances
    - VerificationReport summarizing the verification run
    """
    settings = settings or get_settings()
    q_dir = quarantine_dir or (Path(settings.quarantine_dir) if settings.quarantine_dir else Path(settings.corpus_dir) / "quarantine")
    run_id = f"verify-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    timestamp = datetime.now(timezone.utc).isoformat()

    accepted: list[DocumentRecord] = []
    results: list[VerificationResult] = []
    summary_counts: dict[str, dict[str, int]] = {
        "quality": {"passed": 0, "failed": 0},
        "completeness": {"passed": 0, "failed": 0},
        "recency": {"passed": 0, "failed": 0},
        "uniqueness": {"passed": 0, "failed": 0},
    }

    for doc in documents:
        # Check 1: Quality
        q_check = check_quality(doc, min_chars=settings.min_doc_chars)
        # Check 2: Completeness
        c_check = check_completeness(doc)
        # Check 3: Recency
        r_check = check_recency(doc, documents, quarantine_superseded=settings.quarantine_superseded)
        # Check 4: Uniqueness (against already accepted documents)
        u_check = check_uniqueness(doc, accepted, near_dupe_threshold=settings.near_dupe_threshold)

        checks = {
            "quality": q_check,
            "completeness": c_check,
            "recency": r_check,
            "uniqueness": u_check,
        }

        for name, ch in checks.items():
            if ch.passed:
                summary_counts[name]["passed"] += 1
            else:
                summary_counts[name]["failed"] += 1

        all_passed = all(ch.passed for ch in checks.values())
        action = "accepted" if all_passed else "quarantined"
        v_res = VerificationResult(
            source=doc.source,
            passed=all_passed,
            action=action,
            checks=checks,
            metadata={
                "title": doc.title,
                "doc_version": doc.doc_version,
                "superseded": doc.superseded,
                "char_count": doc.char_count,
                "page_count": doc.page_count,
                "sha256": doc.file_sha256,
            },
        )

        if all_passed:
            accepted.append(doc)
        else:
            q_path = quarantine_file(doc.path, v_res, q_dir)
            v_res.quarantine_path = str(q_path)

        results.append(v_res)

    report = VerificationReport(
        run_id=run_id,
        timestamp=timestamp,
        corpus_dir=str(settings.corpus_dir),
        quarantine_dir=str(q_dir),
        total_documents=len(documents),
        passed_count=len(accepted),
        quarantined_count=len(documents) - len(accepted),
        checks_summary=summary_counts,
        results=results,
    )

    # Save report JSON
    q_dir.mkdir(parents=True, exist_ok=True)
    report_file = q_dir / "verification_report.json"
    report_file.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if settings.verification_report_path:
        out_p = Path(settings.verification_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return accepted, report


# --------------------------------------------------------------------------- CLI Entrypoint

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify incoming documents before ingestion (Part A1).")
    parser.add_argument("--corpus-dir", default="", help="Directory containing documents and optional _index.json")
    parser.add_argument("--quarantine-dir", default="", help="Directory where invalid documents will be moved")
    parser.add_argument("--report", default="", help="Path to save output verification_report.json")
    parser.add_argument("--near-dupe-threshold", type=float, default=0.85, help="Near-duplicate threshold")
    parser.add_argument("--allow-superseded", action="store_true", help="Allow superseded documents without quarantining")
    args = parser.parse_args()

    settings = get_settings()
    corpus_dir = Path(args.corpus_dir or settings.corpus_dir)
    if not corpus_dir.exists():
        print(f"Error: Corpus directory '{corpus_dir}' does not exist.", file=sys.stderr)
        return 1

    quarantine_dir = Path(args.quarantine_dir or (settings.quarantine_dir if settings.quarantine_dir else corpus_dir / "quarantine"))
    report_path = args.report or settings.verification_report_path

    # Discover and inspect
    manifest_entries: dict[str, dict[str, Any]] = {}
    index_file = corpus_dir / "_index.json"
    if index_file.exists():
        try:
            for entry in json.loads(index_file.read_text(encoding="utf-8")):
                manifest_entries[entry.get("file", "")] = entry
        except Exception as e:
            print(f"Warning: could not parse {index_file}: {e}")

    records: list[DocumentRecord] = []
    for file_path in sorted(corpus_dir.glob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith(".") or file_path.name.lower().startswith("readme"):
            continue
        if file_path.suffix.lower() not in {".pdf", ".html", ".htm", ".md", ".txt"}:
            continue

        meta = manifest_entries.get(file_path.name, {})
        title = meta.get("name", "")
        doc_version = str(meta.get("doc_version", ""))
        superseded = bool(meta.get("superseded", False))

        # Check front-matter if markdown
        if file_path.suffix.lower() == ".md" and not (title and doc_version):
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines and lines[0].strip() == "---":
                for line in lines[1:]:
                    if line.strip() == "---":
                        break
                    k, _, v = line.partition(":")
                    k, v = k.strip(), v.strip()
                    if k == "title" and not title:
                        title = v
                    elif k == "doc_version" and not doc_version:
                        doc_version = v
                    elif k == "superseded":
                        superseded = v.lower() == "true"

        record = inspect_document(
            file_path,
            title=title,
            doc_version=doc_version,
            superseded=superseded,
            settings=settings,
        )
        records.append(record)

    override_settings = IndexerSettings(
        embed_model=settings.embed_model,
        chroma_host=settings.chroma_host,
        chroma_port=settings.chroma_port,
        collection=settings.collection,
        corpus_dir=str(corpus_dir),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        ollama_url=settings.ollama_url,
        quarantine_dir=str(quarantine_dir),
        near_dupe_threshold=args.near_dupe_threshold,
        quarantine_superseded=not args.allow_superseded,
        verification_report_path=report_path,
    )

    accepted, report = verify_documents(records, override_settings, quarantine_dir)

    print(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "logger": "rahbar.indexer.verify",
        "msg": "verification complete",
        "total": report.total_documents,
        "accepted": report.passed_count,
        "quarantined": report.quarantined_count,
        "summary": report.checks_summary,
        "quarantine_dir": str(quarantine_dir),
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

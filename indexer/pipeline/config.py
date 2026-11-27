"""Indexer configuration, from the same ConfigMap the API reads (section 4.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexerSettings:
    embed_model: str        # EMBED_MODEL: must equal the API's value (section 4.4)
    chroma_host: str
    chroma_port: int
    collection: str
    corpus_dir: str         # where the documents and _index.json sit inside the Job's pod
    chunk_size: int
    chunk_overlap: int
    ollama_url: str
    # Optional knobs (all have defaults, so the ConfigMap does not have to set them).
    embed_batch_size: int = 32   # chunks per embedding request
    ocr_dpi: int = 300           # resolution a scanned page is rendered at before OCR
    ocr_lang: str = "eng"        # Tesseract language pack
    min_page_chars: int = 40     # a page with less text than this is sent to OCR
    detect_columns: bool = True  # read two-column PDF pages column by column
    quarantine_dir: str = ""     # directory for quarantined invalid documents
    near_dupe_threshold: float = 0.85  # similarity threshold for near-duplicate detection
    min_doc_chars: int = 15      # min length for non-empty document quality check
    quarantine_superseded: bool = False  # whether superseded documents are quarantined
    verification_report_path: str = ""  # path to save JSON verification report
    state_file: str = ""         # path to persistent ingestion state JSON file
    incremental: bool = True     # enable incremental delta updates (Part A2)
    archive_dir: str = ""        # directory for archived superseded documents (Part A3)


def get_settings() -> IndexerSettings:
    return IndexerSettings(
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        chroma_host=os.environ.get("CHROMA_HOST", "chroma"),
        chroma_port=int(os.environ.get("CHROMA_PORT", 8000)),
        collection=os.environ.get("CHROMA_COLLECTION", "rahbar"),
        corpus_dir=os.environ.get("CORPUS_DIR", "/corpus"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", 500)),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", 80)),
        ollama_url=os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434"),
        embed_batch_size=int(os.environ.get("EMBED_BATCH_SIZE", 32)),
        ocr_dpi=int(os.environ.get("OCR_DPI", 300)),
        ocr_lang=os.environ.get("OCR_LANG", "eng"),
        min_page_chars=int(os.environ.get("MIN_PAGE_CHARS", 40)),
        detect_columns=os.environ.get("EXTRACT_COLUMNS", "auto").lower() not in {"off", "0", "false"},
        quarantine_dir=os.environ.get("QUARANTINE_DIR", ""),
        near_dupe_threshold=float(os.environ.get("NEAR_DUPE_THRESHOLD", "0.85")),
        min_doc_chars=int(os.environ.get("MIN_DOC_CHARS", "15")),
        quarantine_superseded=os.environ.get("QUARANTINE_SUPERSEDED", "0").lower() in {"1", "true", "yes"},
        verification_report_path=os.environ.get("VERIFICATION_REPORT_PATH", ""),
        state_file=os.environ.get("STATE_FILE", ""),
        incremental=os.environ.get("INCREMENTAL", "1").lower() not in {"off", "0", "false", "no"},
        archive_dir=os.environ.get("ARCHIVE_DIR", ""),
    )
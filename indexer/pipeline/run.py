"""The indexer Job's entry point. It runs once and exits (section 4.2).

    python -m pipeline.run --dummy      # week 1: three fake chunks, no model needed
    python -m pipeline.run              # the real pipeline: extract, clean, chunk, embed, store
    python -m pipeline.run --dry-run    # extract, clean, chunk only: print the numbers, write nothing
    python -m pipeline.run --dry-run --dump chunks.jsonl   # ...and save every chunk to read

Exit code 0 means the index was written; anything else fails the Job, and
Kubernetes retries it up to the Job's backoffLimit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from .config import IndexerSettings, get_settings
from .models import Chunk

STAGE_LOGGER = "rahbar.indexer"


def log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": STAGE_LOGGER, "msg": msg, **fields}), flush=True)


class _JsonFormatter(logging.Formatter):
    """The stage modules use `logging`; this prints their records as one JSON line each."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
                           "logger": record.name, "level": record.levelname,
                           "msg": record.getMessage()})


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    for noisy in ("pdfminer", "pdfplumber", "PIL"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _front_matter(path: Path) -> dict[str, str]:
    """Read `key: value` lines between leading '---' markers (used by eval/fixtures)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """List (path, doc_version, superseded) for every document to index.

    corpus/fetch.py writes _index.json next to the files it downloads; use it when
    present. Otherwise read Markdown front matter (the CI fixtures work this way).
    """
    root = Path(corpus_dir)
    # If the mounted corpus volume is empty, seed it with bundled fixtures
    if not root.exists() or not any(p for p in root.glob("*") if p.suffix.lower() in {".pdf", ".md", ".html", ".txt"}):
        fixtures_dir = Path(__file__).resolve().parents[1] / "fixtures"
        if fixtures_dir.exists() and any(fixtures_dir.iterdir()):
            import shutil
            try:
                root.mkdir(parents=True, exist_ok=True)
                for item in fixtures_dir.glob("*"):
                    if item.is_file() and not item.name.lower().startswith("readme") and item.suffix.lower() in {".md", ".json", ".yaml"}:
                        dest = root / item.name
                        if not dest.exists():
                            shutil.copy2(item, dest)
                log("seeded empty corpus directory from bundled fixtures", corpus_dir=corpus_dir)
            except Exception as e:
                log("corpus auto-seed skipped or failed", error=str(e))

    index = root / "_index.json"
    if index.exists():
        entries = json.loads(index.read_text(encoding="utf-8"))
        return [(root / e["file"], str(e.get("doc_version", "unknown")),
                 bool(e.get("superseded", False))) for e in entries]
    found = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt"}:
            continue
        if path.name.lower().startswith("readme"):
            continue
        meta = _front_matter(path) if path.suffix.lower() == ".md" else {}
        found.append((path, meta.get("doc_version", "unknown"),
                      meta.get("superseded", "false").lower() == "true"))
    return found


def run_pipeline(settings: IndexerSettings, *, dry_run: bool = False,
                 dump: str | None = None) -> int:
    from .chunk import chunk, token_count
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .store import store

    documents = discover(settings.corpus_dir)
    if not documents:
        raise SystemExit(f"no documents found in {settings.corpus_dir}")
    all_chunks: list[Chunk] = []
    failed: list[str] = []
    original_tokens = 0
    for path, doc_version, superseded in documents:
        try:
            raw = extract(str(path), doc_version, superseded, ocr_dpi=settings.ocr_dpi,
                          ocr_lang=settings.ocr_lang, min_page_chars=settings.min_page_chars,
                          detect_columns=settings.detect_columns)
            doc = clean(raw)
            pieces = chunk(doc, settings.chunk_size, settings.chunk_overlap)
        except Exception as exc:  # one bad document must not stop the others (section 1)
            failed.append(path.name)
            log("document failed, skipped", source=path.name, error=f"{type(exc).__name__}: {exc}")
            continue
        doc_tokens = sum(token_count(p.text) for p in doc.pages)
        original_tokens += doc_tokens
        log("chunked", source=path.name, doc_version=doc_version, superseded=superseded,
            pages=len(raw.pages), ocr_pages=sum(p.ocr for p in raw.pages),
            empty_pages=len(raw.pages) - len(doc.pages), tokens=doc_tokens, chunks=len(pieces))
        all_chunks.extend(pieces)
    if not all_chunks:
        raise SystemExit("no chunks produced: every document failed or was empty")

    stored_tokens = sum(token_count(c.text) for c in all_chunks)
    log("chunking done", documents=len(documents) - len(failed), failed=failed,
        chunks=len(all_chunks), original_tokens=original_tokens, stored_tokens=stored_tokens,
        storage_amplification=round(stored_tokens / max(1, original_tokens), 2),
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    if dump:
        with open(dump, "w", encoding="utf-8") as fh:
            for c in all_chunks:
                fh.write(json.dumps({"id": c.id, **c.metadata(), "text": c.text},
                                    ensure_ascii=False) + "\n")
        log("chunks written", path=dump)
    if dry_run:
        return len(all_chunks)
    vectors = embed([c.text for c in all_chunks], settings)
    return store(all_chunks, vectors, settings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    parser.add_argument("--dry-run", action="store_true",
                        help="extract, clean and chunk only; no embedding, nothing written")
    parser.add_argument("--dump", metavar="FILE", help="also write every chunk to FILE as JSONL")
    args = parser.parse_args()
    _setup_logging()
    settings = get_settings()
    log("indexer starting", dummy=args.dummy, dry_run=args.dry_run, embed_model=settings.embed_model,
        collection=settings.collection, chroma=f"{settings.chroma_host}:{settings.chroma_port}",
        pod=os.environ.get("HOSTNAME", "-"))
    started = time.perf_counter()
    try:
        if args.dummy:
            from .dummy import write_dummy_index
            from .store import get_client

            count = write_dummy_index(settings, get_client(settings))
        else:
            count = run_pipeline(settings, dry_run=args.dry_run, dump=args.dump)
    except NotImplementedError as exc:
        log("indexer stage not implemented yet", error=str(exc))
        return 2
    log("indexer finished", chunks=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
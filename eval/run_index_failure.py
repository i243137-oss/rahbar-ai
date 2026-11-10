"""Run an isolated OCR or chunking failure experiment against a local Chroma index."""

from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pdfplumber
import yaml

from pipeline import chunk, extract
from pipeline.config import get_settings
from pipeline.embed import embed
from pipeline.models import Chunk, Document, Page
from pipeline.store import get_client

FACULTY_SCAN = "faculty-entrepreneurship-and-technical-start-ups-policy.pdf"
WINDOW_SIZE = 100


def without_ocr(path: str, doc_version: str, superseded: bool) -> Document:
    with pdfplumber.open(path) as pdf:
        pages = [
            Page(Path(path).name, number, page.extract_text() or "")
            for number, page in enumerate(pdf.pages, 1)
        ]
    print(json.dumps({
        "experiment": "A",
        "source": Path(path).name,
        "pages": len(pages),
        "pages_with_text": sum(bool(page.text.strip()) for page in pages),
        "ocr_pages": 0,
    }), flush=True)
    return Document(Path(path).name, doc_version, superseded, pages)


def fixed_windows(doc: Document, _size: int, _overlap: int) -> list[Chunk]:
    words = [(word, page.page) for page in doc.pages for word in page.text.split()]
    return [
        Chunk(
            id=f"{doc.source}-fixed-{start // WINDOW_SIZE:04d}",
            text=" ".join(word for word, _ in words[start:start + WINDOW_SIZE]),
            source=doc.source,
            section="Fixed window",
            page=words[start][1],
            doc_version=doc.doc_version,
            superseded=doc.superseded,
        )
        for start in range(0, len(words), WINDOW_SIZE)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("A", "B"))
    parser.add_argument("--baseline", default="rahbar-local-500-80")
    args = parser.parse_args()

    settings = get_settings()
    client = get_client(settings)
    baseline = client.get_collection(args.baseline)
    if not baseline.count() or (baseline.metadata or {}).get("embed_model") != settings.embed_model:
        raise SystemExit("Baseline must be nonempty and use the configured embedding model.")
    if args.stage == "A" and not (Path(settings.corpus_dir) / FACULTY_SCAN).is_file():
        raise SystemExit(f"Missing scanned PDF in CORPUS_DIR: {FACULTY_SCAN}")

    collection_name = f"rahbar-failure-{args.stage.lower()}-{uuid4().hex[:12]}"
    os.environ["CHROMA_COLLECTION"] = collection_name
    os.environ["CHUNK_SIZE"] = str(500 if args.stage == "A" else WINDOW_SIZE)
    os.environ["CHUNK_OVERLAP"] = str(80 if args.stage == "A" else 0)
    original_extract = extract.extract

    def extract_for_a(path: str, doc_version: str, superseded: bool) -> Document:
        if Path(path).name == FACULTY_SCAN:
            return without_ocr(path, doc_version, superseded)
        return original_extract(path, doc_version, superseded)

    print(f"Building experiment {args.stage} in fresh collection {collection_name}", flush=True)
    change = (
        patch.object(extract, "extract", extract_for_a)
        if args.stage == "A"
        else patch.object(chunk, "chunk", fixed_windows)
    )
    with change, patch("sys.argv", ["pipeline.run"]):
        try:
            runpy.run_module("pipeline.run", run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise

    golden_path = Path(__file__).with_name("golden.yaml")
    questions = {q["id"]: q for q in yaml.safe_load(golden_path.read_text())["questions"]}
    probes = (
        {"q08": "research conducted at NUCES campuses"}
        if args.stage == "A"
        else {"q09": "last day of the makeup week", "q10": "last day of classes", "q12": "FA | 0.00"}
    )
    vectors = embed([questions[qid]["question"] for qid in probes], settings)
    broken = client.get_collection(collection_name)
    for name, collection in ((args.baseline, baseline), (collection_name, broken)):
        print(f"\n{name}: {collection.count()} chunks")
        for (qid, phrase), vector in zip(probes.items(), vectors, strict=True):
            source = questions[qid]["relevant"][0]["source"]
            where = {"doc_version": "2023-08"} if qid == "q10" else {"superseded": False}
            results = collection.query(
                query_embeddings=[vector],
                n_results=5,
                where=where,
                include=["documents", "metadatas"],
            )
            print(f"{qid}: expected source {source}; evidence phrase: {phrase}")
            for rank, (metadata, text) in enumerate(
                zip(results["metadatas"][0], results["documents"][0], strict=True), 1
            ):
                evidence = metadata["source"] == source and phrase.casefold() in (
                    " ".join(text.casefold().split())
                )
                print(f"  {rank}. {metadata['source']} p{metadata['page']} evidence={evidence}")


if __name__ == "__main__":
    main()

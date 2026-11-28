import logging
import os
import sys
from dataclasses import replace
from functools import cache
from pathlib import Path

sys.path.insert(0, "/app/eval")
from pipeline import extract
from pipeline.config import get_settings
import eval_retrieval as evaluation

logging.basicConfig(level=logging.INFO)
extract.extract = cache(extract.extract)
embed = evaluation.embed
vectors = {}

def cached_embeddings(texts, settings):
    missing = list(dict.fromkeys(text for text in texts if text not in vectors))
    if missing:
        vectors.update(zip(missing, embed(missing, settings), strict=True))
    return [vectors[text] for text in texts]

evaluation.embed = cached_embeddings
corpus = Path("/app/corpus/raw")
golden = Path("/app/eval/golden.jsonl")
settings = get_settings()
for size, overlap in [(500, 80), (400, 80)]:
    setting = f"{size}-{overlap}"
    print(f"--- Screening {setting} on the unchanged approved corpus ---", flush=True)
    report = evaluation.run_retrieval_suite(
        golden_file=golden, corpus_dir=corpus,
        settings=replace(settings, chunk_size=size, chunk_overlap=overlap),
        results_dir=Path("/screen") / setting,
    )
    if report["embed_model"] != "nomic-embed-text" or report["corpus_dir"] != str(corpus):
        raise RuntimeError("The screening run did not use the approved corpus and model.")

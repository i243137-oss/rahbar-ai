#!/usr/bin/env python3
"""Component-Level Retrieval Evaluation & Noise Robustness (Assignment 3, Part B3).

Evaluates the RAG retrieval tier against the golden dataset (eval/golden.jsonl):
1. Metrics:
   - Recall@3: Fraction of ground truth relevant passages retrieved in top 3.
   - Recall@5: Fraction of ground truth relevant passages retrieved in top 5.
   - MRR (Mean Reciprocal Rank): 1 / rank of first relevant passage.
   - nDCG@5: Normalized Discounted Cumulative Gain at rank 5.
   - Abstention rate on unanswerable questions.
2. Noise Robustness Experiment:
   - Injects realistic query typos (character swaps, drops, common misspellings).
   - Injects distractor documents into the candidate search pool.
   - Compares Clean vs. Noisy retrieval performance and reports deltas.
3. Reporting:
   - Prints structured comparison tables.
   - Outputs machine-readable results to eval/results/retrieval_eval_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexer"))
sys.path.insert(0, str(REPO_ROOT / "api"))

from pipeline.config import IndexerSettings, get_settings  # noqa: E402
from pipeline.embed import embed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("rahbar.eval.retrieval")


# --------------------------------------------------------------------------- Data Models

@dataclass
class RetrievedPassage:
    source: str
    section: str
    score: float
    page: int | None = None
    text: str = ""
    chunk_id: str = ""


@dataclass
class QueryEvalResult:
    id: str
    question: str
    category: str
    difficulty: str
    is_answerable: bool
    relevant_targets: list[dict[str, Any]]
    retrieved: list[RetrievedPassage]
    recall_at_3: float | None
    recall_at_5: float | None
    reciprocal_rank: float | None
    ndcg_at_5: float | None
    first_hit_rank: int | None


@dataclass
class RetrievalMetricsSummary:
    mode: str
    total_queries: int
    answerable_queries: int
    unanswerable_queries: int
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    abstention_accuracy: float
    recall_at_3_by_type: dict[str, float] = field(default_factory=dict)
    recall_at_5_by_type: dict[str, float] = field(default_factory=dict)
    ndcg_at_5_by_type: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- Metric Math

def is_hit(expected_source: str, expected_section: str, actual_source: str, actual_section: str) -> bool:
    """Check if retrieved passage matches expected ground truth reference."""
    if actual_source.strip().lower() != expected_source.strip().lower():
        return False
    exp_sec = expected_section.strip().lower()
    act_sec = actual_section.strip().lower()
    return exp_sec == act_sec or act_sec.startswith((f"{exp_sec} ", f"{exp_sec}:", f"{exp_sec}."))


def compute_dcg_at_k(hit_ranks: list[int], k: int = 5) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for r in hit_ranks:
        if 1 <= r <= k:
            dcg += 1.0 / math.log2(r + 1)
    return dcg


def compute_idcg_at_k(num_relevant: int, k: int = 5) -> float:
    """Compute Ideal Discounted Cumulative Gain at rank k."""
    ideal_hits = min(max(0, num_relevant), k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))


def score_query_retrieval(
    question_record: dict[str, Any],
    retrieved_passages: list[RetrievedPassage],
    top_k: int = 5,
) -> QueryEvalResult:
    """Score retrieval outcomes for a single query."""
    qid = question_record.get("id", "")
    q_text = question_record.get("question", "")
    cat = question_record.get("category", question_record.get("type", "ordinary"))
    diff = question_record.get("difficulty", "medium")
    relevant = question_record.get("relevant", [])
    must_abstain = question_record.get("must_abstain", False) or not question_record.get("answerable", True)

    if must_abstain or not relevant:
        return QueryEvalResult(
            id=qid,
            question=q_text,
            category=cat,
            difficulty=diff,
            is_answerable=False,
            relevant_targets=[],
            retrieved=retrieved_passages[:top_k],
            recall_at_3=None,
            recall_at_5=None,
            reciprocal_rank=None,
            ndcg_at_5=None,
            first_hit_rank=None,
        )

    expected_refs = [(r["source"], r.get("section", "")) for r in relevant]
    num_expected = len(expected_refs)

    # Identify hits at each rank
    hit_ranks: list[int] = []
    satisfied_refs: set[int] = set()

    for rank, passage in enumerate(retrieved_passages[:top_k], start=1):
        matched = False
        for ref_idx, (exp_src, exp_sec) in enumerate(expected_refs):
            if is_hit(exp_src, exp_sec, passage.source, passage.section):
                satisfied_refs.add(ref_idx)
                matched = True
        if matched:
            hit_ranks.append(rank)

    # Recall@3
    hits_at_3 = sum(
        1 for ref_idx, (exp_src, exp_sec) in enumerate(expected_refs)
        if any(is_hit(exp_src, exp_sec, p.source, p.section) for p in retrieved_passages[:3])
    )
    recall_3 = hits_at_3 / num_expected

    # Recall@5
    hits_at_5 = sum(
        1 for ref_idx, (exp_src, exp_sec) in enumerate(expected_refs)
        if any(is_hit(exp_src, exp_sec, p.source, p.section) for p in retrieved_passages[:5])
    )
    recall_5 = hits_at_5 / num_expected

    # MRR (first hit rank)
    first_hit = hit_ranks[0] if hit_ranks else None
    rr = (1.0 / first_hit) if first_hit else 0.0

    # nDCG@5
    dcg = compute_dcg_at_k(hit_ranks, k=5)
    idcg = compute_idcg_at_k(num_expected, k=5)
    ndcg_5 = (dcg / idcg) if idcg > 0.0 else 0.0

    return QueryEvalResult(
        id=qid,
        question=q_text,
        category=cat,
        difficulty=diff,
        is_answerable=True,
        relevant_targets=relevant,
        retrieved=retrieved_passages[:top_k],
        recall_at_3=recall_3,
        recall_at_5=recall_5,
        reciprocal_rank=rr,
        ndcg_at_5=ndcg_5,
        first_hit_rank=first_hit,
    )


def summarize_results(
    eval_results: list[QueryEvalResult],
    mode: str = "clean",
) -> RetrievalMetricsSummary:
    """Aggregate per-query evaluations into dataset-wide retrieval metrics."""
    answerable = [r for r in eval_results if r.is_answerable]
    unanswerable = [r for r in eval_results if not r.is_answerable]

    recalls_3 = [r.recall_at_3 for r in answerable if r.recall_at_3 is not None]
    recalls_5 = [r.recall_at_5 for r in answerable if r.recall_at_5 is not None]
    mrrs = [r.reciprocal_rank for r in answerable if r.reciprocal_rank is not None]
    ndcgs = [r.ndcg_at_5 for r in answerable if r.ndcg_at_5 is not None]

    # Unanswerable correctness (abstains if no relevant passages or low score)
    abstain_correct = sum(
        1 for r in unanswerable
        if not r.retrieved or (r.retrieved and r.retrieved[0].score < 0.25)
    )
    abstain_acc = (abstain_correct / len(unanswerable)) if unanswerable else 1.0

    # Grouped metrics by category
    rec3_by_cat: dict[str, list[float]] = {}
    rec5_by_cat: dict[str, list[float]] = {}
    ndcg_by_cat: dict[str, list[float]] = {}

    for r in answerable:
        cat = r.category
        if r.recall_at_3 is not None:
            rec3_by_cat.setdefault(cat, []).append(r.recall_at_3)
        if r.recall_at_5 is not None:
            rec5_by_cat.setdefault(cat, []).append(r.recall_at_5)
        if r.ndcg_at_5 is not None:
            ndcg_by_cat.setdefault(cat, []).append(r.ndcg_at_5)

    return RetrievalMetricsSummary(
        mode=mode,
        total_queries=len(eval_results),
        answerable_queries=len(answerable),
        unanswerable_queries=len(unanswerable),
        recall_at_3=round(mean(recalls_3) if recalls_3 else 0.0, 4),
        recall_at_5=round(mean(recalls_5) if recalls_5 else 0.0, 4),
        mrr=round(mean(mrrs) if mrrs else 0.0, 4),
        ndcg_at_5=round(mean(ndcgs) if ndcgs else 0.0, 4),
        abstention_accuracy=round(abstain_acc, 4),
        recall_at_3_by_type={c: round(mean(v), 4) for c, v in rec3_by_cat.items()},
        recall_at_5_by_type={c: round(mean(v), 4) for c, v in rec5_by_cat.items()},
        ndcg_at_5_by_type={c: round(mean(v), 4) for c, v in ndcg_by_cat.items()},
    )


# --------------------------------------------------------------------------- In-Memory Vector Retriever

class InMemoryVectorRetriever:
    """Deterministic in-memory vector retriever supporting clean & distractor search."""

    def __init__(self, settings: IndexerSettings) -> None:
        self.settings = settings
        self.chunks: list[dict[str, Any]] = []
        self.vectors: list[list[float]] = []

    def populate_from_corpus(self, corpus_dir: Path | str | list[Any]) -> int:
        """Extract and chunk policy documents to build candidate vector pool."""
        from pipeline.chunk import chunk
        from pipeline.clean import clean
        from pipeline.extract import extract
        from pipeline.run import discover

        target_list = [corpus_dir] if isinstance(corpus_dir, (str, Path)) else list(corpus_dir)
        docs = []
        for target in target_list:
            t_path = Path(target)
            if t_path.is_file():
                docs.append((t_path, "2026-06", False))
            elif t_path.exists():
                docs.extend(discover(str(t_path)))
        raw_chunks = []
        for path, doc_version, superseded in docs:
            try:
                raw = extract(
                    str(path),
                    doc_version=doc_version,
                    superseded=superseded,
                    ocr_dpi=self.settings.ocr_dpi,
                    ocr_lang=self.settings.ocr_lang,
                    min_page_chars=self.settings.min_page_chars,
                    detect_columns=self.settings.detect_columns,
                )
                cleaned = clean(raw)
                pieces = chunk(cleaned, self.settings.chunk_size, self.settings.chunk_overlap)
                for p in pieces:
                    raw_chunks.append({
                        "id": p.id,
                        "source": p.source,
                        "section": p.section,
                        "page": p.page,
                        "doc_version": p.doc_version,
                        "superseded": p.superseded,
                        "text": p.text,
                    })
            except (RuntimeError, ValueError, OSError) as exc:
                log.warning("failed to chunk %s: %s", path.name, exc)

        if not raw_chunks:
            # Fallback to fixtures if corpus had no documents
            fixtures_dir = REPO_ROOT / "eval" / "fixtures"
            if fixtures_dir.exists() and str(fixtures_dir) != str(corpus_dir):
                return self.populate_from_corpus(fixtures_dir)
            raise RuntimeError("no chunks could be extracted from corpus")

        self.chunks = raw_chunks
        texts = [c["text"] for c in self.chunks]
        self.vectors = embed(texts, self.settings)
        log.info("InMemoryRetriever indexed %d chunks across %d documents", len(self.chunks), len(docs))
        return len(self.chunks)

    def inject_distractors(self, distractor_chunks: list[dict[str, Any]]) -> None:
        """Inject distractor documents into the candidate vector space."""
        texts = [d["text"] for d in distractor_chunks]
        vecs = embed(texts, self.settings)
        self.chunks.extend(distractor_chunks)
        self.vectors.extend(vecs)
        log.info("injected %d distractor chunks into candidate pool", len(distractor_chunks))

    def retrieve(self, question: str, k: int = 5) -> list[RetrievedPassage]:
        """Query candidate chunks and return top k best matches."""
        q_vec = embed([question], self.settings)[0]
        scored: list[tuple[float, dict[str, Any]]] = []

        for c_dict, v_vec in zip(self.chunks, self.vectors):
            dot = sum(a * b for a, b in zip(q_vec, v_vec))
            norm_q = math.sqrt(sum(a * a for a in q_vec)) or 1.0
            norm_v = math.sqrt(sum(b * b for b in v_vec)) or 1.0
            sim = dot / (norm_q * norm_v)
            scored.append((sim, c_dict))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[RetrievedPassage] = []

        for sim, c_dict in scored[:k]:
            results.append(
                RetrievedPassage(
                    source=c_dict["source"],
                    section=c_dict["section"],
                    score=round(sim, 4),
                    page=c_dict.get("page"),
                    text=c_dict.get("text", "")[:200],
                    chunk_id=c_dict.get("id", ""),
                )
            )
        return results


# --------------------------------------------------------------------------- Noise Injector

TYPO_RULES = [
    (r"\battendance\b", "attendence"),
    (r"\btuition\b", "tuiton"),
    (r"\bacademic\b", "acedemic"),
    (r"\bregulation\b", "regulatn"),
    (r"\bminimum\b", "minimm"),
    (r"\bwarning\b", "warnng"),
    (r"\bwithdrawal\b", "withdrwl"),
    (r"\brefund\b", "refnud"),
    (r"\bstudent\b", "studnt"),
    (r"\bexamination\b", "exmination"),
    (r"\bsemester\b", "semster"),
    (r"\bundergraduate\b", "undergrad"),
    (r"\bgenerative\b", "genrative"),
    (r"\bintelligence\b", "inteligence"),
    (r"\bhonors\b", "honrs"),
]


def inject_noise_into_query(query: str, seed: int = 42) -> str:
    """Inject realistic typographical errors, transpositions, and dropped vowels."""
    rng = random.Random(seed + len(query))
    noisy = query

    # Apply lexical typo substitutions
    for pattern, replacement in TYPO_RULES:
        if re.search(pattern, noisy, re.IGNORECASE) and rng.random() > 0.3:
            noisy = re.sub(pattern, replacement, noisy, flags=re.IGNORECASE)

    # Apply character transpositions on long words
    words = noisy.split()
    altered_words = []
    for w in words:
        if len(w) > 5 and rng.random() < 0.25:
            # swap two adjacent interior letters
            idx = rng.randint(1, len(w) - 3)
            w_list = list(w)
            w_list[idx], w_list[idx + 1] = w_list[idx + 1], w_list[idx]
            altered_words.append("".join(w_list))
        else:
            altered_words.append(w)

    return " ".join(altered_words)


DISTRACTOR_CHUNKS = [
    {
        "id": "distractor-cafeteria-01",
        "source": "campus-services-guide.md",
        "section": "1. Dining Services & Cafeteria",
        "page": 1,
        "doc_version": "2026",
        "superseded": False,
        "text": "The campus dining facility provides hot lunches, snacks, and tea service from 8:00 AM to 5:00 PM on weekdays. Subsidized meal tokens are available at the accounts counter.",
    },
    {
        "id": "distractor-sports-02",
        "source": "sports-facilities-rules.md",
        "section": "2. Gymnasium and Swimming Pool Access",
        "page": 1,
        "doc_version": "2026",
        "superseded": False,
        "text": "The university gymnasium and indoor swimming pool are accessible to registered students carrying valid university biometric ID cards. Lockers must be cleared every evening by 7:00 PM.",
    },
    {
        "id": "distractor-parking-03",
        "source": "campus-traffic-regulations.md",
        "section": "3. Motorcycle and Vehicle Parking Permits",
        "page": 1,
        "doc_version": "2026",
        "superseded": False,
        "text": "All student motorbikes and four-wheel vehicles entering university gates must display an authorized campus parking sticker. Overnight parking requires written permission from the Chief Security Officer.",
    },
]


# --------------------------------------------------------------------------- Evaluation Runner

def run_retrieval_suite(
    golden_file: Path | None = None,
    corpus_dir: Path | None = None,
    settings: IndexerSettings | None = None,
    results_dir: Path | None = None,
    noise_test: bool = True,
) -> dict[str, Any]:
    """Execute complete retrieval evaluation suite with baseline and noise robustness benchmarks."""
    g_path = golden_file or REPO_ROOT / "eval" / "golden.jsonl"
    r_dir = results_dir or REPO_ROOT / "eval" / "results"
    r_dir.mkdir(parents=True, exist_ok=True)
    c_dir = corpus_dir or REPO_ROOT / "corpus" / "raw"

    if not c_dir.exists() or not any(c_dir.iterdir()):
        c_dir = REPO_ROOT / "eval" / "fixtures"

    active_settings = settings or get_settings()
    if "OLLAMA_URL" not in os.environ and "minikube" in active_settings.ollama_url:
        import urllib.error
        import urllib.request
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
            active_settings = IndexerSettings(**{**active_settings.__dict__, "ollama_url": "http://127.0.0.1:11434"})
        except (OSError, urllib.error.URLError) as exc:
            log.debug("Ollama localhost probe: %s", exc)

    if active_settings.embed_model != "dummy-768":
        import urllib.error
        import urllib.request
        try:
            urllib.request.urlopen(f"{active_settings.ollama_url.rstrip('/')}/api/tags", timeout=1.0)
        except (OSError, urllib.error.URLError):
            log.warning("Ollama is not reachable at %s; falling back to deterministic dummy-768 embeddings", active_settings.ollama_url)
            active_settings = IndexerSettings(**{**active_settings.__dict__, "embed_model": "dummy-768"})

    # Load golden questions
    if not g_path.exists():
        raise FileNotFoundError(f"Golden dataset file not found: {g_path}")

    records: list[dict[str, Any]] = []
    with open(g_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print("=" * 80)
    print("  RAHBAR AI — COMPONENT-LEVEL RETRIEVAL EVALUATION (PART B3)")
    print("=" * 80)
    print(f"Golden Dataset      : {g_path} ({len(records)} questions)")
    print(f"Corpus Directory    : {c_dir}")
    print(f"Embedding Model     : {active_settings.embed_model}")
    print("-" * 80)

    # Initialize retriever
    retriever = InMemoryVectorRetriever(active_settings)
    chunk_count = retriever.populate_from_corpus(c_dir)
    print(f"Candidate Vector Pool: {chunk_count} chunks indexed.")

    # 1. Clean Baseline Evaluation
    print("\n[STEP 1] Evaluating Baseline Retrieval on Clean Golden Queries...")
    clean_eval_results: list[QueryEvalResult] = []
    for rec in records:
        passages = retriever.retrieve(rec["question"], k=5)
        eval_res = score_query_retrieval(rec, passages, top_k=5)
        clean_eval_results.append(eval_res)

    clean_summary = summarize_results(clean_eval_results, mode="clean")

    # 2. Noise Robustness Experiment
    noisy_summary = None
    noise_deltas = {}

    if noise_test:
        print("\n[STEP 2] Running Noise Robustness Experiment (Query Typos + 3 Distractor Chunks)...")
        # Inject 3 distractor chunks into search pool
        retriever.inject_distractors(DISTRACTOR_CHUNKS)

        noisy_eval_results: list[QueryEvalResult] = []
        for idx, rec in enumerate(records):
            noisy_q = inject_noise_into_query(rec["question"], seed=100 + idx)
            passages = retriever.retrieve(noisy_q, k=5)
            eval_res = score_query_retrieval(rec, passages, top_k=5)
            noisy_eval_results.append(eval_res)

        noisy_summary = summarize_results(noisy_eval_results, mode="noisy")

        # Compute metric deltas
        noise_deltas = {
            "delta_recall_at_3": round(noisy_summary.recall_at_3 - clean_summary.recall_at_3, 4),
            "delta_recall_at_5": round(noisy_summary.recall_at_5 - clean_summary.recall_at_5, 4),
            "delta_mrr": round(noisy_summary.mrr - clean_summary.mrr, 4),
            "delta_ndcg_at_5": round(noisy_summary.ndcg_at_5 - clean_summary.ndcg_at_5, 4),
            "recall_5_retention_pct": round((noisy_summary.recall_at_5 / max(0.0001, clean_summary.recall_at_5)) * 100.0, 1),
            "ndcg_5_retention_pct": round((noisy_summary.ndcg_at_5 / max(0.0001, clean_summary.ndcg_at_5)) * 100.0, 1),
        }

    # Print Summary Table
    print("\n" + "=" * 80)
    print("  RETRIEVAL METRICS & NOISE ROBUSTNESS COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Clean Baseline':<16} | {'Noisy / Distractors':<20} | {'Delta':<10}")
    print("-" * 77)

    if noisy_summary:
        print(f"{'Recall@3':<25} | {clean_summary.recall_at_3:<16.4f} | {noisy_summary.recall_at_3:<20.4f} | {noise_deltas['delta_recall_at_3']:<10.4f}")
        print(f"{'Recall@5':<25} | {clean_summary.recall_at_5:<16.4f} | {noisy_summary.recall_at_5:<20.4f} | {noise_deltas['delta_recall_at_5']:<10.4f}")
        print(f"{'MRR (Mean Reciprocal Rank)':<25} | {clean_summary.mrr:<16.4f} | {noisy_summary.mrr:<20.4f} | {noise_deltas['delta_mrr']:<10.4f}")
        print(f"{'nDCG@5':<25} | {clean_summary.ndcg_at_5:<16.4f} | {noisy_summary.ndcg_at_5:<20.4f} | {noise_deltas['delta_ndcg_at_5']:<10.4f}")
        print(f"{'Abstention Accuracy':<25} | {clean_summary.abstention_accuracy:<16.4f} | {noisy_summary.abstention_accuracy:<20.4f} | {'0.0000':<10}")
        print("-" * 77)
        print(f"Recall@5 Robustness Retention  : {noise_deltas['recall_5_retention_pct']}% of baseline capability")
        print(f"nDCG@5 Robustness Retention    : {noise_deltas['ndcg_5_retention_pct']}% of baseline capability")
    else:
        print(f"{'Recall@3':<25} | {clean_summary.recall_at_3:<16.4f} | {'N/A':<20} | {'N/A':<10}")
        print(f"{'Recall@5':<25} | {clean_summary.recall_at_5:<16.4f} | {'N/A':<20} | {'N/A':<10}")
        print(f"{'MRR':<25} | {clean_summary.mrr:<16.4f} | {'N/A':<20} | {'N/A':<10}")
        print(f"{'nDCG@5':<25} | {clean_summary.ndcg_at_5:<16.4f} | {'N/A':<20} | {'N/A':<10}")

    print("=" * 80)

    # Save structured report
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dict: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden_file": str(g_path),
        "corpus_dir": str(c_dir),
        "embed_model": active_settings.embed_model,
        "clean_baseline": asdict(clean_summary),
        "noisy_experiment": asdict(noisy_summary) if noisy_summary else None,
        "noise_deltas": noise_deltas,
        "per_query_clean": [
            {
                "id": r.id,
                "question": r.question,
                "category": r.category,
                "difficulty": r.difficulty,
                "recall_3": r.recall_at_3,
                "recall_5": r.recall_at_5,
                "mrr": r.reciprocal_rank,
                "ndcg_5": r.ndcg_at_5,
                "first_hit_rank": r.first_hit_rank,
            }
            for r in clean_eval_results
        ],
    }

    report_latest = r_dir / "retrieval_eval_report.json"
    report_stamped = r_dir / f"retrieval_eval_report_{timestamp_str}.json"
    report_latest.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    report_stamped.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"Report written to: {report_latest}")
    print(f"Timestamped archive: {report_stamped}")
    print("=" * 80)

    return report_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate component-level retrieval and noise robustness.")
    parser.add_argument("--golden", type=Path, default=None, help="Path to golden dataset JSONL")
    parser.add_argument("--corpus-dir", type=Path, default=None, help="Path to corpus directory")
    parser.add_argument("--model", type=str, default="", help="Embedding model override")
    parser.add_argument("--results-dir", type=Path, default=None, help="Output directory for reports")
    parser.add_argument("--skip-noise", action="store_true", help="Skip noise robustness experiment")
    args = parser.parse_args()

    custom_settings = get_settings()
    if args.model:
        custom_settings = IndexerSettings(**{**custom_settings.__dict__, "embed_model": args.model})
    elif custom_settings.embed_model != "dummy-768":
        try:
            import httpx  # noqa: F401
        except ImportError:
            custom_settings = IndexerSettings(**{**custom_settings.__dict__, "embed_model": "dummy-768"})

    try:
        report = run_retrieval_suite(
            golden_file=args.golden,
            corpus_dir=args.corpus_dir,
            settings=custom_settings,
            results_dir=args.results_dir,
            noise_test=not args.skip_noise,
        )
        return 0 if report["clean_baseline"]["recall_at_5"] > 0.0 else 1
    except Exception:
        log.exception("Retrieval evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

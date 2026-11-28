#!/usr/bin/env python3
"""Part E2 & E3: Coverage Analysis and Reporting Runner (Assignment 3, Part E).

Evaluates knowledge base and golden dataset coverage against simulated live queries:
- For each simulated query in eval/live_queries.jsonl:
  * Computes semantic similarity against golden questions and indexed corpus chunks.
  * Classifies query as IN_DOMAIN_COVERED, IN_DOMAIN_UNCOVERED, or OUT_OF_DOMAIN.
  * Identifies matched evidence (source document, section, and similarity score).
  * Records uncovered topics and builds similarity distribution statistics.
- Justifies similarity threshold based on empirical embedding evaluation results:
  * Paraphrase similarity: mean=0.8539, min=0.7299
  * Unrelated similarity : mean=0.4619, max=0.5733
  * Separation gap       : 0.3919 -> Justified threshold tau = 0.70
- Produces:
  * Formatted console summary and topic breakdown.
  * Machine-readable artifact: eval/results/coverage_report.json.
  * Configurable coverage threshold with non-zero exit code if not met.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("rahbar.eval.coverage")

# Empirically justified similarity threshold based on Part B2 separation gap (0.3919)
JUSTIFIED_SIMILARITY_THRESHOLD = 0.70


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return dot / (norm_a * norm_b)


def run_coverage_analysis(
    live_queries_path: Path,
    golden_path: Path,
    corpus_dir: Path | None = None,
    threshold: float = JUSTIFIED_SIMILARITY_THRESHOLD,
    min_coverage_pct: float = 0.0,
    model_override: str = "",
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute coverage analysis over simulated live queries."""
    r_dir = results_dir or REPO_ROOT / "eval" / "results"
    r_dir.mkdir(parents=True, exist_ok=True)

    if not live_queries_path.exists():
        raise FileNotFoundError(f"Simulated live queries dataset not found: {live_queries_path}")
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {golden_path}")

    # Ensure search paths and import pipeline dependencies
    for p in [str(REPO_ROOT / "indexer"), str(REPO_ROOT / "eval")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from eval_retrieval import InMemoryVectorRetriever
    from pipeline.config import IndexerSettings, get_settings
    from pipeline.embed import embed

    # Load simulated queries
    live_queries: list[dict[str, Any]] = []
    with live_queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                live_queries.append(json.loads(line))

    # Load golden questions
    golden_questions: list[dict[str, Any]] = []
    with golden_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                golden_questions.append(json.loads(line))

    # Setup IndexerSettings and probe local Ollama
    base_settings = get_settings()
    if model_override:
        base_settings = IndexerSettings(**{**base_settings.__dict__, "embed_model": model_override})
    elif "OLLAMA_URL" not in os.environ and "minikube" in base_settings.ollama_url:
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
            base_settings = IndexerSettings(**{**base_settings.__dict__, "ollama_url": "http://127.0.0.1:11434"})
        except (OSError, urllib.error.URLError) as exc:
            log.debug("Ollama localhost probe: %s", exc)

    if base_settings.embed_model != "dummy-768":
        try:
            urllib.request.urlopen(f"{base_settings.ollama_url.rstrip('/')}/api/tags", timeout=1.0)
        except (OSError, urllib.error.URLError):
            log.warning("Ollama offline at %s; using deterministic fallback", base_settings.ollama_url)
            base_settings = IndexerSettings(**{**base_settings.__dict__, "embed_model": "dummy-768"})

    print("=" * 88)
    print("  RAHBAR AI — KNOWLEDGE BASE COVERAGE ANALYSIS (PART E2 & E3)")
    print("=" * 88)
    print(f"Simulated Live Queries : {live_queries_path} ({len(live_queries)} queries)")
    print(f"Golden Dataset Target  : {golden_path} ({len(golden_questions)} reference questions)")
    print(f"Corpus Directory       : {corpus_dir or 'None (Golden-Only Comparison)'}")
    print(f"Embedding Model        : {base_settings.embed_model}")
    print(f"Similarity Threshold   : {threshold:.2f} (Empirically justified by separation gap: 0.3919)")
    print("-" * 88)

    # 1. Embed golden questions
    log.info("Embedding %d golden reference questions...", len(golden_questions))
    g_texts = [g["question"] for g in golden_questions]
    g_vectors = embed(g_texts, base_settings)

    # 2. Optionally index corpus chunks for evidence matching
    corpus_retriever = None
    if corpus_dir:
        targets = [corpus_dir] if isinstance(corpus_dir, (str, Path)) else list(corpus_dir)
        valid_targets = [Path(t) for t in targets if Path(t).exists()]
        if valid_targets:
            log.info("Indexing corpus chunks from %s...", [str(t) for t in valid_targets])
            corpus_retriever = InMemoryVectorRetriever(base_settings)
            chunk_count = corpus_retriever.populate_from_corpus(valid_targets)
            log.info("Indexed %d corpus candidate chunks.", chunk_count)

    # 3. Embed simulated queries
    log.info("Embedding %d simulated live queries...", len(live_queries))
    q_texts = [q["question"] for q in live_queries]
    q_vectors = embed(q_texts, base_settings)

    # 4. Evaluate coverage per query
    per_query_results: list[dict[str, Any]] = []
    uncovered_topic_counts: dict[str, int] = {}
    uncovered_topic_examples: dict[str, list[str]] = {}

    similarity_scores: list[float] = []

    for q_data, q_vec in zip(live_queries, q_vectors):
        qid = q_data.get("id", "")
        q_text = q_data.get("question", "")
        domain = q_data.get("domain", "in_domain")
        topic = q_data.get("topic", "general")

        # Compare against golden questions
        best_g_sim = -1.0
        best_g_match = None
        for g_data, g_vec in zip(golden_questions, g_vectors):
            sim = cosine_similarity(q_vec, g_vec)
            if sim > best_g_sim:
                best_g_sim = sim
                best_g_match = g_data

        # Compare against corpus chunks if indexed
        best_c_sim = -1.0
        best_c_passage = None
        if corpus_retriever:
            top_passages = corpus_retriever.retrieve(q_text, k=1)
            if top_passages:
                best_c_passage = top_passages[0]
                best_c_sim = best_c_passage.score

        # Determine best evidence match
        effective_sim = max(best_g_sim, best_c_sim)
        similarity_scores.append(round(effective_sim, 4))

        matched_evidence = {}
        if best_c_sim >= best_g_sim and best_c_passage:
            matched_evidence = {
                "type": "corpus_chunk",
                "source": best_c_passage.source,
                "section": best_c_passage.section,
                "page": best_c_passage.page,
                "similarity": round(best_c_sim, 4),
            }
        elif best_g_match:
            g_rel = best_g_match.get("relevant", [{}])
            first_rel = g_rel[0] if g_rel else {}
            matched_evidence = {
                "type": "golden_question",
                "golden_id": best_g_match.get("id"),
                "golden_question": best_g_match.get("question"),
                "source": first_rel.get("source"),
                "section": first_rel.get("section"),
                "page": first_rel.get("page"),
                "similarity": round(best_g_sim, 4),
            }

        # Classification Logic
        is_covered = False
        if domain == "out_of_domain":
            classification = "OUT_OF_DOMAIN"
            is_covered = False
        elif effective_sim >= threshold:
            classification = "IN_DOMAIN_COVERED"
            is_covered = True
        else:
            classification = "IN_DOMAIN_UNCOVERED"
            is_covered = False
            uncovered_topic_counts[topic] = uncovered_topic_counts.get(topic, 0) + 1
            if topic not in uncovered_topic_examples:
                uncovered_topic_examples[topic] = []
            if len(uncovered_topic_examples[topic]) < 2:
                uncovered_topic_examples[topic].append(q_text)

        per_query_results.append({
            "id": qid,
            "question": q_text,
            "domain": domain,
            "topic": topic,
            "classification": classification,
            "is_covered": is_covered,
            "similarity": round(effective_sim, 4),
            "matched_evidence": matched_evidence,
        })

    # 5. Aggregate metrics
    total_queries = len(live_queries)
    in_domain_queries = [r for r in per_query_results if r["domain"] == "in_domain"]
    out_of_domain_queries = [r for r in per_query_results if r["domain"] == "out_of_domain"]

    covered_count = sum(1 for r in per_query_results if r["is_covered"])
    uncovered_count = total_queries - covered_count
    in_domain_covered = sum(1 for r in in_domain_queries if r["is_covered"])
    in_domain_uncovered = len(in_domain_queries) - in_domain_covered

    in_domain_coverage_pct = round((in_domain_covered / max(1, len(in_domain_queries))) * 100, 2)
    overall_coverage_pct = round((covered_count / max(1, total_queries)) * 100, 2)

    # 6. Similarity Distribution
    buckets = {
        "< 0.50": sum(1 for s in similarity_scores if s < 0.50),
        "0.50 - 0.59": sum(1 for s in similarity_scores if 0.50 <= s < 0.60),
        "0.60 - 0.69": sum(1 for s in similarity_scores if 0.60 <= s < 0.70),
        "0.70 - 0.79": sum(1 for s in similarity_scores if 0.70 <= s < 0.80),
        "0.80 - 0.89": sum(1 for s in similarity_scores if 0.80 <= s < 0.90),
        ">= 0.90": sum(1 for s in similarity_scores if s >= 0.90),
    }

    mean_sim = round(sum(similarity_scores) / max(1, len(similarity_scores)), 4)
    sorted_sims = sorted(similarity_scores)
    median_sim = round(sorted_sims[len(sorted_sims) // 2], 4)

    # Format Uncovered Topic Breakdown
    uncovered_breakdown: list[dict[str, Any]] = []
    for top, count in sorted(uncovered_topic_counts.items(), key=lambda x: x[1], reverse=True):
        uncovered_breakdown.append({
            "topic": top,
            "uncovered_queries": count,
            "sample_questions": uncovered_topic_examples.get(top, []),
        })

    # Print Summary Table
    print(f"{'Metric':<36} | {'Value':<18} | {'Details'}")
    print("-" * 88)
    print(f"{'Total Simulated Queries':<36} | {total_queries:<18} | Full evaluation dataset")
    print(f"{'In-Domain Queries':<36} | {len(in_domain_queries):<18} | Realistic university domain")
    print(f"{'Out-of-Domain Queries':<36} | {len(out_of_domain_queries):<18} | External queries (must reject/abstain)")
    print(f"{'Covered Queries':<36} | {covered_count:<18} | Queries meeting tau >= {threshold:.2f}")
    print(f"{'Uncovered Queries (Total)':<36} | {uncovered_count:<18} | In-domain gaps + Out-of-domain")
    print(f"{'In-Domain Covered':<36} | {in_domain_covered:<18} | Backed by current knowledge base")
    print(f"{'In-Domain Uncovered (Gaps)':<36} | {in_domain_uncovered:<18} | Knowledge gaps requiring corpus expansion")
    print(f"{'In-Domain Coverage Rate':<36} | {in_domain_coverage_pct:<17}% | In-domain coverage performance")
    print(f"{'Overall Coverage Rate':<36} | {overall_coverage_pct:<17}% | Entire query set coverage")
    print(f"{'Mean Similarity':<36} | {mean_sim:<18} | Across all 44 queries")
    print(f"{'Median Similarity':<36} | {median_sim:<18} | Distribution midpoint")
    print("-" * 88)

    print("\nSimilarity Distribution:")
    for b_label, b_count in buckets.items():
        pct = (b_count / max(1, total_queries)) * 100
        bar = "#" * int(pct // 4)
        print(f"  {b_label:<14} : {b_count:>3} ({pct:>5.1f}%) | {bar}")

    print("\nUncovered Topics Breakdown:")
    for item in uncovered_breakdown:
        print(f"  - Topic: {item['topic']} ({item['uncovered_queries']} queries)")
        for s in item["sample_questions"]:
            print(f"      * \"{s}\"")

    # Structure Report JSON
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "live_queries_path": str(live_queries_path),
            "golden_path": str(golden_path),
            "corpus_dir": (
                [str(d) for d in corpus_dir]
                if isinstance(corpus_dir, (list, tuple))
                else (str(corpus_dir) if corpus_dir else None)
            ),
            "embed_model": base_settings.embed_model,
            "similarity_threshold": threshold,
            "threshold_justification": (
                "Empirically justified by embedding evaluation separation gap of 0.3919 "
                "(Paraphrase mean=0.8539, min=0.7299; Unrelated mean=0.4619, max=0.5733)."
            ),
            "min_required_coverage_pct": min_coverage_pct,
        },
        "summary": {
            "total_queries": total_queries,
            "in_domain_queries": len(in_domain_queries),
            "out_of_domain_queries": len(out_of_domain_queries),
            "covered_queries": covered_count,
            "uncovered_queries": uncovered_count,
            "in_domain_covered": in_domain_covered,
            "in_domain_uncovered": in_domain_uncovered,
            "in_domain_coverage_pct": in_domain_coverage_pct,
            "overall_coverage_pct": overall_coverage_pct,
            "mean_similarity": mean_sim,
            "median_similarity": median_sim,
        },
        "similarity_distribution": buckets,
        "uncovered_topic_breakdown": uncovered_breakdown,
        "queries": per_query_results,
    }

    report_path = r_dir / "coverage_report.json"
    archive_path = r_dir / f"coverage_report_{timestamp_str}.json"

    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    archive_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nMachine-readable coverage report saved to: {report_path}")
    print(f"Timestamped archive saved to: {archive_path}")

    # Check minimum required coverage
    passed_gate = in_domain_coverage_pct >= min_coverage_pct
    status_str = "PASS" if passed_gate else "FAIL"
    print(f"\nCoverage Gate Status: [{status_str}] (Observed: {in_domain_coverage_pct}% >= Required: {min_coverage_pct}%)")
    print("=" * 88)

    report_data["passed_gate"] = passed_gate
    return report_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate simulated live query coverage against knowledge base.")
    parser.add_argument(
        "--live-queries",
        "-q",
        type=Path,
        default=REPO_ROOT / "eval" / "live_queries.jsonl",
        help="Path to simulated live queries JSONL",
    )
    parser.add_argument(
        "--golden",
        "-g",
        type=Path,
        default=REPO_ROOT / "eval" / "golden.jsonl",
        help="Path to golden dataset JSONL",
    )
    parser.add_argument(
        "--corpus-dir",
        "-c",
        type=Path,
        default=REPO_ROOT / "corpus" / "raw",
        help="Path to corpus directory (raw or updated batch)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=JUSTIFIED_SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold (default: {JUSTIFIED_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="Minimum required in-domain coverage percentage (exit non-zero if unmet)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Embedding model override",
    )
    parser.add_argument(
        "--results-dir",
        "-o",
        type=Path,
        default=None,
        help="Results output directory",
    )
    args = parser.parse_args()

    report = run_coverage_analysis(
        live_queries_path=args.live_queries,
        golden_path=args.golden,
        corpus_dir=args.corpus_dir,
        threshold=args.threshold,
        min_coverage_pct=args.min_coverage,
        model_override=args.model,
        results_dir=args.results_dir,
    )
    return 0 if report.get("passed_gate", True) else 1


if __name__ == "__main__":
    sys.exit(main())

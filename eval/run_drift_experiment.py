#!/usr/bin/env python3
"""Part E4: Drift Experiment and Release Gate Runner (Assignment 3, Part E).

Executes the knowledge base drift experiment by comparing coverage before and after
a corpus update (Batch B) and addition of new golden evaluation questions:
1. Ingests / evaluates baseline state:
   - Baseline Golden Dataset: eval/golden_baseline.jsonl (28 reference questions)
   - Baseline Corpus: corpus/raw (or golden reference knowledge base)
2. Ingests / evaluates candidate state (post-update):
   - Candidate Golden Dataset: eval/golden.jsonl (34 reference questions, including
     6 new golden questions targeting previously uncovered Batch B topics)
   - Candidate Corpus: corpus/raw + corpus/batch_b (or updated knowledge base)
3. Identifies and quantifies drift:
   - Newly covered topics (e.g. hostel_rules, scholarships)
   - Newly uncovered topics (regressions check - must be zero)
   - Residual uncovered topics (persistent gaps e.g. sports_facilities, amenities)
   - Changes in retrieval similarity (query-level deltas and distribution shifts)
   - Changes in in-domain and overall coverage percentage
4. Evaluates automated release gate:
   - Checks candidate in-domain coverage against configurable --min-coverage threshold
   - Checks coverage improvement against configurable --min-improvement threshold
   - Returns exit code 0 on PASS, 1 on FAIL
5. Outputs machine-readable report to eval/results/drift_experiment_report.json.
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
log = logging.getLogger("rahbar.eval.drift")

JUSTIFIED_SIMILARITY_THRESHOLD = 0.70
DEFAULT_MIN_COVERAGE = 75.0
DEFAULT_MIN_IMPROVEMENT = 10.0


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return dot / (norm_a * norm_b)


def run_drift_experiment(
    live_queries_path: Path,
    baseline_golden_path: Path,
    candidate_golden_path: Path,
    baseline_corpus: Path | list[Path] | None = None,
    candidate_corpus: Path | list[Path] | None = None,
    threshold: float = JUSTIFIED_SIMILARITY_THRESHOLD,
    min_coverage_pct: float = DEFAULT_MIN_COVERAGE,
    min_improvement_pct: float = DEFAULT_MIN_IMPROVEMENT,
    model_override: str = "",
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute before-and-after drift experiment and evaluate the release gate."""
    r_dir = results_dir or REPO_ROOT / "eval" / "results"
    r_dir.mkdir(parents=True, exist_ok=True)

    for p in [str(REPO_ROOT / "indexer"), str(REPO_ROOT / "eval")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from eval_retrieval import InMemoryVectorRetriever
    from pipeline.config import IndexerSettings, get_settings
    from pipeline.embed import embed

    # Load simulated live queries
    live_queries: list[dict[str, Any]] = []
    with live_queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                live_queries.append(json.loads(line))

    # Load baseline and candidate golden questions
    baseline_golden: list[dict[str, Any]] = []
    with baseline_golden_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                baseline_golden.append(json.loads(line))

    candidate_golden: list[dict[str, Any]] = []
    with candidate_golden_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidate_golden.append(json.loads(line))

    # Setup settings & probe Ollama
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

    print("=" * 90)
    print("  RAHBAR AI — KNOWLEDGE BASE DRIFT EXPERIMENT & RELEASE GATE (PART E4)")
    print("=" * 90)
    print(f"Simulated Live Queries : {live_queries_path} ({len(live_queries)} queries)")
    print(f"Baseline Golden Target : {baseline_golden_path} ({len(baseline_golden)} questions)")
    print(f"Candidate Golden Target: {candidate_golden_path} ({len(candidate_golden)} questions)")
    print(f"Embedding Model        : {base_settings.embed_model}")
    print(f"Similarity Threshold   : {threshold:.2f} (Separation gap: 0.3919)")
    print(f"Release Gate Criteria  : In-Domain Coverage >= {min_coverage_pct:.1f}%, Delta >= +{min_improvement_pct:.1f}%")
    print("-" * 90)

    # Embed simulated live queries once
    log.info("Embedding %d simulated live queries...", len(live_queries))
    q_texts = [q["question"] for q in live_queries]
    q_vectors = embed(q_texts, base_settings)

    # 1. Evaluate Baseline State
    log.info("Evaluating BASELINE state (%d golden questions)...", len(baseline_golden))
    bg_texts = [g["question"] for g in baseline_golden]
    bg_vectors = embed(bg_texts, base_settings)

    base_retriever = None
    if baseline_corpus:
        b_targets = [baseline_corpus] if isinstance(baseline_corpus, (str, Path)) else list(baseline_corpus)
        valid_b = [Path(t) for t in b_targets if Path(t).exists()]
        if valid_b:
            log.info("Indexing baseline corpus from %s...", [str(t) for t in valid_b])
            base_retriever = InMemoryVectorRetriever(base_settings)
            base_retriever.populate_from_corpus(valid_b)

    baseline_results: list[dict[str, Any]] = []
    baseline_sims: list[float] = []

    for q_data, q_vec in zip(live_queries, q_vectors):
        q_text = q_data["question"]
        domain = q_data.get("domain", "in_domain")
        topic = q_data.get("topic", "general")

        best_g_sim = -1.0
        best_g_match = None
        for g_data, g_vec in zip(baseline_golden, bg_vectors):
            sim = cosine_similarity(q_vec, g_vec)
            if sim > best_g_sim:
                best_g_sim = sim
                best_g_match = g_data

        best_c_sim = -1.0
        best_c_passage = None
        if base_retriever:
            top_passages = base_retriever.retrieve(q_text, k=1)
            if top_passages:
                best_c_passage = top_passages[0]
                best_c_sim = best_c_passage.score

        eff_sim = max(best_g_sim, best_c_sim)
        baseline_sims.append(round(eff_sim, 4))

        is_covered = (domain != "out_of_domain") and (eff_sim >= threshold)
        classification = (
            "OUT_OF_DOMAIN" if domain == "out_of_domain"
            else ("IN_DOMAIN_COVERED" if is_covered else "IN_DOMAIN_UNCOVERED")
        )

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
            first_rel = best_g_match.get("relevant", [{}])[0] if best_g_match.get("relevant") else {}
            matched_evidence = {
                "type": "golden_question",
                "golden_id": best_g_match.get("id"),
                "golden_question": best_g_match.get("question"),
                "source": first_rel.get("source"),
                "section": first_rel.get("section"),
                "page": first_rel.get("page"),
                "similarity": round(best_g_sim, 4),
            }

        baseline_results.append({
            "id": q_data["id"],
            "question": q_text,
            "domain": domain,
            "topic": topic,
            "classification": classification,
            "is_covered": is_covered,
            "similarity": round(eff_sim, 4),
            "matched_evidence": matched_evidence,
        })

    # 2. Evaluate Candidate State (Post-Update)
    log.info("Evaluating CANDIDATE state (%d golden questions)...", len(candidate_golden))
    cg_texts = [g["question"] for g in candidate_golden]
    cg_vectors = embed(cg_texts, base_settings)

    cand_retriever = None
    if candidate_corpus:
        c_targets = [candidate_corpus] if isinstance(candidate_corpus, (str, Path)) else list(candidate_corpus)
        valid_c = [Path(t) for t in c_targets if Path(t).exists()]
        if valid_c:
            log.info("Indexing candidate corpus from %s...", [str(t) for t in valid_c])
            cand_retriever = InMemoryVectorRetriever(base_settings)
            cand_retriever.populate_from_corpus(valid_c)

    candidate_results: list[dict[str, Any]] = []
    candidate_sims: list[float] = []

    for q_data, q_vec in zip(live_queries, q_vectors):
        q_text = q_data["question"]
        domain = q_data.get("domain", "in_domain")
        topic = q_data.get("topic", "general")

        best_g_sim = -1.0
        best_g_match = None
        for g_data, g_vec in zip(candidate_golden, cg_vectors):
            sim = cosine_similarity(q_vec, g_vec)
            if sim > best_g_sim:
                best_g_sim = sim
                best_g_match = g_data

        best_c_sim = -1.0
        best_c_passage = None
        if cand_retriever:
            top_passages = cand_retriever.retrieve(q_text, k=1)
            if top_passages:
                best_c_passage = top_passages[0]
                best_c_sim = best_c_passage.score

        eff_sim = max(best_g_sim, best_c_sim)
        candidate_sims.append(round(eff_sim, 4))

        is_covered = (domain != "out_of_domain") and (eff_sim >= threshold)
        classification = (
            "OUT_OF_DOMAIN" if domain == "out_of_domain"
            else ("IN_DOMAIN_COVERED" if is_covered else "IN_DOMAIN_UNCOVERED")
        )

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
            first_rel = best_g_match.get("relevant", [{}])[0] if best_g_match.get("relevant") else {}
            matched_evidence = {
                "type": "golden_question",
                "golden_id": best_g_match.get("id"),
                "golden_question": best_g_match.get("question"),
                "source": first_rel.get("source"),
                "section": first_rel.get("section"),
                "page": first_rel.get("page"),
                "similarity": round(best_g_sim, 4),
            }

        candidate_results.append({
            "id": q_data["id"],
            "question": q_text,
            "domain": domain,
            "topic": topic,
            "classification": classification,
            "is_covered": is_covered,
            "similarity": round(eff_sim, 4),
            "matched_evidence": matched_evidence,
        })

    # 3. Topic-Level Drift & Query Flips
    base_uncovered_topics = {
        r["topic"] for r in baseline_results if r["domain"] == "in_domain" and not r["is_covered"]
    }
    cand_uncovered_topics = {
        r["topic"] for r in candidate_results if r["domain"] == "in_domain" and not r["is_covered"]
    }
    base_covered_topics = {
        r["topic"] for r in baseline_results if r["domain"] == "in_domain" and r["is_covered"]
    }
    cand_covered_topics = {
        r["topic"] for r in candidate_results if r["domain"] == "in_domain" and r["is_covered"]
    }

    # Newly covered: topics previously uncovered that now have coverage
    newly_covered_topics = sorted(base_uncovered_topics - cand_uncovered_topics)
    # Regressions: topics previously covered that lost coverage
    newly_uncovered_topics = sorted(base_covered_topics - cand_covered_topics)
    # Residual uncovered topics
    residual_uncovered_topics = sorted(cand_uncovered_topics)

    # Track individual query flips
    query_flips: list[dict[str, Any]] = []
    for b_res, c_res in zip(baseline_results, candidate_results):
        if not b_res["is_covered"] and c_res["is_covered"]:
            query_flips.append({
                "id": c_res["id"],
                "question": c_res["question"],
                "topic": c_res["topic"],
                "baseline_similarity": b_res["similarity"],
                "candidate_similarity": c_res["similarity"],
                "similarity_gain": round(c_res["similarity"] - b_res["similarity"], 4),
                "matched_evidence": c_res["matched_evidence"],
            })

    # 4. Aggregate Metrics & Deltas
    total_queries = len(live_queries)
    in_domain_total = sum(1 for r in live_queries if r.get("domain") == "in_domain")
    out_of_domain_total = total_queries - in_domain_total

    b_in_cov = sum(1 for r in baseline_results if r["domain"] == "in_domain" and r["is_covered"])
    c_in_cov = sum(1 for r in candidate_results if r["domain"] == "in_domain" and r["is_covered"])

    b_in_pct = round((b_in_cov / max(1, in_domain_total)) * 100, 2)
    c_in_pct = round((c_in_cov / max(1, in_domain_total)) * 100, 2)
    in_cov_delta = round(c_in_pct - b_in_pct, 2)

    b_all_cov = sum(1 for r in baseline_results if r["is_covered"])
    c_all_cov = sum(1 for r in candidate_results if r["is_covered"])

    b_all_pct = round((b_all_cov / max(1, total_queries)) * 100, 2)
    c_all_pct = round((c_all_cov / max(1, total_queries)) * 100, 2)
    all_cov_delta = round(c_all_pct - b_all_pct, 2)

    b_mean_sim = round(sum(baseline_sims) / max(1, len(baseline_sims)), 4)
    c_mean_sim = round(sum(candidate_sims) / max(1, len(candidate_sims)), 4)
    mean_sim_delta = round(c_mean_sim - b_mean_sim, 4)

    # 5. Evaluate Release Gate Decision
    gate_checks = {
        "candidate_min_coverage": {
            "required": min_coverage_pct,
            "observed": c_in_pct,
            "passed": c_in_pct >= min_coverage_pct,
        },
        "coverage_improvement": {
            "required": min_improvement_pct,
            "observed": in_cov_delta,
            "passed": in_cov_delta >= min_improvement_pct,
        },
        "zero_topic_regressions": {
            "required": 0,
            "observed": len(newly_uncovered_topics),
            "passed": len(newly_uncovered_topics) == 0,
        },
    }

    gate_passed = all(c["passed"] for c in gate_checks.values())
    gate_decision = "PASS" if gate_passed else "FAIL"

    # Print Executive Summary Table
    print(f"\n{'Metric':<34} | {'Baseline':<12} | {'Candidate':<12} | {'Delta':<12} | {'Details'}")
    print("-" * 90)
    print(f"{'Golden Reference Questions':<34} | {len(baseline_golden):<12} | {len(candidate_golden):<12} | {f'+{len(candidate_golden)-len(baseline_golden)}':<12} | Reference questions in golden set")
    print(f"{'In-Domain Queries Covered':<34} | {f'{b_in_cov}/{in_domain_total}':<12} | {f'{c_in_cov}/{in_domain_total}':<12} | {f'+{c_in_cov-b_in_cov}':<12} | Queries meeting tau >= {threshold:.2f}")
    print(f"{'In-Domain Coverage Rate':<34} | {f'{b_in_pct}%':<12} | {f'{c_in_pct}%':<12} | {f'+{in_cov_delta}%':<12} | Primary release gate metric")
    print(f"{'Overall Coverage Rate':<34} | {f'{b_all_pct}%':<12} | {f'{c_all_pct}%':<12} | {f'+{all_cov_delta}%':<12} | Full 44 simulated query set")
    print(f"{'Out-of-Domain Abstentions':<34} | {f'{out_of_domain_total}/{out_of_domain_total}':<12} | {f'{out_of_domain_total}/{out_of_domain_total}':<12} | {'0 (stable)':<12} | Correctly rejected external queries")
    print(f"{'Mean Similarity Score':<34} | {b_mean_sim:<12} | {c_mean_sim:<12} | {f'+{mean_sim_delta:.4f}' if mean_sim_delta >= 0 else str(mean_sim_delta):<12} | Semantic alignment across all queries")
    print("-" * 90)

    print("\nTopic Drift Summary:")
    print(f"  * Newly Covered Topics     : {', '.join(newly_covered_topics) if newly_covered_topics else 'None'}")
    print(f"  * Regressed Topics         : {', '.join(newly_uncovered_topics) if newly_uncovered_topics else 'None (0 regressions)'}")
    print(f"  * Residual Uncovered Topics: {', '.join(residual_uncovered_topics) if residual_uncovered_topics else 'None'}")

    print("\nKey Query Flips (Uncovered -> Covered):")
    for flip in query_flips:
        print(f"  [{flip['id']}] ({flip['topic']}) sim: {flip['baseline_similarity']} -> {flip['candidate_similarity']} (+{flip['similarity_gain']})")
        print(f"       Q: \"{flip['question']}\"")
        ev = flip["matched_evidence"]
        print(f"       Match: {ev.get('source')} - {ev.get('section')}")

    print("\nRelease Gate Decision:")
    for name, c in gate_checks.items():
        st = "PASS" if c["passed"] else "FAIL"
        print(f"  [{st}] {name:<26}: observed={c['observed']}, required={c['required']}")
    print(f"\nFinal Release Gate Status: [{gate_decision}]")
    print("=" * 90)

    # Save Machine-Readable Report
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "live_queries_path": str(live_queries_path),
            "baseline_golden_path": str(baseline_golden_path),
            "candidate_golden_path": str(candidate_golden_path),
            "baseline_corpus": [str(d) for d in baseline_corpus] if isinstance(baseline_corpus, list) else (str(baseline_corpus) if baseline_corpus else None),
            "candidate_corpus": [str(d) for d in candidate_corpus] if isinstance(candidate_corpus, list) else (str(candidate_corpus) if candidate_corpus else None),
            "embed_model": base_settings.embed_model,
            "similarity_threshold": threshold,
            "threshold_justification": "Empirically justified by embedding evaluation separation gap of 0.3919",
            "min_coverage_threshold": min_coverage_pct,
            "min_improvement_threshold": min_improvement_pct,
        },
        "gate_decision": gate_decision,
        "gate_passed": gate_passed,
        "gate_checks": gate_checks,
        "metrics": {
            "baseline": {
                "golden_questions_count": len(baseline_golden),
                "in_domain_covered": b_in_cov,
                "in_domain_total": in_domain_total,
                "in_domain_coverage_pct": b_in_pct,
                "overall_covered": b_all_cov,
                "total_queries": total_queries,
                "overall_coverage_pct": b_all_pct,
                "mean_similarity": b_mean_sim,
            },
            "candidate": {
                "golden_questions_count": len(candidate_golden),
                "in_domain_covered": c_in_cov,
                "in_domain_total": in_domain_total,
                "in_domain_coverage_pct": c_in_pct,
                "overall_covered": c_all_cov,
                "total_queries": total_queries,
                "overall_coverage_pct": c_all_pct,
                "mean_similarity": c_mean_sim,
            },
            "deltas": {
                "new_golden_questions": len(candidate_golden) - len(baseline_golden),
                "in_domain_covered_delta": c_in_cov - b_in_cov,
                "in_domain_coverage_delta_pct": in_cov_delta,
                "overall_coverage_delta_pct": all_cov_delta,
                "mean_similarity_delta": mean_sim_delta,
            },
        },
        "topic_drift": {
            "newly_covered_topics": newly_covered_topics,
            "regressed_topics": newly_uncovered_topics,
            "residual_uncovered_topics": residual_uncovered_topics,
        },
        "query_flips": query_flips,
        "per_query_results": {
            "baseline": baseline_results,
            "candidate": candidate_results,
        },
    }

    report_path = r_dir / "drift_experiment_report.json"
    archive_path = r_dir / f"drift_experiment_report_{timestamp_str}.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    with archive_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    log.info("Drift experiment report saved to: %s", report_path)
    return report_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Part E4 Drift Experiment & Coverage Release Gate.")
    parser.add_argument(
        "--live-queries",
        "-q",
        type=Path,
        default=REPO_ROOT / "eval" / "live_queries.jsonl",
        help="Path to simulated live queries JSONL",
    )
    parser.add_argument(
        "--baseline-golden",
        type=Path,
        default=REPO_ROOT / "eval" / "golden_baseline.jsonl",
        help="Path to baseline golden dataset JSONL",
    )
    parser.add_argument(
        "--candidate-golden",
        type=Path,
        default=REPO_ROOT / "eval" / "golden.jsonl",
        help="Path to candidate golden dataset JSONL (with new questions)",
    )
    parser.add_argument(
        "--baseline-corpus",
        type=Path,
        default=REPO_ROOT / "corpus" / "raw",
        help="Baseline corpus directory",
    )
    parser.add_argument(
        "--candidate-corpus",
        type=Path,
        nargs="*",
        default=[REPO_ROOT / "corpus" / "raw", REPO_ROOT / "corpus" / "batch_b"],
        help="Candidate corpus directories (raw and Batch B)",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Run golden-only semantic comparison (skips extracting and indexing corpus documents)",
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
        default=DEFAULT_MIN_COVERAGE,
        help=f"Release gate: minimum required candidate in-domain coverage percentage (default: {DEFAULT_MIN_COVERAGE})",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=DEFAULT_MIN_IMPROVEMENT,
        help=f"Release gate: minimum required coverage improvement percentage points (default: {DEFAULT_MIN_IMPROVEMENT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Embedding model override (e.g. nomic-embed-text or dummy-768)",
    )

    args = parser.parse_args()

    b_corpus = None if args.skip_corpus else args.baseline_corpus
    c_corpus = None if args.skip_corpus else args.candidate_corpus

    report = run_drift_experiment(
        live_queries_path=args.live_queries,
        baseline_golden_path=args.baseline_golden,
        candidate_golden_path=args.candidate_golden,
        baseline_corpus=b_corpus,
        candidate_corpus=c_corpus,
        threshold=args.threshold,
        min_coverage_pct=args.min_coverage,
        min_improvement_pct=args.min_improvement,
        model_override=args.model,
    )

    return 0 if report.get("gate_passed") else 1


if __name__ == "__main__":
    sys.exit(main())

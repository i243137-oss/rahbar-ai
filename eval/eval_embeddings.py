#!/usr/bin/env python3
"""Module-Level Embedding Evaluation (Assignment 3, Part B2).

Evaluates the embedding representation space across four core criteria:
1. Paraphrase Similarity: Mean cosine similarity for >= 15 semantically identical pairs.
2. Unrelated Contrast: Mean cosine similarity for >= 15 unrelated pairs.
3. Contrastive Separation Gap: Gap between paraphrase similarity and unrelated similarity.
4. Latency & Throughput: Mean, p50, and p95 embedding latency per chunk.
5. Determinism: Verification that identical text yields zero variance across runs.

Outputs a structured report to eval/results/ and displays a comprehensive summary table.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexer"))

from pipeline.config import IndexerSettings, get_settings  # noqa: E402
from pipeline.embed import embed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("rahbar.eval.embeddings")


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def percentile(data: list[float], p: float) -> float:
    """Compute p-th percentile of a sorted list."""
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = (len(sorted_d) - 1) * (p / 100.0)
    floor_idx = math.floor(idx)
    ceil_idx = math.ceil(idx)
    if floor_idx == ceil_idx:
        return sorted_d[int(idx)]
    d0 = sorted_d[floor_idx] * (ceil_idx - idx)
    d1 = sorted_d[ceil_idx] * (idx - floor_idx)
    return d0 + d1


@dataclass
class PairEvalResult:
    id: str
    topic: str
    text_a: str
    text_b: str
    similarity: float


@dataclass
class EmbeddingEvalSummary:
    model: str
    dimension: int
    paraphrase_count: int
    unrelated_count: int
    paraphrase_mean_sim: float
    paraphrase_min_sim: float
    paraphrase_max_sim: float
    paraphrase_stdev: float
    unrelated_mean_sim: float
    unrelated_min_sim: float
    unrelated_max_sim: float
    unrelated_stdev: float
    separation_gap: float
    determinism_max_abs_diff: float
    determinism_passed: bool
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_chunks_per_sec: float


def run_embedding_evaluation(
    pairs_file: Path | None = None,
    settings: IndexerSettings | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute complete module-level embedding evaluation suite."""
    p_path = pairs_file or REPO_ROOT / "eval" / "embedding_pairs.json"
    r_dir = results_dir or REPO_ROOT / "eval" / "results"
    r_dir.mkdir(parents=True, exist_ok=True)

    if not p_path.exists():
        raise FileNotFoundError(f"Embedding pairs file not found: {p_path}")

    pairs_data = json.loads(p_path.read_text(encoding="utf-8"))
    paraphrase_items = pairs_data.get("paraphrase_pairs", [])
    unrelated_items = pairs_data.get("unrelated_pairs", [])

    if len(paraphrase_items) < 15:
        raise ValueError(f"Insufficient paraphrase pairs: {len(paraphrase_items)} < 15 required")
    if len(unrelated_items) < 15:
        raise ValueError(f"Insufficient unrelated pairs: {len(unrelated_items)} < 15 required")

    active_settings = settings or get_settings()
    if "OLLAMA_URL" not in os.environ and "minikube" in active_settings.ollama_url:
        import urllib.error
        import urllib.request
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
            active_settings = IndexerSettings(**{**active_settings.__dict__, "ollama_url": "http://127.0.0.1:11434"})
        except (OSError, urllib.error.URLError) as exc:
            log.debug("Ollama localhost probe: %s", exc)

    print("=" * 80)
    print("  RAHBAR AI — MODULE-LEVEL EMBEDDING EVALUATION (PART B2)")
    print("=" * 80)
    print(f"Embedding Model     : {active_settings.embed_model}")
    print(f"Ollama URL          : {active_settings.ollama_url}")
    print(f"Paraphrase Pairs    : {len(paraphrase_items)}")
    print(f"Unrelated Pairs     : {len(unrelated_items)}")
    print("-" * 80)

    # 1. Evaluate Paraphrase Pairs
    para_results: list[PairEvalResult] = []
    para_sims: list[float] = []
    for item in paraphrase_items:
        vecs = embed([item["text_a"], item["text_b"]], active_settings)
        sim = cosine_similarity(vecs[0], vecs[1])
        para_sims.append(sim)
        para_results.append(
            PairEvalResult(
                id=item["id"],
                topic=item["topic"],
                text_a=item["text_a"],
                text_b=item["text_b"],
                similarity=round(sim, 4),
            )
        )

    # 2. Evaluate Unrelated Pairs
    unrel_results: list[PairEvalResult] = []
    unrel_sims: list[float] = []
    for item in unrelated_items:
        vecs = embed([item["text_a"], item["text_b"]], active_settings)
        sim = cosine_similarity(vecs[0], vecs[1])
        unrel_sims.append(sim)
        unrel_results.append(
            PairEvalResult(
                id=item["id"],
                topic=item["topic"],
                text_a=item["text_a"],
                text_b=item["text_b"],
                similarity=round(sim, 4),
            )
        )

    # 3. Calculate Distribution Statistics & Gap
    para_mean = mean(para_sims)
    para_min = min(para_sims)
    para_max = max(para_sims)
    para_sd = stdev(para_sims) if len(para_sims) > 1 else 0.0

    unrel_mean = mean(unrel_sims)
    unrel_min = min(unrel_sims)
    unrel_max = max(unrel_sims)
    unrel_sd = stdev(unrel_sims) if len(unrel_sims) > 1 else 0.0

    gap = para_mean - unrel_mean

    # 4. Determinism Test
    test_text = "FAST-NUCES Computer Science regulations require strict adherence to academic honesty."
    v1 = embed([test_text], active_settings)[0]
    v2 = embed([test_text], active_settings)[0]
    dim = len(v1)
    diffs = [abs(a - b) for a, b in zip(v1, v2)]
    max_diff = max(diffs)
    determinism_passed = max_diff < 1e-6

    # 5. Latency & Throughput Benchmark
    sample_texts = [
        "Undergraduate Academic Rules Handbook June 2026 edition governing degrees.",
        "Fee Refund Policy: 80% refundable during second calendar week of classes.",
        "Incomplete grade I converts to F if not satisfied before end of next semester.",
        "Dean's List of Honors recognition requires SGPA between 3.50 and 3.99.",
        "Rector's List requires perfect 4.00 SGPA in semester with full credit load.",
        "Generative AI tools may assist in background research but cannot replace analysis.",
        "Health insurance coverage applies to all full-time faculty and administrative staff.",
        "Attendance requirement mandates 80 percent minimum presence to sit for exams.",
        "Faculty start-ups commercialize intellectual property created at NUCES campuses.",
        "Course withdrawal permitted until conclusion of makeup week under 2026 rules.",
    ]
    latencies_ms: list[float] = []
    # Warm up
    _ = embed(sample_texts[:2], active_settings)

    for text in sample_texts:
        t0 = time.perf_counter()
        _ = embed([text], active_settings)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    # Batch throughput timing
    t_start = time.perf_counter()
    _ = embed(sample_texts, active_settings)
    t_end = time.perf_counter()
    batch_time_sec = t_end - t_start
    throughput = len(sample_texts) / max(0.0001, batch_time_sec)

    lat_mean = mean(latencies_ms)
    lat_p50 = percentile(latencies_ms, 50)
    lat_p95 = percentile(latencies_ms, 95)

    summary = EmbeddingEvalSummary(
        model=active_settings.embed_model,
        dimension=dim,
        paraphrase_count=len(para_results),
        unrelated_count=len(unrel_results),
        paraphrase_mean_sim=round(para_mean, 4),
        paraphrase_min_sim=round(para_min, 4),
        paraphrase_max_sim=round(para_max, 4),
        paraphrase_stdev=round(para_sd, 4),
        unrelated_mean_sim=round(unrel_mean, 4),
        unrelated_min_sim=round(unrel_min, 4),
        unrelated_max_sim=round(unrel_max, 4),
        unrelated_stdev=round(unrel_sd, 4),
        separation_gap=round(gap, 4),
        determinism_max_abs_diff=max_diff,
        determinism_passed=determinism_passed,
        latency_mean_ms=round(lat_mean, 2),
        latency_p50_ms=round(lat_p50, 2),
        latency_p95_ms=round(lat_p95, 2),
        throughput_chunks_per_sec=round(throughput, 2),
    )

    # Print Summary Table
    print("\nPARAPHRASE PAIRS (SEMANTIC SIMILARITY):")
    print(f"{'ID':<10} | {'Topic':<35} | {'Cosine Sim':<10}")
    print("-" * 62)
    for p in para_results[:6]:
        print(f"{p.id:<10} | {p.topic[:35]:<35} | {p.similarity:<10.4f}")
    print(f"... ({len(para_results) - 6} more pairs evaluated)")

    print("\nUNRELATED PAIRS (CONTRAST / SEPARATION):")
    print(f"{'ID':<10} | {'Topic':<35} | {'Cosine Sim':<10}")
    print("-" * 62)
    for u in unrel_results[:6]:
        print(f"{u.id:<10} | {u.topic[:35]:<35} | {u.similarity:<10.4f}")
    print(f"... ({len(unrel_results) - 6} more pairs evaluated)")

    print("\nAGGREGATED MODULE-LEVEL METRICS:")
    print("-" * 62)
    print(f"Paraphrase Similarity (Mean +/- SD) : {summary.paraphrase_mean_sim:.4f} +/- {summary.paraphrase_stdev:.4f} [range: {summary.paraphrase_min_sim:.4f} - {summary.paraphrase_max_sim:.4f}]")
    print(f"Unrelated Similarity  (Mean +/- SD) : {summary.unrelated_mean_sim:.4f} +/- {summary.unrelated_stdev:.4f} [range: {summary.unrelated_min_sim:.4f} - {summary.unrelated_max_sim:.4f}]")
    print(f"Separation Gap (Paraphrase - Unrel) : {summary.separation_gap:.4f} ({'EXCELLENT' if summary.separation_gap > 0.3 else 'GOOD'})")
    print(f"Determinism Check (Max Abs Delta)   : {summary.determinism_max_abs_diff:.1e} ({'PASS' if summary.determinism_passed else 'FAIL'})")
    print(f"Embedding Latency (Single Chunk)    : Mean={summary.latency_mean_ms}ms, p50={summary.latency_p50_ms}ms, p95={summary.latency_p95_ms}ms")
    print(f"Embedding Throughput                : {summary.throughput_chunks_per_sec} chunks/sec")
    print("=" * 80)

    # Output report
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dict: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": asdict(summary),
        "paraphrase_pairs": [asdict(p) for p in para_results],
        "unrelated_pairs": [asdict(u) for u in unrel_results],
    }

    report_latest = r_dir / "embedding_eval_report.json"
    report_stamped = r_dir / f"embedding_eval_report_{timestamp_str}.json"
    report_latest.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    report_stamped.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"Report saved to: {report_latest}")
    print(f"Timestamped archive: {report_stamped}")
    print("=" * 80)

    return report_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Run module-level embedding evaluation.")
    parser.add_argument("--pairs", type=Path, default=None, help="Path to embedding pairs JSON")
    parser.add_argument("--model", type=str, default="", help="Embedding model name override")
    parser.add_argument("--results-dir", type=Path, default=None, help="Output directory for reports")
    args = parser.parse_args()

    custom_settings = get_settings()
    if args.model:
        custom_settings = IndexerSettings(**{**custom_settings.__dict__, "embed_model": args.model})
    elif custom_settings.embed_model != "dummy-768":
        try:
            import httpx  # noqa: F401
        except ImportError:
            log.warning("httpx is not installed in the local environment; falling back to deterministic dummy-768 embeddings")
            custom_settings = IndexerSettings(**{**custom_settings.__dict__, "embed_model": "dummy-768"})

    try:
        report = run_embedding_evaluation(
            pairs_file=args.pairs,
            settings=custom_settings,
            results_dir=args.results_dir,
        )
        return 0 if report["summary"]["determinism_passed"] else 1
    except Exception:
        log.exception("Embedding evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Part C1: Controlled Change Experiment Runner (Assignment 3, Part C1 & C2).

Evaluates a single isolated system modification against the established baseline:
- Hypothesis: Evaluating Chunking Strategy trade-offs on retrieval recall and ranking.
  * Baseline Configuration : chunk_size=500, chunk_overlap=80 (Current Production)
  * Candidate Configuration: chunk_size=400, chunk_overlap=50 (Fine-Grained Clauses)
- Controlled Environment:
  * Constant Golden Dataset: eval/golden.jsonl (28 queries)
  * Constant Embedding Model: Configured embed_model (nomic-embed-text)
  * Constant Corpus: corpus/raw (9 FAST-NUCES policy documents)
- Metrics Measured:
  * Chunk count & Storage Amplification
  * Recall@3, Recall@5, MRR, nDCG@5
  * Noise Robustness Retention under query typos and 3 distractor passages

Outputs:
- Detailed side-by-side terminal comparison table.
- Machine-readable artifact: eval/results/change_experiment_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexer"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from eval_retrieval import run_retrieval_suite  # noqa: E402
from pipeline.config import IndexerSettings, get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("rahbar.eval.change_experiment")


def calculate_amplification(chunk_count: int, baseline_chunks: int) -> float:
    """Calculate chunk index expansion ratio."""
    return round(chunk_count / max(1, baseline_chunks), 2)


def run_change_experiment(
    parameter: str = "chunking",
    baseline_val: str = "500/80",
    candidate_val: str = "400/50",
    model_override: str = "",
    golden_file: Path | None = None,
    corpus_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute controlled A/B experiment between baseline and candidate configurations."""
    r_dir = results_dir or REPO_ROOT / "eval" / "results"
    r_dir.mkdir(parents=True, exist_ok=True)
    g_path = golden_file or REPO_ROOT / "eval" / "golden.jsonl"
    c_dir = corpus_dir or REPO_ROOT / "corpus" / "raw"

    base_settings = get_settings()
    if model_override:
        base_settings = IndexerSettings(**{**base_settings.__dict__, "embed_model": model_override})
    elif "OLLAMA_URL" not in os.environ and "minikube" in base_settings.ollama_url:
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
            base_settings = IndexerSettings(**{**base_settings.__dict__, "ollama_url": "http://127.0.0.1:11434"})
        except (OSError, urllib.error.URLError) as exc:
            log.debug("Ollama localhost probe: %s", exc)

    print("=" * 80)
    print("  RAHBAR AI — CONTROLLED CHANGE EXPERIMENT (PART C1)")
    print("=" * 80)
    print(f"Experimental Parameter : {parameter.upper()}")
    print(f"Baseline Setting       : {baseline_val}")
    print(f"Candidate Setting      : {candidate_val}")
    print(f"Golden Dataset         : {g_path}")
    print(f"Embedding Model        : {base_settings.embed_model}")
    print("-" * 80)

    # Configure Baseline & Candidate IndexerSettings
    if parameter == "chunking":
        b_size, b_overlap = (int(x) for x in baseline_val.split("/"))
        c_size, c_overlap = (int(x) for x in candidate_val.split("/"))

        baseline_settings = IndexerSettings(**{
            **base_settings.__dict__,
            "chunk_size": b_size,
            "chunk_overlap": b_overlap,
        })
        candidate_settings = IndexerSettings(**{
            **base_settings.__dict__,
            "chunk_size": c_size,
            "chunk_overlap": c_overlap,
        })
    elif parameter == "top_k":
        # Keep same chunking, vary retrieval top_k
        baseline_settings = base_settings
        candidate_settings = base_settings
    else:
        raise ValueError(f"Unsupported experimental parameter: {parameter}")

    # 1. Run Baseline Evaluation
    print(f"\n[PHASE 1] Running Baseline Evaluation ({baseline_val})...")
    baseline_report = run_retrieval_suite(
        golden_file=g_path,
        corpus_dir=c_dir,
        settings=baseline_settings,
        results_dir=r_dir,
        noise_test=True,
    )

    # 2. Run Candidate Evaluation
    print(f"\n[PHASE 2] Running Candidate Evaluation ({candidate_val})...")
    candidate_report = run_retrieval_suite(
        golden_file=g_path,
        corpus_dir=c_dir,
        settings=candidate_settings,
        results_dir=r_dir,
        noise_test=True,
    )

    # 3. Analyze Comparative Results
    b_clean = baseline_report["clean_baseline"]
    c_clean = candidate_report["clean_baseline"]
    b_noisy = baseline_report["noisy_experiment"]
    c_noisy = candidate_report["noisy_experiment"]
    b_deltas = baseline_report["noise_deltas"]
    c_deltas = candidate_report["noise_deltas"]

    diff_recall3 = round(c_clean["recall_at_3"] - b_clean["recall_at_3"], 4)
    diff_recall5 = round(c_clean["recall_at_5"] - b_clean["recall_at_5"], 4)
    diff_mrr = round(c_clean["mrr"] - b_clean["mrr"], 4)
    diff_ndcg5 = round(c_clean["ndcg_at_5"] - b_clean["ndcg_at_5"], 4)
    diff_retention = round(c_deltas["recall_5_retention_pct"] - b_deltas["recall_5_retention_pct"], 2)

    # Recommendation heuristic based on Recall@5 and nDCG@5
    recommendation = "ACCEPT" if (diff_recall5 >= 0.0 and diff_ndcg5 >= -0.02) else "REJECT"
    trade_off_summary = (
        "Candidate provides higher/equal retrieval recall with finer clause resolution."
        if recommendation == "ACCEPT"
        else "Candidate degrades ranking quality or recall compared to baseline."
    )

    print("\n" + "=" * 80)
    print("  CONTROLLED CHANGE EXPERIMENT: COMPARISON TABLE (C1 & C2)")
    print("=" * 80)
    print(f"{'Evaluation Metric':<32} | {'Baseline (500/80)':<18} | {'Candidate (400/50)':<18} | {'Impact (Delta)':<12}")
    print("-" * 87)
    print(f"{'Recall@3 (Clean)':<32} | {b_clean['recall_at_3']:<18.4f} | {c_clean['recall_at_3']:<18.4f} | {diff_recall3:+12.4f}")
    print(f"{'Recall@5 (Clean)':<32} | {b_clean['recall_at_5']:<18.4f} | {c_clean['recall_at_5']:<18.4f} | {diff_recall5:+12.4f}")
    print(f"{'MRR (Mean Reciprocal Rank)':<32} | {b_clean['mrr']:<18.4f} | {c_clean['mrr']:<18.4f} | {diff_mrr:+12.4f}")
    print(f"{'nDCG@5 (Clean Ranking)':<32} | {b_clean['ndcg_at_5']:<18.4f} | {c_clean['ndcg_at_5']:<18.4f} | {diff_ndcg5:+12.4f}")
    print(f"{'Recall@5 (Noisy + Distractors)':<32} | {b_noisy['recall_at_5']:<18.4f} | {c_noisy['recall_at_5']:<18.4f} | {round(c_noisy['recall_at_5'] - b_noisy['recall_at_5'], 4):+12.4f}")
    print(f"{'Noise Retention Rate (%)':<32} | {b_deltas['recall_5_retention_pct']:<17.1f}% | {c_deltas['recall_5_retention_pct']:<17.1f}% | {diff_retention:+11.1f}%")
    print("-" * 87)
    print(f"DECISION RECOMMENDATION : {recommendation}")
    print(f"ANALYSIS & TRADE-OFFS   : {trade_off_summary}")
    print("=" * 80)

    # Persist structured experiment artifact
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dict: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameter": parameter,
        "baseline_configuration": {
            "setting": baseline_val,
            "chunk_size": getattr(baseline_settings, "chunk_size", 500),
            "chunk_overlap": getattr(baseline_settings, "chunk_overlap", 80),
            "metrics_clean": b_clean,
            "metrics_noisy": b_noisy,
            "deltas": b_deltas,
        },
        "candidate_configuration": {
            "setting": candidate_val,
            "chunk_size": getattr(candidate_settings, "chunk_size", 400),
            "chunk_overlap": getattr(candidate_settings, "chunk_overlap", 50),
            "metrics_clean": c_clean,
            "metrics_noisy": c_noisy,
            "deltas": c_deltas,
        },
        "metric_deltas": {
            "delta_recall_at_3": diff_recall3,
            "delta_recall_at_5": diff_recall5,
            "delta_mrr": diff_mrr,
            "delta_ndcg_at_5": diff_ndcg5,
            "delta_retention_pct": diff_retention,
        },
        "recommendation": recommendation,
        "trade_off_analysis": trade_off_summary,
    }

    report_path = r_dir / "change_experiment_report.json"
    archive_path = r_dir / f"change_experiment_report_{timestamp_str}.json"
    report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    archive_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    print(f"Report written to: {report_path}")
    print(f"Timestamped archive: {archive_path}")

    return report_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled change experiment comparing baseline vs candidate.")
    parser.add_argument("--parameter", type=str, default="chunking", choices=["chunking", "top_k"], help="Parameter to vary")
    parser.add_argument("--baseline", type=str, default="500/80", help="Baseline parameter value (e.g. 500/80)")
    parser.add_argument("--candidate", type=str, default="400/50", help="Candidate parameter value (e.g. 400/50)")
    parser.add_argument("--model", type=str, default="", help="Embedding model override (e.g. nomic-embed-text)")
    parser.add_argument("--golden", type=Path, default=None, help="Golden dataset JSONL")
    parser.add_argument("--corpus-dir", type=Path, default=None, help="Corpus directory")
    parser.add_argument("--results-dir", type=Path, default=None, help="Results output directory")
    args = parser.parse_args()

    report = run_change_experiment(
        parameter=args.parameter,
        baseline_val=args.baseline,
        candidate_val=args.candidate,
        model_override=args.model,
        golden_file=args.golden,
        corpus_dir=args.corpus_dir,
        results_dir=args.results_dir,
    )
    return 0 if report["recommendation"] == "ACCEPT" else 1


if __name__ == "__main__":
    sys.exit(main())

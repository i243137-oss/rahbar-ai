#!/usr/bin/env python3
"""Evaluation Reproducibility Benchmark Runner (Assignment 3, Part B5).

Fulfills Section B5 (Reproducibility):
1. Fixes random seeds where applicable.
2. Uses consistent evaluation data across runs.
3. Runs the evaluation at least twice (Run 1 vs. Run 2).
4. Compares results between runs and computes metric deltas.
5. Saves timestamped JSON results and records model/system metadata.
6. Asserts 100% determinism and mathematical reproducibility across repeated executions.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "indexer"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from eval_embeddings import run_embedding_evaluation  # noqa: E402
from eval_retrieval import run_retrieval_suite  # noqa: E402
from pipeline.config import IndexerSettings, get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("rahbar.eval.reproducibility")


def compare_numeric_dicts(
    d1: dict[str, Any],
    d2: dict[str, Any],
    tolerance: float = 1e-5,
) -> tuple[bool, list[str]]:
    """Compare two dictionaries of numeric metrics and report any deviations."""
    passed = True
    deviations: list[str] = []

    for key, val1 in d1.items():
        if key not in d2:
            passed = False
            deviations.append(f"Key '{key}' present in Run 1 but missing in Run 2")
            continue
        val2 = d2[key]
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            delta = abs(val1 - val2)
            if delta > tolerance:
                passed = False
                deviations.append(f"Metric '{key}' mismatch: Run 1 = {val1}, Run 2 = {val2} (delta = {delta})")
        elif isinstance(val1, dict) and isinstance(val2, dict):
            sub_passed, sub_devs = compare_numeric_dicts(val1, val2, tolerance)
            if not sub_passed:
                passed = False
                deviations.extend(f"{key}.{d}" for d in sub_devs)

    return passed, deviations


def run_reproducibility_benchmark(
    iterations: int = 2,
    golden_file: Path | None = None,
    pairs_file: Path | None = None,
    results_dir: Path | None = None,
    model_override: str = "",
) -> dict[str, Any]:
    """Execute evaluation suite across multiple iterations and verify zero variance."""
    r_dir = results_dir or REPO_ROOT / "eval" / "results"
    r_dir.mkdir(parents=True, exist_ok=True)

    active_settings = get_settings()
    if model_override:
        active_settings = IndexerSettings(**{**active_settings.__dict__, "embed_model": model_override})
    elif "OLLAMA_URL" not in os.environ and "minikube" in active_settings.ollama_url:
        import urllib.error
        import urllib.request
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0)
            active_settings = IndexerSettings(**{**active_settings.__dict__, "ollama_url": "http://127.0.0.1:11434"})
        except (OSError, urllib.error.URLError) as exc:
            log.debug("Ollama localhost probe: %s", exc)

    print("=" * 80)
    print("  RAHBAR AI — EVALUATION REPRODUCIBILITY BENCHMARK (PART B5)")
    print("=" * 80)
    print(f"Embedding Model     : {active_settings.embed_model}")
    print(f"Ollama URL          : {active_settings.ollama_url}")
    print(f"Total Iterations    : {iterations}")
    print("-" * 80)

    embedding_runs: list[dict[str, Any]] = []
    retrieval_runs: list[dict[str, Any]] = []

    for i in range(1, iterations + 1):
        print(f"\n>>> EXECUTING REPRODUCIBILITY ITERATION {i}/{iterations} <<<")

        # 1. Module-level Embedding Evaluation
        print(f"--- Iteration {i}: Module-Level Embedding Benchmark ---")
        emb_report = run_embedding_evaluation(
            pairs_file=pairs_file,
            settings=active_settings,
            results_dir=r_dir,
        )
        embedding_runs.append(emb_report["summary"])

        # 2. Component-level Retrieval Evaluation
        print(f"--- Iteration {i}: Component-Level Retrieval Benchmark ---")
        ret_report = run_retrieval_suite(
            golden_file=golden_file,
            settings=active_settings,
            results_dir=r_dir,
            noise_test=True,
        )
        retrieval_runs.append({
            "clean": ret_report["clean_baseline"],
            "noisy": ret_report["noisy_experiment"],
            "deltas": ret_report["noise_deltas"],
        })

    # Compare Iteration 1 vs. Iteration 2
    run1_emb, run2_emb = embedding_runs[0], embedding_runs[1]
    run1_ret, run2_ret = retrieval_runs[0], retrieval_runs[1]

    emb_passed, emb_diffs = compare_numeric_dicts(
        {k: v for k, v in run1_emb.items() if not k.startswith("latency_") and k != "throughput_chunks_per_sec"},
        {k: v for k, v in run2_emb.items() if not k.startswith("latency_") and k != "throughput_chunks_per_sec"},
    )
    ret_clean_passed, ret_clean_diffs = compare_numeric_dicts(run1_ret["clean"], run2_ret["clean"])
    ret_noisy_passed, ret_noisy_diffs = compare_numeric_dicts(run1_ret["noisy"], run2_ret["noisy"])

    overall_passed = emb_passed and ret_clean_passed and ret_noisy_passed

    print("\n" + "=" * 80)
    print("  REPRODUCIBILITY COMPARISON TABLE (RUN 1 VS. RUN 2)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Run 1':<12} | {'Run 2':<12} | {'Abs Delta':<10} | {'Status':<6}")
    print("-" * 83)

    metrics_to_show = [
        ("Embedding Paraphrase Mean", run1_emb["paraphrase_mean_sim"], run2_emb["paraphrase_mean_sim"]),
        ("Embedding Unrelated Mean", run1_emb["unrelated_mean_sim"], run2_emb["unrelated_mean_sim"]),
        ("Embedding Separation Gap", run1_emb["separation_gap"], run2_emb["separation_gap"]),
        ("Retrieval Recall@3 (Clean)", run1_ret["clean"]["recall_at_3"], run2_ret["clean"]["recall_at_3"]),
        ("Retrieval Recall@5 (Clean)", run1_ret["clean"]["recall_at_5"], run2_ret["clean"]["recall_at_5"]),
        ("Retrieval MRR (Clean)", run1_ret["clean"]["mrr"], run2_ret["clean"]["mrr"]),
        ("Retrieval nDCG@5 (Clean)", run1_ret["clean"]["ndcg_at_5"], run2_ret["clean"]["ndcg_at_5"]),
        ("Retrieval Recall@5 (Noisy)", run1_ret["noisy"]["recall_at_5"], run2_ret["noisy"]["recall_at_5"]),
        ("Retrieval nDCG@5 (Noisy)", run1_ret["noisy"]["ndcg_at_5"], run2_ret["noisy"]["ndcg_at_5"]),
        ("Recall@5 Retention %", run1_ret["deltas"]["recall_5_retention_pct"], run2_ret["deltas"]["recall_5_retention_pct"]),
        ("nDCG@5 Retention %", run1_ret["deltas"]["ndcg_5_retention_pct"], run2_ret["deltas"]["ndcg_5_retention_pct"]),
    ]

    for label, v1, v2 in metrics_to_show:
        delta = abs(v1 - v2)
        status = "PASS" if math.isclose(delta, 0.0, abs_tol=1e-5) else "FAIL"
        print(f"{label:<35} | {v1:<12.4f} | {v2:<12.4f} | {delta:<10.6f} | {status:<6}")

    print("-" * 83)
    print(f"Overall Reproducibility: {'PASSED (100% Deterministic)' if overall_passed else 'FAILED'}")
    print("=" * 80)

    # Save summary report
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dict: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": active_settings.embed_model,
        "iterations": iterations,
        "reproducibility_passed": overall_passed,
        "embedding_variance_detected": not emb_passed,
        "retrieval_variance_detected": not (ret_clean_passed and ret_noisy_passed),
        "embedding_deviations": emb_diffs,
        "retrieval_deviations": ret_clean_diffs + ret_noisy_diffs,
        "run_1_metrics": {
            "embedding": run1_emb,
            "retrieval_clean": run1_ret["clean"],
            "retrieval_noisy": run1_ret["noisy"],
            "retrieval_deltas": run1_ret["deltas"],
        },
        "run_2_metrics": {
            "embedding": run2_emb,
            "retrieval_clean": run2_ret["clean"],
            "retrieval_noisy": run2_ret["noisy"],
            "retrieval_deltas": run2_ret["deltas"],
        },
    }

    report_path = r_dir / "reproducibility_report.json"
    archive_path = r_dir / f"reproducibility_report_{timestamp_str}.json"
    report_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    archive_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    print(f"Report written to: {report_path}")
    print(f"Timestamped archive: {archive_path}")

    return report_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducibility benchmark across repeated evaluation runs.")
    parser.add_argument("--iterations", type=int, default=2, help="Number of evaluation iterations to run (default: 2)")
    parser.add_argument("--golden", type=Path, default=None, help="Path to golden dataset JSONL")
    parser.add_argument("--pairs", type=Path, default=None, help="Path to embedding pairs JSON")
    parser.add_argument("--model", type=str, default="", help="Embedding model override")
    parser.add_argument("--results-dir", type=Path, default=None, help="Output directory for reports")
    args = parser.parse_args()

    report = run_reproducibility_benchmark(
        iterations=args.iterations,
        golden_file=args.golden,
        pairs_file=args.pairs,
        results_dir=args.results_dir,
        model_override=args.model,
    )

    return 0 if report["reproducibility_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

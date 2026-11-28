#!/usr/bin/env python3
"""Unified Coverage and Drift Evaluation CLI Module (Assignment 3, Part E).

Entry point for running Part E evaluations:
  python -m eval.coverage
  python -m eval.coverage --drift
  python -m eval.coverage --golden eval/golden.jsonl --threshold 0.70
  python -m eval.coverage --drift --min-coverage 40.0 --min-improvement 10.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "eval"))

from check_coverage import JUSTIFIED_SIMILARITY_THRESHOLD, run_coverage_analysis
from run_drift_experiment import (
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_IMPROVEMENT,
    run_drift_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.coverage",
        description="RAHBAR AI — Coverage and Drift Evaluation Suite (Part E)",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Run comparative before-and-after drift experiment (Part E4)",
    )
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
        help="Path to target golden dataset JSONL",
    )
    parser.add_argument(
        "--baseline-golden",
        type=Path,
        default=REPO_ROOT / "eval" / "golden_baseline.jsonl",
        help="Path to baseline golden dataset JSONL (for --drift)",
    )
    parser.add_argument(
        "--corpus-dir",
        "-c",
        type=Path,
        default=None,
        help="Path to corpus directory (optional, for indexing corpus passages)",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Skip extracting/indexing corpus files and evaluate against golden reference questions",
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
        help=f"Minimum required in-domain coverage percentage (default: {DEFAULT_MIN_COVERAGE})",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=DEFAULT_MIN_IMPROVEMENT,
        help=f"Minimum required coverage improvement delta percentage (for --drift, default: {DEFAULT_MIN_IMPROVEMENT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Embedding model override (e.g. nomic-embed-text or dummy-768)",
    )

    args = parser.parse_args()

    if args.drift:
        b_corpus = None if args.skip_corpus else args.corpus_dir
        c_corpus = None if args.skip_corpus else (
            [REPO_ROOT / "corpus" / "raw", REPO_ROOT / "corpus" / "batch_b"]
            if args.corpus_dir is None
            else args.corpus_dir
        )
        report = run_drift_experiment(
            live_queries_path=args.live_queries,
            baseline_golden_path=args.baseline_golden,
            candidate_golden_path=args.golden,
            baseline_corpus=b_corpus,
            candidate_corpus=c_corpus,
            threshold=args.threshold,
            min_coverage_pct=args.min_coverage,
            min_improvement_pct=args.min_improvement,
            model_override=args.model,
        )
        return 0 if report.get("gate_passed") else 1
    else:
        corpus = None if args.skip_corpus else args.corpus_dir
        report = run_coverage_analysis(
            live_queries_path=args.live_queries,
            golden_path=args.golden,
            corpus_dir=corpus,
            threshold=args.threshold,
            min_coverage_pct=args.min_coverage,
            model_override=args.model,
        )
        return 0 if report.get("gate_passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())

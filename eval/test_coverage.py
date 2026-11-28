#!/usr/bin/env python3
"""Unit tests for Part E2 & E3 Coverage Analysis and Reporting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.check_coverage import (  # noqa: E402
    JUSTIFIED_SIMILARITY_THRESHOLD,
    cosine_similarity,
    run_coverage_analysis,
)

__all__ = [
    "test_cosine_similarity_identical",
    "test_cosine_similarity_orthogonal",
    "test_coverage_gate_threshold_failure",
    "test_justified_similarity_threshold",
    "test_run_coverage_analysis_dummy_mode",
]


def test_cosine_similarity_identical() -> None:
    vec = [1.0, 2.0, 3.0]
    sim = cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_orthogonal() -> None:
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    sim = cosine_similarity(vec_a, vec_b)
    assert abs(sim - 0.0) < 1e-6


def test_justified_similarity_threshold() -> None:
    # Empirically verified separation gap: 0.3919
    # Paraphrase min: 0.7299, Unrelated max: 0.5733
    assert JUSTIFIED_SIMILARITY_THRESHOLD == 0.70


def test_run_coverage_analysis_dummy_mode(tmp_path: Path) -> None:
    live_queries_file = REPO_ROOT / "eval" / "live_queries.jsonl"
    golden_file = REPO_ROOT / "eval" / "golden.jsonl"
    results_dir = tmp_path / "results"

    report = run_coverage_analysis(
        live_queries_path=live_queries_file,
        golden_path=golden_file,
        corpus_dir=None,  # Golden-only comparison for fast offline unit test
        threshold=0.70,
        min_coverage_pct=0.0,
        model_override="dummy-768",
        results_dir=results_dir,
    )

    assert "summary" in report
    summary = report["summary"]
    assert summary["total_queries"] == 44
    assert summary["in_domain_queries"] == 30
    assert summary["out_of_domain_queries"] == 14
    assert "in_domain_coverage_pct" in summary
    assert "similarity_distribution" in report
    assert "uncovered_topic_breakdown" in report
    assert "queries" in report

    # Verify per-query classification
    queries = report["queries"]
    assert len(queries) == 44
    classes = {q["classification"] for q in queries}
    assert "OUT_OF_DOMAIN" in classes
    assert all(q["classification"] == "OUT_OF_DOMAIN" for q in queries if q["domain"] == "out_of_domain")

    # Verify report written to disk
    out_file = results_dir / "coverage_report.json"
    assert out_file.exists()
    disk_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert disk_data["summary"]["total_queries"] == 44


def test_coverage_gate_threshold_failure(tmp_path: Path) -> None:
    live_queries_file = REPO_ROOT / "eval" / "live_queries.jsonl"
    golden_file = REPO_ROOT / "eval" / "golden.jsonl"
    results_dir = tmp_path / "results"

    # Require 100% in-domain coverage (which should fail given known gaps)
    report = run_coverage_analysis(
        live_queries_path=live_queries_file,
        golden_path=golden_file,
        corpus_dir=None,
        threshold=0.70,
        min_coverage_pct=100.0,
        model_override="dummy-768",
        results_dir=results_dir,
    )
    assert report["passed_gate"] is False

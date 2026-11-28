#!/usr/bin/env python3
"""Automated unit and integration tests for Part E4 Drift Experiment and Release Gate."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "eval"))

from eval.run_drift_experiment import (
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_IMPROVEMENT,
    JUSTIFIED_SIMILARITY_THRESHOLD,
    run_drift_experiment,
)


def test_drift_experiment_runs_dummy_mode(tmp_path: Path) -> None:
    """Verify run_drift_experiment executes deterministically with dummy embeddings."""
    live_queries_path = REPO_ROOT / "eval" / "live_queries.jsonl"
    baseline_golden_path = REPO_ROOT / "eval" / "golden_baseline.jsonl"
    candidate_golden_path = REPO_ROOT / "eval" / "golden.jsonl"

    report = run_drift_experiment(
        live_queries_path=live_queries_path,
        baseline_golden_path=baseline_golden_path,
        candidate_golden_path=candidate_golden_path,
        baseline_corpus=None,
        candidate_corpus=None,
        threshold=JUSTIFIED_SIMILARITY_THRESHOLD,
        min_coverage_pct=DEFAULT_MIN_COVERAGE,
        min_improvement_pct=DEFAULT_MIN_IMPROVEMENT,
        model_override="dummy-768",
        results_dir=tmp_path,
    )

    assert isinstance(report, dict)
    assert "timestamp" in report
    assert "configuration" in report
    assert "metrics" in report
    assert "deltas" in report["metrics"]
    assert "topic_drift" in report
    assert "query_flips" in report
    assert "gate_decision" in report
    assert "gate_passed" in report
    assert "gate_checks" in report

    # Verify deltas structure
    deltas = report["metrics"]["deltas"]
    assert deltas["new_golden_questions"] == 6  # 34 - 28 = 6
    assert isinstance(deltas["in_domain_coverage_delta_pct"], (int, float))

    # Verify report written to disk
    out_file = tmp_path / "drift_experiment_report.json"
    assert out_file.exists()


def test_drift_gate_fails_on_unmet_coverage(tmp_path: Path) -> None:
    """Verify release gate status is FAIL and gate_passed is False when min coverage is unmet."""
    live_queries_path = REPO_ROOT / "eval" / "live_queries.jsonl"
    baseline_golden_path = REPO_ROOT / "eval" / "golden_baseline.jsonl"
    candidate_golden_path = REPO_ROOT / "eval" / "golden.jsonl"

    report = run_drift_experiment(
        live_queries_path=live_queries_path,
        baseline_golden_path=baseline_golden_path,
        candidate_golden_path=candidate_golden_path,
        baseline_corpus=None,
        candidate_corpus=None,
        threshold=JUSTIFIED_SIMILARITY_THRESHOLD,
        min_coverage_pct=99.9,  # Unattainable threshold to force gate FAIL
        min_improvement_pct=DEFAULT_MIN_IMPROVEMENT,
        model_override="dummy-768",
        results_dir=tmp_path,
    )

    assert report["gate_decision"] == "FAIL"
    assert report["gate_passed"] is False
    assert report["gate_checks"]["candidate_min_coverage"]["passed"] is False


def test_drift_gate_passes_when_criteria_satisfied(tmp_path: Path) -> None:
    """Verify release gate status is PASS when criteria are satisfied."""
    live_queries_path = REPO_ROOT / "eval" / "live_queries.jsonl"
    baseline_golden_path = REPO_ROOT / "eval" / "golden_baseline.jsonl"
    candidate_golden_path = REPO_ROOT / "eval" / "golden.jsonl"

    report = run_drift_experiment(
        live_queries_path=live_queries_path,
        baseline_golden_path=baseline_golden_path,
        candidate_golden_path=candidate_golden_path,
        baseline_corpus=None,
        candidate_corpus=None,
        threshold=JUSTIFIED_SIMILARITY_THRESHOLD,
        min_coverage_pct=5.0,  # Achievable in dummy mode
        min_improvement_pct=0.0,
        model_override="dummy-768",
        results_dir=tmp_path,
    )

    assert report["gate_decision"] == "PASS"
    assert report["gate_passed"] is True
    assert report["gate_checks"]["candidate_min_coverage"]["passed"] is True


def test_query_flips_tracking(tmp_path: Path) -> None:
    """Verify that query flips correctly identify queries improving from uncovered to covered."""
    live_queries_path = REPO_ROOT / "eval" / "live_queries.jsonl"
    baseline_golden_path = REPO_ROOT / "eval" / "golden_baseline.jsonl"
    candidate_golden_path = REPO_ROOT / "eval" / "golden.jsonl"

    report = run_drift_experiment(
        live_queries_path=live_queries_path,
        baseline_golden_path=baseline_golden_path,
        candidate_golden_path=candidate_golden_path,
        baseline_corpus=None,
        candidate_corpus=None,
        threshold=JUSTIFIED_SIMILARITY_THRESHOLD,
        min_coverage_pct=1.0,
        min_improvement_pct=0.0,
        model_override="dummy-768",
        results_dir=tmp_path,
    )

    flips = report["query_flips"]
    assert isinstance(flips, list)
    for flip in flips:
        assert "id" in flip
        assert "question" in flip
        assert "topic" in flip
        assert "baseline_similarity" in flip
        assert "candidate_similarity" in flip
        assert flip["candidate_similarity"] >= JUSTIFIED_SIMILARITY_THRESHOLD
        assert flip["similarity_gain"] >= 0.0

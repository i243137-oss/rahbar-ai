#!/usr/bin/env python3
"""Unit and integration tests for RAHBAR AI Release Gate (Part C3)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from eval.release_gate import (
    evaluate_gate,
    extract_evaluation_metrics,
    load_config,
)


@pytest.fixture
def sample_config() -> dict[str, Any]:
    return {
        "version": "1.0",
        "name": "test-release-gate",
        "retrieval": {
            "min_recall_at_3": 0.65,
            "min_recall_at_5": 0.70,
            "min_mrr": 0.60,
            "min_ndcg_at_5": 0.70,
            "max_recall_at_5_drop": 0.05,
            "max_ndcg_at_5_drop": 0.05,
            "min_noise_retention_pct": 85.0,
            "max_chunk_amplification_ratio": 1.50,
        },
        "generation": {
            "min_groundedness": 0.80,
            "min_citation_precision": 0.80,
            "max_hallucination_rate": 0.15,
            "min_answer_relevance": 0.75,
            "require_generation_metrics": False,
        },
    }


@pytest.fixture
def passing_metrics() -> dict[str, Any]:
    return {
        "candidate": {
            "clean": {
                "recall_at_3": 0.75,
                "recall_at_5": 0.75,
                "mrr": 0.68,
                "ndcg_at_5": 0.81,
            },
            "noisy": {
                "recall_at_5": 0.66,
            },
            "deltas": {
                "recall_5_retention_pct": 88.0,
            },
            "chunk_count": 380,
        },
        "baseline": {
            "clean": {
                "recall_at_3": 0.68,
                "recall_at_5": 0.72,
                "mrr": 0.64,
                "ndcg_at_5": 0.74,
            },
            "noisy": {
                "recall_at_5": 0.65,
            },
            "deltas": {
                "recall_5_retention_pct": 90.0,
            },
            "chunk_count": 328,
        },
        "deltas": {
            "delta_recall_at_3": 0.07,
            "delta_recall_at_5": 0.03,
            "delta_mrr": 0.04,
            "delta_ndcg_at_5": 0.07,
        },
        "generation": {},
    }


def test_evaluate_gate_passes_cleanly(sample_config: dict[str, Any], passing_metrics: dict[str, Any]) -> None:
    decision = evaluate_gate(passing_metrics, sample_config)
    assert decision["verdict"] == "PASS"
    assert decision["overall_passed"] is True
    assert decision["rules_failed"] == 0
    assert decision["rules_passed"] == 8


def test_evaluate_gate_fails_on_low_recall(sample_config: dict[str, Any], passing_metrics: dict[str, Any]) -> None:
    passing_metrics["candidate"]["clean"]["recall_at_5"] = 0.55  # Below 0.70
    decision = evaluate_gate(passing_metrics, sample_config)
    assert decision["verdict"] == "FAIL"
    assert decision["overall_passed"] is False
    failed_rules = [r["rule"] for r in decision["details"] if r["status"] == "FAIL"]
    assert "min_recall_at_5" in failed_rules


def test_evaluate_gate_fails_on_regression(sample_config: dict[str, Any], passing_metrics: dict[str, Any]) -> None:
    # Baseline was 0.72, candidate is 0.65 -> drop is 0.07 (which exceeds max allowable 0.05)
    passing_metrics["candidate"]["clean"]["recall_at_5"] = 0.65
    passing_metrics["deltas"]["delta_recall_at_5"] = -0.07
    decision = evaluate_gate(passing_metrics, sample_config)
    assert decision["verdict"] == "FAIL"
    failed_rules = [r["rule"] for r in decision["details"] if r["status"] == "FAIL"]
    assert "max_recall_at_5_drop" in failed_rules


def test_evaluate_gate_fails_on_storage_explosion(sample_config: dict[str, Any], passing_metrics: dict[str, Any]) -> None:
    passing_metrics["candidate"]["chunk_count"] = 700  # 700 / 328 = 2.13 > 1.50
    decision = evaluate_gate(passing_metrics, sample_config)
    assert decision["verdict"] == "FAIL"
    failed_rules = [r["rule"] for r in decision["details"] if r["status"] == "FAIL"]
    assert "max_chunk_amplification_ratio" in failed_rules


def test_generation_metrics_evaluation(sample_config: dict[str, Any], passing_metrics: dict[str, Any]) -> None:
    passing_metrics["generation"] = {
        "groundedness": 0.92,
        "citation_precision": 0.88,
        "hallucination_rate": 0.08,
        "answer_relevance": 0.85,
    }
    decision = evaluate_gate(passing_metrics, sample_config)
    assert decision["verdict"] == "PASS"

    # Now make hallucination rate too high
    passing_metrics["generation"]["hallucination_rate"] = 0.25  # > 0.15 limit
    decision_fail = evaluate_gate(passing_metrics, sample_config)
    assert decision_fail["verdict"] == "FAIL"
    failed = [r["rule"] for r in decision_fail["details"] if r["status"] == "FAIL"]
    assert "max_hallucination_rate" in failed


def test_required_generation_metrics_missing(sample_config: dict[str, Any], passing_metrics: dict[str, Any]) -> None:
    sample_config["generation"]["require_generation_metrics"] = True
    passing_metrics["generation"] = {}
    decision = evaluate_gate(passing_metrics, sample_config)
    assert decision["verdict"] == "FAIL"
    failed = [r["rule"] for r in decision["details"] if r["status"] == "FAIL"]
    assert "generation_metrics_presence" in failed


def test_extract_evaluation_metrics_from_change_report(tmp_path: Path) -> None:
    report_file = REPO_ROOT / "eval" / "results" / "change_experiment_report.json"
    if not report_file.exists():
        pytest.skip("change_experiment_report.json not found")
    data = json.loads(report_file.read_text(encoding="utf-8"))
    metrics = extract_evaluation_metrics(data)
    assert "clean" in metrics["candidate"]
    assert metrics["candidate"]["clean"]["recall_at_5"] == 0.75
    assert metrics["baseline"]["clean"]["recall_at_5"] == 0.7273


def test_cli_gate_execution(tmp_path: Path) -> None:
    report_file = REPO_ROOT / "eval" / "results" / "change_experiment_report.json"
    config_file = REPO_ROOT / "eval" / "gate_config.yaml"
    out_file = tmp_path / "test_decision.json"

    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "eval" / "release_gate.py"),
            "--candidate",
            str(report_file),
            "--config",
            str(config_file),
            "--output",
            str(out_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "GATE PASSED" in res.stdout
    assert out_file.exists()
    decision = json.loads(out_file.read_text(encoding="utf-8"))
    assert decision["verdict"] == "PASS"


def test_load_config_valid() -> None:
    cfg_path = REPO_ROOT / "eval" / "gate_config.yaml"
    cfg = load_config(cfg_path)
    assert "retrieval" in cfg
    assert cfg["retrieval"]["min_recall_at_5"] == 0.70

#!/usr/bin/env python3
"""RAHBAR AI — Automated Evaluation Release Gate (Assignment 3, Part C3).

Evaluates candidate model/chunking configurations against defined quality thresholds
and baseline performance to determine deployment readiness.

Intended for use in:
- Member 1 evaluation workflows (Change experiment gating)
- Member 2 (Ahmed) Kubernetes CI/CD release and deployment pipelines

Exit Codes:
- 0: GATE PASSED — Candidate meets or exceeds all quality thresholds; approved for release.
- 1: GATE FAILED — Candidate violates one or more quality thresholds; release blocked.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("rahbar.eval.release_gate")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load release gate configuration from YAML or JSON."""
    if not config_path.exists():
        raise FileNotFoundError(f"Gate config not found: {config_path}")

    content = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in [".yaml", ".yml"]:
        if yaml is None:
            raise RuntimeError("PyYAML required to parse .yaml configuration files.")
        return yaml.safe_load(content)  # type: ignore[no-any-return]
    return json.loads(content)  # type: ignore[no-any-return]


def load_report(report_path: Path) -> dict[str, Any]:
    """Load evaluation report JSON file."""
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def extract_evaluation_metrics(
    candidate_data: dict[str, Any], baseline_data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Normalize metrics from either change_experiment_report or standard retrieval reports."""
    extracted: dict[str, Any] = {
        "candidate": {},
        "baseline": {},
        "deltas": {},
        "generation": {},
    }

    # Case 1: change_experiment_report.json (contains both candidate and baseline)
    if "candidate_configuration" in candidate_data and "baseline_configuration" in candidate_data:
        c_conf = candidate_data["candidate_configuration"]
        b_conf = candidate_data["baseline_configuration"]

        extracted["candidate"]["clean"] = c_conf.get("metrics_clean", {})
        extracted["candidate"]["noisy"] = c_conf.get("metrics_noisy", {})
        extracted["candidate"]["deltas"] = c_conf.get("deltas", {})
        extracted["candidate"]["chunk_count"] = c_conf.get("chunk_count")

        extracted["baseline"]["clean"] = b_conf.get("metrics_clean", {})
        extracted["baseline"]["noisy"] = b_conf.get("metrics_noisy", {})
        extracted["baseline"]["deltas"] = b_conf.get("deltas", {})
        extracted["baseline"]["chunk_count"] = b_conf.get("chunk_count")

        extracted["deltas"] = candidate_data.get("metric_deltas", {})

    # Case 2: standard retrieval_eval_report.json
    else:
        extracted["candidate"]["clean"] = candidate_data.get("clean_baseline", {})
        extracted["candidate"]["noisy"] = candidate_data.get("noisy_experiment", {})
        extracted["candidate"]["deltas"] = candidate_data.get("noise_deltas", {})
        extracted["candidate"]["chunk_count"] = candidate_data.get("system_config", {}).get("total_chunks")

        if baseline_data:
            extracted["baseline"]["clean"] = baseline_data.get("clean_baseline", {})
            extracted["baseline"]["noisy"] = baseline_data.get("noisy_experiment", {})
            extracted["baseline"]["deltas"] = baseline_data.get("noise_deltas", {})
            extracted["baseline"]["chunk_count"] = baseline_data.get("system_config", {}).get("total_chunks")

            c_clean = extracted["candidate"]["clean"]
            b_clean = extracted["baseline"]["clean"]
            if c_clean and b_clean:
                extracted["deltas"] = {
                    "delta_recall_at_3": round(c_clean.get("recall_at_3", 0.0) - b_clean.get("recall_at_3", 0.0), 4),
                    "delta_recall_at_5": round(c_clean.get("recall_at_5", 0.0) - b_clean.get("recall_at_5", 0.0), 4),
                    "delta_mrr": round(c_clean.get("mrr", 0.0) - b_clean.get("mrr", 0.0), 4),
                    "delta_ndcg_at_5": round(c_clean.get("ndcg_at_5", 0.0) - b_clean.get("ndcg_at_5", 0.0), 4),
                }

    # Extract Generation metrics if provided (Member 2 contract)
    if "generation_metrics" in candidate_data:
        extracted["generation"] = candidate_data["generation_metrics"]
    elif "generation" in candidate_data:
        extracted["generation"] = candidate_data["generation"]

    return extracted


def evaluate_gate(
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate candidate metrics against the release gate configuration rules."""
    rules_results: list[dict[str, Any]] = []
    overall_passed = True

    retrieval_cfg = config.get("retrieval", {})
    generation_cfg = config.get("generation", {})

    c_clean = metrics["candidate"].get("clean", {})
    c_deltas = metrics["candidate"].get("deltas", {})
    b_clean = metrics["baseline"].get("clean", {})
    deltas = metrics.get("deltas", {})
    gen_metrics = metrics.get("generation", {})

    # Helper function to append checks
    def record_check(
        name: str,
        observed: Any,
        operator_str: str,
        threshold: Any,
        passed: bool,
        rationale: str,
    ) -> None:
        nonlocal overall_passed
        if not passed:
            overall_passed = False
        rules_results.append({
            "rule": name,
            "observed": observed,
            "condition": f"{operator_str} {threshold}",
            "threshold": threshold,
            "status": "PASS" if passed else "FAIL",
            "rationale": rationale,
        })

    # Rule 1: min_recall_at_3
    if "min_recall_at_3" in retrieval_cfg and "recall_at_3" in c_clean:
        val = c_clean["recall_at_3"]
        thresh = retrieval_cfg["min_recall_at_3"]
        record_check(
            "min_recall_at_3",
            round(val, 4),
            ">=",
            thresh,
            val >= thresh,
            "Ensure at least specified proportion of target chunks in top 3",
        )

    # Rule 2: min_recall_at_5
    if "min_recall_at_5" in retrieval_cfg and "recall_at_5" in c_clean:
        val = c_clean["recall_at_5"]
        thresh = retrieval_cfg["min_recall_at_5"]
        record_check(
            "min_recall_at_5",
            round(val, 4),
            ">=",
            thresh,
            val >= thresh,
            "Ensure baseline retrieval recall quality at top 5",
        )

    # Rule 3: min_mrr
    if "min_mrr" in retrieval_cfg and "mrr" in c_clean:
        val = c_clean["mrr"]
        thresh = retrieval_cfg["min_mrr"]
        record_check(
            "min_mrr",
            round(val, 4),
            ">=",
            thresh,
            val >= thresh,
            "Mean Reciprocal Rank must place relevant chunk near top ranks",
        )

    # Rule 4: min_ndcg_at_5
    if "min_ndcg_at_5" in retrieval_cfg and "ndcg_at_5" in c_clean:
        val = c_clean["ndcg_at_5"]
        thresh = retrieval_cfg["min_ndcg_at_5"]
        record_check(
            "min_ndcg_at_5",
            round(val, 4),
            ">=",
            thresh,
            val >= thresh,
            "Normalized Discounted Cumulative Gain ranking quality",
        )

    # Rule 5: max_recall_at_5_drop (regression check against baseline)
    if "max_recall_at_5_drop" in retrieval_cfg and b_clean and "recall_at_5" in b_clean:
        delta_r5 = deltas.get("delta_recall_at_5")
        if delta_r5 is None:
            delta_r5 = c_clean.get("recall_at_5", 0.0) - b_clean.get("recall_at_5", 0.0)
        max_drop = retrieval_cfg["max_recall_at_5_drop"]
        # delta_r5 should be >= -max_drop (e.g. >= -0.05)
        passed = delta_r5 >= (-abs(max_drop))
        record_check(
            "max_recall_at_5_drop",
            round(delta_r5, 4),
            ">=",
            round(-abs(max_drop), 4),
            passed,
            f"Recall@5 regression relative to baseline must not exceed {max_drop}",
        )

    # Rule 6: max_ndcg_at_5_drop (regression check against baseline)
    if "max_ndcg_at_5_drop" in retrieval_cfg and b_clean and "ndcg_at_5" in b_clean:
        delta_ndcg = deltas.get("delta_ndcg_at_5")
        if delta_ndcg is None:
            delta_ndcg = c_clean.get("ndcg_at_5", 0.0) - b_clean.get("ndcg_at_5", 0.0)
        max_drop = retrieval_cfg["max_ndcg_at_5_drop"]
        passed = delta_ndcg >= (-abs(max_drop))
        record_check(
            "max_ndcg_at_5_drop",
            round(delta_ndcg, 4),
            ">=",
            round(-abs(max_drop), 4),
            passed,
            f"nDCG@5 ranking degradation relative to baseline must not exceed {max_drop}",
        )

    # Rule 7: min_noise_retention_pct
    if "min_noise_retention_pct" in retrieval_cfg and "recall_5_retention_pct" in c_deltas:
        val = c_deltas["recall_5_retention_pct"]
        thresh = retrieval_cfg["min_noise_retention_pct"]
        record_check(
            "min_noise_retention_pct",
            round(val, 1),
            ">=",
            thresh,
            val >= thresh,
            "Robustness against typos and distractor noise",
        )

    # Rule 8: max_chunk_amplification_ratio
    c_chunks = metrics["candidate"].get("chunk_count")
    b_chunks = metrics["baseline"].get("chunk_count")
    if "max_chunk_amplification_ratio" in retrieval_cfg and c_chunks and b_chunks:
        ratio = round(c_chunks / max(1, b_chunks), 2)
        thresh = retrieval_cfg["max_chunk_amplification_ratio"]
        record_check(
            "max_chunk_amplification_ratio",
            ratio,
            "<=",
            thresh,
            ratio <= thresh,
            "Chunk expansion must not overload vector storage memory",
        )

    # Generation Checks (Member 2 / Ahmed Integration)
    require_gen = generation_cfg.get("require_generation_metrics", False)
    if require_gen and not gen_metrics:
        record_check(
            "generation_metrics_presence",
            "MISSING",
            "==",
            "PRESENT",
            False,
            "Generation metrics required by policy but not present in report",
        )
    elif gen_metrics:
        if "min_groundedness" in generation_cfg and "groundedness" in gen_metrics:
            val = gen_metrics["groundedness"]
            thresh = generation_cfg["min_groundedness"]
            record_check("min_groundedness", round(val, 3), ">=", thresh, val >= thresh, "Grounded answer claims")
        if "min_citation_precision" in generation_cfg and "citation_precision" in gen_metrics:
            val = gen_metrics["citation_precision"]
            thresh = generation_cfg["min_citation_precision"]
            record_check("min_citation_precision", round(val, 3), ">=", thresh, val >= thresh, "Valid citation mapping")
        if "max_hallucination_rate" in generation_cfg and "hallucination_rate" in gen_metrics:
            val = gen_metrics["hallucination_rate"]
            thresh = generation_cfg["max_hallucination_rate"]
            record_check("max_hallucination_rate", round(val, 3), "<=", thresh, val <= thresh, "Hallucination bound")
        if "min_answer_relevance" in generation_cfg and "answer_relevance" in gen_metrics:
            val = gen_metrics["answer_relevance"]
            thresh = generation_cfg["min_answer_relevance"]
            record_check("min_answer_relevance", round(val, 3), ">=", thresh, val >= thresh, "Semantic answer relevance")

    verdict = "PASS" if overall_passed else "FAIL"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "overall_passed": overall_passed,
        "rules_evaluated": len(rules_results),
        "rules_passed": sum(1 for r in rules_results if r["status"] == "PASS"),
        "rules_failed": sum(1 for r in rules_results if r["status"] == "FAIL"),
        "details": rules_results,
    }


def print_gate_summary(decision: dict[str, Any], config_name: str) -> None:
    """Print formatted gate evaluation summary to stdout."""
    print("=" * 88)
    print("  RAHBAR AI — EVALUATION RELEASE GATE DECISION REPORT")
    print(f"  Configuration : {config_name}")
    print(f"  Timestamp     : {decision['timestamp']}")
    print("=" * 88)
    print(f"{'Rule Name':<30} | {'Observed':<12} | {'Condition':<16} | {'Status':<8} | {'Rationale'}")
    print("-" * 88)

    for r in decision["details"]:
        status_str = f"\033[92m{r['status']}\033[0m" if r["status"] == "PASS" else f"\033[91m{r['status']}\033[0m"
        # Fallback without ANSI if piped/redirected
        if not sys.stdout.isatty():
            status_str = r["status"]
        obs = str(r["observed"])
        cond = str(r["condition"])
        rat = str(r["rationale"])
        print(f"{r['rule']:<30} | {obs:<12} | {cond:<16} | {status_str:<8} | {rat}")

    print("-" * 88)
    v_text = "GATE PASSED — RELEASE APPROVED" if decision["overall_passed"] else "GATE FAILED — RELEASE REJECTED"
    v_str = f"\033[92m{v_text}\033[0m" if (decision["overall_passed"] and sys.stdout.isatty()) else (f"\033[91m{v_text}\033[0m" if sys.stdout.isatty() else v_text)
    print(f"OVERALL VERDICT : {v_str}")
    print(f"SUMMARY         : {decision['rules_passed']}/{decision['rules_evaluated']} rules passed ({decision['rules_failed']} failed)")
    print("=" * 88)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAHBAR AI Release Gate for CI/CD and deployment promotion.")
    parser.add_argument(
        "--candidate",
        "-c",
        type=Path,
        required=True,
        help="Path to candidate evaluation report JSON (or change_experiment_report.json)",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        type=Path,
        default=None,
        help="Path to baseline evaluation report JSON (optional if using change_experiment_report)",
    )
    parser.add_argument(
        "--config",
        "-g",
        type=Path,
        default=Path(__file__).resolve().parent / "gate_config.yaml",
        help="Path to release gate configuration YAML/JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "gate_decision_report.json",
        help="Path to save output gate decision report JSON",
    )
    args = parser.parse_args()

    # 1. Load configuration and reports
    try:
        config = load_config(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        log.error("Failed to load gate config from %s: %s", args.config, exc)
        return 2

    try:
        candidate_data = load_report(args.candidate)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        log.error("Failed to load candidate report from %s: %s", args.candidate, exc)
        return 2

    baseline_data = None
    if args.baseline:
        try:
            baseline_data = load_report(args.baseline)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            log.error("Failed to load baseline report from %s: %s", args.baseline, exc)
            return 2

    # 2. Extract normalized metrics
    metrics = extract_evaluation_metrics(candidate_data, baseline_data)

    # 3. Evaluate gate
    decision = evaluate_gate(metrics, config)
    decision["candidate_file"] = str(args.candidate)
    decision["baseline_file"] = str(args.baseline) if args.baseline else None
    decision["config_file"] = str(args.config)

    # 4. Print summary
    cfg_name = config.get("name", "rahbar-rag-release-gate")
    print_gate_summary(decision, cfg_name)

    # 5. Save machine-readable decision report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"\nMachine-readable gate decision saved to: {args.output}")

    # Return exit code: 0 for PASS, 1 for FAIL
    return 0 if decision["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

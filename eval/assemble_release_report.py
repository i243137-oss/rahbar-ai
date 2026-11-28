"""Combine retrieval metrics and reviewed live answers for the release gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from run_eval import provider_failed


def assemble_report(
    retrieval: dict[str, Any],
    answers: dict[str, Any],
    reviews: dict[str, bool],
) -> dict[str, Any]:
    rows = answers["per_question"]
    if not rows or answers["questions"] != len(rows):
        raise ValueError("The detailed answer report must include every evaluated question.")
    if any(row["response"]["provider"] == "stub" for row in rows):
        raise ValueError("Stub responses cannot supply release evidence.")
    if not retrieval["clean_baseline"] or retrieval["embed_model"] == "dummy-768":
        raise ValueError("The retrieval report must contain metrics from real embeddings.")
    if any(type(value) is not bool for value in reviews.values()):
        raise ValueError("Each faithfulness review must be true or false.")

    provided_answers = {
        row["response"]["request_id"]
        for row in rows
        if row["response"]["answered_from_context"] or row["response"]["citations"]
    }
    if reviews.keys() - provided_answers:
        raise ValueError("Faithfulness reviews must refer to answers in this run by request ID.")
    missing = sorted(provided_answers - reviews.keys())

    generation = {
        "citation_precision": answers["citation_validity"],
        "abstention_correctness": answers["abstention_correctness"],
        "latency_p50_seconds": answers["latency_p50_seconds"],
        "latency_p95_seconds": answers["latency_p95_seconds"],
    }
    if provided_answers and not missing:
        generation["groundedness"] = mean(reviews.values())

    return {
        **retrieval,
        "assembled_at": datetime.now(timezone.utc).isoformat(),
        "generation_metrics": {name: value for name, value in generation.items() if value is not None},
        "unreviewed_request_ids": missing,
        "provider_failures": sum(provider_failed(row["response"]) for row in rows),
        "live_answer_evaluation": answers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--reviews", type=Path, help="JSON mapping request IDs to faithfulness judgments")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
        answers = json.loads(args.answers.read_text(encoding="utf-8"))
        reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else {}
        report = assemble_report(retrieval, answers, reviews)
    except (OSError, AttributeError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Cannot assemble release report: {exc}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report written to {args.output}")
    required = {"groundedness", "citation_precision", "abstention_correctness"}
    if answers["questions"] < 25 or required - report["generation_metrics"].keys() or report["provider_failures"]:
        print("Release evidence incomplete or provider failures detected. See the report for details.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

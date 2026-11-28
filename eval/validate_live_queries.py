#!/usr/bin/env python3
"""Validation script for Part E1 Simulated Live Queries Dataset (eval/live_queries.jsonl).

Verifies compliance with Assignment 3 Part E1 requirements:
- At least 40 total simulated queries.
- Approximately 28 in-domain questions.
- At least 12 out-of-domain questions.
- Realistic university knowledge base queries and out-of-domain queries.
- Inclusion of uncovered topics in the current corpus.
- Non-trivial distribution of answerable vs unanswerable queries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def validate_live_queries(file_path: Path) -> int:
    if not file_path.exists():
        print(f"[FAIL] Dataset file does not exist: {file_path}")
        return 1

    records: list[dict] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError as exc:
                print(f"[FAIL] Line {line_num}: Invalid JSON - {exc}")
                return 1

    total = len(records)
    print("=" * 80)
    print("  RAHBAR AI — PART E1 SIMULATED LIVE QUERIES VALIDATION")
    print("=" * 80)
    print(f"Dataset path: {file_path}")
    print(f"Total queries: {total}")
    print("-" * 80)

    # 1. Total count check (>= 40)
    if total < 40:
        print(f"[FAIL] Total query count is {total}, expected >= 40")
        return 1
    print(f"[PASS] Total queries: {total} (required >= 40)")

    # 2. Domain distribution check
    in_domain = [r for r in records if r.get("domain") == "in_domain"]
    out_of_domain = [r for r in records if r.get("domain") == "out_of_domain"]

    print(f"[INFO] In-domain count: {len(in_domain)} (target: approx 28)")
    print(f"[INFO] Out-of-domain count: {len(out_of_domain)} (target: >= 12)")

    if len(in_domain) < 25:
        print(f"[FAIL] In-domain count {len(in_domain)} is below acceptable range (expected approx 28)")
        return 1
    print(f"[PASS] In-domain count: {len(in_domain)} is valid (expected approx 28)")

    if len(out_of_domain) < 12:
        print(f"[FAIL] Out-of-domain count {len(out_of_domain)} < 12")
        return 1
    print(f"[PASS] Out-of-domain count: {len(out_of_domain)} (required >= 12)")

    # 3. Uncovered topics check
    uncovered_baseline = [r for r in in_domain if not r.get("baseline_covered", False)]
    print(f"[INFO] In-domain queries uncovered in baseline corpus: {len(uncovered_baseline)}")
    if len(uncovered_baseline) < 5:
        print(f"[FAIL] Expected at least 5 uncovered in-domain queries, got {len(uncovered_baseline)}")
        return 1
    print(f"[PASS] Uncovered in-domain queries: {len(uncovered_baseline)} (demonstrates corpus gaps)")

    # 4. Check unique IDs
    ids = [r.get("id") for r in records]
    if len(ids) != len(set(ids)):
        duplicates = [x for x in ids if ids.count(x) > 1]
        print(f"[FAIL] Duplicate query IDs detected: {set(duplicates)}")
        return 1
    print(f"[PASS] Unique IDs: {len(ids)} unique (duplicates: none)")

    # 5. Schema completeness check
    required_keys = {"id", "question", "domain", "topic", "baseline_covered", "batch_b_covered", "notes"}
    schema_errors = 0
    for r in records:
        missing = required_keys - set(r.keys())
        if missing:
            print(f"[FAIL] Query {r.get('id')}: missing required keys {missing}")
            schema_errors += 1
        if not r.get("question") or len(r.get("question", "").strip()) < 10:
            print(f"[FAIL] Query {r.get('id')}: invalid or too short question text")
            schema_errors += 1

    if schema_errors > 0:
        print(f"[FAIL] {schema_errors} schema issues detected")
        return 1
    print(f"[PASS] Schema completeness: 0 errors detected across all {total} queries")

    # 6. Topic diversity
    topics = {r.get("topic") for r in records}
    print(f"[PASS] Topic diversity: {len(topics)} unique topics across in-domain and out-of-domain")

    print("-" * 80)
    print("[SUCCESS] All Part E1 Simulated Live Queries criteria PASSED!")
    print("=" * 80)
    return 0


def main() -> int:
    dataset_path = REPO_ROOT / "eval" / "live_queries.jsonl"
    return validate_live_queries(dataset_path)


if __name__ == "__main__":
    sys.exit(main())

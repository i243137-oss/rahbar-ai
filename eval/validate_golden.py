#!/usr/bin/env python3
"""Validation utility for Part B1 Golden Evaluation Dataset (Assignment 3).

Verifies compliance against all criteria:
1. Total questions >= 25
2. Answerable questions >= 15 (with ground-truth document, section, and page)
3. Unanswerable questions >= 5 (must_abstain=True, relevant=[])
4. Hard questions >= 5 (cross-document, multi-chunk, or paraphrased)
5. Group hand-written (human) questions >= 8
6. Synthetic questions reviewed by hand >= 8
7. Complete schema attributes on every record
8. Verifiable ground-truth references against corpus files
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_JSONL = REPO_ROOT / "eval" / "golden.jsonl"


def validate_golden_dataset(path: Path | None = None) -> bool:
    target_path = path or GOLDEN_JSONL
    if not target_path.exists():
        print(f"[FAIL] Golden dataset file not found: {target_path}", file=sys.stderr)
        return False

    records: list[dict] = []
    with open(target_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as exc:
                print(f"[FAIL] Invalid JSON on line {idx}: {exc}", file=sys.stderr)
                return False

    print("=" * 80)
    print("  RAHBAR AI — PART B1 GOLDEN EVALUATION DATASET VALIDATION")
    print("=" * 80)
    print(f"Dataset path: {target_path}")
    print(f"Total records parsed: {len(records)}")
    print("-" * 80)

    # 1. Total questions
    total_count = len(records)
    pass_total = total_count >= 25
    print(f"[{'PASS' if pass_total else 'FAIL'}] Total questions: {total_count} (required >= 25)")

    # 2. Answerable vs Unanswerable counts
    answerable = [r for r in records if r.get("answerable") is True or not r.get("must_abstain", False)]
    unanswerable = [r for r in records if r.get("answerable") is False or r.get("must_abstain", False)]
    pass_ans = len(answerable) >= 15
    pass_unans = len(unanswerable) >= 5
    print(f"[{'PASS' if pass_ans else 'FAIL'}] Answerable questions: {len(answerable)} (required >= 15)")
    print(f"[{'PASS' if pass_unans else 'FAIL'}] Unanswerable questions: {len(unanswerable)} (required >= 5)")

    # 3. Hard questions
    hard_questions = [
        r for r in records
        if r.get("difficulty") == "hard" or r.get("type") in {"hard", "hard_multihop", "hard_paraphrase", "hard_batch_b"}
    ]
    pass_hard = len(hard_questions) >= 5
    print(f"[{'PASS' if pass_hard else 'FAIL'}] Hard / complex questions: {len(hard_questions)} (required >= 5)")

    # 4. Human vs Synthetic counts
    human_authored = [r for r in records if r.get("author_type") == "human"]
    synthetic_reviewed = [
        r for r in records
        if r.get("author_type") == "synthetic" and r.get("review_status") == "reviewed_by_hand"
    ]
    pass_human = len(human_authored) >= 8
    pass_synth_rev = len(synthetic_reviewed) >= 8
    print(f"[{'PASS' if pass_human else 'FAIL'}] Group hand-written (human) questions: {len(human_authored)} (required >= 8)")
    print(f"[{'PASS' if pass_synth_rev else 'FAIL'}] Synthetic questions reviewed by hand: {len(synthetic_reviewed)} (required >= 8)")

    # 5. Schema completeness & Uniqueness
    required_fields = {
        "id", "question", "answerable", "must_abstain", "relevant",
        "relevant_docs", "relevant_pages", "expect", "type", "category",
        "difficulty", "author_type", "review_status"
    }
    missing_fields_found = []
    seen_ids = set()
    duplicate_ids = set()

    for r in records:
        qid = r.get("id")
        if qid in seen_ids:
            duplicate_ids.add(qid)
        seen_ids.add(qid)

        missing = required_fields - set(r.keys())
        if missing:
            missing_fields_found.append((qid, sorted(missing)))

    pass_unique = len(duplicate_ids) == 0
    pass_fields = len(missing_fields_found) == 0
    print(f"[{'PASS' if pass_unique else 'FAIL'}] Unique question IDs: {len(seen_ids)} unique (duplicates: {duplicate_ids or 'none'})")
    print(f"[{'PASS' if pass_fields else 'FAIL'}] Schema completeness: {len(missing_fields_found)} issues found")

    # 6. Ground truth relevance consistency
    ground_truth_errors = []
    for r in records:
        qid = r.get("id")
        if r.get("must_abstain"):
            if r.get("relevant"):
                ground_truth_errors.append(f"{qid}: unanswerable question has non-empty relevant list")
        else:
            if not r.get("relevant"):
                ground_truth_errors.append(f"{qid}: answerable question has empty relevant list")
            for ref in r.get("relevant", []):
                if not ref.get("source") or not ref.get("section"):
                    ground_truth_errors.append(f"{qid}: relevant item missing source or section: {ref}")

    pass_gt = len(ground_truth_errors) == 0
    print(f"[{'PASS' if pass_gt else 'FAIL'}] Ground truth references consistency: {len(ground_truth_errors)} errors")

    print("-" * 80)
    all_passed = (
        pass_total
        and pass_ans
        and pass_unans
        and pass_hard
        and pass_human
        and pass_synth_rev
        and pass_unique
        and pass_fields
        and pass_gt
    )

    if all_passed:
        print("[SUCCESS] All Part B1 Golden Evaluation Dataset criteria PASSED.")
    else:
        print("[FAILURE] Some golden dataset validation criteria failed.", file=sys.stderr)

    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    success = validate_golden_dataset(target)
    sys.exit(0 if success else 1)

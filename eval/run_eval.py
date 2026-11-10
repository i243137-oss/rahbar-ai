#!/usr/bin/env python3
"""Score a running RAHBAR against a golden set, and gate CI on the result (section 5).

    python eval/run_eval.py --api http://<ingress>                 # your corpus
    python eval/run_eval.py --api ... --golden eval/fixtures/golden.fixture.yaml
    python eval/run_eval.py --api ... --update-baseline            # after a deliberate improvement

Exit code 0 when every gate passes, 1 when any fails. CI relies on that exit code.

The runner sends golden questions to the API, writes results.json, and applies
the evaluation gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from math import isclose
from pathlib import Path
from statistics import mean

import httpx
import yaml

HERE = Path(__file__).resolve().parent

# The gate from section 5.4. Change these only with a reason written in docs/PIPELINE.md.
MAX_RECALL5_DROP = 0.05        # recall@5 may fall at most 5 points below the baseline
MIN_CITATION_VALIDITY = 0.95   # at least 95% of citations must match retrieved passages
REQUIRED_ABSTENTION = 1.0      # every must-abstain question must be declined


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []
    if not questions:
        sys.exit(f"no questions in {path}")
    return questions


def ask(client: httpx.Client, api: str, question: str) -> dict:
    """POST one question and return the AskResponse as a dict (see api/app/schemas.py)."""
    response = client.post(f"{api.rstrip('/')}/api/ask", json={"question": question})
    response.raise_for_status()
    return response.json()


def recall_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> float | None:
    """Fraction of the relevant (source, section) pairs found in the top k retrieved.
    Return None when the question has no relevant passages (must-abstain questions)."""
    if not relevant:
        return None

    expected = {(ref["source"], ref["section"].strip()) for ref in relevant}
    top_k = [(ref["source"], ref["section"].strip()) for ref in retrieved[:k]]
    hits = sum(
        any(
            source == actual_source
            and (section == actual_section or actual_section.startswith((f"{section} ", f"{section}:")))
            for actual_source, actual_section in top_k
        )
        for source, section in expected
    )
    return hits / len(expected)


def _citation_counts(response: dict) -> tuple[int, int]:
    citations = response["citations"]
    retrieved = response["retrieved"]
    valid = sum(
        any(
            citation["source"] == passage["source"]
            and citation["section"] == passage["section"]
            and (citation.get("page") is None or citation["page"] == passage.get("page"))
            for passage in retrieved
        )
        for citation in citations
    )
    return valid, len(citations)


def citation_validity(response: dict) -> float | None:
    """Fraction of the response's citations that match a passage in response['retrieved'].
    Return None when there are no citations."""
    valid, total = _citation_counts(response)
    return valid / total if total else None


def summarise(rows: list[dict]) -> dict:
    """Aggregate recall by question, citation validity by citation, and abstention."""
    results = {}
    for k in (3, 5):
        scores_by_type = defaultdict(list)
        for row in rows:
            scores = scores_by_type[row["type"]]
            if not row["must_abstain"]:
                score = recall_at_k(row["relevant"], row["response"]["retrieved"], k)
                if score is not None:
                    scores.append(score)

        scores = [score for group in scores_by_type.values() for score in group]
        results[f"recall@{k}"] = mean(scores) if scores else None
        results[f"recall@{k}_by_type"] = {
            kind: mean(group) if group else None for kind, group in scores_by_type.items()
        }

    citation_counts = [_citation_counts(row["response"]) for row in rows]
    valid_citations = sum(valid for valid, _ in citation_counts)
    total_citations = sum(total for _, total in citation_counts)
    results["citation_validity"] = valid_citations / total_citations if total_citations else None

    abstentions = [
        row["response"]["answered_from_context"] is False for row in rows if row["must_abstain"]
    ]
    results["abstention_correctness"] = mean(abstentions) if abstentions else None
    return results


def gate(results: dict, baseline: dict | None) -> list[str]:
    """Return one message per failed gate (section 5.4); an empty list means pass."""
    failures = []
    recall5 = results["recall@5"]
    baseline_recall5 = baseline.get("recall@5") if baseline is not None else None
    if recall5 is not None and baseline_recall5 is not None:
        drop = baseline_recall5 - recall5
        if drop > MAX_RECALL5_DROP and not isclose(drop, MAX_RECALL5_DROP, abs_tol=1e-12):
            failures.append(f"Recall@5 fell from {baseline_recall5:.1%} to {recall5:.1%}")

    abstention = results["abstention_correctness"]
    if abstention is not None and abstention < REQUIRED_ABSTENTION:
        failures.append(f"Must-abstain questions answered from context (correct abstention: {abstention:.1%})")

    validity = results["citation_validity"]
    if validity is not None and validity < MIN_CITATION_VALIDITY:
        failures.append(f"Citation validity {validity:.1%} is below {MIN_CITATION_VALIDITY:.0%}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAHBAR against a golden set.")
    parser.add_argument("--api", required=True, help="base URL, e.g. http://192.168.49.2")
    parser.add_argument("--host", help="Host header, when your Ingress rule names a host")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.yaml")
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--update-baseline", action="store_true",
                        help="write these results as the new baseline")
    args = parser.parse_args()

    questions = load_golden(args.golden)
    rows = []
    headers = {"Host": args.host} if args.host else {}
    with httpx.Client(timeout=120.0, headers=headers) as client:
        for q in questions:
            started = time.perf_counter()
            response = ask(client, args.api, q["question"])
            rows.append({
                "id": q["id"], "type": q.get("type", "ordinary"),
                "must_abstain": bool(q.get("must_abstain", False)),
                "relevant": q.get("relevant") or [],
                "expect": q.get("expect"),
                "response": response,
                "seconds": round(time.perf_counter() - started, 2),
            })
            print(f"  {q['id']:<5} {rows[-1]['seconds']:>6.2f}s  {q['question'][:60]}")

    try:
        results = summarise(rows)
        baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
        failures = gate(results, baseline)
    except NotImplementedError as exc:
        print(f"\nnot implemented yet: {exc}")
        return 2

    results["golden"] = str(args.golden)
    results["questions"] = len(rows)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {args.out}")
    if args.update_baseline:
        args.baseline.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"baseline updated: {args.baseline}")

    if failures:
        print("\nGATE FAILED")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\ngate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

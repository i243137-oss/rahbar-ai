"""Print the stored prompt for a trace ID or summarize a time window."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--id")
    mode.add_argument("--summary", action="store_true")
    parser.add_argument("--since", type=datetime.fromisoformat)
    parser.add_argument("--until", type=datetime.fromisoformat)
    args = parser.parse_args()
    if any(x is not None and x.tzinfo is None for x in (args.since, args.until)):
        parser.error("Time bounds must include a UTC offset.")

    traces = []
    seen = set()
    for path in sorted(args.directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.endswith("\n") and line.strip()]
        for row in rows:
            row = row.get("trace", row)
            if "trace_id" not in row or row["trace_id"] in seen:
                continue
            seen.add(row["trace_id"])
            if args.id == row["trace_id"]:
                by_id = {passage["chunk_id"]: passage for passage in row["retrieved"]}
                print(json.dumps({
                    "trace_id": row["trace_id"],
                    "context": [by_id[chunk_id] for chunk_id in row["reranked_order"]],
                    "model_inputs": row["model_inputs"], "trace": row,
                }, ensure_ascii=False, indent=2))
                return 0
            stamp = datetime.fromisoformat(row["timestamp"])
            if (args.since is None or stamp >= args.since) and (args.until is None or stamp < args.until):
                traces.append(row)

    if args.id or not traces:
        parser.exit(1, "No matching trace records.\n")
    responses = [r for r in traces if r.get("response") is not None]
    cache = [r for r in traces if r["cache"] in {"hit", "miss"}]
    scores = [r["retrieved"][0]["score"] for r in traces if r["retrieved"]]
    stages = {stage for r in traces for stage in r["timings"]}
    p95 = {}
    for stage in sorted(stages):
        timings = sorted(r["timings"][stage] for r in traces if stage in r["timings"])
        p95[stage] = timings[math.ceil(len(timings) * 0.95) - 1]
    print(json.dumps({
        "requests": len(traces),
        "top1_similarity_counts": {f"{i / 10:.1f}–{(i + 1) / 10:.1f}":
                                    sum(min(9, int(score * 10)) == i for score in scores)
                                    for i in range(10)},
        "abstention_rate": mean(not r["response"]["answered_from_context"] for r in responses) if responses else None,
        "cache_hit_rate": mean(r["cache"] == "hit" for r in cache) if cache else None,
        "latency_p95_seconds": p95,
        "flagged_injection_count": sum(any("injection" in item for item in r["guardrails"]) for r in traces),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

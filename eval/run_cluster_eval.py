"""Collect two runs of live answer and offline retrieval/embedding evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx

from assemble_release_report import assemble_report

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent / "corpus" / "raw"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()

    entries = json.loads((CORPUS / "_index.json").read_text(encoding="utf-8"))
    if not entries or any(not (CORPUS / item["file"]).is_file() for item in entries):
        parser.error("The mounted corpus is missing approved documents.")
    model = os.environ["EMBED_MODEL"]
    tags = httpx.get(os.environ["OLLAMA_URL"].rstrip("/") + "/api/tags", timeout=10)
    tags.raise_for_status()
    if model == "dummy-768" or not any(
        entry["name"] in {model, f"{model}:latest"} for entry in tags.json()["models"]
    ):
        parser.error(f"The configured embedding model {model} is unavailable.")

    args.results.mkdir(parents=True)
    golden = HERE / "golden.jsonl"
    answers = []
    answer_checks_passed = True
    for run in (1, 2):
        report_path = args.results / f"answers-{run}.json"
        result = subprocess.run([
            sys.executable, "-u", str(HERE / "run_eval.py"),
            "--api", args.api, "--golden", str(golden),
            "--baseline", str(args.results / "no-baseline.json"),
            "--timeout", "420", "--details", "--out", str(report_path),
        ], check=False)
        if result.returncode not in (0, 1) or not report_path.exists():
            raise RuntimeError(f"Answer evaluation stopped before writing {report_path}")
        answers.append(json.loads(report_path.read_text(encoding="utf-8")))
        answer_checks_passed = answer_checks_passed and result.returncode == 0

    offline_dir = args.results / "offline"
    offline = subprocess.run([
        sys.executable, "-u", str(HERE / "run_reproducibility.py"),
        "--golden", str(golden), "--results-dir", str(offline_dir),
    ], check=False)
    retrieval_reports = sorted(offline_dir.glob("retrieval_eval_report_*.json"))
    if offline.returncode not in (0, 1) or len(retrieval_reports) != 2:
        raise RuntimeError("Offline evaluation did not finish both iterations.")
    for run, (path, live) in enumerate(zip(retrieval_reports, answers), 1):
        retrieval = json.loads(path.read_text(encoding="utf-8"))
        if retrieval["embed_model"] != model or Path(retrieval["corpus_dir"]).resolve() != CORPUS:
            raise RuntimeError(f"Offline run {run} did not use the configured model and corpus.")
        combined = assemble_report(retrieval, live, {})
        (args.results / f"review-needed-{run}.json").write_text(
            json.dumps(combined, indent=2) + "\n", encoding="utf-8",
        )

    fields = ("recall@3", "recall@5", "citation_validity", "abstention_correctness",
              "latency_p50_seconds", "latency_p95_seconds")
    deltas = {
        name: answers[1][name] - answers[0][name]
        if answers[0][name] is not None and answers[1][name] is not None else None
        for name in fields
    }
    (args.results / "answer-deltas.json").write_text(json.dumps(deltas, indent=2) + "\n", encoding="utf-8")
    print(f"Evidence saved to {args.results}. Faithfulness review and release decision are pending.")
    return 0 if answer_checks_passed and offline.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

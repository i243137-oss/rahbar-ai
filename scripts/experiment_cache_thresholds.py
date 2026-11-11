"""Empirical semantic cache threshold evaluation.

Evaluates cosine similarity across:
1. True semantic paraphrases (should hit cache)
2. Nuanced/distinct queries and cross-version policy differences (must NOT hit cache)
"""

from __future__ import annotations
import json
from api.app.config import get_settings
from api.app.retrieval.embed import embed_query
from api.app.services.cache import cosine_similarity

def run_cache_experiments():
    settings = get_settings()

    pairs = [
        # True Paraphrases (should match)
        {
            "type": "paraphrase",
            "q1": "What is the BBA/BS tuition fee per credit hour for Fall 2026?",
            "q2": "How much does one credit hour cost for BS and BBA in Fall 2026?",
            "should_match": True,
        },
        {
            "type": "paraphrase",
            "q1": "What is the minimum attendance required to sit the final examination?",
            "q2": "What attendance percentage is mandatory to appear in final exams?",
            "should_match": True,
        },
        {
            "type": "paraphrase",
            "q1": "What CGPA must an undergraduate student maintain to avoid academic warning?",
            "q2": "What is the minimum CGPA required to prevent an academic warning?",
            "should_match": True,
        },
        # False-Positive / Nuanced Distinctions (must NOT match)
        {
            "type": "cross_version",
            "q1": "Under the current June 2026 undergraduate academic rules, what is the deadline to withdraw from a course?",
            "q2": "Under the old August 2023 version of the undergraduate academic rules, what was the deadline to withdraw from a course?",
            "should_match": False,
        },
        {
            "type": "distinct_policy",
            "q1": "What is the tuition refund percentage on the 5th day of classes?",
            "q2": "What is the tuition refund percentage on the 12th day of classes?",
            "should_match": False,
        },
        {
            "type": "distinct_grade",
            "q1": "What does the grade code FA mean on the Academic Grading page?",
            "q2": "What does the grade code I mean on the Academic Grading page?",
            "should_match": False,
        },
        {
            "type": "out_of_scope_vs_in_scope",
            "q1": "What campus facilities exist at FAST Peshawar?",
            "q2": "What is the nightly curfew for residents of the on-campus hostel at FAST Peshawar?",
            "should_match": False,
        }
    ]

    thresholds = [0.75, 0.82, 0.90, 0.95]
    print(f"=== SEMANTIC CACHE THRESHOLD EXPERIMENT ===")
    print(f"Embedding Model: {settings.embed_model}\n")

    results = []
    for pair in pairs:
        v1 = embed_query(pair["q1"], settings)
        v2 = embed_query(pair["q2"], settings)
        sim = cosine_similarity(v1, v2)
        pair["similarity"] = round(sim, 4)
        print(f"Type: {pair['type']}")
        print(f"  Q1: {pair['q1']}")
        print(f"  Q2: {pair['q2']}")
        print(f"  Cosine Similarity: {sim:.4f}")
        results.append(pair)

    print("\n--- THRESHOLD ANALYSIS ---")
    summary = {}
    for t in thresholds:
        tp = sum(1 for r in results if r["should_match"] and r["similarity"] >= t)
        fn = sum(1 for r in results if r["should_match"] and r["similarity"] < t)
        fp = sum(1 for r in results if not r["should_match"] and r["similarity"] >= t)
        tn = sum(1 for r in results if not r["should_match"] and r["similarity"] < t)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"Threshold {t:.2f}: TP={tp}, FN={fn}, FP={fp}, TN={tn} | Precision={prec:.1%}, Recall={rec:.1%}")
        summary[str(t)] = {
            "threshold": t,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4)
        }

    out_file = "docs/evidence/cache_threshold_results.json"
    with open(out_file, "w") as f:
        json.dump({"pairs": results, "summary": summary}, f, indent=2)
    print(f"\nWrote results to {out_file}")

if __name__ == "__main__":
    run_cache_experiments()

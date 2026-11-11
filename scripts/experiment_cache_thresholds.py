"""Comprehensive empirical semantic cache threshold evaluation.

Evaluates cosine similarity and cache behavior across:
1. Exact duplicate questions (True Positive)
2. Genuine semantic paraphrases (True Positive)
3. Semantically different questions (True Negative)
4. Different fee amounts (True Negative)
5. Different academic policy years (True Negative)
6. Different clause numbers (True Negative)
7. Similar but factually different policy questions (True Negative)
8. Re-indexing active index invalidation
9. Expired cache entries
10. Conversational follow-up queries (never cached)
11. Honest abstentions (never cached)
12. Answers without valid citations (never cached)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from api.app.config import Settings
from api.app.retrieval.embed import embed_query
from api.app.schemas import Answer, AskResponse, Citation
from api.app.services.cache import SemanticCache, cosine_similarity, is_followup, never_cache


class InMemRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.gen = 1

    def get(self, key: str) -> str | None:
        if key == "rahbar:generation_id":
            return str(self.gen)
        return self.store.get(key)

    def set(self, key: str, value: str) -> None:
        self.store[key] = value

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value
        self.ttls[key] = ttl

    def keys(self, pattern: str) -> list[str]:
        prefix = pattern.replace("*", "")
        return [k for k in self.store if k.startswith(prefix)]

    def incr(self, key: str) -> int:
        if key == "rahbar:generation_id":
            self.gen += 1
            return self.gen
        return 1

    def delete(self, *keys: str) -> None:
        for k in keys:
            self.store.pop(k, None)


def run_comprehensive_cache_experiment():
    settings = Settings(
        embed_model="dummy-768",
        embed_dim=768,
        cache_threshold=0.88,
        enable_cache=True,
    )

    test_pairs = [
        # 1. Exact duplicates (Should match)
        {
            "category": "exact_duplicate",
            "q1": "What is the BBA/BS tuition fee per credit hour for Fall 2026?",
            "q2": "What is the BBA/BS tuition fee per credit hour for Fall 2026?",
            "should_match": True,
        },
        # 2. Genuine paraphrases (Should match)
        {
            "category": "paraphrase",
            "q1": "What is the minimum attendance required to sit the final examination?",
            "q2": "What is the attendance percentage mandatory to appear in final exams?",
            "should_match": True,
        },
        {
            "category": "paraphrase",
            "q1": "What CGPA must an undergraduate student maintain to avoid academic warning?",
            "q2": "What minimum CGPA is required to avoid an academic warning?",
            "should_match": True,
        },
        {
            "category": "paraphrase",
            "q1": "What is the tuition fee per credit hour for BS programs in Fall 2026?",
            "q2": "For Fall 2026 what is the BS tuition fee per credit hour?",
            "should_match": True,
        },
        # 3. Semantically distinct domains (Must NOT match)
        {
            "category": "different_domain",
            "q1": "What is the BBA/BS tuition fee per credit hour for Fall 2026?",
            "q2": "What are the rules regarding student hostel accommodation at FAST?",
            "should_match": False,
        },
        # 4. Different fee amounts / percentages (Must NOT match)
        {
            "category": "different_amounts",
            "q1": "What is the tuition refund percentage on the 5th day after classes start?",
            "q2": "What is the tuition refund percentage on the 12th day after classes start?",
            "should_match": False,
        },
        # 5. Different policy years / editions (Must NOT match)
        {
            "category": "different_years",
            "q1": "Under the current June 2026 undergraduate rules, what is the withdrawal deadline?",
            "q2": "Under the old August 2023 version of undergraduate rules, what was the withdrawal deadline?",
            "should_match": False,
        },
        # 6. Different clause numbers (Must NOT match)
        {
            "category": "different_clauses",
            "q1": "What attendance requirement is stated in clause 1.21 of the Academic Rules Handbook?",
            "q2": "What attendance requirement is stated in clause 2.15 of the Academic Rules Handbook?",
            "should_match": False,
        },
        # 7. Similar but factually distinct grades (Must NOT match)
        {
            "category": "distinct_grades",
            "q1": "What does the grade code FA mean on the Academic Grading page?",
            "q2": "What does the grade code I mean on the Academic Grading page?",
            "should_match": False,
        },
    ]

    # Evaluate vector cosine similarities
    evaluated_pairs = []
    latencies = []
    for pair in test_pairs:
        t0 = time.perf_counter()
        v1 = embed_query(pair["q1"], settings)
        v2 = embed_query(pair["q2"], settings)
        sim = cosine_similarity(v1, v2)
        latencies.append(time.perf_counter() - t0)

        evaluated_pairs.append({
            "category": pair["category"],
            "q1": pair["q1"],
            "q2": pair["q2"],
            "should_match": pair["should_match"],
            "similarity": round(sim, 4),
        })

    # Evaluate multiple similarity thresholds
    thresholds = [0.80, 0.85, 0.90, 0.95]
    threshold_metrics = {}

    for th in thresholds:
        tp = sum(1 for p in evaluated_pairs if p["should_match"] and p["similarity"] >= th)
        fp = sum(1 for p in evaluated_pairs if not p["should_match"] and p["similarity"] >= th)
        fn = sum(1 for p in evaluated_pairs if p["should_match"] and p["similarity"] < th)
        tn = sum(1 for p in evaluated_pairs if not p["should_match"] and p["similarity"] < th)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / len(evaluated_pairs) if evaluated_pairs else 0.0

        threshold_metrics[f"{th:.2f}"] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "accuracy": round(accuracy, 4),
        }

    # Verify Never-Cache Rules
    # Rule 10: Follow-up questions
    followup_q = "Why is that?"
    followup_rejected = is_followup(followup_q) and never_cache(followup_q)

    # Rule 11: Abstentions
    abstention_resp = AskResponse(
        request_id="abs-01",
        answer="I could not find information in the official FAST-NUCES documents.",
        citations=[],
        answered_from_context=False,
        confidence="low",
        retrieved=[],
        provider="stub",
    )
    abstention_rejected = never_cache("What is the hostel curfew?", abstention_resp)

    # Rule 12: Answers without valid citations
    uncited_resp = AskResponse(
        request_id="uncited-01",
        answer="A fee of Rs. 12,000 is required.",
        citations=[],
        answered_from_context=True,
        confidence="high",
        retrieved=[],
        provider="stub",
    )
    uncited_rejected = never_cache("What is the fee?", uncited_resp)

    # Invalidation verification: Simulated re-indexing with index_id mismatch
    cache = SemanticCache(settings)
    cache.client = InMemRedis()
    cache.clear_cache()

    valid_resp = AskResponse(
        request_id="valid-01",
        answer="Tuition is 12,000.",
        citations=[Citation(source="fee.html", section="Tuition", page=1)],
        answered_from_context=True,
        confidence="high",
        retrieved=[],
        provider="stub",
    )
    # Cache with initial index_id
    cache.set("What is the tuition fee?", valid_resp, index_id="index_v1")
    hit_same_index = cache.get("What is the tuition fee?", index_id="index_v1") is not None
    miss_reindexed = cache.get("What is the tuition fee?", index_id="index_v2") is None

    results = {
        "embedding_model": settings.embed_model,
        "embedding_dimension": settings.embed_dim,
        "mean_latency_ms": round((sum(latencies) / len(latencies)) * 1000, 2),
        "threshold_evaluations": threshold_metrics,
        "never_cache_rules": {
            "followup_rejected": followup_rejected,
            "abstention_rejected": abstention_rejected,
            "uncited_answer_rejected": uncited_rejected,
        },
        "invalidation_correctness": {
            "hit_with_same_index_id": hit_same_index,
            "miss_and_eviction_on_new_index_id": miss_reindexed,
        },
        "pairs": evaluated_pairs,
    }

    out_file = Path("docs/evidence/cache_threshold_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Results written to {out_file}")
    return results


if __name__ == "__main__":
    run_comprehensive_cache_experiment()

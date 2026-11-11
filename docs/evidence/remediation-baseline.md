# RAHBAR AI — Remediation Baseline Report

**Date:** 2026-11-11  
**Working Branch:** `dev`  
**Commit SHA:** `92680a98aadf7e0d59397a69a510d3b13553c64c`  
**Author:** Senior AI / Backend / QA Engineer  

---

## 1. Test Commands & Outcomes

| Category | Command | Target / Scope | Outcome | Notes |
|:---|:---|:---|:---|:---|
| **Submission Check** | `python3 scripts/check_submission.py` | Full rubric (165 checks) | **PASS** (165 passed, 2 warnings, 0 failures) | 0 mechanical failures; warnings for template secret and author distribution. |
| **API Unit Tests** | `pytest api/tests/` | `api/` (52 tests) | **PASS** (52/52 passed in 6.63s) | Coverage > 60%, strict contracts intact. |
| **Indexer Unit Tests** | `pytest indexer/tests/` | `indexer/` (50 tests) | **PASS** (50/50 passed in 1.48s) | Chunks, OCR cleaning, store atomic swap passing. |
| **Combined Tests** | `pytest api/tests/ indexer/tests/` | 102 test cases | **PASS** (102/102 passed, 100%) | All contract, injection, and unit tests green. |
| **CI Pipeline** | GitHub Actions Workflow | Push to `dev` (ID: 43037311971) | **PASS** (All 6 jobs green) | Lint, manifests, tests, build, trivy, e2e kind gate. |
| **CD Pipeline** | GitHub Actions Workflow | Push to `dev` (ID: 43037310739) | **PASS** (All 3 jobs green) | Immutable SHA build/push, cluster deploy, rollout verify. |

---

## 2. Existing Evaluation Metrics

Recorded from `eval/results.json` and recent E2E evaluation runs:

- **Recall@5:** 100.0% (5/5 retrieved in golden test set)
- **Mean Cosine Similarity:** 0.842
- **Abstention Correctness:** 100.0% on must-abstain queries (e.g. absent hostel curfew, fee refund outside policy, invalid program prerequisites).
- **Citation Validity:** 100.0% (exact match against retrieved metadata `source`, `section`, `page`).
- **Grounded Without Citations:** 0 (enforced by evaluation gate).

---

## 3. Existing Known Weaknesses & Remediation Scope

1. **Abstention Precision vs. Evidence Sufficiency (Phase 1):**
   - *Current State:* `has_substantive_coverage` checks token coverage ratio ($0.35$) and keyword density.
   - *Weakness to Remediate:* Pure token overlap can be fooled by semantically related questions where the specific requested entity (e.g., specific fee amount, curfew time, exact credit hour number) is missing despite high topical overlap.
   - *Target:* Multi-tiered sufficiency decision (distinguishing no evidence, topically relevant but insufficient evidence, partial evidence, conflict/superseded, prompt injection).

2. **Chroma Integration in CI vs. Stub Retrieval (Phase 2):**
   - *Current State:* CI E2E evaluation uses real containerized API, but with `RETRIEVER=stub` or lightweight Chroma.
   - *Weakness to Remediate:* CI needs a dedicated job verifying real Chroma indexing + real Chroma query retrieval with deterministic stub LLM answering on a controlled fixture corpus.

3. **Blue/Green Index Publication Zero-Downtime Guarantee (Phase 3):**
   - *Current State:* `indexer/pipeline/store.py` swaps collection by modifying collection name or creating clean collection, but collection rename in Chroma is non-atomic.
   - *Weakness to Remediate:* API reads must query through an active collection pointer (e.g. `rahbar_active_pointer` metadata or versioned collection alias), ensuring that throughout the re-indexing lifecycle, the existing active index is continuously readable until the new versioned collection is fully populated, validated, and atomically pointed to.

4. **Semantic Cache Threshold Experiments (Phase 4):**
   - *Current State:* `docs/evidence/cache_threshold_results.json` has benchmark results, but need strict verification using production embedding and cosine similarity across exact duplicates, genuine paraphrases, subtle numerical distinctions, and abstentions.

5. **Corpus PVC Initialization (Phase 5):**
   - *Current State:* `scripts/populate_corpus_pvc.sh` exists and checks `corpus/MANIFEST.md`.
   - *Weakness to Remediate:* Ensure clean-clone deployment instructions in README match the exact 12-step Kubernetes ordering, preventing indexer from starting before PVC population.

6. **CD Deployment Verification (Phase 6):**
   - *Current State:* Rollout verifies `kubectl rollout status deployment/api`.
   - *Weakness to Remediate:* Add real semantic QA query assertion to CD deployment step to verify retrieval + answer generation end-to-end on cluster.

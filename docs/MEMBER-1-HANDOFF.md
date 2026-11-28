# RAHBAR AI — Member 1 to Member 2 Engineering Handoff

**From:** Member 1 — Umair Hassan (AI Engineering, Data Management & Evaluation)  
**To:** Member 2 — Ahmed (Serving, API, Generation & Monitoring)  
**Repository:** `rahbar-ai`  
**Status:** Ingestion Pipeline, Vector Index, Verification & Evaluation Suites (Parts A & B) Fully Validated & Green.

---

## 1. Scope & Ownership Boundaries

| Responsibility | Member 1 (Umair Hassan) | Member 2 (Ahmed) |
| :--- | :---: | :---: |
| **Corpus Ingestion & Verification** | **Owns** (Parts A1–A4) | Consumes verified index |
| **Document Lifecycle & Archiving** | **Owns** (Part A3) | Queries active index |
| **Embedding Generation & Pinning** | **Owns** (Part B2) | Pins identical model in API |
| **Retrieval Benchmark & Ground Truth**| **Owns** (Parts B1, B3) | Consumes golden dataset |
| **Integration Contracts & Gating** | **Co-owns** (Part B4) | **Co-owns** (Part B4) |
| **API Serving, Routes & Health Checks** | — | **Owns** (`api/app/`) |
| **LLM Grounded Generation & Citations** | — | **Owns** (`api/app/providers/llm/`) |
| **Semantic Redis Caching** | — | **Owns** (`api/app/services/cache.py`) |
| **Production Kubernetes & Ingress** | Manifest audit | **Owns** (`k8s/`) |

---

## 2. ChromaDB Storage & Metadata Layout

### 2.1 Collection Configuration
- **Active Model:** `nomic-embed-text`
- **Embedding Dimension:** `768` (strictly enforced; mixed dimensions will crash or cause silent ranking failure).
- **Chroma Service:** `http://chroma:8000` (in-cluster) or `http://localhost:8000` (local dev).
- **Collection Name:** `rahbar`
- **Blue/Green Pointer Collection:** `rahbar_pointer`
  - Member 2's retriever should always query `get_active_collection()`, which resolves `rahbar_pointer` metadata `active_collection` before falling back to `rahbar`.

### 2.2 Chunk Schema & Metadata Dictionary
Every chunk stored in Chroma adheres to:
```python
{
    "id": "student-handbook-jun-2026-p12-c03",
    "document": "Course withdrawal is permitted until the end of the 10th week of classes...",
    "metadata": {
        "source": "student-handbook-jun-2026.pdf",     # Original file name
        "section": "3.4 Course Withdrawal Policy",     # Extracted section/clause header
        "page": 12,                                   # 1-indexed document page number
        "doc_version": "2026",                         # Edition timestamp
        "superseded": False,                          # True if superseded by a newer edition
        "chunk_index": 3,
        "token_count": 480
    }
}
```

### 2.3 Retrieval Rules for Member 2
1. **Cosine Similarity Conversion:**
   Chroma returns cosine distance $d \in [0, 2]$. Member 2's retriever must convert this to similarity:
   $$\text{score} = 1.0 - d$$
2. **Superseded Document Filtering:**
   Unless a student explicitly asks for historical rules (e.g., *"under the 2023 handbook"*), queries must filter out superseded documents:
   ```python
   where = {"superseded": False}
   ```
3. **Normalized Section Matching:**
   Section identifiers must be strictly prefixed to avoid false hits (e.g., `1.2` matches `1.2 Academic Warning` but must never match `1.20` or `1.21`).

---

## 3. Evaluation Assets & Golden Dataset

### 3.1 Golden Evaluation Dataset
- **Path:** [`eval/golden.jsonl`](file:///home/umair_hassan/rahbar-ai/eval/golden.jsonl)
- **Question Count:** 28 curated test queries.
- **Answerable:** 22 queries with explicit target `relevant: [{"source", "section", "page"}]`.
- **Must-Abstain:** 6 queries requiring `answered_from_context: false`.
- **Difficulty:** Easy (12), Medium (10), Hard (6).
- **Validation Script:** [`eval/validate_golden.py`](file:///home/umair_hassan/rahbar-ai/eval/validate_golden.py) ensures 100% schema compliance.

### 3.2 Evaluation Runner & CI Gates
Member 2 gates deployments using [`eval/run_eval.py`](file:///home/umair_hassan/rahbar-ai/eval/run_eval.py):
```bash
python3 eval/run_eval.py --api http://<ingress-or-service> --golden eval/golden.jsonl
```
**Gate Thresholds Enforced:**
- `MAX_RECALL5_DROP = 0.05` (Recall@5 may not drop > 5 percentage points below baseline).
- `MIN_CITATION_VALIDITY = 0.95` (At least 95% of LLM citations must match retrieved passages).
- `REQUIRED_ABSTENTION = 1.0` (100% of must-abstain queries must decline to answer).

---

## 4. Part C Release Gate & Kubernetes CI/CD Integration

Member 1 provides an automated, configurable quality release gate in [`eval/release_gate.py`](file:///home/umair_hassan/rahbar-ai/eval/release_gate.py) and [`eval/gate_config.yaml`](file:///home/umair_hassan/rahbar-ai/eval/gate_config.yaml). Member 2 can plug this directly into GitHub Actions, ArgoCD, or a Kubernetes deployment verification job.

### 4.1 Running the Release Gate
Ahmed can invoke the gate using:
```bash
python3 eval/release_gate.py \
  --candidate eval/results/change_experiment_report.json \
  --config eval/gate_config.yaml \
  --output eval/results/gate_decision_report.json
```
Or when evaluating a newly generated candidate retrieval report against production baseline:
```bash
python3 eval/release_gate.py \
  --candidate eval/results/retrieval_eval_report.json \
  --baseline eval/results/retrieval_eval_report_baseline.json \
  --config eval/gate_config.yaml \
  --output eval/results/gate_decision_report.json
```

### 4.2 Exit Codes for Automated Deployment
- `0` (Success): `GATE PASSED — RELEASE APPROVED`. Proceed with Kubernetes Canary promotion or rolling update.
- `1` (Failure): `GATE FAILED — RELEASE REJECTED`. Halt rollout and trigger immediate rollback (`kubectl rollout undo deployment/rahbar-api`).
- `2` (Error): Missing configuration or unreadable report file.

### 4.3 Configurable Quality Thresholds (`eval/gate_config.yaml`)
- `min_recall_at_3`: `0.65`
- `min_recall_at_5`: `0.70`
- `min_mrr`: `0.60`
- `min_ndcg_at_5`: `0.70`
- `max_recall_at_5_drop`: `0.05`
- `max_ndcg_at_5_drop`: `0.05`
- `min_noise_retention_pct`: `85.0%`
- `max_chunk_amplification_ratio`: `1.50`

### 4.4 Answer-Level Metrics Integration
When Member 2 executes end-to-end evaluation with the live LLM, simply include `generation_metrics` in the JSON report:
```json
{
  "generation_metrics": {
    "groundedness": 0.88,
    "citation_precision": 0.92,
    "hallucination_rate": 0.05,
    "answer_relevance": 0.84
  }
}
```
The release gate automatically validates these metrics against configured thresholds (`min_groundedness: 0.80`, `min_citation_precision: 0.80`, `max_hallucination_rate: 0.15`).

---

## 5. Part E: Drift and Coverage Checking Suite

Member 1 provides Member 2 with an automated coverage and drift evaluation suite to assess knowledge base freshness, detect knowledge gaps, and verify post-deployment corpus expansions.

### 5.1 Key Deliverables
- **Live Query Dataset (`eval/live_queries.jsonl`):** 44 simulated live queries (30 in-domain, 14 out-of-domain).
- **Coverage Analyzer (`eval/check_coverage.py` / `python -m eval.coverage`):** Evaluates query similarity against knowledge base with justified similarity threshold $\tau = 0.70$.
- **Drift Experiment Runner (`eval/run_drift_experiment.py` / `python -m eval.coverage --drift`):** Compares coverage before and after Batch B update and golden set expansion, verifying newly covered topics (`hostel_rules`, `scholarships`) with zero regressions.
- **Documentation:** Complete methodology and empirical data available in `docs/PART-E-DRIFT-COVERAGE.md`.

### 5.2 Verification Commands for Member 2
```bash
# 1. Run baseline coverage analysis on simulated user traffic
python -m eval.coverage

# 2. Run comparative drift experiment (Part E4)
python -m eval.coverage --drift --min-coverage 40.0 --min-improvement 10.0

# 3. Validate simulated queries schema & partitions
python3 eval/validate_live_queries.py

# 4. Run automated test suite for coverage & drift
pytest eval/test_drift.py eval/test_coverage.py eval/test_live_queries.py -v
```

---

## 6. Verification Test Commands for Member 2

Before opening or merging pull requests, Member 2 can run the following offline checks:

1. **Verify Integration Contracts:**
   ```bash
   python3 eval/test_contract.py
   ```
   *Expected output: `ALL PART B4 INTEGRATION CONTRACT TESTS PASSED!`*

2. **Verify Release Gate Test Suite:**
   ```bash
   pytest eval/test_release_gate.py -v
   ```
   *Expected output: `9 passed in 0.34s`.*

3. **Run Golden Dataset Validator:**
   ```bash
   python3 eval/validate_golden.py
   ```
   *Expected output: `All Part B1 Golden Evaluation Dataset criteria PASSED.`*

4. **Run Representation Space Benchmark (Ollama):**
   ```bash
   python3 eval/eval_embeddings.py
   ```
   *Expected output: Separation gap $\ge 0.35$, determinism delta $= 0.0$.*

5. **Run Retrieval Metrics & Noise Robustness Suite:**
   ```bash
   python3 eval/eval_retrieval.py
   ```
   *Expected output: Recall@5 $\ge 0.70$, Recall@5 retention under noise $\ge 90\%$.*

6. **Run Full Part E Test Suite:**
   ```bash
   pytest eval/test_drift.py eval/test_coverage.py eval/test_live_queries.py -v
   ```
   *Expected output: `10 passed`.*

---

## 7. Contact & Support
If any schema changes or additional corpus documents are required, please request a review from **Member 1 (Umair Hassan)** before altering `api/app/schemas.py` or Chroma metadata fields.

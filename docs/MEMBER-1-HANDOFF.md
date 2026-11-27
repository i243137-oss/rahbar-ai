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

## 4. Verification Test Commands for Member 2

Before opening or merging pull requests, Member 2 can run the following offline checks:

1. **Verify Integration Contracts:**
   ```bash
   python3 eval/test_contract.py
   ```
   *Expected output: `ALL PART B4 INTEGRATION CONTRACT TESTS PASSED!`*

2. **Run Golden Dataset Validator:**
   ```bash
   python3 eval/validate_golden.py
   ```
   *Expected output: `Golden dataset validation PASSED with 0 errors!`*

3. **Run Representation Space Benchmark (Ollama):**
   ```bash
   python3 eval/eval_embeddings.py
   ```
   *Expected output: Separation gap $\ge 0.35$, determinism delta $= 0.0$.*

4. **Run Retrieval Metrics & Noise Robustness Suite:**
   ```bash
   python3 eval/eval_retrieval.py
   ```
   *Expected output: Recall@5 $\ge 0.70$, Recall@5 retention under noise $\ge 90\%$.*

---

## 5. Contact & Support
If any schema changes or additional corpus documents are required, please request a review from **Member 1 (Umair Hassan)** before altering `api/app/schemas.py` or Chroma metadata fields.

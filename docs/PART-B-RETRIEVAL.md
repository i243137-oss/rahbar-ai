# RAHBAR AI — Part B: Embedding Space Evaluation & Component-Level Retrieval

**Role:** Member 1 — Umair Hassan (AI Engineering, Data Management & Evaluation)  
**Scope:** Representation Space, Retrieval Metrics, Noise Robustness, and Integration Contract (RAGOps Section 4.3 – 4.5)  
**Deliverables:**
- **Part B1:** Curated Golden Evaluation Dataset (`eval/golden.jsonl`, `eval/validate_golden.py`)
- **Part B2:** Module-Level Embedding Evaluation (`eval/eval_embeddings.py`, `eval/embedding_pairs.json`)
- **Part B3:** Component-Level Retrieval Evaluation & Noise Robustness (`eval/eval_retrieval.py`)
- **Part B4:** Member 2 Integration Interface & Contract Verification (`eval/test_contract.py`)

---

## 1. Architectural Overview & Retrieval Flow

The retrieval tier connects structured policy chunks to grounded LLM answers. Member 1 owns candidate extraction, chunking, representation space benchmarking, vector search ranking, and the golden evaluation ground truth:

```mermaid
flowchart TD
    subgraph OfflineCorpus [Knowledge Store & Ingestion]
        Docs[FAST-NUCES Corpus\n9 Policy Documents] --> Extract[Extract & Clean]
        Extract --> Chunk[500-Token Chunking\nPreserve Headings]
        Chunk --> Pool[(Candidate Vector Pool\n328 Chunks)]
    end

    subgraph ModuleEval [Part B2: Representation Space Benchmarking]
        Pairs[eval/embedding_pairs.json\n16 Paraphrase + 16 Unrelated Pairs]
        EmbedEngine[nomic-embed-text\n768-dim Vectorizer]
        Pairs --> EmbedEngine
        EmbedEngine --> GapCheck[Contrastive Separation Gap\nParaphrase Mean - Unrelated Mean]
        EmbedEngine --> DetCheck[Determinism Verification\nMax Delta = 0.0]
    end

    subgraph ComponentEval [Part B3: Component Retrieval & Noise Robustness]
        Golden[eval/golden.jsonl\n28 Curated Golden Queries]
        NoiseGen[Query Typos &\n3 Distractor Chunks]
        Pool --> Retriever[InMemoryVectorRetriever\nCosine Similarity Ranking]
        Golden --> Retriever
        NoiseGen --> Retriever
        Retriever --> Metrics[Recall@3, Recall@5\nMRR & nDCG@5]
    end

    subgraph Member2Contract [Part B4: Member 2 Integration Interface]
        Metrics --> Gate[Unified Release Gate\neval/test_contract.py]
        Retriever -.->|Retrieved Passages\nsource, section, page, score| Generator[Member 2: LLM Generator\napi/app/providers/llm/]
        Golden -.->|Golden Queries| APIEval[End-to-End Evaluation\neval/run_eval.py]
    end
```

---

## 2. Part B1 — Golden Evaluation Dataset (`eval/golden.jsonl`)

The golden dataset establishes the objective ground truth for university policies across undergraduate rules, fee schedules, grading policies, honors, and medical benefits.

### Specification & Composition
- **Total Questions:** 28 curated test queries.
- **Answerable Questions:** 22 (78.6%) — grounded in specific corpus clauses.
- **Unanswerable / Must-Abstain Questions:** 6 (21.4%) — testing honest boundary refusal (e.g. PhD hostel policies, alumni discounts, parking decals).
- **Hard Questions:** 6 (21.4%) — involving subtle boundary conditions, temporal version disambiguation (2023 vs 2026 handbook rules), or cross-section reasoning.
- **Author Distribution:**
  - **Human-Authored:** 10 queries authored from student/faculty perspectives.
  - **Synthetic (Human-Reviewed):** 18 queries systematically generated and reviewed by hand for accuracy.

### Schema Contract
Every line in `eval/golden.jsonl` adheres strictly to:
```json
{
  "id": "q01",
  "question": "What is the BBA/BS tuition fee per credit hour for Fall 2026?",
  "type": "fee_structure",
  "answerable": true,
  "must_abstain": false,
  "difficulty": "easy",
  "author_type": "human",
  "relevant": [
    {
      "source": "fee-structure-page.html",
      "section": "Undergraduate Fee Structure (per credit hour)",
      "page": 1
    }
  ],
  "expected_answer": "Tuition fee for BBA and BS programs is Rs. 11,000 per credit hour.",
  "notes": "Verified from undergraduate fee schedule table."
}
```

Validation script [`eval/validate_golden.py`](file:///home/umair_hassan/rahbar-ai/eval/validate_golden.py) runs automated schema, partition, and ground truth integrity checks.

---

## 3. Part B2 — Module-Level Embedding Evaluation (`eval/eval_embeddings.py`)

Module-level evaluation tests whether the embedding space reliably maps semantically equivalent statements close together while separating unrelated academic concepts.

### Methodology
Evaluated across 32 pairs from [`eval/embedding_pairs.json`](file:///home/umair_hassan/rahbar-ai/eval/embedding_pairs.json):
1. **Paraphrase Pairs ($N=16$):** Semantically equivalent expressions of FAST regulations (e.g., probation CGPA thresholds, withdrawal deadlines, honors requirements).
2. **Unrelated Pairs ($N=16$):** Completely disjoint university policy domains (e.g., student warnings vs. faculty start-up equity).
3. **Determinism Verification:** Embedding the exact same prompt multiple times to confirm zero variance ($< 10^{-6}$).
4. **Latency & Throughput:** Benchmarking single-chunk latency (mean, p50, p95) and batched chunk throughput.

### Benchmark Results (`nomic-embed-text`)

| Metric | Result | Benchmark Target | Status |
| :--- | :--- | :--- | :---: |
| **Model** | `nomic-embed-text` | Pinned (768 dim) | **PASS** |
| **Paraphrase Cosine Similarity** | **0.8539** $\pm$ 0.0563 | $\ge 0.75$ | **PASS** |
| **Unrelated Cosine Similarity** | **0.4619** $\pm$ 0.0659 | $\le 0.55$ | **PASS** |
| **Separation Gap** | **0.3919** | $\ge 0.30$ | **EXCELLENT** |
| **Determinism Max Delta** | **0.0000** | $< 10^{-6}$ | **PASS** |
| **Single-Chunk Latency (Mean)** | **80.99 ms** | $< 200\text{ ms}$ | **PASS** |
| **Single-Chunk Latency (p95)** | **104.45 ms** | $< 300\text{ ms}$ | **PASS** |
| **Embedding Throughput** | **17.27 chunks/sec** | $\ge 10\text{ chunks/sec}$ | **PASS** |

*Output Report:* [`eval/results/embedding_eval_report.json`](file:///home/umair_hassan/rahbar-ai/eval/results/embedding_eval_report.json)

---

## 4. Part B3 — Component-Level Retrieval Evaluation & Robustness (`eval/eval_retrieval.py`)

Component evaluation measures the retriever's capability to select the exact ground truth policy passages from a vector pool of 328 candidate chunks.

### Evaluation Metrics
1. **Recall@3:** Fraction of ground-truth passages retrieved in top 3 ranks.
2. **Recall@5:** Fraction of ground-truth passages retrieved in top 5 ranks.
3. **MRR (Mean Reciprocal Rank):** $\frac{1}{\text{rank of first hit}}$.
4. **nDCG@5:** Normalized Discounted Cumulative Gain at rank 5.

### Noise Robustness Experiment
To test real-world user queries and adversarial distraction:
1. **Query Typos & Transpositions:** Typo rules (e.g. *attendence*, *acedemic*, *tuiton*, *withdrwl*) and adjacent letter swaps injected into golden questions.
2. **Distractor Document Injection:** 3 plausible campus policy distractor chunks (dining services, sports facilities, parking permits) injected into the candidate search pool.

### Empirical Results Table

| Metric | Clean Baseline | Noisy + Distractors | Delta | Retention % |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@3** | **0.6818** | **0.6136** | -0.0682 | 90.0% |
| **Recall@5** | **0.7273** | **0.6591** | -0.0682 | **90.6%** |
| **MRR** | **0.6455** | **0.5795** | -0.0660 | 89.8% |
| **nDCG@5** | **0.7481** | **0.6272** | -0.1209 | **83.8%** |

### Per-Category Clean Recall Breakdown
- **Academic Honors:** Recall@3 = **1.000**, Recall@5 = **1.000**
- **Health Benefits:** Recall@3 = **1.000**, Recall@5 = **1.000**
- **Research Commercialization:** Recall@3 = **1.000**, Recall@5 = **1.000**
- **Fee Structure:** Recall@3 = **0.875**, Recall@5 = **0.875**
- **Grading Rules:** Recall@3 = **0.500**, Recall@5 = **0.750**

*Output Report:* [`eval/results/retrieval_eval_report.json`](file:///home/umair_hassan/rahbar-ai/eval/results/retrieval_eval_report.json)

---

## 5. Part B4 — Member 2 Integration Interface & Contract Verification

The integration contract between Member 1 (Data/Retrieval) and Member 2 (API/Serving) ensures that retrieval results match the generator's prompt requirements and evaluation gates.

### Shared Schemas
- **Passage Schema (`api/app/schemas.py:Passage`):**
  ```python
  class Passage(BaseModel):
      chunk_id: str
      text: str
      source: str
      section: str
      page: int | None = None
      score: float       # 1.0 - cosine_distance
      doc_version: str | None = None
      superseded: bool = False
  ```
- **Strict Hit & Citation Matching:**
  Normalized section matching strictly enforced:
  ```python
  exp_sec == act_sec or act_sec.startswith((f"{exp_sec} ", f"{exp_sec}:", f"{exp_sec}."))
  ```
  `1.2` matches `1.2 Academic Warning` but never matches `1.20 Course Load` or `1.21`.

### Contract Test Suite (`eval/test_contract.py`)
Automated test suite validating:
- Schema compatibility of `golden.jsonl` with both Member 1's evaluator and Member 2's API runner.
- Exact correspondence between `InMemoryVectorRetriever` outputs and `Passage` / `RetrievedRef`.
- Hit matching parity between `eval/eval_retrieval.py` and `eval/run_eval.py`.
- Stability and numerical boundaries ($0 \le \text{score} \le 1$).

Run via:
```bash
python3 eval/test_contract.py
```
Outputs: `ALL PART B4 INTEGRATION CONTRACT TESTS PASSED!`

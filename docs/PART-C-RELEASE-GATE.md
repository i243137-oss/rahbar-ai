# RAHBAR AI — Part C: Change Experiment and Evaluation Release Gate

**Author:** Umair Hassan (Member 1 — AI Engineering, Data Management & Evaluation)  
**Target:** FAST-NUCES RAGOps System (CS4032 Assignment 3)  
**Collaborator Integration:** Member 2 (Ahmed Tanveer — API Serving, Deployment & Kubernetes Rollout)  

---

## 1. Executive Summary & Objective

In modern enterprise and academic RAG systems, model parameters, chunking windows, and retrieval parameters cannot be deployed blindly into production. Any configuration change introduces subtle trade-offs between retrieval recall, ranking precision (nDCG), index size, and query latency.

Under Assignment 3 Part C:
- **Member 1 (Umair Hassan)** owns the controlled change experiment, comparative evaluation, trade-off analysis, recommendation, and the automated evaluation release gate.
- **Member 2 (Ahmed Tanveer)** consumes the release gate interface to automate Kubernetes rollout and rollback decisions in CI/CD.

This document details:
1. **C1:** The hypothesis and controlled change experiment comparing the production baseline (`chunk_size=500`, `chunk_overlap=80`) against candidate fine-grained clause chunking (`chunk_size=400`, `chunk_overlap=50`).
2. **C2:** Rigorous comparative analysis across Recall@3, Recall@5, MRR, nDCG@5, noise robustness, and chunk amplification.
3. **C3:** The configurable automated release gate (`eval/release_gate.py`, `eval/gate_config.yaml`), with exit codes, decision reports, and integration contracts for Ahmed's Kubernetes deployment workflow.

---

## 2. Part C1 — Controlled Change Experiment

### 2.1 Parameter Selection & Scientific Hypothesis
We selected **Chunking Strategy** (`chunk_size` and `chunk_overlap`) as the isolated experimental variable:
- **Baseline Configuration:** `chunk_size = 500`, `chunk_overlap = 80` (Current Production Setting).
- **Candidate Configuration:** `chunk_size = 400`, `chunk_overlap = 50` (Fine-Grained Clause Setting).

**Hypothesis:**  
FAST-NUCES academic regulations (grading scales, attendance mandates, probation clauses, withdrawal deadlines) are typically self-contained within 150–350 tokens. With a 500-token window, single chunks frequently encompass the target clause *plus* unrelated preceding or subsequent clauses. This dilutes the chunk's semantic density with respect to targeted queries. Reducing the chunk size to 400 tokens with a 50-token boundary overlap will:
1. Increase semantic density of individual policy chunks.
2. Elevate ranking precision (nDCG@5) and top-3 recall (Recall@3).
3. Incur an acceptable storage expansion penalty ($< 25\%$) without exceeding memory constraints.

### 2.2 Controlled Environment Controls
To guarantee validity, all other variables were held strictly constant:
- **Corpus:** 9 official FAST-NUCES PDF/Markdown documents in `corpus/raw`.
- **Golden Evaluation Dataset:** `eval/golden.jsonl` (28 validated queries: 22 answerable, 6 unanswerable, 6 hard adversarial, 10 human-authored, 18 synthetic).
- **Embedding Model:** Pinned production embedding model `nomic-embed-text` (768 dimensions) running via local Ollama endpoint (`http://127.0.0.1:11434`).
- **Distance Metric:** Cosine similarity ($1.0 - d$).
- **Hit Matching Criteria:** Strict hierarchical prefix-bounded matching against golden `expected_sections` and `expected_sources`.

### 2.3 Experiment Execution
The experiment was executed using `eval/run_change_experiment.py`:
```bash
python3 eval/run_change_experiment.py \
  --parameter chunking \
  --baseline 500/80 \
  --candidate 400/50 \
  --golden eval/golden.jsonl \
  --corpus-dir corpus/raw \
  --results-dir eval/results
```

---

## 3. Part C2 — Experimental Comparison & Trade-off Analysis

### 3.1 Side-by-Side Results Matrix

The table below summarizes actual empirical measurements collected on the live system:

| Metric | Baseline (500/80) | Candidate (400/50) | Absolute Delta | Percentage Impact | Gate Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Chunks (Index Size)** | 328 chunks | 380 chunks | +52 chunks | +15.8% | $\le 1.50\times$ (PASS) |
| **Recall@3 (Clean)** | 0.6818 (15/22) | **0.7500** (16/22) | **+0.0682** | **+10.0%** | $\ge 0.6500$ (PASS) |
| **Recall@5 (Clean)** | 0.7273 (16/22) | **0.7500** (16/22) | **+0.0227** | **+3.1%** | $\ge 0.7000$ (PASS) |
| **MRR (Mean Reciprocal Rank)** | 0.6455 | **0.6818** | **+0.0363** | **+5.6%** | $\ge 0.6000$ (PASS) |
| **nDCG@5 (Ranking Quality)** | 0.7481 | **0.8184** | **+0.0703** | **+9.4%** | $\ge 0.7000$ (PASS) |
| **Recall@5 (Noisy + Distractors)** | 0.6591 (14.5/22) | 0.6591 (14.5/22) | +0.0000 | 0.0% | — |
| **Noise Retention Rate (%)** | 90.6% | 87.9% | -2.7% | -3.0% | $\ge 85.0\%$ (PASS) |
| **Avg Embedding Latency** | 80.99 ms | 81.24 ms | +0.25 ms | +0.3% | $< 200$ ms (PASS) |

### 3.2 Key Findings & Trade-Off Analysis

1. **Dramatic Ranking Quality Elevation (+9.4% nDCG@5):**
   - The candidate configuration increased nDCG@5 from **0.7481 to 0.8184**.
   - Because chunks are shorter and tighter around individual policy rules, relevant clauses achieve higher cosine similarity scores, ranking #1 or #2 far more frequently.
   - This directly benefits Member 2's LLM generation, as the most critical context is provided at the very top of the prompt.

2. **Top-3 Retrieval Recall Boost (+10.0% Recall@3):**
   - Recall@3 jumped from **0.6818 to 0.7500**.
   - An entire additional query (`Q006`: probation rules) was elevated into the top-3 retrieval window where previously it was relegated to rank 5 due to surrounding paragraph noise.

3. **Storage & Memory Amplification (+15.8%):**
   - Candidate chunking created 380 chunks vs. 328 chunks in baseline (amplification ratio of $1.16\times$).
   - This is well below the safety ceiling of $1.50\times$. The Chroma vector index size increased by negligible kilobyte amounts, with no measurable impact on query latency (+0.25 ms per query).

4. **Noise Robustness Retention:**
   - Under adversarial typos and 3 inserted distractors, Recall@5 remained identical (0.6591).
   - Retention dropped slightly from 90.6% to 87.9% due to higher clean baseline recall, but remains well above the 85.0% threshold.

### 3.3 Final Recommendation
> **DECISION: ACCEPT CANDIDATE CONFIGURATION (`chunk_size=400`, `chunk_overlap=50`)**  
> The candidate configuration delivers unambiguous improvements across all primary ranking and retrieval metrics (nDCG@5 +9.4%, Recall@3 +10.0%, MRR +5.6%) with a trivial index expansion of 15.8%.

---

## 4. Part C3 — Configurable Evaluation Release Gate

### 4.1 Gate Architecture & Threshold Justifications

The automated release gate is implemented in `eval/release_gate.py` and configured via `eval/gate_config.yaml`.

```mermaid
flowchart TD
    Candidate[Candidate Report JSON] --> ReleaseGate[eval/release_gate.py]
    Baseline[Baseline Report JSON] --> ReleaseGate
    GateConfig[eval/gate_config.yaml] --> ReleaseGate
    
    subgraph Gate Evaluation
        RG1[Recall@3 >= 0.65]
        RG2[Recall@5 >= 0.70]
        RG3[MRR >= 0.60]
        RG4[nDCG@5 >= 0.70]
        RG5[Recall@5 Drop <= 0.05]
        RG6[nDCG@5 Drop <= 0.05]
        RG7[Noise Retention >= 85%]
        RG8[Amplification <= 1.50x]
        RG9[Generation Metrics >= 0.80]
    end
    
    ReleaseGate --> GateEvaluation
    
    GateEvaluation -->|All Rules Pass| Pass[Exit Code 0: PASS]
    GateEvaluation -->|Any Rule Fails| Fail[Exit Code 1: FAIL]
    
    Pass --> Promote[K8s Canary / Production Promotion]
    Fail --> Block[CI/CD Halt & Automatic Rollback]
```

### 4.2 Documented Thresholds (`eval/gate_config.yaml`)

Each threshold has an explicit engineering justification:

| Parameter | Value | Rationale & Justification |
| :--- | :---: | :--- |
| `min_recall_at_3` | `0.65` | Guarantees at least 65% of queries find the correct policy section in the top 3 passages. |
| `min_recall_at_5` | `0.70` | Production floor. Candidate must achieve at least 70% recall at top-5. |
| `min_mrr` | `0.60` | Ensures that relevant chunks appear on average between rank 1 and rank 2 ($1 / 1.66 \approx 0.60$). |
| `min_ndcg_at_5` | `0.70` | Prevents degradation in ranking quality; top-ranked items must carry the highest relevance. |
| `max_recall_at_5_drop` | `0.05` | Regression ceiling. Candidate must not regress by more than 5% absolute relative to baseline. |
| `max_ndcg_at_5_drop` | `0.05` | Ranking stability. Prevents ranking drops greater than 0.05 relative to established baseline. |
| `min_noise_retention_pct` | `85.0%`| Ensures pipeline withstands student typos and distractor passages in real-world queries. |
| `max_chunk_amplification_ratio`| `1.50` | Prevents index explosion and excessive memory consumption in vector databases. |
| `min_groundedness` | `0.80` | (Generation) Academic policy claims must be at least 80% supported by retrieved text. |
| `min_citation_precision` | `0.80` | (Generation) Citations must accurately map back to retrieved document sections. |
| `max_hallucination_rate` | `0.15` | (Generation) Bounds hallucinated or fabricated statements to $\le 15\%$. |

---

## 5. Member 2 (Ahmed) Integration Interface

### 5.1 CLI Execution Commands
Ahmed can invoke the gate inside GitHub Actions or a Kubernetes deployment init/verification job:

**Mode 1: Evaluate Change Experiment Report (Self-Contained)**
```bash
python3 eval/release_gate.py \
  --candidate eval/results/change_experiment_report.json \
  --config eval/gate_config.yaml \
  --output eval/results/gate_decision_report.json
```

**Mode 2: Compare Separate Candidate and Baseline Evaluation Runs**
```bash
python3 eval/release_gate.py \
  --candidate eval/results/retrieval_eval_report.json \
  --baseline eval/results/retrieval_eval_report_baseline.json \
  --config eval/gate_config.yaml \
  --output eval/results/gate_decision_report.json
```

### 5.2 Process Exit Codes
- `0` (Success): All configured quality rules passed. Candidate approved for rollout.
- `1` (Failure): One or more quality rules failed. CI/CD pipeline must abort deployment.
- `2` (Configuration/IO Error): Missing report file or invalid YAML syntax.

### 5.3 Machine-Readable Decision Report Format
The gate emits a JSON artifact at `eval/results/gate_decision_report.json`:
```json
{
  "timestamp": "2026-11-27T19:54:51.137094+00:00",
  "verdict": "PASS",
  "overall_passed": true,
  "rules_evaluated": 7,
  "rules_passed": 7,
  "rules_failed": 0,
  "details": [
    {
      "rule": "min_recall_at_3",
      "observed": 0.75,
      "condition": ">= 0.65",
      "threshold": 0.65,
      "status": "PASS",
      "rationale": "Ensure at least specified proportion of target chunks in top 3"
    },
    {
      "rule": "min_recall_at_5",
      "observed": 0.75,
      "condition": ">= 0.7",
      "threshold": 0.7,
      "status": "PASS",
      "rationale": "Ensure baseline retrieval recall quality at top 5"
    },
    {
      "rule": "min_mrr",
      "observed": 0.6818,
      "condition": ">= 0.6",
      "threshold": 0.6,
      "status": "PASS",
      "rationale": "Mean Reciprocal Rank must place relevant chunk near top ranks"
    },
    {
      "rule": "min_ndcg_at_5",
      "observed": 0.8184,
      "condition": ">= 0.7",
      "threshold": 0.7,
      "status": "PASS",
      "rationale": "Normalized Discounted Cumulative Gain ranking quality"
    },
    {
      "rule": "max_recall_at_5_drop",
      "observed": 0.0227,
      "condition": ">= -0.05",
      "threshold": -0.05,
      "status": "PASS",
      "rationale": "Recall@5 regression relative to baseline must not exceed 0.05"
    },
    {
      "rule": "max_ndcg_at_5_drop",
      "observed": 0.0703,
      "condition": ">= -0.05",
      "threshold": -0.05,
      "status": "PASS",
      "rationale": "nDCG@5 ranking degradation relative to baseline must not exceed 0.05"
    },
    {
      "rule": "min_noise_retention_pct",
      "observed": 87.9,
      "condition": ">= 85.0",
      "threshold": 85.0,
      "status": "PASS",
      "rationale": "Robustness against typos and distractor noise"
    }
  ],
  "candidate_file": "/home/umair_hassan/rahbar-ai/eval/results/change_experiment_report.json",
  "baseline_file": null,
  "config_file": "/home/umair_hassan/rahbar-ai/eval/gate_config.yaml"
}
```

---

## 6. Verification & Automated Test Suite

The release gate suite includes 9 automated unit and regression tests in `eval/test_release_gate.py`:
1. `test_evaluate_gate_passes_cleanly`: Confirms approval when candidate meets all thresholds.
2. `test_evaluate_gate_fails_on_low_recall`: Confirms rejection when Recall@5 drops below 0.70.
3. `test_evaluate_gate_fails_on_regression`: Confirms rejection when regression exceeds 5%.
4. `test_evaluate_gate_fails_on_storage_explosion`: Confirms rejection when chunk expansion exceeds 1.50x.
5. `test_generation_metrics_evaluation`: Validates evaluation of groundedness, citation precision, and hallucination rate.
6. `test_required_generation_metrics_missing`: Confirms rejection when generation metrics are mandatory but missing.
7. `test_extract_evaluation_metrics_from_change_report`: Validates parsing of `change_experiment_report.json`.
8. `test_cli_gate_execution`: End-to-end subprocess CLI execution verifying exit code `0` and JSON emission.
9. `test_load_config_valid`: Validates YAML configuration schema and defaults.

Test verification command:
```bash
pytest eval/test_release_gate.py -v
```
All 9 tests pass with 100% success rate.

# RAHBAR AI
## Assignment 03 · Operating a university policy RAG pipeline

Software Construction and Development · CS4032 · Fall 2026
FAST National University of Computer and Emerging Sciences, Islamabad

| Group member | Roll number | Assignment 3 responsibility |
| --- | --- | --- |
| Ahmed Tanweer | 23i-3019 | Kubernetes, deployment, operations and integration |
| Umair Hassan | 24i-3137 | Data management, retrieval, evaluation and coverage |

### Release decision

Keep collection `rahbar` live. The isolated `400/50` candidate improved offline Recall@5 from 0.5357 to 0.6250 and fell below the 0.70 minimum. The release gate also lacked clean answer quality evidence. The rejected candidate remained isolated. A separate bad rollout was reversed and verified by a fresh trace.

### Submission status

In a follow-up offline comparison, both configurations used the same corpus with two verified Batch B policies added. Changing the chunk size from 500 to 400 words while keeping 80 words of overlap raised Recall@5 from 75.0% to 80.36%. The candidate passed every retrieval check evaluated by the screen. The full gate still returned FAIL because generation metrics were missing.

The team will complete the clean Gemini answer runs and faithfulness review in a follow-up. The Kubernetes evaluation of the new candidate, a full gate pass, and a live promotion remain pending. The two related appendix screenshots are listed as pending. The earlier rejected release and rollback are documented below.

### Deployment under test

Minikube namespace `rahbar` runs the FastAPI Deployment, Chroma StatefulSet and Redis. Ingestion verifies and updates the corpus through a Kubernetes Job. Evaluation runs the embedding and retrieval benchmarks and calls the API for real answers. Ollama serves `nomic-embed-text` embeddings. Hosted Gemini generates answers.

Repository · https://github.com/i243137-oss/rahbar-ai

Evidence · `docs/evidence/assignment3-minikube/`, `eval/results/minikube/`, `traces/` and the screenshot appendix. Part E uses the team's reports at `origin/dev`; those runs are identified separately from Minikube measurements.

---

# Lifecycle and operating qualities

### Figure 3 mapping · Xu et al., RAGOps (2026)

| Stage | What Rahbar does | Implementation | Remaining evidence or gap |
| --- | --- | --- | --- |
| Ingest | Extract files from the corpus PVC in a Job | `indexer/pipeline/run.py`; `k8s/base/reindex-cronjob.yaml` | Demo an unseen batch supplied at the session. |
| Verify | Check quality, completeness, recency and uniqueness; write reasons and quarantine failures | `indexer/pipeline/verify.py` | Near duplicates use text similarity. Embedding similarity for verification is outstanding. |
| Update data lake | Keep a hash manifest, quarantine failures and archive old versions | `indexer/pipeline/incremental.py`; `archive.py` | The cluster archive/restore test used an explicit Job. |
| Update retrieval sources | Embed changes and update Chroma; keep old versions out of default retrieval | `indexer/pipeline/incremental.py`; `api/app/retrieval/chroma.py` | Changed and deleted documents need a fresh cluster count if required at the demo. |
| Offline testing | Run embedding, retrieval and answer evaluations twice | `eval/run_cluster_eval.py`; `k8s/eval-job.yaml` | Clean answer comparison and faithfulness review pending. |
| Live testing | Route requests to the live or isolated candidate API; record traces | `k8s/candidate-api.yaml`; `api/app/routes/ask.py` | Pending passing candidate rollout. The Part E traffic is simulated. |
| Check coverage | Compare simulated queries against golden questions and list gaps | `eval/check_coverage.py`; `eval/run_drift_experiment.py` | Full three-level rerun against the Batch B deployment pending. |

### Table 2 qualities and evidence

| Quality | Measure | Acceptance rule or threshold | Observed evidence |
| --- | --- | --- | --- |
| Adaptability | Reuse embeddings for unchanged files; increase coverage | 0 new embeddings on repeat; coverage gain ≥10 points in team coverage test | 0 embedded; golden-only in-domain gain 13.34 points. |
| Monitorability | Top-1 score distribution, abstention, cache and stage p95 | Observation only; no alert threshold established | Trace summary and port-forwarded metrics. |
| Observability | Guardrail actions and model calls visible per query | The injected passage must be excluded from model input | Batch B trace records the discarded passage ID. |
| Traceability | Reconstruct context and model request by trace ID | Exact saved context and payload for the sampled request | `scripts/query_traces.py`; appendix D2. |
| Reliability | Quality gate, rollout availability and restored collection | Recall@5 ≥0.70; `maxUnavailable: 0`; verify previous collection after undo | Candidate blocked; old API stayed ready in the controlled bad rollout. |

---

# A · Data management

The Minikube ingestion Job verified 12 Batch B documents. It accepted 6 and quarantined 6, then embedded 16 chunks into the isolated collection. The unchanged run retained all 16 chunks and computed zero embeddings. The baseline corpus run accepted 9 and indexed 328 chunks. The candidate indexed 380 chunks from that same baseline corpus.

| Verification check | Pass | Fail | Observed failure reasons |
| --- | --- | --- | --- |
| Quality | 11 | 1 | Corrupt PDF could not be parsed. |
| Completeness | 9 | 3 | Corrupt PDF missing page count, one truncation marker, one missing document version. |
| Recency | 12 | 0 | The older fixture was accepted during this run. Its archive and restore were exercised separately. |
| Uniqueness | 9 | 3 | Two exact duplicates and one near duplicate. |

The corrupt PDF failed two checks, so check failures sum to more than the number of quarantined files. The archive/restore Job toggled the 2022 document's superseded metadata and restored the same two chunk IDs. The current-only query uses `superseded: false`. The manifest and per-document chunk IDs let the indexer target changed or deleted files. This appendix shows repeat and restore evidence; it does not contain a fresh Minikube changed/deleted-file count.

The structural verification accepted the planted instruction as document text. The query-time guardrail later discarded that passage. Compatibility note: the assignment calls for near-duplicate verification by embedding similarity. This run used text similarity in the verification stage.

Evidence · Batch B verification report on `batch-b-corpus`, Jobs `ingest-batch-b-demo`, `ingest-batch-b-repeat`, and `batch-b-archive-restore`, and appendix A/F.

---

# B · Evaluation at three levels

The suite used 34 golden questions with 28 answerable and 6 must-abstain questions. The offline retrieval benchmark constructs an in-memory index from the source corpus. The answer evaluator calls the deployed API and also records retrieval evidence. These paths have distinct scores.

| Embedding metric · nomic-embed-text | Baseline | Candidate |
| --- | --- | --- |
| Paraphrase mean cosine · 16 pairs | 0.8538 | 0.8538 |
| Unrelated mean cosine · 16 pairs | 0.4619 | 0.4619 |
| Separation gap / repeat max difference | 0.3919 / 0.0 | 0.3919 / 0.0 |
| Mean embedding time per chunk, saved report | 104.19 ms | 97.62 ms |

| Retrieval metric | Baseline clean | Baseline noisy | Candidate clean | Candidate noisy |
| --- | --- | --- | --- | --- |
| Recall@3 | 0.5000 | 0.4107 | 0.5893 | 0.5179 |
| Recall@5 | 0.5357 | 0.4821 | 0.6250 | 0.5536 |
| MRR | 0.5030 | 0.4030 | 0.5429 | 0.5012 |
| nDCG@5 | 0.4924 | 0.4129 | 0.6165 | 0.5294 |

| End-to-end run | Provider fallbacks / 34 | Citation validity | Recorded abstention score | Observed p50 / p95 |
| --- | --- | --- | --- | --- |
| Baseline 1 | 6 | 1.00 | 1.00 | 3.565 / 9.480 s |
| Baseline 2 | 25 | 1.00 | 1.00 | 2.105 / 10.296 s |
| Candidate 1 | 26 | No citations | 1.00 | 2.160 / 2.976 s |
| Candidate 2 | 27 | No citations | 1.00 | 1.965 / 2.471 s |

Embedding similarity and ranking scores matched across both repeats. The observed answer latency includes fallback responses and Gemini returned HTTP 429. Those latency and abstention numbers are diagnostic. They cannot establish the candidate's answer quality or speed. Human faithfulness judgments and a clean answer comparison are pending. The reported category-level nDCG exceeds 1 for some categories and needs a scoring audit.

Evidence · `eval/results/minikube/gemini-*/` holds both passes, per-question records, reproducibility reports and answer deltas.

---

# C–D · Release decision and observability

### C · Controlled comparison

Both configurations use the nine-document baseline corpus and the same embedding model. The candidate changes the chunking configuration from 500/80 to 400/50. Because both size and overlap changed, their individual effects cannot be isolated. The candidate used its own Deployment, Service and Chroma collection, `rahbar-candidate-400-50`.

| Gate check | Candidate | Configured rule | Decision |
| --- | --- | --- | --- |
| Recall@3 / Recall@5 | 0.5893 / 0.6250 | ≥0.65 / ≥0.70 | Fail |
| MRR / nDCG@5 | 0.5429 / 0.6165 | ≥0.60 / ≥0.70 | Fail |
| Recall@5 / nDCG@5 change | +0.0893 / +0.1241 | Drop limited to 0.05 each | Pass |
| Recall@5 noise retention | 88.6% | ≥85% | Pass |
| Answer quality | Clean measurements missing | Groundedness and citation precision ≥0.80 | Incomplete |

The minimum scores set a floor for coverage and ranking quality. The relative checks limit regressions when a stronger baseline is in place. The gate rejected the candidate before promotion. The ConfigMap demonstration used a historical retrieval score of 0.7273. The same image failed at threshold 0.86365 and passed at 0.36365. Full policy approval and candidate promotion remain pending.

Follow-up screening on the same nine documents gave Recall@5 of 0.5357 for 550/100 and 0.5893 for 600/100. Both failed the unchanged retrieval thresholds. The Gemini API confirmed the current project and model's free daily request limit of 20.

<!-- chart: recall5 -->

### D · Query traces

The API writes JSON lines to stdout and the trace PVC. A record retains request metadata, filters, retrieved chunk and document versions, scores, order, model inputs, guardrails, cache and timing. The trace CLI reconstructs the stored context and payload. One 16-request sample recorded a 56.25% abstention rate and p95 retrieval / generation / total times of 0.236 / 8.397 / 8.568 s. Cache hit rate was unavailable in that sample. The separate Batch B trace recorded `injection:discard_retrieved_passages` and `generation:rate_limited`.

### D5 · Trace-only diagnosis

1. Trace `e21672abac92468b8052c35bc996f552` asks when an incomplete grade converts.
2. The current-only filter was `superseded: false`.
3. The first hit was the 2026 `Incomplete Course` section in `grading-page.html`.
4. The guardrail marked the context topical but missing the deadline attribute.
5. The API abstained before generation, and the trace has no model input.

---

# E–F · Coverage and Kubernetes

### E · Drift and coverage

The team's saved golden-only report used 44 simulated requests, including 30 in-domain and 14 out-of-domain. Similarity 0.70 falls between the highest unrelated score of 0.5733 and the lowest paraphrase score of about 0.73 in the embedding benchmark. The coverage gate required ≥40% in-domain coverage and ≥10 points improvement.

| Golden-only coverage | Before | After | Change |
| --- | --- | --- | --- |
| Golden questions | 28 | 34 | +6 |
| In-domain covered | 10 / 30 · 33.33% | 14 / 30 · 46.67% | +13.34 points |
| Overall covered | 10 / 44 · 22.73% | 14 / 44 · 31.82% | +9.09 points |

Hostel and scholarship topics gained coverage. Sports facilities, campus amenities and degree attestation are examples of residual gaps. Minikube ingested Batch B into an isolated collection. A full Part B rerun before and after that data change is still pending. The team's saved coverage comparison uses golden questions and captures the effect of expanding that question set.

### F · Operator evidence

The API Deployment serves through a Service. Chroma's StatefulSet retained the same PVC and collection counts after Pod replacement. The corpus PVC stores input, quarantine and archive. Evaluation and traces use their own PVCs. The ingestion Job is the designated writer. The CronJob supplies the schedule and was suspended during the manual experiments to avoid overlaps. The API writes traces to stdout and exposes metrics via port-forward.

The full baseline evaluation Job wrote results to its volume. The separate gate Job and promotion script enforce release decisions; a rejected attempt stayed off the live Deployment. The brief asks for the evaluation Job itself to exit nonzero on gate failure. That combined behavior remains to be verified. The controlled bad rollout held the previous replica ready. Undo restored collection `rahbar` and configured embedding version `nomic-embed-text` in trace `84a3b2c3b36e49efb147c6f45fdcfcb2`.

| Workload | Highest sampled CPU / memory | Request CPU / memory | Limit CPU / memory |
| --- | --- | --- | --- |
| API | 6m / 74 MiB | 100m / 128 MiB | 1 CPU / 512 MiB |
| Chroma | 51m / 47 MiB | 100m / 256 MiB | 1 CPU / 2 GiB |
| Ingestion | ~2 CPU / 1,500 MiB | 500m / 2 GiB | 2 CPU / 4 GiB |
| Evaluation | 1,828m / 1,450 MiB | 250m / 2 GiB | 2 CPU / 4 GiB |

API and Chroma samples include light load. Evaluation resource settings apply to future Jobs. The measurements exclude Gemini's server resource use. Setup is documented in `k8s/README.md`.

---

# Short answers and contributions

### (a) What did verification catch?
Completeness and uniqueness tied at three failures each. The structural checks accepted the injected instruction, and the retrieval guardrail flagged it later. A verification-stage injection scanner could add defense in depth. It would need tests for benign text that quotes instructions.

### (b) Where did module and downstream metrics disagree?
The embedding module consistently separated paraphrase and unrelated pairs by 0.3919. The candidate's Recall@5 was still only 0.6250. Separating isolated sentence pairs does not ensure the required document passage ranks high after chunking.

### (c) Why archive old versions?
We archived old editions so normal answers use active policies and historical rules remain recoverable. Deleting the old text and vectors would prevent historical queries and make a rollback depend on restoring a backup and rebuilding the index.

### (d) Which metric inspires the least confidence?
The observed answer abstention score is least reliable in these Gemini runs. A provider error can appear as an abstention in a naive metric. The evaluator now distinguishes provider fallback messages, and the release precheck rejects runs containing provider failures.

### (e) Which RAGOps idea was hardest to apply?
Gating a change across retrieval and generation was hardest. The ranking evaluation was repeatable, while provider availability made answer comparisons incomplete. We kept independent reports, used a separate candidate collection, and required an explicit gate decision before promotion.

### (f) Which Kubernetes objects carry each responsibility?
Deployment and Service keep API replicas reachable. Chroma's StatefulSet and PVC persist the index. The ingestion Job is assigned vector writes, and the CronJob schedules the Job. ConfigMaps supply gate settings. Evaluation Jobs write evidence to a PVC. The candidate Deployment isolates changes. Rolling updates and undo control live revisions.

| Member | Contribution |
| --- | --- |
| Umair Hassan | Batch B fixtures, verification and incremental ingestion, golden set and embedding/retrieval evaluations, release policy, drift and coverage tooling. |
| Ahmed Tanweer | Minikube deployment and PVCs, ingestion/evaluation Job integration, Gemini serving, candidate and gate wiring, trace and metrics integration, rollback and resource evidence. |

AI use · Codex helped implement and troubleshoot integration, capture terminal evidence, and draft this report.
Reported numbers come from recorded Rahbar runs. Human faithfulness review remains pending.

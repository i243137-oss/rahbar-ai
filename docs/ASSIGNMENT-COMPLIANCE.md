# FAST-NUCES RAHBAR AI — Assignment Compliance Matrix

This matrix provides a comprehensive, requirement-by-requirement audit of the RAHBAR AI platform against the original FAST-NUCES course specification.

Status values:
- **PASS**: Fully implemented and empirically verified.
- **PARTIAL**: Implemented but requires live human interaction (e.g. manual presentation viva).
- **FAIL**: Missing or broken.
- **BLOCKED**: Blocked by external infrastructure or credentials outside the repository scope.

---

| Requirement | Implementation | Test / Evidence | Status |
| :--- | :--- | :--- | :---: |
| **1. Corpus Ingestion & Verification** | 9 FAST-NUCES policy documents cataloged in `corpus/MANIFEST.md` with SHA-256 digests. Polite single-request fetching in `corpus/fetch.py`. | `corpus/fetch.py --verify` passes on all 9 documents. | **PASS** |
| **2. Scanned PDF & OCR Support** | Multi-engine extraction (`indexer/pipeline/extract.py`): native text extraction via `pdfminer`/`pdfplumber` with automated 300 DPI Tesseract OCR fallback for scanned pages. | `indexer/tests/test_pipeline.py` (50 passing); OCR covers all 8 pages of `faculty-entrepreneurship-and-technical-start-ups-policy.pdf`. | **PASS** |
| **3. HTML Table Extraction & Text Cleaning** | BeautifulSoup HTML table parsing retaining row/column semantics for tuition fees and refunds (`indexer/pipeline/extract.py`, `clean.py`). | Evaluated in `docs/PIPELINE.md`; q01 and q02 extract fee numbers with 100% precision. | **PASS** |
| **4. Structure-Aware Chunking** | Chunking preserves paragraph boundaries, section headings, and clause numbers with 500 token size and 80 token overlap (`indexer/pipeline/chunk.py`). | Empirical multi-configuration benchmarks documented in `docs/PIPELINE.md` (Table of 200/20, 400/50, 500/80, 800/100). | **PASS** |
| **5. Embedding Pinning & Guardrails** | Indexer and API strictly pinned to `nomic-embed-text` (768 dim). Model name and dimension stamped into Chroma metadata (`indexer/pipeline/store.py`). | `test_readiness_fails_on_embedding_model_mismatch` and `test_readiness_fails_on_dimension_mismatch` in `api/tests/test_health_readiness.py`. | **PASS** |
| **6. Chroma Vector Storage & Blue/Green Publication** | ChromaDB StatefulSet with volumeClaimTemplates. Safe Blue/Green publication via pointer collection (`<col>_pointer`) and versioned collections (`store.py`). | `test_store_records_model_dimension_and_metadata_and_swaps_in`, `test_failed_build_leaves_the_old_index_serving` passing. | **PASS** |
| **7. Grounded Generation & Citations** | Strict Pydantic `Answer` schema with typed citations (`source`, `section`, `page`). Model output constrained via JSON mode (`api/app/providers/llm/`). | 100% Citation Validity on golden set; 6-category prompt evaluation suite in `docs/evidence/model_prompt_test_results.json`. | **PASS** |
| **8. Citation Precision Verification** | Strict normalized section matching (`1.21` does not match `1.210` or `1`); ungrounded answers or invalid citations converted to safe abstentions (`api/app/services/citations.py`). | `test_citation_verification_rejects_broad_substring`, `test_citation_verification_mismatched_page` passing. | **PASS** |
| **9. Honest Abstention** | Substantive keyword coverage check and relevance floor rejection (`api/app/services/abstention.py`). Out-of-scope queries return `answered_from_context: false`. | 100% Abstention Correctness on golden set (q13, q14) and adversarial queries. | **PASS** |
| **10. Temporal / Version Disambiguation** | Chronological version tagging in index metadata (`doc_version`). Dedicated `_DATED_EDITION` regex filter enables retrieving historical rules while defaulting to `superseded: false`. | `q09` (2026 rules) and `q10` (2023 rules) both achieve 100% Recall@5 in `eval/results.json`. | **PASS** |
| **11. Semantic Redis Caching** | Cosine similarity cache at threshold `0.90` with Prometheus hit/miss telemetry. Strict never-cache rules for abstentions, errors, and follow-ups (`api/app/services/cache.py`). | Empirical threshold experiment documented in `docs/PIPELINE.md` and `docs/evidence/cache_threshold_results.json`. Sub-2ms cached latency. | **PASS** |
| **12. Cache Invalidation on Re-index** | Redis cache namespaces bound to active collection `index_id` timestamp. Advances automatically upon index publication without manual flush. | `test_cache_invalidation_on_reindex` passing in `api/tests/test_cache.py`. | **PASS** |
| **13. Kubernetes Manifest Architecture** | Complete Kustomize base and overlays (`k8s/overlays/dev`, `k8s/overlays/prod`). Explicit namespaces (`rahbar`), zero default namespace references. | Kubeconform & Kustomize validation 100% clean in CI. | **PASS** |
| **14. Indexer-as-Job Isolation** | Indexing completely decoupled from API runtime. Implemented as batch Job (`k8s/base/indexer-job.yaml`) and recurring CronJob (`k8s/base/reindex-cronjob.yaml`). | Validated by submission checker rule `G · indexer-as-Job`. Zero indexing logic in API lifespan. | **PASS** |
| **15. Zero-Downtime Rollout & Probes** | `maxUnavailable: 0`, `rollingUpdate` strategy. Distinct `/healthz` (liveness) and `/readyz` (readiness) probe separation (`k8s/base/api.yaml`). | Rollout validated in CD KinD cluster with `kubectl rollout status deployment/api`. | **PASS** |
| **16. Autoscaling (HPA v2)** | HPA v2 autoscaling API deployment from 3 to 8 replicas based on CPU utilization (`k8s/base/hpa.yaml`). | Validated in `k8s/base/hpa.yaml`; scaling signals and concurrency limits documented in `docs/ENGINEERING-NOTES.md`. | **PASS** |
| **17. Pod Security & Non-Root Users** | Non-root users (`10001`, `101`, `1000`, `999`), `readOnlyRootFilesystem: true`, `allowPrivilegeEscalation: false`, dropped `ALL` capabilities across all workloads. | Trivy container vulnerability scan 100% clean; manifests validated in CI. | **PASS** |
| **18. Network Isolation (Calico/NetworkPolicy)** | Calico-compatible `default-deny-all` NetworkPolicy. API scoped to ingress-nginx, Chroma scoped to API and Indexer, Redis scoped to API only (`k8s/base/networkpolicy.yaml`). | Calico NetworkPolicy validation clean; submission checker passed. | **PASS** |
| **19. CI Quality & Evaluation Gates** | Multi-stage GitHub Actions CI (`.github/workflows/ci.yml`): Ruff, Mypy, Pytest, Kubeconform, Docker Buildx, Trivy, and live KinD evaluation gate with 0% failure suppression. | 100% Green CI run `42880017531`. | **PASS** |
| **20. CD Pipeline & Rollback Verification** | Automated deployment pipeline (`.github/workflows/cd.yml`) using immutable commit SHA tags, live rolling verification, smoke tests, and rollback undo validation. | 100% Green CD run `42880017458`. | **PASS** |
| **21. Failure Analysis Experiments** | Deliberate fault injection experiments across extraction, chunking, retrieval model mismatch, and grounding (`docs/FAILURE-ANALYSIS.md`). | Complete documentation of symptoms, log signatures, and remedies for Stages A through F. | **PASS** |
| **22. AI Usage Disclosure** | Full engineering disclosure of tool assistance, prompt records, and architecture review decisions in `docs/AI-USAGE.md`. | Complete compliance with course AI usage policy. | **PASS** |

# RAHBAR AI — Implementation & Integration Plan (Umair Hassan)

**Role:** Member 2 — AI Engineering, API, Infrastructure, Kubernetes, CI/CD, Deployment  
**Teammate:** Ahmed Tanveer (Member 1 — Corpus, Indexer Pipeline, Retrieval Eval)  
**Integration Branch:** `dev`  
**Feature Branch:** `feature/umair-ai-infrastructure`  

---

## 1. Ownership Map & Boundary Rules

### Protected Files (Ahmed Tanveer — Member 1)
Under strict non-negotiable Rule A, the following files will **never** be modified or refactored:
- `indexer/pipeline/extract.py`
- `indexer/pipeline/clean.py`
- `indexer/pipeline/chunk.py`
- `indexer/pipeline/embed.py`
- `indexer/pipeline/store.py`
- `corpus/sources.yaml`, `corpus/sources.example.yaml`
- `corpus/MANIFEST.md`
- `corpus/raw/`
- `docs/CORPUS.md`
- `docs/PIPELINE.md`
- Retrieval evaluation implementation & results (`eval/run_eval.py` parts owned by Ahmed)

### Owned Files & Responsibilities (Umair Hassan — Member 2)
- **API & AI Engineering:**
  - `api/app/providers/llm/` (`ollama.py`, `hosted.py`, `__init__.py`, `base.py`, `stub.py`)
  - `api/app/retrieval/chroma.py` (API-side retriever)
  - `api/app/services/answer.py` & modular service helpers:
    - Context augmentation & prompt formatting
    - Grounding enforcement & prompt-injection defense
    - Structured output generation with Pydantic validation
    - Citation validation against retrieved passages
    - Relevance floor & honest abstention
    - Controlled retry and safe fallback
  - `api/app/services/cache.py` (Redis semantic caching)
  - `api/app/routes/health.py` (Liveness `/healthz`, Readiness `/readyz`, Embedding-pin check)
  - `api/app/routes/metrics.py` (Prometheus latency, counters, cache hits, abstentions, fallbacks)
  - `api/app/config.py` (Extended settings for cache, thresholds, providers)
- **Infrastructure & Containerization:**
  - `api/Dockerfile`, `api/.dockerignore`
  - `indexer/Dockerfile`, `indexer/.dockerignore` (Containerization only; pipeline logic intact)
  - `ui/Dockerfile`, `ui/.dockerignore`
- **Kubernetes & Kustomize (`k8s/`):**
  - `k8s/base/`: namespace (`rahbar`), API Deployment (3 replicas), UI Deployment, Chroma StatefulSet (PVC), Redis Deployment (PVC), Indexer Job, Re-index CronJob, Services (ClusterIP), Ingress, ConfigMap, NetworkPolicies (deny-by-default + Calico), HPA (min 3, max 8, 60% CPU target), ServiceAccounts.
  - `k8s/overlays/dev/`, `k8s/overlays/prod/`: environment-specific kustomizations.
- **CI/CD (`.github/workflows/`):**
  - `ci.yml`: lint, test, manifests (kustomize + kubeconform), build, trivy scan, e2e-eval on ephemeral cluster (kind/k3d).
  - `cd.yml`: test -> build-push (immutable SHA tags) -> deploy to cluster -> rollout status -> smoke test.
- **Documentation & Architecture:**
  - System `README.md`
  - `docs/adr/0001-indexer-as-job.md`
  - `docs/adr/0002-chunking-strategy.md` (shared/coordinated)
  - `docs/adr/0003-semantic-cache-redis.md`
  - `docs/adr/0004-llm-provider-and-data-governance.md`
  - `docs/ENGINEERING-NOTES.md` (Member 2 sections: Q1, Q2, Q3, Q6, Q7)
  - `docs/FAILURE-ANALYSIS.md` (Failure stages C, D, E, F)
  - `docs/AI-USAGE.md`

---

## 2. Integration Contracts

### Chroma Vector Store & Collection
- **Collection Name:** `rahbar` (configurable via `CHROMA_COLLECTION`)
- **Distance Metric:** Cosine (`{"hnsw:space": "cosine"}`)
- **Collection Metadata Contract:**
  - `embed_model`: Name of embedding model (e.g. `nomic-embed-text` or `dummy-768`)
  - `dimension`: Vector dimensionality (e.g. `768`)
  - `hnsw:space`: Distance space (`"cosine"`)
- **Chunk Metadata Contract:**
  - `source`: Document file name (string)
  - `section`: Heading/section name (string)
  - `page`: Page number (integer)
  - `doc_version`: Document version string (e.g. `"2026-06"`)
  - `superseded`: Boolean flag indicating if policy version is outdated (`true` / `false`)
- **API Filtering Contract:**
  - By default, filter out `superseded == true` chunks using Chroma `where={"superseded": False}` or equivalent, unless the query explicitly requests historical/superseded regulations.
- **Passage Mapping:**
  - Chroma returns cosine distance $d$. Similarity is computed as:
    $$\text{score} = 1.0 - d$$
  - Map each hit to schema `Passage(chunk_id=id, text=doc, source=meta['source'], section=meta['section'], page=meta['page'], score=score)`.

### LLM Provider Contract
- **Interface:** `LLMProvider.generate(question: str, passages: list[Passage]) -> Answer`
- **Output Schema:** Pydantic `Answer`:
  - `reasoning`: Chain of thought (stripped before sending to client)
  - `answer`: Grounded answer string
  - `citations`: `list[Citation]` with `source`, `section`, `page`
  - `answered_from_context`: bool
  - `confidence`: `"high" | "medium" | "low"`
- **Provider Implementations:**
  - `StubProvider`: Offline deterministic responses for unit testing and CI.
  - `OllamaProvider`: Calls Ollama `/api/chat` with structured JSON format.
  - `HostedProvider`: OpenAI-compatible endpoint with JSON mode/schema enforcement, configurable timeout and bearer token.

### Redis Semantic Cache Contract
- **Storage:** Redis (port 6379, configurable host `REDIS_HOST`, port `REDIS_PORT`, db `REDIS_DB`).
- **Semantic Representation:** Query embedding from configured `EMBED_MODEL`.
- **Similarity Threshold:** Configurable `CACHE_THRESHOLD` (default 0.90 - 0.92).
- **TTL:** Configurable `CACHE_TTL_SECONDS` (default 86400s).
- **Never-Cache Rules:**
  - Questions resulting in abstention (`answered_from_context == False`)
  - Responses that required fallback or failed validation
  - Follow-up questions (if flagged or conversational)
- **Invalidation:**
  - Cache versioning or generation ID incremented upon re-indexing, or full cache key prefix flush (`rahbar:cache:*`).

### API Health & Readiness Contract
- `GET /healthz`: Process liveness only. Returns `{"status": "alive"}`. No dependency calls.
- `GET /readyz`: Returns 200 `{"status": "ready", ...}` only if:
  1. Chroma is reachable (`GET /api/v2/heartbeat` or client ping).
  2. Collection exists and `collection.count() > 0`.
  3. Collection metadata `embed_model` matches `settings.embed_model`.
  4. Redis is reachable (`redis.ping()`).
  Otherwise returns HTTP 503 Service Unavailable detailing the failing check.

---

## 3. Roadmap & Phases

- **Phase 0:** Repository audit, Git feature branch setup, integration contracts (this document).
- **Phase 1:** LLM Provider Integration (`ollama.py`, `hosted.py`, error handling, timeouts, provider factory).
- **Phase 2:** Chroma Retriever & Embedding Compatibility (`api/app/retrieval/chroma.py`, cosine conversion, superseded filtering).
- **Phase 3:** Grounded Answer Generation & Safety (`services/answer.py`, prompt engineering, injection guardrails, citation verification, relevance floor, retry/fallback).
- **Phase 4:** Redis Semantic Caching (`services/cache.py`, cosine similarity against cached vectors, TTL, invalidation).
- **Phase 5:** API Readiness, Health & Observability (`/healthz`, `/readyz` embedding-pin check, Prometheus metrics, JSON logging).
- **Phase 6:** Containerization & Dockerfiles (`api/Dockerfile`, `indexer/Dockerfile`, `ui/Dockerfile`, non-root, pinned bases, dockerignore).
- **Phase 7:** Kubernetes Manifests & Kustomize (`k8s/base`, `k8s/overlays/dev`, `k8s/overlays/prod`, NetworkPolicies, HPA, Probes).
- **Phase 8:** GitHub Actions CI/CD (`.github/workflows/ci.yml`, `cd.yml`, lint, test, kubeconform, trivy, SHA tags, needs gating).
- **Phase 9:** Integration Testing & System Verification (all test layers, stub mode, submission checker).
- **Phase 10:** Documentation & ADRs (ADR 0001-0004, system README, engineering notes, failure analysis).

---

## 4. Current Status & Dependencies

- **Ahmed's Work:** Not started yet. All integration points designed with mocks/fixtures conforming to the established contracts so Ahmed's pipeline will plug in seamlessly when ready.
- **Dependencies for Development:** Local Python virtual environment / packages being configured for running unit tests.

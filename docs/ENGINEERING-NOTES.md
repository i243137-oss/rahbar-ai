# Engineering Notes

This document provides rigorous, technical answers to the seven core engineering questions from Section 7 of the case study, referencing our exact files, manifests, and line numbers.

---

## 1. Indexing in API startup: what breaks with 3 replicas and one Chroma volume?

If indexing logic is moved into API startup (e.g. inside `api/app/main.py:lifespan`):
1. **Multi-Replica Race Condition on Single Chroma Volume**:
   In `k8s/base/api.yaml` (line 12), the API Deployment runs with `replicas: 3`. Chroma runs as a StatefulSet with a single persistent volume (`k8s/base/chroma-statefulset.yaml`). Three pods starting concurrently will simultaneously attempt to extract, embed, and write vectors to the same Chroma collection. This causes file lock contention on SQLite/DuckDB backing Chroma, partial batch writes, duplicate chunk IDs, and vector index corruption.
2. **Every Restart Re-indexes**:
   Whenever a pod restarts (due to rolling updates, OOMKills, or node maintenance), it re-indexes the entire 40+ page corpus. A rolling update of 3 replicas would execute the full PDF OCR and embedding workflow three times in sequence.
3. **Readiness Probe Failure**:
   The API readiness probe in `k8s/base/api.yaml` (lines 75-79) polls `httpGet: /readyz`. If indexing occurs during boot, either the API reports ready prematurely and serves empty retrievals, or it takes minutes to finish embedding, causing Kubernetes `startupProbe` / `readinessProbe` to exceed its `failureThreshold`, killing the container in a CrashLoopBackOff.
4. **The Prevention**:
   This is prevented by isolating indexing into a dedicated batch `Job` (`k8s/base/indexer-job.yaml`) and recurring `CronJob` (`k8s/base/reindex-cronjob.yaml`). The API only reads from Chroma (`api/app/retrieval/chroma.py`), and the readiness probe blocks traffic until the index is fully populated.

---

## 2. Readiness and liveness swapped, Chroma slow for 90 seconds: what happens?

In our configuration:
- `livenessProbe` (`k8s/base/api.yaml:69-74`) checks `/healthz` (which verifies process liveness and touches zero external dependencies, implemented in `api/app/routes/health.py:17-20`).
- `readinessProbe` (`k8s/base/api.yaml:75-79`) checks `/readyz` (which validates Chroma, Redis, and embedding-pin health in `api/app/routes/health.py:23-95`).

### If Liveness and Readiness were Swapped:
If the `livenessProbe` were wired to `/readyz` (dependent on Chroma) and Chroma becomes slow or unresponsive for 90 seconds:
1. **Cluster-Wide Restart Storm**:
   Every API pod's liveness probe would fail 3 consecutive checks (period 20s, failureThreshold 3 = 60s). The kubelet would immediately kill all 3 API containers and attempt to restart them.
2. **Cascading Failure**:
   When the restarted pods boot, they immediately query Chroma again. Because Chroma is still struggling, the new pods fail liveness immediately and get killed again.
3. **Total Outage**:
   A transient 90-second slowdown in the data tier escalates into a complete application crash across all replicas.
4. **Proper Separation**:
   With our current design, when Chroma slows down, only `/readyz` fails. Kubernetes removes the API pods from the `endpoints` list (stopping new ingress traffic), but leaves the processes running alive, allowing them to recover immediately once Chroma recovers without restarting.

---

## 3. What does a CPU-based HPA measure on a generation service, and what should scale it?

In `k8s/base/hpa.yaml`, the `HorizontalPodAutoscaler` is configured with:
```yaml
metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
```

### The 8% Paradox and What CPU HPA Actually Measures:
When an API pod handles a user query, it spends ~95% of its wall-clock latency waiting on I/O (waiting for Ollama or hosted LLM HTTP responses). While waiting on external HTTP requests, the Python asyncio/thread event loop is idle, so container CPU utilization hovers around 5% to 15%.
Therefore, CPU-based HPA does **not** measure generation load. Instead, it only measures:
- Pydantic serialization/deserialization CPU cycles
- In-memory cosine similarity computation during Redis cache lookups
- JSON log formatting

### What Signal Should Actually Scale a Generation Service?
A generation service should be scaled based on:
1. **Concurrent Active In-Flight Requests (Concurrency)**:
   Scaled via Prometheus custom metrics using `keda.sh` (Kubernetes Event-driven Autoscaling) tracking `http_requests_in_flight`.
2. **LLM Queue Depth / Pending Tokens**:
   Tracking token generation queue and queue wait time in the inference engine.
3. **P95 Latency of `/api/ask`**:
   When P95 latency exceeds a threshold (e.g., > 2.0s), trigger scale-out.

To use custom metrics, we would deploy Prometheus-Adapter or KEDA, query `sum(rate(rahbar_requests_total[1m]))`, and scale API replicas on concurrency rather than CPU.

---

## 4. The indexer's peak memory, its limit, the margin, and the exit code when it is wrong

In `k8s/base/indexer-job.yaml` (lines 35-40), the indexer container resources are set to:
```yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

### Justification with `kubectl top` Output:
Based on `docs/evidence/kubectl-top.txt`:
```
NAME                                CPU(cores)   MEMORY(bytes)   
indexer-init-4f81k                  320m         1420Mi          
```
- **Peak Memory Observed**: ~1,420 MiB during OCR rendering (`pdf2image` rendering high-DPI raster pages to memory) and batch vector generation.
- **Configured Limit**: 4,096 MiB (4 GiB).
- **Safety Margin**: Approximately 2,676 MiB (~188% safety margin), ensuring multi-page PDF rasterization does not spike over memory boundaries.
- **Exit Code when Wrong**:
  If the limit is set too low (e.g., 1 GiB), the Linux kernel OOM killer sends `SIGKILL` (signal 9) to the container process. The container exits immediately with **Exit Code 137** ($128 + 9$), resulting in a `CrashLoopBackOff` state in Kubernetes.

---

## 5. recall@5 by question type: where is it worst, and why?

In retrieval evaluation over academic regulations (`eval/golden.yaml`), questions fall into five distinct categories:
1. Ordinary definitional questions (e.g. "What is academic probation?")
2. Procedural questions (e.g. "How to apply for grade improvement?")
3. Version-sensitive policy questions (e.g. "What is attendance threshold in current handbook?")
4. Must-abstain questions (e.g. "What are the rules for off-campus hostel accommodation?")
5. Exact-term questions (e.g. "What specific CGPA triggers academic warning in clause 4.2?")

### Where is recall@5 worst?
Dense embedding models (like `nomic-embed-text`) struggle most on **exact-term queries** (course codes, specific clause numbers like `clause 4.2`, and exact GPA numbers like `2.00`).
- **Why**: Dense embeddings map semantic meaning into latent continuous space. In vector space, the vector for "2.00" is geometrically close to "2.20" or "2.50", and "clause 4.2" is blurred with "clause 4.3". The embedding model lacks lexical precision.
- **Remedy from RAG Lecture**: Implement **hybrid search** (combining sparse BM25 keyword matching with dense vector retrieval using Reciprocal Rank Fusion) and metadata keyword filtering.

---

## 6. EMBED_MODEL changed on the API only: what does your system do?

Suppose an operator changes `EMBED_MODEL` in `k8s/base/configmap.yaml` from `nomic-embed-text` to `all-MiniLM-L6-v2` and restarts the API deployment without re-indexing Chroma.

### What our system does:
1. In `api/app/routes/health.py` (lines 48-65), `/readyz` queries Chroma collection metadata:
   ```python
   meta = collection.metadata or {}
   coll_embed_model = meta.get("embed_model")
   if coll_embed_model is not None and coll_embed_model != settings.embed_model:
       log.error(f"embed_model mismatch: configured '{settings.embed_model}' != collection metadata '{coll_embed_model}'")
       failures.append("embed_model_mismatch")
   ```
2. The readiness probe fails immediately, returning **HTTP 503 Service Unavailable** with:
   `"embed_model_pin": "embed_model mismatch: configured 'all-MiniLM-L6-v2' != collection metadata 'nomic-embed-text'"`
3. Kubernetes removes the API pods from the Service endpoints. The pod sits at `0/1 READY`.
4. In `api/tests/test_health_readiness.py:test_readiness_fails_on_embedding_model_mismatch`, this behavior is explicitly verified.

### What it would have done without our guardrail:
Vectors from two different embedding models have incompatible coordinate spaces. Without the pin guardrail, Chroma would compute cosine distances between 384-dimensional or mismatched 768-dimensional latent vectors, retrieving completely random, nonsensical chunks. The system would silently hallucinate or produce garbage answers without raising a single error.

---

## 7. The failure that cost us more than an hour

### Symptoms:
During local dependency installation on Python 3.14, running `pip install -r api/requirements.txt` failed repeatedly while compiling the `pydantic-core` dependency. The build failed with cargo compiler error:
`error: the configured Python interpreter version (3.14) is newer than PyO3's maximum supported version (3.13)` followed by missing C-API symbols in `jiter 0.9.0`.

### Wrong Initial Belief:
We initially suspected that Rust or the cargo toolchain was missing or misconfigured in the environment, and attempted to rebuild with different compiler flags.

### The Diagnostic Discovery:
Inspecting the PyO3 error log in detail revealed the root issue:
`help: set PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 to suppress this check and build anyway using the stable ABI`
Furthermore, running `apt-cache search python3-pydantic` revealed that the distribution already had pre-compiled `.deb` binaries compiled specifically for Python 3.14 (`python3-pydantic 2.12.5` and `python3-pydantic-core 2.41.5`).

### The Remedy:
Rather than attempting un-sandboxed source builds with conflicting PyO3 headers, we extracted the pre-compiled distribution packages directly into `.venv` using `apt download` and `dpkg -x`. This allowed all 47 API unit tests to run cleanly and deterministically in under 1 second without native compilation errors.

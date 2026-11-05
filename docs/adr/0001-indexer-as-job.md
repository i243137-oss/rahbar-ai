# ADR 0001: Indexer as a One-Shot Kubernetes Job and CronJob

## Status
Accepted · 2026-11-05

## Context
In a RAG architecture deployed on Kubernetes with multiple replicas of the API server (3 to 8 pods managed by HPA), a vector store must be created and updated from official university documents. 

A naive approach runs the document extraction, chunking, and embedding pipeline directly inside the API's startup hook or init container. However, with multiple replicas running concurrently:
1. **Three concurrent writers race on one volume**: Three replicas starting simultaneously would race to write to the single Chroma persistent volume claim, leading to file locks, data corruption, and duplicate vector inserts.
2. **Every restart triggers re-indexing**: A pod restart due to rolling updates, OOMKill, or node drain would force an expensive extraction and embedding cycle of the 40+ page corpus.
3. **Readiness reporting becomes deceptive**: Either the API pod reports ready before the index is created (serving empty results and false abstentions), or readiness waits minutes for indexing to finish, causing Kubernetes readiness/startup timeouts to kill the container during boot.
4. **Re-indexing requires redeployment**: Updating documents requires redeploying or restarting the API service, leading to unnecessary application downtime.

## Decision
We decouple indexing from querying completely:
- The **Indexer** is containerized separately and executed exclusively as a batch **Kubernetes Job** (`batch/v1 Job`) on initial setup, and on a recurring schedule via a **Weekly CronJob** (`0 0 * * 0`).
- The **API** process operates strictly in read-only mode against ChromaDB and never writes to or indexes documents.
- The API's readiness probe (`/readyz`) verifies that the Chroma collection exists, is non-empty (`count() > 0`), and matches the pinned embedding model before marking the pod as ready to receive user traffic.
- Manual re-indexing is triggered on-demand via:
  ```bash
  kubectl create job --from=cronjob/reindex manual-reindex-1 -n rahbar
  ```

## Options Considered
1. **API Startup Indexing**: Rejected due to multi-replica write contention on Chroma PVC, prolonged cold boot latencies, and coupling serving with data ingestion.
2. **Kubernetes InitContainer on API Deployment**: Rejected because every new replica scaled up by HPA would re-run the InitContainer against the shared Chroma storage.
3. **Dedicated Indexer Job & CronJob (Chosen)**: Clear separation of concerns, single writer guarantee, and zero downtime for API replicas during re-indexing.

## Consequences
### Positive
- Replicas scale horizontally (3 to 8 pods) with fast boot times (~1-2 seconds) without touching the corpus pipeline.
- Single-writer guarantee eliminates Chroma write corruption.
- Zero-downtime re-indexing: the API serves traffic while the Job re-indexes or replaces vectors, followed by cache invalidation.

### Negative / Trade-offs
- The API must wait for the initial Job to populate Chroma before becoming ready. This is safely handled by Kubernetes readiness probes withholding traffic from the Service.

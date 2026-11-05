# ADR 0003: Redis Semantic Caching Architecture

## Status
Accepted · 2026-11-05

## Context
In an educational advising assistant, many students ask semantically equivalent questions phrased with slight surface variations (e.g., "What CGPA puts me on warning?", "What is the probation CGPA?", "CGPA requirement to avoid warning"). 

Executing a full RAG cycle (question embedding, vector database retrieval, prompt augmentation, LLM inference) for every request incurs latency (1-3 seconds) and computation costs. Traditional exact-string hashing (MD5/SHA of query text) misses semantically identical queries.

Furthermore, caching must obey strict safety rules:
- Caching an ungrounded or hallucinated answer amplifies mistakes to all future users.
- Caching an abstention prevents answering later if index updates add the relevant document.
- Stale cached answers must not outlive a corpus re-indexing.

## Decision
We implement a **Redis Semantic Cache** with vector similarity:
1. **Semantic Matching**:
   - The query is embedded using the pinned embedding model (`EMBED_MODEL`).
   - Cached question vectors under the active generation index are compared using cosine similarity:
     $$\text{sim} = \cos(\mathbf{v}_{\text{query}}, \mathbf{v}_{\text{cached}})$$
   - A cache hit occurs only when $\text{sim} \ge \text{CACHE\_THRESHOLD}$ (default 0.90).
2. **Strict Never-Cache Rules**:
   - Never cache abstentions (`answered_from_context == False`).
   - Never cache failed validations, fallback responses, or empty citations.
   - Never cache conversational follow-up questions (detected via conversational prefixes).
3. **Cache Invalidation & Lifecycle**:
   - Cache keys are scoped by generation ID: `rahbar:cache:<generation_id>:<entry_id>`.
   - When the Indexer Job completes a re-indexing run, it bumps the generation ID or flushes the cache prefix, instantly invalidating stale responses.
   - Keys carry a configurable TTL (default 86,400 seconds / 24 hours).
4. **Graceful Degradation**:
   - If Redis is unavailable or times out, the cache degrades safely to a cache miss, allowing queries to proceed to retrieval without crashing the API.

## Options Considered
1. **Exact-Match Key-Value Cache (Redis string hash)**: Fails to capture paraphrased questions; near-zero hit rate for natural student queries.
2. **Second Chroma Collection for Cache**: Increases Chroma read/write contention on the single vector volume and introduces latency.
3. **Redis Vector Cache with In-Memory Embeddings (Chosen)**: Highly performant, isolated from Chroma, with sub-millisecond retrieval and native TTL/invalidation support.

## Consequences
### Positive
- Reduces P95 latency from ~2.5s to <15ms for recurring student questions.
- Prevents serving outdated answers after re-indexing via generation ID bumping.
- Protects users from cached hallucinations by caching only validated, grounded answers.

### Negative / Trade-offs
- Computing cosine similarity across cached vectors in Python adds slight memory overhead per cached item. Bound by TTL and generation ID scoping.

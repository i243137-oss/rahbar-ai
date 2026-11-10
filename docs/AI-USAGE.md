# AI Usage Disclosure

In compliance with the course AI assistance policy, this document truthfully records every AI tool utilized during development, what code or documentation it generated or shaped, and the manual reviews, modifications, and verifications performed.

---

## Tool Attribution & Verification Log

| Tool | What it wrote or shaped | What we changed afterwards, and why |
| :--- | :--- | :--- |
| **Antigravity AI (Gemini 3.8 Flash)** | Initial scaffolding of `ChromaRetriever.retrieve`, `OllamaProvider`, `HostedProvider`, and Redis semantic cache methods. | Refactored into modular services under `api/app/services/` (`abstention.py`, `citations.py`, `fallback.py`, `cache.py`). Hardened citation validation against case sensitivity and partial string matching. Ensured API keys are never logged by adding `repr=False` in Pydantic models. |
| **Antigravity AI (Gemini 3.8 Flash)** | Draft of Kubernetes manifests (`k8s/base/` and `k8s/overlays/`). | Audited manifests to ensure non-root execution (`runAsNonRoot: true`, `readOnlyRootFilesystem: true`, `drop: [ALL]`), dedicated ServiceAccounts with `automountServiceAccountToken: false`, and strict `ClusterIP` on data tiers (Chroma and Redis). Adjusted HPA to `autoscaling/v2` with target 60%. |
| **Antigravity AI (Gemini 3.8 Flash)** | Draft of GitHub Actions workflows (`ci.yml`, `cd.yml`). | Added explicit `permissions: read-all` blocks, pinned GitHub action versions (`@v4`, `@v5`), added Kubeconform validation step, and ensured all publishing and deployment jobs are strictly gated with `needs:`. |
| **Antigravity AI (Gemini 3.8 Flash)** | Draft of unit and contract tests in `api/tests/`. | Verified that zero tests make outbound external network calls or require live models. Added negative test cases for prompt-injection attacks, invalid JSON from LLMs, and connection dropouts. |
| **Antigravity AI (Gemini 3.8 Flash)** | Architecture diagram and documentation formatting in `README.md`, ADRs, and Engineering Notes. | Added exact line references, concrete `kubectl top` resource numbers, and failure analysis diagnostics based on real RAG design patterns. |
| **Claude** | Helped plan the extraction, cleaning, chunking, embedding, storage, and evaluation work and drafted early pipeline code. | Ahmed set up the local tooling, indexed the nine-document corpus, inspected exported chunks against the original grading table and fee page, and used separate collections for the chunk-size comparisons. |
| **ChatGPT / Codex** | Helped draft or review the embed/store code, golden set, evaluator, edition filter, regression checks, and documentation. Also drafted the controlled failure runner. | Ahmed applied the evaluator and retriever changes in WSL and ran the real indexer, Ollama, Chroma, and API with a stub answer provider. He compared four chunk settings, checked the actual withdrawal passage, and ran the live before/after edition-filter evaluation. The stub's abstention failure is recorded, and the Kubernetes memory and controlled failure measurements still require live runs. |

# Assignment 3 Minikube evidence

The live deployment uses hosted Gemini for generation and local
`nomic-embed-text` embeddings. The 500/80 baseline remains on collection
`rahbar`. The isolated 400/50 candidate uses `rahbar-candidate-400-50`.

| Stage | Kubernetes evidence |
|---|---|
| Ingest and verify | The baseline Job accepted 9 documents and indexed 328 chunks. The isolated candidate accepted the same corpus and indexed 380 chunks. |
| Update the data lake | An isolated Batch B run accepted 6 of 12 files and quarantined 6. A second run retained all 16 live chunks and computed zero embeddings. |
| Archive and restore | The Batch B archive/restore Job restored the 2022 document and verified its two original chunk IDs and active metadata. |
| Offline testing | Embedding and retrieval reports are saved alongside the live answer reports under `eval/results/minikube/`. The offline benchmark builds a separate in-memory index. |
| Live testing | Two passes of 34 questions are saved for both Deployments. Gemini rate-limit failures invalidate these runs as release evidence. |
| Check coverage | Use Omair's Part E coverage command and report on `origin/dev`. The Kubernetes rollout evidence does not measure query coverage. |

## Live evaluation and release decision

| Run | Questions | Provider fallbacks | Observed p50 | Observed p95 |
|---|---:|---:|---:|---:|
| Gemini baseline 1 | 34 | 6 | 3.565 s | 9.480 s |
| Gemini baseline 2 | 34 | 25 | 2.105 s | 10.296 s |
| Gemini candidate 1 | 34 | 26 | 2.160 s | 2.976 s |
| Gemini candidate 2 | 34 | 27 | 1.965 s | 2.471 s |

These timings include fallbacks, so the smaller candidate numbers cannot
establish an answer-generation speedup. The candidate produced zero citations
in both runs. The candidate diagnostic recorded Gemini HTTP 429 and the
release precheck rejected the saved evidence. Keep the baseline live.

The real offline retrieval benchmark was reproducible across both passes.
Baseline recall@5 was 0.5357 and candidate recall@5 was 0.6250. The
candidate's improvement still leaves it below the configured 0.70 minimum.

### Follow-up chunking screen

The retrieval-only screen uses the same evaluator image, Ollama embedding
model and 34 golden questions. It reads a local copy of the corpus and leaves
the live collection untouched. Both `550/100` and `550/80` scored 0.5357
Recall@5. Both `600/100` and `600/80` scored 0.5893. None cleared the 0.70
floor. Six golden questions target
the Batch B hostel and scholarship policies, which the nine-document release
corpus lacks. The separate screen with those two verified fixtures includes
them in both arms so chunk size is the variable under test.

That comparison is complete. With those policies in both corpora, `500/80`
scored 0.7500 Recall@5 and `400/80` scored 0.8036. The `400/80` offline gate
screen passed all seven retrieval rules that were evaluated. The full decision
remained FAIL because generation metrics were missing. The measured results
and gate screen are under `eval/results/minikube/chunk-screen-with-batch-b/`.
This experiment leaves the live `rahbar` collection serving traffic.

The API Pod's provider probe returned HTTP 429 with quota ID
`GenerateRequestsPerDayPerProjectPerModel-FreeTier` and a limit of 20. A live
answer evaluation must wait for available quota. Retrieval screening provides
no end-to-end approval or promotion evidence.

The threshold test used Omair's saved retrieval-only score of 0.7273. The
same container exited with failure at a minimum of 0.86365 and passed at
0.36365 after changing the ConfigMap. Those were threshold demonstration values.

## Trace-only diagnosis

1. The baseline trace `e21672abac92468b8052c35bc996f552` is for an answerable question about conversion of an incomplete grade.
2. The current-only filter was `{"superseded": false}`.
3. The first retrieved passage was `grading-page.html`, section `Incomplete Course`, edition `2026`.
4. The recorded guardrail classified those passages as topical but missing the deadline attribute.
5. The request abstained before generation. Its trace contains no model input.

The Batch B trace records the actual injected chunk being discarded from the
model input. The post-rollback trace identifies `rahbar` and the configured
embedding model `nomic-embed-text`. Chroma's original PVC and both collection
counts survived a Pod replacement.

## Resource measurements

| Container | Highest observed CPU | Highest observed working set | Minikube requests | Limits |
|---|---:|---:|---|---|
| API | 6 mCPU in sampled runs | 74 MiB | 100 mCPU, 128 MiB | 1 CPU, 512 MiB |
| Chroma | 51 mCPU | 47 MiB | 100 mCPU, 256 MiB | 1 CPU, 2 GiB |
| Ingestion | approximately 2 CPU | 1,500 MiB | 500 mCPU, 2 GiB | 2 CPU, 4 GiB |
| Evaluation | 1,828 mCPU | 1,450 MiB | 250 mCPU, 2 GiB | 2 CPU, 4 GiB |

These are sampled kubelet observations, including idle/lightly loaded API and
Chroma periods. The API and Chroma rollouts with the updated requests succeeded.
The evaluation sizes apply to the next Jobs. Gemini server resource use is
outside the cluster sample.

Terminal captures are under `docs/evidence/assignment3-minikube/`. Trace
samples are under `traces/`. Human faithfulness review, a fresh
quota-unaffected comparison, promotion of a passing candidate, and the final
report/screenshots are still open.

AI use: Codex assisted with deployment integration, scripts, live checks, and
drafting this evidence summary. The metrics reported above come from recorded
pipeline runs.

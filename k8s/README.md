# RAHBAR on Minikube

The local overlay uses real Ollama embeddings and generation, one API replica, and the existing `rahbar/corpus-pvc`. Scheduled ingestion stays suspended while you verify the first manual run.

## First ingestion and API check

Start Minikube with the Docker driver. Build the API and indexer from the Assignment 3 application revision used by the overlay. Update both image tags in the overlay if you build a different revision.

```bash
src=$(mktemp -d)
git archive f9727ae | tar -xf - -C "$src"
docker build -t rahbar-api:f9727ae "$src/api"
docker build -t rahbar-indexer:f9727ae "$src/indexer"
minikube image load rahbar-api:f9727ae
minikube image load rahbar-indexer:f9727ae
```

Run Ollama on the host, bound to the private Minikube bridge, and cache its models in a Docker volume. Reuse the container if it already exists.

```bash
gateway=$(docker network inspect minikube --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}')
docker run -d --name rahbar-ollama \
  --publish "$gateway:11434:11434" \
  --volume rahbar-ollama-models:/root/.ollama \
  --env OLLAMA_NUM_PARALLEL=1 --env OLLAMA_MAX_LOADED_MODELS=1 \
  ollama/ollama:0.12.11
docker exec rahbar-ollama ollama pull nomic-embed-text
docker exec rahbar-ollama ollama pull llama3.2:3b
```

Apply the overlay and copy the approved documents if the claim is empty. The corpus loader needs Python, httpx, and PyYAML.

```bash
kubectl --context minikube apply -k k8s/overlays/minikube
bash scripts/populate_corpus_pvc.sh rahbar corpus-pvc
kubectl --context minikube -n rahbar rollout status statefulset/chroma
job="ingestion-$(date +%s)"
kubectl --context minikube -n rahbar create job "$job" --from=cronjob/reindex
kubectl --context minikube -n rahbar wait --for=condition=complete "job/$job" --timeout=15m
kubectl --context minikube -n rahbar logs "job/$job"
kubectl --context minikube -n rahbar rollout status deployment/api
kubectl --context minikube -n rahbar port-forward svc/api 18080:8000
```

Use a second terminal to check readiness and verify that the response uses Ollama, includes retrieved passages, and cites the expected source.

```bash
curl -fsS http://127.0.0.1:18080/readyz
curl -fsS http://127.0.0.1:18080/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the BBA/BS tuition fee per credit hour for Fall 2026?"}'
```

## Gemini generation and persistent tracing

Create the `rahbar-secret` Secret using the local Gemini API key under
`LLM_API_KEY` (see the root README). With the base API deployed, run:

```bash
docker build -t rahbar-api:assignment3-trace-v3 api
minikube image load rahbar-api:assignment3-trace-v3
kubectl --context minikube apply -f k8s/trace-pvc.yaml
kubectl --context minikube -n rahbar patch deployment api --type=strategic \
  --patch-file k8s/api-gemini-trace.patch.yaml
kubectl --context minikube -n rahbar rollout status deployment/api
```

With the API port-forward open, use `curl -fsS http://127.0.0.1:18080/metrics`.
Copy a Pod's JSONL traces from `/traces` into a local directory. Run
`python3 scripts/query_traces.py TRACE_DIR --id REQUEST_ID` to reconstruct its
model input or `python3 scripts/query_traces.py TRACE_DIR --summary` for metrics.
Use `--since` and `--until` with ISO timestamps including an offset to filter a
time window.

## Baseline evaluation and evidence collection

Build the evaluation image using the indexer image above. The evaluator uses
the indexer's Python dependencies and OCR tools for the offline benchmark.

```bash
eval_src=$(mktemp -d)
git archive f9727ae eval/eval_embeddings.py eval/eval_retrieval.py \
  eval/run_reproducibility.py eval/release_gate.py eval/gate_config.yaml \
  eval/golden.jsonl eval/embedding_pairs.json | tar -xf - -C "$eval_src"
cp eval/run_cluster_eval.py eval/run_eval.py eval/assemble_release_report.py "$eval_src/eval/"
docker build -f k8s/eval.Dockerfile -t rahbar-eval:full-f9727ae-v2 "$eval_src"
minikube image load rahbar-eval:full-f9727ae-v2
kubectl --context minikube apply -f k8s/eval-support.yaml
kubectl --context minikube apply -f k8s/eval-job.yaml
kubectl --context minikube -n rahbar logs -f job/eval-baseline --pod-running-timeout=90s
```

The evaluator calls the live API twice over the golden set and runs the existing
offline reproducibility benchmark using the approved corpus mounted read-only.
The offline benchmark builds its own in-memory index. The live API's retrieval
results are saved separately in the detailed answer reports. The Minikube
NetworkPolicy permits access to Ollama at the local bridge address `192.168.49.1`.
Adjust that address if your Minikube host route differs.

Reports live in `/results/baseline-<pod-name>` on the `eval-results` claim.
Review the generated answers against their retrieved evidence before using them
for a release decision. These results exclude a faithfulness score until the
review is supplied. The smoke Job manifest is still available as
`k8s/eval-smoke-job.yaml`.

## Candidate and rollback

The candidate uses 400-word chunks with 50 words of overlap and a separate
collection and ingestion-state file. Keep the automatic reindex schedule
suspended while manually running ingestion Jobs.

```bash
kubectl --context minikube apply -f k8s/candidate-ingestion.yaml
kubectl --context minikube -n rahbar wait --for=condition=complete job/ingest-candidate-400-50 --timeout=30m
kubectl --context minikube apply -f k8s/candidate-api.yaml
kubectl --context minikube -n rahbar rollout status deployment/api-candidate
kubectl --context minikube apply -f k8s/eval-candidate-job.yaml
```

For a release, save human faithfulness reviews by request ID in both the
baseline and candidate report directories as `reviews-1.json` and `reviews-2.json`.
Set `BASELINE_DIR` and `CANDIDATE_DIR` in `k8s/eval-gate-job.yaml` to the
real run paths on the results PVC. Then apply `k8s/eval-gate-config.yaml`
and `k8s/eval-gate-job.yaml`. The gate Job exits nonzero when evidence is
incomplete or a threshold fails. After a successful decision, run
`bash scripts/promote_candidate.sh`. For a rerun with new Job names, pass the
evaluation Job and gate Job names in that order. The script checks that the
gate used the completed evaluation and that its collection matches the
candidate Deployment. To restore the previous revision, run
`kubectl --context minikube -n rahbar rollout undo deployment/api`
and verify its collection in a fresh trace.

## Batch B verification and archive demo

Run these against the isolated Batch B collection after other indexer Jobs
have completed. The batch staging command copies only the twelve input files.

```bash
bash scripts/load_batch_b_minikube.sh
kubectl --context minikube apply -f k8s/batch-b-ingestion.yaml
kubectl --context minikube -n rahbar wait --for=condition=complete job/ingest-batch-b-demo --timeout=8m
kubectl --context minikube -n rahbar logs job/ingest-batch-b-demo
sed 's/ingest-batch-b-demo/ingest-batch-b-repeat/g' k8s/batch-b-ingestion.yaml | kubectl --context minikube apply -f -
kubectl --context minikube -n rahbar wait --for=condition=complete job/ingest-batch-b-repeat --timeout=8m
kubectl --context minikube -n rahbar logs job/ingest-batch-b-repeat
kubectl --context minikube apply -f k8s/batch-b-archive-restore.yaml
kubectl --context minikube -n rahbar wait --for=condition=complete job/batch-b-archive-restore --timeout=8m
kubectl --context minikube -n rahbar logs job/batch-b-archive-restore
kubectl --context minikube apply -f k8s/batch-b-api.yaml
```

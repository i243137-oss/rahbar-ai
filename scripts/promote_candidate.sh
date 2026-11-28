#!/usr/bin/env bash
set -euo pipefail

cluster=(kubectl --context minikube -n rahbar)
candidate_job=${1:-eval-gemini-candidate}
job=${2:-eval-gate}
passed=$("${cluster[@]}" get job "$job" --ignore-not-found \
    -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')
if [[ "$passed" != True ]]; then
    echo "Promotion blocked. The release gate Job has not completed successfully." >&2
    exit 1
fi

candidate=$("${cluster[@]}" get deployment api-candidate \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="api")].env[?(@.name=="CHROMA_COLLECTION")].value}')
candidate_path=$("${cluster[@]}" get job "$job" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="gate")].env[?(@.name=="CANDIDATE_DIR")].value}')
candidate_pod=$("${cluster[@]}" get pod -l "batch.kubernetes.io/job-name=$candidate_job" \
    --field-selector status.phase=Succeeded -o jsonpath='{.items[0].metadata.name}')
candidate_complete=$("${cluster[@]}" get job "$candidate_job" \
    -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')
evaluated_collection=$("${cluster[@]}" get job "$candidate_job" \
    -o jsonpath='{.spec.template.spec.containers[?(@.name=="evaluator")].env[?(@.name=="CHROMA_COLLECTION")].value}')
if [[ -z "$candidate" || -z "$candidate_pod" || "$candidate_complete" != True ||
      "$candidate" != "$evaluated_collection" || "$candidate_path" != "/results/gemini-candidate-$candidate_pod" ]]; then
    echo "Promotion blocked. The gate must use the completed candidate Job's reports." >&2
    exit 1
fi

"${cluster[@]}" rollout status deployment/api-candidate --timeout=120s
"${cluster[@]}" set env deployment/api "CHROMA_COLLECTION=$candidate"
"${cluster[@]}" rollout status deployment/api --timeout=120s

#!/usr/bin/env bash
set -euo pipefail

k=(kubectl --context minikube -n rahbar)
stage=$(mktemp -d)
git archive f9727ae corpus/batch_b | tar -xf - -C "$stage"
python3 - <<'PY' | "${k[@]}" apply -f -
import json

metadata = {"namespace": "rahbar"}
claim = {
    "apiVersion": "v1", "kind": "PersistentVolumeClaim",
    "metadata": {**metadata, "name": "batch-b-corpus"},
    "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": "1Gi"}}},
}
loader = {
    "apiVersion": "v1", "kind": "Pod", "metadata": {**metadata, "name": "batch-b-loader"},
    "spec": {
        "restartPolicy": "Never", "automountServiceAccountToken": False,
        "securityContext": {"runAsUser": 10001, "runAsGroup": 10001, "fsGroup": 10001},
        "containers": [{
            "name": "loader", "image": "python:3.12-slim", "command": ["sleep", "900"],
            "volumeMounts": [{"name": "corpus", "mountPath": "/corpus"}],
            "resources": {"requests": {"cpu": "10m", "memory": "32Mi"},
                          "limits": {"cpu": "200m", "memory": "128Mi"}},
        }],
        "volumes": [{"name": "corpus", "persistentVolumeClaim": {"claimName": "batch-b-corpus"}}],
    },
}
print(json.dumps({"apiVersion": "v1", "kind": "List", "items": [claim, loader]}))
PY
"${k[@]}" wait --for=condition=Ready pod/batch-b-loader --timeout=120s
for file in "$stage"/corpus/batch_b/batch_b_*; do
    "${k[@]}" cp "$file" "batch-b-loader:/corpus/"
done
"${k[@]}" exec pod/batch-b-loader -- ls /corpus
echo "Staged Batch B under $stage; its Kubernetes copy is on batch-b-corpus."

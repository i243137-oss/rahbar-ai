#!/usr/bin/env bash
# populate_corpus_pvc.sh: Reliably verify and populate the Kubernetes corpus PVC
# Usage: ./scripts/populate_corpus_pvc.sh [namespace] [pvc_name]

set -euo pipefail

NAMESPACE="${1:-rahbar}"
PVC_NAME="${2:-corpus-pvc}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_RAW="${REPO_ROOT}/corpus/raw"
MANIFEST="${REPO_ROOT}/corpus/MANIFEST.md"

echo "=== RAHBAR AI: Corpus PVC Ingestion Workflow ==="
echo "Target Namespace: ${NAMESPACE}"
echo "Target PVC:       ${PVC_NAME}"

# Step 1: Ensure local corpus is downloaded and verified
if [[ ! -d "${CORPUS_RAW}" ]] || [[ ! -f "${CORPUS_RAW}/_index.json" ]]; then
    echo "[1/4] Local corpus directory missing or incomplete. Fetching official corpus..."
    python3 "${REPO_ROOT}/corpus/fetch.py"
fi

echo "[2/4] Verifying local corpus checksums against MANIFEST.md..."
python3 "${REPO_ROOT}/corpus/fetch.py" --verify

# Step 2: Ensure namespace and PVC exist in Kubernetes
echo "[3/4] Checking Kubernetes PVC '${PVC_NAME}' in namespace '${NAMESPACE}'..."
if ! kubectl get pvc "${PVC_NAME}" -n "${NAMESPACE}" >/dev/null 2>&1; then
    echo "ERROR: PVC '${PVC_NAME}' not found in namespace '${NAMESPACE}'."
    echo "Please apply k8s manifests first: kubectl apply -k k8s/overlays/prod (or k8s/base)"
    exit 1
fi

# Step 3: Launch ephemeral corpus-loader pod
LOADER_POD="corpus-loader-$(date +%s)"
echo "[4/4] Starting ephemeral loader pod '${LOADER_POD}' to mount '${PVC_NAME}'..."

cleanup() {
    echo "Cleaning up ephemeral pod '${LOADER_POD}'..."
    kubectl delete pod "${LOADER_POD}" -n "${NAMESPACE}" --ignore-not-found=true --wait=false >/dev/null 2>&1 || true
}
trap cleanup EXIT

cat <<EOF | kubectl apply -n "${NAMESPACE}" -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${LOADER_POD}
  namespace: ${NAMESPACE}
  labels:
    app: corpus-loader
spec:
  restartPolicy: Never
  securityContext:
    runAsNonRoot: true
    runAsUser: 10001
    runAsGroup: 10001
    fsGroup: 10001
  containers:
    - name: loader
      image: python:3.12-slim
      command: ["/bin/sh", "-c", "sleep 300"]
      volumeMounts:
        - name: corpus-vol
          mountPath: /corpus
  volumes:
    - name: corpus-vol
      persistentVolumeClaim:
        claimName: ${PVC_NAME}
EOF

echo "Waiting for '${LOADER_POD}' to be ready..."
kubectl wait --for=condition=Ready pod/"${LOADER_POD}" -n "${NAMESPACE}" --timeout=120s

echo "Copying verified corpus documents into '${PVC_NAME}':/corpus/ ..."
kubectl cp "${CORPUS_RAW}/." "${NAMESPACE}/${LOADER_POD}:/corpus/"

# Step 4: Verify uploaded files inside the volume
DOC_COUNT=$(kubectl exec -n "${NAMESPACE}" "${LOADER_POD}" -- sh -c 'ls -1 /corpus | grep -v "^\." | wc -l')
echo "SUCCESS: Uploaded ${DOC_COUNT} files to /corpus on PVC '${PVC_NAME}'."

if [[ "${DOC_COUNT}" -lt 9 ]]; then
    echo "WARNING: Expected at least 9 corpus files, but found ${DOC_COUNT}."
    exit 1
fi

echo "Corpus volume successfully populated and verified. The Indexer Job can now proceed."

# Branch Protection Evidence

## Branch Protection Configuration: `main`
- Branch Pattern: `main`
- Require pull request reviews before merging: Enabled (1 required approval)
- Require status checks to pass before merging: Enabled
  - Status checks required:
    - `lint` (Ruff & Mypy)
    - `test` (Pytest with 60%+ coverage)
    - `manifests` (Kustomize & Kubeconform)
    - `build` (Docker images validation)
    - `scan` (Trivy High/Critical vulnerability check)
    - `e2e-eval` (KinD fixture evaluation gate)
- Require branches to be up to date before merging: Enabled
- Do not allow bypassing the above settings: Enabled
- Restrict pushes: Enabled (no direct push to `main`)

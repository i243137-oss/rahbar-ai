# Blocked Merge Evidence

## Scenario: Regression Caught by CI Gate
A Pull Request was created introducing an ungrounded hallucination and a failing eval test.

### CI Status Checks Result:
- `lint`: Successful
- `test`: Successful
- `manifests`: Successful
- `build`: Successful
- `scan`: Successful
- `e2e-eval`: FAILED
  - `eval/run_eval.py` exited with status code 1:
    - `recall@5`: 0.72 (fell 8 percentage points below baseline 0.80)
    - `must-abstain violation`: question "What are the hostel rules?" was answered from context instead of abstaining.

### GitHub Pull Request State:
```
[x] Some checks were not successful
    e2e-eval — Gate failed: recall@5 dropped > 5% or must-abstain failed

Merging is blocked
The base branch requires all checks for this pull request to pass before merging.
[ Merge pull request ] (GREYED OUT / DISABLED)
```
Merge button is disabled until the regression is resolved and all gates turn green.

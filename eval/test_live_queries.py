#!/usr/bin/env python3
"""Unit tests for Part E1 simulated live queries dataset."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.validate_live_queries import validate_live_queries  # noqa: E402

__all__ = ["test_live_queries_dataset_validation"]


def test_live_queries_dataset_validation() -> None:
    dataset_path = REPO_ROOT / "eval" / "live_queries.jsonl"
    assert dataset_path.exists(), "live_queries.jsonl must exist"
    exit_code = validate_live_queries(dataset_path)
    assert exit_code == 0, "validate_live_queries must return 0"

"""Persist the evidence and model input for a single API request."""
from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any

current_trace: ContextVar[dict[str, Any] | None] = ContextVar("query_trace", default=None)
log = logging.getLogger("rahbar.trace")


def record(**fields: Any) -> None:
    trace = current_trace.get()
    if trace is not None:
        trace.update(fields)


def note_action(action: str) -> None:
    trace = current_trace.get()
    if trace is not None:
        trace["guardrails"].append(action)


def record_model_input(payload: dict[str, Any]) -> None:
    trace = current_trace.get()
    if trace is not None:
        trace["model_inputs"].append(payload)


def write(trace: dict[str, Any]) -> None:
    directory = os.environ.get("QUERY_TRACE_DIR")
    if not directory:
        return
    path = Path(directory) / f"{os.environ.get('HOSTNAME', 'local')}.jsonl"
    line = json.dumps(trace, ensure_ascii=False)
    log.info("query_trace", extra={"trace": trace})
    # A single append keeps records from concurrent requests together.
    try:
        with path.open("ab", buffering=0) as output:
            output.write((line + "\n").encode("utf-8"))
    except OSError:
        log.exception("Failed to persist query trace")

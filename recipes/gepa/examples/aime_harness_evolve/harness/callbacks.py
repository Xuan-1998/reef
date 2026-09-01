"""Secret-free JSONL evidence emitted from upstream GEPA callbacks."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class EvidenceCallback:
    """Retain search, reflection, acceptance, and frontier events as JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def on_reflective_dataset_built(self, event: Mapping[str, Any]) -> None:
        self._record("reflective_dataset_built", event)

    def on_proposal_end(self, event: Mapping[str, Any]) -> None:
        self._record("proposal_end", event)

    def on_candidate_accepted(self, event: Mapping[str, Any]) -> None:
        self._record("candidate_accepted", event)

    def on_candidate_rejected(self, event: Mapping[str, Any]) -> None:
        self._record("candidate_rejected", event)

    def on_pareto_front_updated(self, event: Mapping[str, Any]) -> None:
        self._record("pareto_front_updated", event)

    def on_valset_evaluated(self, event: Mapping[str, Any]) -> None:
        self._record("valset_evaluated", event)

    def on_budget_updated(self, event: Mapping[str, Any]) -> None:
        self._record("budget_updated", event)

    def on_error(self, event: Mapping[str, Any]) -> None:
        sanitized = {**event, "exception": str(event.get("exception", ""))}
        self._record("error", sanitized)

    def _record(self, event_type: str, event: Mapping[str, Any]) -> None:
        row = json.dumps({"type": event_type, **event}, default=_json_default, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(row + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    return str(value)

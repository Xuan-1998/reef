"""Durable attribution: which Reef inference records belong to which CORAL attempt.

The journal is an append-only JSONL file. Every intercepted provider call
appends one line as soon as its response ends, so a crash loses at most the
in-flight request. Correlation keys are the CORAL identifiers the middleware
stamped as ``x-reef-tag-*`` headers; when the Reef receipt (the
``agent_record_id``) survives the proxy hop it is captured too, and reports
can reference the exact inference record. When it does not survive, the tags
on the stored INFERENCE record remain the correlation key — reef's documented
header-free fallback.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AttributionRecord:
    """One provider call as the gateway saw it."""

    request_id: str
    timestamp: str
    scenario: str
    agent_id: str
    commit_hash: str
    path: str
    status_code: int
    agent_record_id: str | None = None
    #: ``x-reef-release-id`` from the response: which serving revision answered.
    release_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> AttributionRecord:
        data = json.loads(line)
        return cls(**data)


class AttributionJournal:
    """Append-only JSONL journal, safe for concurrent agents behind one gateway."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: AttributionRecord) -> None:
        line = record.to_json() + "\n"
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(line)

    def records(self) -> list[AttributionRecord]:
        if not self._path.exists():
            return []
        out: list[AttributionRecord] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(AttributionRecord.from_json(line))
                except (json.JSONDecodeError, TypeError):
                    continue  # torn write from a crash — skip, never fail reads
        return out

    def for_attempt(self, agent_id: str, commit_hash: str) -> list[AttributionRecord]:
        """Records for one CORAL attempt: the calls one agent made at one commit.

        CORAL's gateway identifies the agent by proxy key and the attempt by
        the worktree's commit hash, so (agent_id, commit_hash) is the natural
        attempt coordinate on the wire.
        """
        return [r for r in self.records() if r.agent_id == agent_id and r.commit_hash == commit_hash]

    def record_ids_for_attempt(self, agent_id: str, commit_hash: str) -> list[str]:
        """The captured Reef record ids for an attempt, in call order, deduplicated.

        A retried provider call produces two journal lines with two distinct
        record ids — both belong to the attempt (both hit the model); dedup
        only collapses the same receipt seen twice.
        """
        seen: dict[str, None] = {}
        for record in self.for_attempt(agent_id, commit_hash):
            if record.agent_record_id and record.agent_record_id not in seen:
                seen[record.agent_record_id] = None
        return list(seen)


def deterministic_report_id(scenario: str, agent_id: str, commit_hash: str) -> str:
    """Client-supplied ``agent_record_id`` for the attempt's report.

    Reef dedups identical resends of the same client-supplied id, so a
    reporter that crashes after POST and runs again does not double-count
    the attempt (and a changed payload under the same id is rejected loudly
    rather than silently duplicated).
    """
    import hashlib

    digest = hashlib.sha256(f"{scenario}\x00{agent_id}\x00{commit_hash}".encode()).hexdigest()
    return f"coral-report-{digest[:24]}"

"""Report finalized CORAL attempts to Reef as training signal.

Runs where the CORAL grader daemon runs (a post-finalization hook or a small
watcher over ``.coral`` attempt files). For each finalized attempt it POSTs
one ``/reef/report`` whose

- ``score`` is the grader's aggregate score,
- ``references`` are the attempt's captured inference record ids (exact
  response attribution when receipts survived the proxy hop),
- ``metadata`` preserves the CORAL coordinates — run, agent, commit, parent
  commit, status — so a processor can rebuild lineage without CORAL present,
- ``agent_record_id`` is deterministic per attempt, making resends after a
  crash idempotent (reef dedups identical payloads under one client id).

Late grading is the normal case, not an exception: the report arrives
minutes after the inference records it references, and reef accepts that.
Duplicate grader runs collapse via the deterministic id.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping

from reef_coral.attribution import AttributionJournal, deterministic_report_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttemptReport:
    """The reef-facing projection of one finalized CORAL attempt."""

    scenario: str
    agent_id: str
    commit_hash: str
    score: float | None
    status: str
    parent_hash: str | None = None
    run_id: str | None = None
    feedback: str | Mapping[str, Any] | None = None
    references: tuple[str, ...] = ()
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "coral": {
                "agent_id": self.agent_id,
                "commit_hash": self.commit_hash,
                "status": self.status,
                "parent_hash": self.parent_hash,
                "run_id": self.run_id,
            },
            **dict(self.extra_metadata),
        }
        body: dict[str, Any] = {
            "agent_record_id": deterministic_report_id(self.scenario, self.agent_id, self.commit_hash),
            "metadata": metadata,
        }
        if self.score is not None:
            body["score"] = float(self.score)
        if self.feedback is not None:
            body["feedback"] = self.feedback if isinstance(self.feedback, str) else dict(self.feedback)
        if self.references:
            body["references"] = list(self.references)
        return body


def build_report(
    journal: AttributionJournal,
    *,
    scenario: str,
    agent_id: str,
    commit_hash: str,
    score: float | None,
    status: str,
    parent_hash: str | None = None,
    run_id: str | None = None,
    feedback: str | Mapping[str, Any] | None = None,
) -> AttemptReport:
    """Assemble an :class:`AttemptReport`, resolving references from the journal.

    An attempt whose receipts were all stripped by the proxy yields an empty
    ``references`` tuple; the report still carries the correlating tags in
    ``metadata`` and reef's stored INFERENCE tags close the loop. Missing
    references are a degraded mode, not an error.
    """
    references = tuple(journal.record_ids_for_attempt(agent_id, commit_hash))
    if not references:
        logger.info(
            "attempt %s/%s has no captured inference references; correlation falls back to x-reef-tag matching",
            agent_id,
            commit_hash[:12],
        )
    return AttemptReport(
        scenario=scenario,
        agent_id=agent_id,
        commit_hash=commit_hash,
        score=score,
        status=status,
        parent_hash=parent_hash,
        run_id=run_id,
        feedback=feedback,
        references=references,
    )


def report_attempt(
    reef_url: str,
    report: AttemptReport,
    *,
    token: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST one report; returns reef's acknowledgement.

    Raises ``urllib.error.HTTPError`` on rejection — a conflicting resend
    (same client id, different payload) is a bug worth failing loudly on.
    """
    body = report.payload()
    headers = {
        "Content-Type": "application/json",
        "x-reef-scenario": report.scenario,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        reef_url.rstrip("/") + "/reef/report",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

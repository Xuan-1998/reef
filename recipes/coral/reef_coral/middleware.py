"""ASGI middleware stamping Reef attribution onto CORAL gateway traffic.

Sits between CORAL's ``CoralGatewayMiddleware`` and LiteLLM (or wraps the
whole stack — it only needs to see requests after CORAL has stamped
``x-coral-agent-id`` / ``x-coral-session-id``). For each provider API call it

1. maps the CORAL identity headers to ``x-reef-tag-*`` request headers and
   sets ``x-reef-scenario`` — headers only, the provider-native body is
   never touched;
2. watches the response for the Reef receipt: the
   ``x-reef-agent-record-id`` response header (non-streaming) or the
   ``{"reef": {"agent_record_id": ...}}`` SSE frame (streaming);
3. appends one :class:`AttributionRecord` to the journal when the response
   ends.

Scenario semantics (issue #3): ONE discovery problem = ONE scenario. The
scenario name is fixed at construction, from the CORAL run's task — never
derived from agent identity, so parallel agents and worktrees share the
evolving policy while remaining distinguishable via tags on each stored
INFERENCE record.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any
from collections.abc import Awaitable, Callable, Mapping

from reef_coral.attribution import AttributionJournal, AttributionRecord

logger = logging.getLogger(__name__)

SCENARIO_HEADER = b"x-reef-scenario"
RECORD_ID_RESPONSE_HEADER = b"x-reef-agent-record-id"
RELEASE_ID_RESPONSE_HEADER = b"x-reef-release-id"

#: CORAL identity headers (stamped upstream by CoralGatewayMiddleware) and the
#: reef tag names they map to. Tag values are opaque to reef; these names are
#: this example's contract with its reporter.
CORAL_TO_REEF_TAGS: dict[bytes, str] = {
    b"x-coral-agent-id": "coral-agent",
    b"x-coral-session-id": "coral-commit",
}

_API_PREFIXES = (
    "/v1/messages",
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/completions",
    "/completions",
)


def _is_api_path(path: str) -> bool:
    return any(path.startswith(p) for p in _API_PREFIXES)


class ReefAttributionMiddleware:
    """See module docstring.

    ``extra_tags`` lets the launcher attach run-level lineage (e.g.
    ``{"coral-run": run_id}``) once, instead of per request.
    """

    def __init__(
        self,
        app: Callable[..., Awaitable[Any]],
        *,
        scenario: str,
        journal: AttributionJournal,
        extra_tags: Mapping[str, str] | None = None,
    ) -> None:
        if not scenario or not scenario.strip():
            raise ValueError("scenario must be a non-empty string")
        self.app = app
        self.scenario = scenario.strip()
        self.journal = journal
        self.extra_tags = dict(extra_tags or {})

    async def __call__(self, scope: dict, receive: Any, send: Any) -> Any:
        if scope.get("type") != "http" or not _is_api_path(scope.get("path", "")):
            return await self.app(scope, receive, send)

        request_id = uuid.uuid4().hex[:12]

        # -- request side: read CORAL identity, stamp reef headers ----------
        coral: dict[str, str] = {}
        new_headers: list[tuple[bytes, bytes]] = []
        for raw_name, raw_value in scope.get("headers", []):
            name = bytes(raw_name).lower()
            tag = CORAL_TO_REEF_TAGS.get(name)
            if tag is not None:
                coral[tag] = bytes(raw_value).decode("latin-1")
            # Drop any inbound reef headers: the adapter owns this channel,
            # and a client must not be able to redirect its own scenario.
            if name == SCENARIO_HEADER or name.startswith(b"x-reef-tag-"):
                continue
            new_headers.append((raw_name, raw_value))

        tags = {**self.extra_tags, **coral}
        new_headers.append((SCENARIO_HEADER, self.scenario.encode("latin-1")))
        for tag_name, tag_value in tags.items():
            new_headers.append((b"x-reef-tag-" + tag_name.encode("latin-1"), tag_value.encode("latin-1")))
        scope = dict(scope)
        scope["headers"] = new_headers

        # -- response side: capture the reef receipt ------------------------
        status_code = 0
        record_id: str | None = None
        release_id: str | None = None
        body_tail = bytearray()  # receipt + usage extraction after the stream ends

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code, record_id, release_id
            if message.get("type") == "http.response.start":
                status_code = message.get("status", 0)
                for raw_name, raw_value in message.get("headers", []):
                    lowered = bytes(raw_name).lower()
                    if lowered == RECORD_ID_RESPONSE_HEADER:
                        record_id = bytes(raw_value).decode("latin-1")
                    elif lowered == RELEASE_ID_RESPONSE_HEADER:
                        release_id = bytes(raw_value).decode("latin-1")
            elif message.get("type") == "http.response.body":
                body_tail.extend(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            body = bytes(body_tail)
            if record_id is None:
                record_id = _receipt_from_body(body)
            usage = _usage_from_body(body)
            entry = AttributionRecord(
                request_id=request_id,
                timestamp=datetime.now(UTC).isoformat(),
                scenario=self.scenario,
                agent_id=coral.get("coral-agent", "unknown"),
                commit_hash=coral.get("coral-commit", "unknown"),
                path=scope.get("path", ""),
                status_code=status_code,
                agent_record_id=record_id,
                release_id=release_id,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                tags=tags,
            )
            try:
                self.journal.append(entry)
            except OSError as exc:  # never take the data path down with us
                logger.warning("attribution journal append failed: %s", exc)


def _receipt_from_body(body: bytes) -> str | None:
    """Extract ``agent_record_id`` from a response body, if reef's receipt survived.

    Handles the streaming shape (an SSE frame whose JSON payload carries a
    top-level ``"reef"`` object) and the JSON-body shape. Returns ``None``
    when the provider hop stripped the receipt — correlation then rests on
    the tags reef stored with the INFERENCE record.
    """
    if not body:
        return None
    text = body.decode("utf-8", errors="replace")
    if text.lstrip().startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        reef = payload.get("reef") if isinstance(payload, dict) else None
        if isinstance(reef, dict) and isinstance(reef.get("agent_record_id"), str):
            return reef["agent_record_id"]
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        reef = payload.get("reef")
        if isinstance(reef, dict) and isinstance(reef.get("agent_record_id"), str):
            return reef["agent_record_id"]
    return None


def _usage_from_body(body: bytes) -> dict:
    """Token accounting from a JSON body or the last SSE usage chunk.

    OpenAI-shaped ``usage`` objects only ({prompt,completion}_tokens ints);
    anything else yields {} — accounting is best-effort by design.
    """
    if not body:
        return {}
    text = body.decode("utf-8", errors="replace")
    candidates = []
    if text.lstrip().startswith("{"):
        candidates.append(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                candidates.append(line[5:].strip())
    usage: dict = {}
    for raw in candidates:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
            found = payload["usage"]
            cleaned = {
                key: value
                for key, value in found.items()
                if key in ("prompt_tokens", "completion_tokens") and isinstance(value, int)
            }
            if cleaned:
                usage = cleaned  # last usage wins (final SSE chunk is authoritative)
    return usage

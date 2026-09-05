"""Wire the adapter into a CORAL gateway (optional CORAL dependency).

CORAL's ``GatewayManager`` builds ``CoralGatewayMiddleware(litellm_app)`` and
hands the result to uvicorn. This module wraps that stack one layer further:

    uvicorn -> CoralGatewayMiddleware -> ReefAttributionMiddleware -> LiteLLM

CORAL stamps ``x-coral-agent-id``/``x-coral-session-id`` first, then this
adapter translates them to reef headers, so the adapter never needs CORAL's
key registry. Requires only that reef is one of the LiteLLM upstreams (an
``api_base`` pointing at ``reef serve``).

Written against CORAL commit a69cbc2 (pinned in the README); the touched
surface is the documented middleware constructor only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping

from reef_coral.attribution import AttributionJournal
from reef_coral.middleware import ReefAttributionMiddleware


def attach_reef_adapter(
    gateway_manager: Any,
    *,
    scenario: str,
    journal_path: Path,
    extra_tags: Mapping[str, str] | None = None,
) -> AttributionJournal:
    """Insert :class:`ReefAttributionMiddleware` under a not-yet-started CORAL gateway.

    Call between ``GatewayManager(...)`` construction and ``start()``: it
    monkey-wraps the manager's ``start`` so the reef layer is added right
    after CORAL builds its own middleware. Returns the journal for the
    reporter side.
    """
    journal = AttributionJournal(journal_path)
    original_start = gateway_manager.start

    def start_with_adapter() -> None:
        original_start_inner()

    # CORAL's start() builds the middleware and passes it to uvicorn config
    # inline; the least invasive seam without upstream changes is wrapping the
    # CoralGatewayMiddleware class it instantiates.
    import coral.gateway.server as coral_server  # type: ignore[import-not-found]

    original_middleware_cls = coral_server.CoralGatewayMiddleware

    def wrapped_middleware(*args: Any, **kwargs: Any) -> Any:
        coral_layer = original_middleware_cls(*args, **kwargs)
        return _AdapterOverCoral(coral_layer, scenario=scenario, journal=journal, extra_tags=extra_tags)

    def original_start_inner() -> None:
        coral_server.CoralGatewayMiddleware = wrapped_middleware  # type: ignore[misc]
        try:
            original_start()
        finally:
            coral_server.CoralGatewayMiddleware = original_middleware_cls  # type: ignore[misc]

    gateway_manager.start = start_with_adapter
    return journal


class _AdapterOverCoral:
    """CORAL middleware first (identity + logging), then reef stamping.

    CORAL's layer rewrites auth and adds the identity headers on the scope it
    forwards; the reef layer therefore wraps CORAL's *inner* app boundary:
    this object presents CORAL's interface to uvicorn and inserts the reef
    layer between CORAL and LiteLLM.
    """

    def __init__(
        self,
        coral_layer: Any,
        *,
        scenario: str,
        journal: AttributionJournal,
        extra_tags: Mapping[str, str] | None,
    ) -> None:
        self._coral = coral_layer
        coral_layer.app = ReefAttributionMiddleware(
            coral_layer.app,
            scenario=scenario,
            journal=journal,
            extra_tags=extra_tags,
        )

    def register_agent(self, *args: Any, **kwargs: Any) -> Any:
        return self._coral.register_agent(*args, **kwargs)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> Any:
        return await self._coral(scope, receive, send)

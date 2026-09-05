"""Reef adapter for CORAL multi-agent test-time training (issue #3).

The package has three pieces, one per integration seam:

- ``middleware``: an ASGI layer for CORAL's LiteLLM gateway that stamps
  Reef scenario/tag headers onto provider requests without touching the
  provider-native body, and captures Reef inference receipts.
- ``attribution``: the durable request -> ``agent_record_id`` journal that
  correlation and reporting read.
- ``reporter``: turns finalized CORAL attempts into ``POST /reef/report``
  calls with idempotent, client-supplied record ids.
"""

from reef_coral.attribution import AttributionJournal, AttributionRecord
from reef_coral.middleware import ReefAttributionMiddleware
from reef_coral.reporter import AttemptReport, report_attempt

__all__ = [
    "AttemptReport",
    "AttributionJournal",
    "AttributionRecord",
    "ReefAttributionMiddleware",
    "report_attempt",
]

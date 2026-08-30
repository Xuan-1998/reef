"""Persistent observed-cost stop for live GEPA reproduction calls."""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SpendCapReached(RuntimeError):
    """Raised before a new model call once observed spend reaches the cap."""


class ObservedCostLedger:
    """Persist completed-call cost and refuse to begin calls past a fixed cap."""

    def __init__(self, path: Path, max_observed_cost_usd: float) -> None:
        if not math.isfinite(max_observed_cost_usd) or max_observed_cost_usd <= 0:
            raise ValueError("max_observed_cost_usd must be finite and positive")
        self.path = Path(path)
        self.max_observed_cost_usd = float(max_observed_cost_usd)
        self._lock = threading.Lock()
        state = self._read()
        self._observed_cost_usd = float(state.get("observed_cost_usd", 0.0))
        self._completed_calls = int(state.get("completed_calls", 0))
        if not math.isfinite(self._observed_cost_usd) or self._observed_cost_usd < 0:
            raise ValueError("observed cost ledger contains an invalid total")
        if self._completed_calls < 0:
            raise ValueError("observed cost ledger contains an invalid call count")
        self._write()

    def before_call(self) -> None:
        with self._lock:
            if self._observed_cost_usd >= self.max_observed_cost_usd:
                raise SpendCapReached(
                    f"observed cost ${self._observed_cost_usd:.6f} reached the "
                    f"${self.max_observed_cost_usd:.6f} cap; no new model call was started"
                )

    def record_call(self, observed_cost_usd: float) -> None:
        if not math.isfinite(observed_cost_usd) or observed_cost_usd < 0:
            raise ValueError("observed call cost must be finite and non-negative")
        with self._lock:
            self._observed_cost_usd += float(observed_cost_usd)
            self._completed_calls += 1
            self._write()

    @property
    def observed_cost_usd(self) -> float:
        with self._lock:
            return self._observed_cost_usd

    @property
    def completed_calls(self) -> int:
        with self._lock:
            return self._completed_calls

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("observed cost ledger must contain a JSON object")
        return value

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completed_calls": self._completed_calls,
            "max_observed_cost_usd": self.max_observed_cost_usd,
            "observed_cost_usd": self._observed_cost_usd,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

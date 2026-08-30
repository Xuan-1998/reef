"""Durable per-example checkpoints for evaluation after search is sealed."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from gepa.core.adapter import EvaluationBatch

from .adapter import AIMEExample

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class BatchAdapter(Protocol):
    def evaluate(
        self,
        batch: list[AIMEExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Any, Any]: ...


class HeldoutEvaluator(Protocol):
    def evaluate(
        self,
        label: str,
        batch: Sequence[AIMEExample],
        candidate: Mapping[str, str],
    ) -> EvaluationBatch[Any, Any]: ...


class CheckpointedHeldoutEvaluator:
    """Evaluate a sealed batch without repeating completed examples on resume."""

    def __init__(self, adapter: BatchAdapter, root: Path) -> None:
        self.adapter = adapter
        self.root = Path(root).resolve()

    def evaluate(
        self,
        label: str,
        batch: Sequence[AIMEExample],
        candidate: Mapping[str, str],
    ) -> EvaluationBatch[Any, Any]:
        if not _SAFE_LABEL.fullmatch(label):
            raise ValueError(f"invalid held-out checkpoint label {label!r}")
        if not batch:
            raise ValueError("held-out checkpoint batch must not be empty")
        normalized_candidate = dict(candidate)
        candidate_sha256 = _fingerprint(normalized_candidate)
        outputs: list[Any] = []
        scores: list[float] = []
        for index, example in enumerate(batch):
            path = self.root / label / f"example-{index:04d}.json"
            example_sha256 = _fingerprint(example)
            if path.is_file():
                payload = _read_checkpoint(path, index, example_sha256, candidate_sha256)
            else:
                evaluated = self.adapter.evaluate([example], normalized_candidate, capture_traces=False)
                if len(evaluated.outputs) != 1 or len(evaluated.scores) != 1:
                    raise RuntimeError("held-out adapter must return exactly one result per example")
                score = float(evaluated.scores[0])
                if not math.isfinite(score):
                    raise RuntimeError("held-out adapter returned a non-finite score")
                payload = {
                    "schema_version": 1,
                    "index": index,
                    "example_sha256": example_sha256,
                    "candidate_sha256": candidate_sha256,
                    "score": score,
                    "output": _jsonable(evaluated.outputs[0]),
                }
                _write_json(path, payload)
            outputs.append(payload["output"])
            scores.append(float(payload["score"]))
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=None,
            num_metric_calls=len(batch),
        )


def _read_checkpoint(
    path: Path,
    index: int,
    example_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"held-out checkpoint is not an object: {path}")
    expected = {
        "schema_version": 1,
        "index": index,
        "example_sha256": example_sha256,
        "candidate_sha256": candidate_sha256,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError(f"held-out checkpoint identity changed across resume: {path}")
    score = value.get("score")
    if not isinstance(score, int | float) or not math.isfinite(float(score)) or "output" not in value:
        raise RuntimeError(f"held-out checkpoint payload is invalid: {path}")
    return value


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

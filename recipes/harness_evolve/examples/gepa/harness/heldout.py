"""Durable per-example checkpoints for evaluation after search is sealed."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

from gepa.core.adapter import EvaluationBatch

from .adapter import AIMEExample

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
Candidate = Mapping[str, str] | str


class BatchAdapter(Protocol):
    def evaluate(
        self,
        batch: list[AIMEExample],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch[Any, Any]: ...


class HeldoutEvaluator(Protocol):
    def evaluate(
        self,
        label: str,
        batch: Sequence[AIMEExample],
        candidate: Candidate,
    ) -> EvaluationBatch[Any, Any]: ...


class CheckpointedHeldoutEvaluator:
    """Evaluate a sealed batch without repeating completed examples on resume."""

    def __init__(
        self,
        adapter: BatchAdapter,
        root: Path,
        *,
        max_workers: int = 1,
        failure_score: float | None = None,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("held-out max_workers must be positive")
        self.adapter = adapter
        self.root = Path(root).resolve()
        self.max_workers = max_workers
        self.failure_score = failure_score

    def evaluate(
        self,
        label: str,
        batch: Sequence[AIMEExample],
        candidate: Candidate,
    ) -> EvaluationBatch[Any, Any]:
        if not _SAFE_LABEL.fullmatch(label):
            raise ValueError(f"invalid held-out checkpoint label {label!r}")
        if not batch:
            raise ValueError("held-out checkpoint batch must not be empty")
        normalized_candidate = dict(candidate) if isinstance(candidate, Mapping) else candidate
        candidate_sha256 = _fingerprint(normalized_candidate)
        payloads: list[dict[str, Any] | None] = [None] * len(batch)
        pending = []
        for index, example in enumerate(batch):
            path = self.root / label / f"example-{index:04d}.json"
            example_sha256 = _fingerprint(example)
            if path.is_file():
                payloads[index] = _read_checkpoint(path, index, example_sha256, candidate_sha256)
            else:
                pending.append((index, example, path, example_sha256))
        if self.max_workers == 1:
            for index, example, path, example_sha256 in pending:
                payloads[index] = self._evaluate_one(
                    index,
                    example,
                    path,
                    example_sha256,
                    candidate_sha256,
                    normalized_candidate,
                )
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_one,
                        index,
                        example,
                        path,
                        example_sha256,
                        candidate_sha256,
                        normalized_candidate,
                    ): index
                    for index, example, path, example_sha256 in pending
                }
                for future in as_completed(futures):
                    payloads[futures[future]] = future.result()
        if any(payload is None for payload in payloads):
            raise RuntimeError("held-out evaluation did not produce every result")
        complete = [payload for payload in payloads if payload is not None]
        return EvaluationBatch(
            outputs=[payload["output"] for payload in complete],
            scores=[float(payload["score"]) for payload in complete],
            trajectories=None,
            num_metric_calls=len(batch),
        )

    def _evaluate_one(
        self,
        index: int,
        example: AIMEExample,
        path: Path,
        example_sha256: str,
        candidate_sha256: str,
        candidate: Candidate,
    ) -> dict[str, Any]:
        try:
            evaluated = self.adapter.evaluate([example], candidate, capture_traces=False)
            if len(evaluated.outputs) != 1 or len(evaluated.scores) != 1:
                raise RuntimeError("held-out adapter must return exactly one result per example")
            score = float(evaluated.scores[0])
            if not math.isfinite(score):
                raise RuntimeError("held-out adapter returned a non-finite score")
            output = _jsonable(evaluated.outputs[0])
        except Exception as exc:
            if self.failure_score is None:
                raise
            score = self.failure_score
            output = {"error": type(exc).__name__}
        payload = {
            "schema_version": 1,
            "index": index,
            "example_sha256": example_sha256,
            "candidate_sha256": candidate_sha256,
            "score": score,
            "output": output,
        }
        _write_json(path, payload)
        return payload


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

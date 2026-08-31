"""Machine-readable GEPA reproduction artifacts."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gepa.core.result import GEPAResult

from .models import ModelPrice
from .search import SealedSearchOutcome, pareto_candidate_indices


def write_search_report(
    *,
    output_dir: Path,
    cell: str,
    seed: int,
    outcome: SealedSearchOutcome,
    config: Mapping[str, Any],
    task_usage: Mapping[str, int],
    reflection_usage: Mapping[str, int],
    task_price: ModelPrice,
    reflection_price: ModelPrice,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = outcome.result
    summary = {
        "cell": cell,
        "seed": seed,
        "promotion": asdict(outcome.promotion),
        "frozen_test_score": outcome.frozen_test_score,
        "selected_test_score": outcome.selected_test_score,
        "test_delta": outcome.selected_test_score - outcome.frozen_test_score,
        "frozen_test_scores": outcome.frozen_test_scores,
        "selected_test_scores": outcome.selected_test_scores,
        "pareto_candidate_indices": pareto_candidate_indices(result),
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "wall_time_s": outcome.wall_time_s,
        "usage": {
            "task": dict(task_usage),
            "reflection": dict(reflection_usage),
        },
        "estimated_cost_usd": {
            "task": task_price.estimate(task_usage),
            "reflection": reflection_price.estimate(reflection_usage),
            "total": task_price.estimate(task_usage) + reflection_price.estimate(reflection_usage),
        },
        "pricing": {
            "task": asdict(task_price),
            "reflection": asdict(reflection_price),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "raw_result.json", result.to_dict())
    _write_json(
        output_dir / "heldout.json",
        {
            "frozen_scores": outcome.frozen_test_scores,
            "frozen_outputs": outcome.frozen_test_outputs,
            "selected_scores": outcome.selected_test_scores,
            "selected_outputs": outcome.selected_test_outputs,
        },
    )
    _write_json(output_dir / "selected_candidate.json", result.candidates[outcome.promotion.candidate_idx])
    _write_json(output_dir / "learning_curve.json", _learning_curve(result))
    (output_dir / "candidate-lineage.dot").write_text(result.candidate_tree_dot(), encoding="utf-8")


def write_aggregate_report(*, output_dir: Path, cells: Sequence[str], seeds: Sequence[int]) -> None:
    """Aggregate only completed cell summaries into one comparison artifact."""
    aggregates: dict[str, Any] = {}
    for cell in cells:
        runs = [_read_json(output_dir / cell / f"seed-{seed}" / "summary.json") for seed in seeds]
        normalized = [_normalize_run(cell, seed, summary) for seed, summary in zip(seeds, runs, strict=True)]
        selected_scores = [run["selected_test_score"] for run in normalized]
        frozen_scores = [run["frozen_test_score"] for run in normalized]
        deltas = [run["test_delta"] for run in normalized]
        promotions = [run["promoted"] for run in normalized if run["promoted"] is not None]
        aggregates[cell] = {
            "runs": normalized,
            "frozen_test_score_mean": statistics.fmean(frozen_scores),
            "selected_test_score_mean": statistics.fmean(selected_scores),
            "selected_test_score_sample_stdev": (
                statistics.stdev(selected_scores) if len(selected_scores) > 1 else None
            ),
            "test_delta_mean": statistics.fmean(deltas),
            "promotion_rate": statistics.fmean(promotions) if promotions else None,
            "estimated_cost_usd_total": sum(run["estimated_cost_usd"] for run in normalized),
            "wall_time_s_total": sum(run["wall_time_s"] for run in normalized),
        }
    _write_json(output_dir / "results.json", {"cells": aggregates})


def _normalize_run(cell: str, seed: int, summary: Mapping[str, Any]) -> dict[str, Any]:
    if cell == "frozen":
        frozen_score = selected_score = float(summary["test_score"])
        promoted = None
        estimated_cost = float(summary["estimated_cost_usd"])
        started = summary.get("started_at")
        finished = summary.get("finished_at")
        wall_time = 0.0
        if isinstance(started, str) and isinstance(finished, str):
            wall_time = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    else:
        frozen_score = float(summary["frozen_test_score"])
        selected_score = float(summary["selected_test_score"])
        promotion = summary.get("promotion", {})
        promoted = bool(promotion.get("selected")) if isinstance(promotion, Mapping) else False
        cost = summary.get("estimated_cost_usd", {})
        estimated_cost = float(cost.get("total", 0.0)) if isinstance(cost, Mapping) else float(cost)
        wall_time = float(summary["wall_time_s"])
    return {
        "seed": seed,
        "frozen_test_score": frozen_score,
        "selected_test_score": selected_score,
        "test_delta": selected_score - frozen_score,
        "promoted": promoted,
        "estimated_cost_usd": estimated_cost,
        "wall_time_s": wall_time,
    }


def _learning_curve(result: GEPAResult[Any, Any]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_idx": idx,
            "metric_calls_at_discovery": result.discovery_eval_counts[idx],
            "validation_score": result.val_aggregate_scores[idx],
            "parents": result.parents[idx],
        }
        for idx in range(result.num_candidates)
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)

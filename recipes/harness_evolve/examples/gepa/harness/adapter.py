"""Upstream GEPA adapter that evaluates a Reef rules node through Pi."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from gepa.core.adapter import EvaluationBatch

from reef.harness import (
    AdapterDescriptor,
    EpisodeError,
    EpisodeResult,
    TrajectoryError,
    render_composition,
    run_episode,
)
from reef.harness.model_binding import ModelBinding

from .models import TASK_MODEL_PRICE, TokenUsage, UsageLedger, empty_usage, trajectory_usage

RULES_COMPONENT = "rules"
_FINAL_ANSWER = re.compile(r"(?m)^\s*###\s+([^\s]+)\s*$")
_COMPONENT_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_NODE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_TEXT_NODE_KINDS = ("rules", "skill", "agent_command")


class AIMEExample(TypedDict, total=False):
    """The stable subset of GEPA's AIME example schema that Reef consumes."""

    input: str
    answer: str
    additional_context: dict[str, str]


class HarnessRollout(TypedDict):
    assistant_response: str
    exit_code: int
    stderr: str
    residue: list[str]
    usage: TokenUsage


class HarnessTrajectory(TypedDict):
    input: str
    expected_answer: str
    assistant_response: str
    feedback: str
    exit_code: int
    stderr: str
    residue: list[str]
    events: list[dict[str, Any]]
    usage: TokenUsage


EpisodeRunner = Callable[..., EpisodeResult]


@dataclass(frozen=True)
class TextComponent:
    """One named GEPA parameter mapped to one fixed Reef text node."""

    key: str
    kind: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not _COMPONENT_KEY.fullmatch(self.key):
            raise ValueError(f"invalid GEPA component key {self.key!r}")
        if self.kind not in _TEXT_NODE_KINDS:
            known = ", ".join(_TEXT_NODE_KINDS)
            raise ValueError(f"text component kind must be one of {known}")
        if self.kind == "rules" and self.name is not None:
            raise ValueError("rules components are unnamed")
        if self.kind != "rules" and (self.name is None or not _NODE_NAME.fullmatch(self.name)):
            raise ValueError(f"{self.kind} components require a safe node name")

    def node(self, text: str) -> tuple[str, Mapping[str, str]]:
        config = {"text": text}
        if self.name is not None:
            config["name"] = self.name
        return self.kind, config

    @property
    def role(self) -> str:
        if self.kind == "rules":
            return "global rules loaded for every episode"
        return f"{self.kind} node {self.name!r}"


RULES_ONLY_COMPONENTS = (TextComponent(RULES_COMPONENT, "rules"),)
MULTI_NODE_COMPONENTS = (
    TextComponent(RULES_COMPONENT, "rules"),
    TextComponent("skill", "skill", "aime-solver"),
)


class ReefCompositionAdapter:
    """Expose GEPA text components as one complete rendered Reef harness.

    Component topology is fixed when the adapter is built. Endpoint
    configuration is supplied separately through a Reef :class:`ModelBinding`;
    it is rendered into each throwaway episode but never becomes part of the
    GEPA candidate.
    """

    propose_new_texts = None

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        task_model: ModelBinding,
        components: Sequence[TextComponent],
        binary: str | None = None,
        timeout_s: float = 600.0,
        episode_runner: EpisodeRunner = run_episode,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not components:
            raise ValueError("composition adapter requires at least one text component")
        component_keys = [component.key for component in components]
        if len(set(component_keys)) != len(component_keys):
            raise ValueError("composition component keys must be unique")
        self.descriptor = descriptor
        self.task_model = task_model
        self.components = tuple(components)
        self._components_by_key = {component.key: component for component in self.components}
        self.binary = binary
        self.timeout_s = timeout_s
        self._episode_runner = episode_runner
        self._binding_nodes = task_model.compose_nodes(descriptor)
        self.usage = UsageLedger(TASK_MODEL_PRICE)

    def candidate_nodes(self, candidate: Mapping[str, str]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        """Map a GEPA candidate to Reef nodes without mutating either input."""
        self._validate_candidate(candidate)
        return tuple(component.node(candidate[component.key]) for component in self.components)

    def render_candidate(self, candidate: Mapping[str, str]) -> dict[str, str]:
        """Render a provider-free candidate suitable for Reef publication."""
        return render_composition(self.candidate_nodes(candidate), self.descriptor)

    def render_episode_candidate(self, candidate: Mapping[str, str]) -> dict[str, str]:
        """Render the transient endpoint binding plus the candidate for evaluation."""
        return render_composition((*self._binding_nodes, *self.candidate_nodes(candidate)), self.descriptor)

    def evaluate(
        self,
        batch: list[AIMEExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[HarnessTrajectory, HarnessRollout]:
        files = self.render_episode_candidate(candidate)
        outputs: list[HarnessRollout] = []
        scores: list[float] = []
        trajectories: list[HarnessTrajectory] | None = [] if capture_traces else None

        for raw_example in batch:
            task, expected = _validate_example(raw_example)
            try:
                result = self._episode_runner(
                    self.descriptor,
                    files,
                    task,
                    binary=self.binary,
                    timeout=self.timeout_s,
                )
                assistant_response = final_assistant_text(result.trajectory) or result.stdout
                score = (
                    score_aime_answer(expected, assistant_response)
                    if result.exit_code == 0 and not result.residue
                    else 0.0
                )
                feedback = _feedback(expected, assistant_response, result)
                output: HarnessRollout = {
                    "assistant_response": assistant_response,
                    "exit_code": result.exit_code,
                    "stderr": result.stderr,
                    "residue": list(result.residue),
                    "usage": trajectory_usage(result.trajectory),
                }
                events = [dict(event) for event in result.trajectory]
            except (EpisodeError, TrajectoryError) as exc:
                assistant_response = ""
                score = 0.0
                feedback = f"The harness episode failed before producing a gradeable answer: {exc}"
                output = {
                    "assistant_response": "",
                    "exit_code": -1,
                    "stderr": str(exc),
                    "residue": [],
                    "usage": empty_usage(),
                }
                events = []

            outputs.append(output)
            self.usage.add(output["usage"])
            scores.append(score)
            if trajectories is not None:
                trajectories.append(
                    {
                        "input": task,
                        "expected_answer": expected,
                        "assistant_response": assistant_response,
                        "feedback": feedback,
                        "exit_code": output["exit_code"],
                        "stderr": output["stderr"],
                        "residue": output["residue"],
                        "events": events,
                        "usage": output["usage"],
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[HarnessTrajectory, HarnessRollout],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        self._validate_candidate(candidate)
        unknown = set(components_to_update) - set(self._components_by_key)
        if unknown:
            raise ValueError(f"unknown reflection components: {sorted(unknown)!r}")
        if not components_to_update:
            raise ValueError("at least one reflection component is required")
        if eval_batch.trajectories is None:
            raise ValueError("captured trajectories are required for reflection")
        return {
            component_key: [
                {
                    "Inputs": trajectory["input"],
                    "Generated Outputs": trajectory["assistant_response"],
                    "Feedback": trajectory["feedback"],
                    "Component role": self._components_by_key[component_key].role,
                    "Harness trajectory": trajectory["events"],
                }
                for trajectory in eval_batch.trajectories
            ]
            for component_key in components_to_update
        }

    def _validate_candidate(self, candidate: Mapping[str, str]) -> None:
        expected = set(self._components_by_key)
        if set(candidate) != expected:
            raise ValueError(f"candidate components must be exactly {sorted(expected)!r}")
        for key, value in candidate.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"candidate component {key!r} must contain non-empty text")


class ReefRulesAdapter(ReefCompositionAdapter):
    """Single-rules-node conformance adapter."""

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        task_model: ModelBinding,
        binary: str | None = None,
        timeout_s: float = 600.0,
        episode_runner: EpisodeRunner = run_episode,
    ) -> None:
        super().__init__(
            descriptor=descriptor,
            task_model=task_model,
            components=RULES_ONLY_COMPONENTS,
            binary=binary,
            timeout_s=timeout_s,
            episode_runner=episode_runner,
        )


def score_aime_answer(expected: str, response: str) -> float:
    """Score the last ``### value`` line against the exact expected value."""
    expected_value = expected.removeprefix("###").strip()
    matches = _FINAL_ANSWER.findall(response)
    return 1.0 if matches and matches[-1] == expected_value else 0.0


def final_assistant_text(trajectory: Sequence[Mapping[str, Any]]) -> str | None:
    """Extract the final assistant text from Pi's wrapped or flat events."""
    for event in reversed(trajectory):
        wrapped = event.get("message")
        message = wrapped if isinstance(wrapped, Mapping) else event
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                part.get("text")
                for part in content
                if isinstance(part, Mapping) and part.get("type") == "text" and isinstance(part.get("text"), str)
            ]
            if texts:
                return "\n".join(texts)
    return None


def _validate_example(example: Mapping[str, Any]) -> tuple[str, str]:
    task, expected = example.get("input"), example.get("answer")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("AIME example requires a non-empty input")
    if not isinstance(expected, str) or not expected.removeprefix("###").strip():
        raise ValueError("AIME example requires a non-empty answer")
    return task, expected


def _feedback(expected: str, response: str, result: EpisodeResult) -> str:
    if result.exit_code != 0:
        return f"The harness exited with code {result.exit_code}. Expected the final line {expected!r}."
    if result.residue:
        return f"The harness left unexpected residue {list(result.residue)!r}. Expected the final line {expected!r}."
    if score_aime_answer(expected, response):
        return f"Correct: the final answer exactly matched {expected!r}."
    return f"Incorrect: the final non-empty answer line must be exactly {expected!r}."

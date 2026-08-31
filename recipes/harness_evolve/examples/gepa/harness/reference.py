"""Current upstream GEPA AIME evaluator, adapted only for Reef accounting."""

from __future__ import annotations

from typing import Any

import dspy
from gepa.core.adapter import EvaluationBatch
from gepa.optimize_anything import SideInfo

from .adapter import AIMEExample

OFFICIAL_SEED_PROMPT = (
    "Solve the math problem carefully. Break down the steps and provide the final answer as a single number."
)


class MathSolverSignature(dspy.Signature):
    input = dspy.InputField(desc="The math problem to solve.")
    answer = dspy.OutputField(desc="The final numerical answer.")


def evaluate(candidate: str, example: AIMEExample) -> tuple[float, SideInfo]:
    """Run the official DSPy ChainOfThought solver and return GEPA feedback."""
    signature = MathSolverSignature.with_instructions(candidate)
    prediction = dspy.ChainOfThought(signature)(input=example["input"])
    score, feedback = math_metric(example, prediction)
    return score, {
        "score": score,
        "input": example["input"],
        "output": prediction.answer,
        "reasoning": getattr(prediction, "reasoning", ""),
        "execution_feedback": feedback,
    }


def math_metric(example: AIMEExample, prediction: Any) -> tuple[float, str]:
    """Exact integer scoring and diagnostic feedback from upstream."""
    correct_answer = int(example["answer"].removeprefix("### "))
    context = example.get("additional_context")
    written_solution = context.get("solution", "") if isinstance(context, dict) else ""
    solution_suffix = (
        f" Here's the full step-by-step solution:\n{written_solution}\n\n"
        "Think about what takeaways you can learn from this solution to improve your future "
        "answers and approach to similar problems"
        if written_solution
        else ""
    )
    try:
        llm_answer = int(prediction.answer)
    except (ValueError, TypeError):
        feedback = (
            "The final answer must be a valid integer and nothing else. You responded with "
            f"'{prediction.answer}', which couldn't be parsed as a python integer. Please ensure "
            "your answer is a valid integer without any additional text or formatting. The correct "
            f"answer is '{correct_answer}'.{solution_suffix}"
            f"{' and ensure your final answer is a valid integer.' if written_solution else ''}"
        )
        return 0.0, feedback

    score = float(correct_answer == llm_answer)
    status = "correct" if score == 1.0 else "incorrect"
    return score, f"Your answer is {status}. The correct answer is '{correct_answer}'.{solution_suffix}"


class OfficialAIMEAdapter:
    """Small adapter used only by the durable held-out evaluator."""

    def evaluate(
        self,
        batch: list[AIMEExample],
        candidate: str,
        capture_traces: bool = False,
    ) -> EvaluationBatch[Any, Any]:
        outputs = []
        scores = []
        for example in batch:
            score, side_info = evaluate(candidate, example)
            scores.append(score)
            outputs.append(
                {
                    "answer": side_info["output"],
                    "reasoning": side_info["reasoning"],
                    "feedback": side_info["execution_feedback"],
                }
            )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=None, num_metric_calls=len(batch))

# chapter_generation/evaluation_service.py
"""Service for running evaluation cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from kg_maintainer.models import EvaluationResult, ProblemDetail


@dataclass
class EvaluationCycleResult:
    """Result from a single evaluation cycle."""

    evaluation: EvaluationResult
    continuity_problems: list[ProblemDetail]
    eval_usage: dict[str, int] | None
    continuity_usage: dict[str, int] | None
    repetition_problems: list[ProblemDetail]


class EvaluationService:
    """Run comprehensive draft evaluations via :class:`ComprehensiveEvaluatorAgent`."""

    def __init__(self, agent: ComprehensiveEvaluatorAgent | None = None) -> None:
        self.agent = agent or ComprehensiveEvaluatorAgent()

    async def evaluate(
        self,
        plot_outline: dict[str, Any],
        draft_text: str,
        chapter_number: int,
        plot_point_focus: str,
        plot_point_index: int,
        chapter_context: str,
        ignore_spans: list[tuple[int, int]] | None = None,
    ) -> EvaluationCycleResult:
        """Return evaluation details for a chapter draft."""

        evaluation, usage = await self.agent.evaluate_chapter_draft(
            plot_outline,
            draft_text,
            chapter_number,
            plot_point_focus,
            plot_point_index,
            chapter_context,
            ignore_spans=ignore_spans,
        )
        return EvaluationCycleResult(
            evaluation=evaluation,
            continuity_problems=[],
            eval_usage=usage,
            continuity_usage=None,
            repetition_problems=[],
        )

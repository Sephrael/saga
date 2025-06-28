# processing/evaluation_pipeline.py
"""Unified evaluation pipeline coordinating multiple checks."""

from __future__ import annotations

from typing import Any

import structlog
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from agents.world_continuity_agent import WorldContinuityAgent

from models import ProblemDetail

from .repetition_analyzer import RepetitionAnalyzer

logger = structlog.get_logger(__name__)


class EvaluationPipeline:
    """Run evaluator agents and algorithmic checks."""

    def __init__(self) -> None:
        self.comp_agent = ComprehensiveEvaluatorAgent()
        self.continuity_agent = WorldContinuityAgent()
        self.repetition_analyzer = RepetitionAnalyzer()

    async def run(
        self,
        plot_outline: dict[str, Any],
        draft_text: str,
        chapter_number: int,
        previous_chapters_context: str,
    ) -> list[ProblemDetail]:
        """Return consolidated problem list for ``draft_text``."""
        logger.info("EvaluationPipeline running", chapter=chapter_number)
        problems: list[ProblemDetail] = []
        eval_result, _ = await self.comp_agent.evaluate_chapter_draft(
            plot_outline,
            [],
            {},
            draft_text,
            chapter_number,
            None,
            0,
            previous_chapters_context,
        )
        problems.extend(eval_result.problems_found)

        continuity_probs, _ = await self.continuity_agent.check_consistency(
            plot_outline,
            draft_text,
            chapter_number,
            previous_chapters_context,
        )
        problems.extend(continuity_probs)

        repetition_probs = await self.repetition_analyzer.analyze(draft_text)
        problems.extend(repetition_probs)

        return sorted(problems, key=lambda p: p.get("issue_category", ""))

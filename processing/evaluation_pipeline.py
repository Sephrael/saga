# processing/evaluation_pipeline.py
"""Unified evaluation pipeline coordinating multiple checks."""

from __future__ import annotations

from typing import Any, cast

import structlog
from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent

from processing.revision_manager import RevisionManager

from .repetition_analyzer import RepetitionAnalyzer

logger = structlog.get_logger(__name__)


class EvaluationPipeline:
    """Run evaluator agents and algorithmic checks."""

    def __init__(self) -> None:
        self.comp_agent = ComprehensiveEvaluatorAgent()
        self.repetition_analyzer = RepetitionAnalyzer()
        self.revision_manager = RevisionManager()

    async def run(
        self,
        plot_outline: dict[str, Any],
        draft_text: str,
        chapter_number: int,
        previous_chapters_context: str,
    ) -> tuple[tuple[str, str | None, list[tuple[int, int]]] | None, Any | None]:
        """Run checks then send problems to the revision manager."""
        logger.info("EvaluationPipeline running", chapter=chapter_number)
        eval_result, _ = await self.comp_agent.evaluate_chapter_draft(
            plot_outline,
            draft_text,
            chapter_number,
            None,
            0,
            previous_chapters_context,
        )

        repetition_probs = await self.repetition_analyzer.analyze(draft_text)

        eval_result.problems_found.extend(repetition_probs)
        if repetition_probs:
            eval_result.needs_revision = True
            eval_result.reasons.append("Issues found by repetition analyzer")

        return await self.revision_manager.revise_chapter(
            plot_outline,
            {},
            {},
            draft_text,
            chapter_number,
            eval_result,
            previous_chapters_context,
            None,
            continuity_problems=[],
            repetition_problems=cast(list[dict[str, Any]], repetition_probs),
        )

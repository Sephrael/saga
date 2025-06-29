"""Service for running evaluation cycles."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from config import settings
from kg_maintainer.models import EvaluationResult, ProblemDetail
from processing.repetition_analyzer import RepetitionAnalyzer

if TYPE_CHECKING:  # pragma: no cover - type hint import
    from orchestration.nana_orchestrator import NANA_Orchestrator
    from storage.file_manager import FileManager

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationCycleResult:
    """Result from a single evaluation cycle."""

    evaluation: EvaluationResult
    continuity_problems: list[ProblemDetail]
    eval_usage: dict[str, int] | None
    continuity_usage: dict[str, int] | None
    repetition_problems: list[ProblemDetail]


class EvaluationService:
    """Run evaluation cycles for drafted chapters."""

    def __init__(
        self, orchestrator: NANA_Orchestrator, file_manager: FileManager
    ) -> None:
        self.orchestrator = orchestrator
        self.file_manager = file_manager
        self.repetition_analyzer = RepetitionAnalyzer(
            tracker=self.orchestrator.repetition_tracker
        )

    async def run_cycle(
        self,
        chapter_number: int,
        attempt: int,
        current_text: str,
        plot_point_focus: str,
        plot_point_index: int,
        hybrid_context_for_draft: str,
        patched_spans: list[tuple[int, int]],
    ) -> EvaluationCycleResult:
        self.orchestrator._update_rich_display(
            step=f"Ch {chapter_number} - Evaluation Cycle {attempt} (Parallel)"
        )

        tasks_to_run: list[asyncio.Future] = []
        task_names: list[str] = []

        ignore_spans = patched_spans

        if settings.ENABLE_COMPREHENSIVE_EVALUATION:
            tasks_to_run.append(
                self.orchestrator.evaluator_agent.evaluate_chapter_draft(
                    self.orchestrator.plot_outline,
                    current_text,
                    chapter_number,
                    plot_point_focus,
                    plot_point_index,
                    hybrid_context_for_draft,
                    ignore_spans=ignore_spans,
                )
            )
            task_names.append("evaluation")

        results = await asyncio.gather(*tasks_to_run)

        eval_result_obj: EvaluationResult | None = None
        eval_usage = None
        continuity_problems: list[ProblemDetail] = []
        continuity_usage = None

        result_idx = 0
        if "evaluation" in task_names:
            eval_result_obj, eval_usage = results[result_idx]

        if isinstance(eval_result_obj, EvaluationResult):
            evaluation_result: EvaluationResult = eval_result_obj
        else:
            data = eval_result_obj or {}
            evaluation_result = EvaluationResult(
                needs_revision=data.get("needs_revision", False),
                reasons=data.get("reasons", []),
                problems_found=data.get("problems_found", []),
                coherence_score=data.get("coherence_score"),
                consistency_issues=data.get("consistency_issues"),
                plot_deviation_reason=data.get("plot_deviation_reason"),
                thematic_issues=data.get("thematic_issues"),
                narrative_depth_issues=data.get("narrative_depth_issues"),
            )

        repetition_probs = await self.repetition_analyzer.analyze(current_text)
        evaluation_result.problems_found.extend(repetition_probs)
        if repetition_probs:
            evaluation_result.needs_revision = True
            evaluation_result.reasons.append("Repetition issues detected")

        return EvaluationCycleResult(
            evaluation=evaluation_result,
            continuity_problems=continuity_problems,
            eval_usage=eval_usage,
            continuity_usage=continuity_usage,
            repetition_problems=repetition_probs,
        )

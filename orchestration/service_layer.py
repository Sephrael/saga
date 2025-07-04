# orchestration/service_layer.py
"""Service layer that orchestrates agent interactions."""

from __future__ import annotations

from typing import Any

from agents.comprehensive_evaluator_agent import ComprehensiveEvaluatorAgent
from agents.drafting_agent import DraftingAgent
from agents.finalize_agent import FinalizeAgent
from agents.kg_maintainer_agent import KGMaintainerAgent
from chapter_generation.drafting_service import DraftingService, DraftResult
from chapter_generation.evaluation_service import (
    EvaluationCycleResult,
    EvaluationService,
)
from chapter_generation.finalization_service import (
    FinalizationService,
    FinalizationServiceResult,
)
from chapter_generation.revision_service import RevisionResult, RevisionService
from kg_maintainer.models import (
    CharacterProfile,
    EvaluationResult,
    SceneDetail,
    WorldItem,
)
from processing.revision_manager import RevisionManager


class ChapterServiceLayer:
    """Coordinate high level chapter generation services."""

    def __init__(
        self,
        drafting_agent: DraftingAgent | None = None,
        evaluator_agent: ComprehensiveEvaluatorAgent | None = None,
        revision_manager: RevisionManager | None = None,
        finalize_agent: FinalizeAgent | None = None,
    ) -> None:
        kg_agent = KGMaintainerAgent()
        self.drafting_service = DraftingService(drafting_agent)
        self.evaluation_service = EvaluationService(evaluator_agent)
        self.revision_service = RevisionService(revision_manager)
        self.finalization_service = FinalizationService(
            finalize_agent or FinalizeAgent(kg_agent)
        )

    async def draft_chapter(
        self,
        plot_outline: dict[str, Any],
        chapter_number: int,
        plot_point_focus: str,
        hybrid_context: str,
        chapter_plan: list[SceneDetail] | None,
    ) -> DraftResult:
        """Create a draft using the drafting service."""
        return await self.drafting_service.draft(
            plot_outline,
            chapter_number,
            plot_point_focus,
            hybrid_context,
            chapter_plan,
        )

    async def evaluate_draft(
        self,
        plot_outline: dict[str, Any],
        draft_text: str,
        chapter_number: int,
        plot_point_focus: str,
        plot_point_index: int,
        chapter_context: str,
        ignore_spans: list[tuple[int, int]] | None = None,
    ) -> EvaluationCycleResult:
        """Evaluate a draft using the evaluation service."""
        return await self.evaluation_service.evaluate(
            plot_outline,
            draft_text,
            chapter_number,
            plot_point_focus,
            plot_point_index,
            chapter_context,
            ignore_spans,
        )

    async def revise_draft(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        original_text: str,
        chapter_number: int,
        evaluation_result: EvaluationResult,
        hybrid_context: str,
        chapter_plan: list[SceneDetail] | None,
        revision_cycle: int = 0,
    ) -> RevisionResult:
        """Revise a draft using the revision service."""
        return await self.revision_service.revise(
            plot_outline,
            character_profiles,
            world_building,
            original_text,
            chapter_number,
            evaluation_result,
            hybrid_context,
            chapter_plan,
            revision_cycle,
        )

    async def finalize_chapter(
        self,
        plot_outline: dict[str, Any],
        character_profiles: dict[str, CharacterProfile],
        world_building: dict[str, dict[str, WorldItem]],
        chapter_number: int,
        final_text: str,
        raw_llm_output: str | None,
        from_flawed_draft: bool,
        fill_in_context: str | None,
    ) -> FinalizationServiceResult:
        """Finalize and persist chapter output."""
        return await self.finalization_service.finalize(
            plot_outline,
            character_profiles,
            world_building,
            chapter_number,
            final_text,
            raw_llm_output,
            from_flawed_draft,
            fill_in_context,
        )

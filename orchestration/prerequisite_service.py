# orchestration/prerequisite_service.py
"""Service for gathering chapter prerequisites."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from chapter_generation.context_models import ContextProfileName, SceneDetail
from chapter_generation.prerequisites_service import PrerequisiteData
from config import settings
from core.db_manager import neo4j_manager
from utils.plot import get_plot_point_info

from orchestration.token_accountant import Stage

if TYPE_CHECKING:  # pragma: no cover - type hints
    from .nana_orchestrator import NANA_Orchestrator

logger = structlog.get_logger(__name__)


class PrerequisiteService:
    """Generate planning context and scene plans."""

    def __init__(self, orchestrator: NANA_Orchestrator) -> None:
        self.orchestrator = orchestrator

    async def gather(self, novel_chapter_number: int) -> PrerequisiteData:
        """Return planning data required before drafting."""
        o = self.orchestrator
        o._update_rich_display(
            step=f"Ch {novel_chapter_number} - Preparing Prerequisites"
        )

        plot_point_focus, plot_point_index = get_plot_point_info(
            o.plot_outline, novel_chapter_number
        )
        if plot_point_focus is None:
            logger.error(
                "NANA: Ch %s prerequisite check failed: no concrete plot point focus (index %s).",
                novel_chapter_number,
                plot_point_index,
            )
            return PrerequisiteData(None, -1, None, None)

        o._update_novel_props_cache()

        planning_context = o.next_chapter_context
        if planning_context is None:
            planning_context = await o.context_service.build_hybrid_context(
                o,
                novel_chapter_number,
                None,
                None,
                profile_name=ContextProfileName.DEFAULT,
            )
        chapter_plan_result, plan_usage = await o.planner_agent.plan_chapter_scenes(
            o.plot_outline,
            novel_chapter_number,
            plot_point_focus,
            plot_point_index,
            (novel_chapter_number - 1) % settings.PLOT_POINT_CHAPTER_SPAN + 1,
            planning_context,
            list(o.completed_plot_points),
        )
        o._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.CHAPTER_PLANNING.value}", plan_usage
        )

        chapter_plan: list[SceneDetail] | None = chapter_plan_result

        if (
            settings.ENABLE_SCENE_PLAN_VALIDATION
            and chapter_plan is not None
            and settings.ENABLE_WORLD_CONTINUITY_CHECK
        ):
            (
                plan_problems,
                usage,
            ) = await o.evaluator_agent.check_scene_plan_consistency(
                o.plot_outline,
                chapter_plan,
                novel_chapter_number,
                planning_context,
            )
            o._accumulate_tokens(
                f"Ch{novel_chapter_number}-{Stage.PLAN_CONSISTENCY.value}", usage
            )
            await o.output_service.save_debug_output(
                novel_chapter_number,
                "scene_plan_consistency_problems",
                plan_problems,
            )
            if plan_problems:
                logger.warning(
                    "NANA: Ch %s scene plan has %s consistency issues.",
                    novel_chapter_number,
                    len(plan_problems),
                )

        o.missing_references["characters"].clear()
        o.missing_references["locations"].clear()
        prev_state = await o._load_previous_end_state(novel_chapter_number - 1)
        if chapter_plan and prev_state is not None:
            known_chars = {c.name for c in prev_state.character_states}
            known_locs = {c.location for c in prev_state.character_states}
            known_locs.update(prev_state.key_world_changes.keys())
            for scene in chapter_plan:
                for name in scene.get("characters_involved", []):
                    if name not in known_chars:
                        o.missing_references["characters"].add(name)
                setting = scene.get("setting_details")
                if setting and setting not in known_locs:
                    o.missing_references["locations"].add(setting)

        hybrid_context_for_draft = o.next_chapter_context
        if hybrid_context_for_draft is None:
            await o.refresh_plot_outline()
            if neo4j_manager.driver is not None:
                await o.refresh_knowledge_cache()
            else:
                logger.warning(
                    "Neo4j driver not initialized. Skipping knowledge cache refresh."
                )
            hybrid_context_for_draft = await o.context_service.build_hybrid_context(
                o,
                novel_chapter_number,
                chapter_plan,
                None,
                profile_name=ContextProfileName.DEFAULT,
                missing_entities=list(
                    o.missing_references["characters"]
                    | o.missing_references["locations"]
                ),
            )
        else:
            o.next_chapter_context = None

        fill_in_lines = [
            chunk.text for chunk in o.context_service.llm_fill_chunks if chunk.text
        ]
        if o.pending_fill_ins:
            fill_in_lines = o.pending_fill_ins + fill_in_lines
            o.pending_fill_ins = []
        fill_in_context = "\n".join(fill_in_lines) or None

        if settings.ENABLE_AGENTIC_PLANNING and chapter_plan is None:
            logger.warning(
                "NANA: Ch %s: Planning Agent failed or plan invalid. Proceeding with plot point focus only.",
                novel_chapter_number,
            )
        await o.output_service.save_debug_output(
            novel_chapter_number,
            "scene_plan",
            chapter_plan if chapter_plan else "No plan generated.",
        )
        await o.output_service.save_debug_output(
            novel_chapter_number,
            "hybrid_context_for_draft",
            hybrid_context_for_draft,
        )

        return PrerequisiteData(
            plot_point_focus=plot_point_focus,
            plot_point_index=plot_point_index,
            chapter_plan=chapter_plan,
            hybrid_context_for_draft=hybrid_context_for_draft,
            fill_in_context=fill_in_context,
        )

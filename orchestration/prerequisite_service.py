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

    async def _get_plot_point_focus(
        self, novel_chapter_number: int
    ) -> tuple[str | None, int]:
        """Gets the plot point focus and index for the chapter."""
        return get_plot_point_info(
            self.orchestrator.plot_outline, novel_chapter_number
        )

    async def _generate_chapter_plan(
        self,
        novel_chapter_number: int,
        plot_point_focus: str,
        plot_point_index: int,
        planning_context: str | None,
    ) -> tuple[list[SceneDetail] | None, str | None]:
        """Generates the chapter scene plan."""
        o = self.orchestrator
        if planning_context is None:
            # This implies o.next_chapter_context was None initially
            planning_context = await o.context_service.build_hybrid_context(
                o,
                novel_chapter_number,
                None,  # No chapter plan yet
                None,  # No specific hints initially for planning context
                profile_name=ContextProfileName.DEFAULT,
            )

        chapter_plan_result, plan_usage = await o.planner_agent.plan_chapter_scenes(
            o.plot_outline,
            novel_chapter_number,
            plot_point_focus,
            plot_point_index,
            (novel_chapter_number - 1) % settings.PLOT_POINT_CHAPTER_SPAN + 1,
            planning_context, # Use the resolved planning_context
            list(o.completed_plot_points),
        )
        o._accumulate_tokens(
            f"Ch{novel_chapter_number}-{Stage.CHAPTER_PLANNING.value}", plan_usage
        )
        return chapter_plan_result, planning_context

    async def _validate_scene_plan(
        self,
        novel_chapter_number: int,
        chapter_plan: list[SceneDetail] | None,
        planning_context: str | None, # Context used for plan generation
    ):
        """Validates the scene plan if enabled and necessary."""
        o = self.orchestrator
        if not (
            settings.ENABLE_SCENE_PLAN_VALIDATION
            and chapter_plan is not None
            and settings.ENABLE_WORLD_CONTINUITY_CHECK
            and planning_context is not None # Validation needs the context
        ):
            return

        (
            plan_problems,
            usage,
        ) = await o.evaluator_agent.check_scene_plan_consistency(
            o.plot_outline,
            chapter_plan, # chapter_plan is not None here
            novel_chapter_number,
            planning_context, # planning_context is not None here
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

    async def _identify_missing_references(
        self, novel_chapter_number: int, chapter_plan: list[SceneDetail] | None
    ) -> tuple[set[str], set[str]]:
        """Identifies missing characters and locations based on the chapter plan and previous state."""
        o = self.orchestrator
        missing_chars: set[str] = set()
        missing_locs: set[str] = set()

        prev_state = await o._load_previous_end_state(novel_chapter_number - 1)
        if not chapter_plan or not prev_state:
            return missing_chars, missing_locs

        known_chars = {c.name for c in prev_state.character_states}
        known_locs = {c.location for c in prev_state.character_states}
        known_locs.update(prev_state.key_world_changes.keys())

        for scene in chapter_plan:
            for name in scene.get("characters_involved", []):
                if name not in known_chars:
                    missing_chars.add(name)
            setting = scene.get("setting_details")
            if setting and setting not in known_locs:
                missing_locs.add(setting)

        # Update orchestrator's missing references directly
        o.missing_references["characters"].update(missing_chars)
        o.missing_references["locations"].update(missing_locs)
        return missing_chars, missing_locs

    async def _prepare_hybrid_context_for_draft(
        self,
        novel_chapter_number: int,
        chapter_plan: list[SceneDetail] | None,
        initial_hybrid_context: str | None,
    ) -> str | None:
        """Prepares the hybrid context for drafting, generating if necessary."""
        o = self.orchestrator
        if initial_hybrid_context is not None:
            o.next_chapter_context = None # Consume the pre-loaded context
            return initial_hybrid_context

        # If not pre-loaded, generate it now.
        # This block is similar to what was in the main gather function.
        await o.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await o.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh during hybrid context prep."
            )

        # Use the orchestrator's missing_references which should have been updated
        # by _identify_missing_references before this call.
        missing_entities_for_context = list(
            o.missing_references["characters"] | o.missing_references["locations"]
        )

        return await o.context_service.build_hybrid_context(
            o,
            novel_chapter_number,
            chapter_plan,
            None, # No specific hints here, context service will use its defaults
            profile_name=ContextProfileName.DEFAULT,
            missing_entities=missing_entities_for_context,
        )

    def _collect_fill_in_context(self) -> str | None:
        """Collects and clears pending fill-in context from the orchestrator."""
        o = self.orchestrator
        fill_in_lines = [
            chunk.text for chunk in o.context_service.llm_fill_chunks if chunk.text
        ]
        # Prepend any fill-ins that were pending from a previous context build
        # that didn't result in a chapter generation (e.g. if only planning context was built)
        if o.pending_fill_ins:
            fill_in_lines = o.pending_fill_ins + fill_in_lines
            o.pending_fill_ins = [] # Clear after collecting

        # Clear current llm_fill_chunks from context_service as they are now collected
        # This is important because llm_fill_chunks are accumulated by context_service.
        # If not cleared, they would be re-added in subsequent context builds.
        # The NANA_Orchestrator._finalize_and_log method also has logic for o.pending_fill_ins
        # which might need review for consistency with this clearing.
        # For now, assuming PrerequisiteService is the primary point of collecting these for a chapter.
        if hasattr(o.context_service, 'llm_fill_chunks'): # Ensure attribute exists
            o.context_service.llm_fill_chunks = []

        return "\n".join(fill_in_lines) or None

    async def gather(self, novel_chapter_number: int) -> PrerequisiteData:
        """Return planning data required before drafting."""
        o = self.orchestrator
        o._update_rich_display(
            step=f"Ch {novel_chapter_number} - Preparing Prerequisites"
        )

        plot_point_focus, plot_point_index = await self._get_plot_point_focus(
            novel_chapter_number
        )
        if plot_point_focus is None:
            logger.error(
                "NANA: Ch %s prerequisite check failed: no concrete plot point focus (index %s).",
                novel_chapter_number,
                plot_point_index,
            )
            return PrerequisiteData(None, -1, None, None)

        o._update_novel_props_cache()

        initial_planning_context = o.next_chapter_context
        chapter_plan, planning_context_used = await self._generate_chapter_plan(
            novel_chapter_number,
            plot_point_focus,
            plot_point_index,
            initial_planning_context,
        )

        await self._validate_scene_plan(
            novel_chapter_number, chapter_plan, planning_context_used
        )

        o.missing_references["characters"].clear()
        o.missing_references["locations"].clear()
        await self._identify_missing_references(novel_chapter_number, chapter_plan)

        hybrid_context_for_draft = await self._prepare_hybrid_context_for_draft(
            novel_chapter_number,
            chapter_plan,
            o.next_chapter_context, # Pass orchestrator's next_chapter_context again
                                    # _prepare_hybrid_context_for_draft will consume it if it exists
        )

        fill_in_context = self._collect_fill_in_context()

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

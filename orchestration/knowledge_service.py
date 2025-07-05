# orchestration/knowledge_service.py
"""Service for managing plot and knowledge caches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from config import settings
from data_access import (
    chapter_repository,
    character_queries,
    plot_queries,
    world_queries,
)
from initialization.models import PlotOutline

from orchestration.token_accountant import Stage

if TYPE_CHECKING:  # pragma: no cover - for type hints
    from .nana_orchestrator import NANA_Orchestrator

logger = structlog.get_logger(__name__)


class KnowledgeService:
    """Handle plot outline and knowledge cache operations."""

    def __init__(self, orchestrator: NANA_Orchestrator) -> None:
        self.orchestrator = orchestrator

    def update_novel_props_cache(self) -> None:
        """Refresh novel property cache from the current plot outline managed by StateManagementService."""
        sm = self.orchestrator.state_manager
        current_plot_outline = sm.get_plot_outline()

        cache_data = {
            "title": current_plot_outline.get("title", settings.DEFAULT_PLOT_OUTLINE_TITLE),
            "genre": current_plot_outline.get("genre", settings.CONFIGURED_GENRE),
            "theme": current_plot_outline.get("theme", settings.CONFIGURED_THEME),
            "protagonist_name": current_plot_outline.get(
                "protagonist_name", settings.DEFAULT_PROTAGONIST_NAME
            ),
            "character_arc": current_plot_outline.get("character_arc", "N/A"),
            "logline": current_plot_outline.get("logline", "N/A"),
            "setting": current_plot_outline.get(
                "setting", settings.CONFIGURED_SETTING_DESCRIPTION
            ),
            "narrative_style": current_plot_outline.get("narrative_style", "N/A"),
            "tone": current_plot_outline.get("tone", "N/A"),
            "pacing": current_plot_outline.get("pacing", "N/A"),
            "plot_points": current_plot_outline.get("plot_points", []),
            "plot_outline_full": current_plot_outline,
        }
        sm.set_novel_props_cache(cache_data)
        self.orchestrator._update_rich_display()

    async def refresh_plot_outline(self) -> None:
        """Reload the plot outline from the database and update StateManagementService."""
        result = await plot_queries.get_plot_outline_from_db()
        if isinstance(result, dict):
            sm = self.orchestrator.state_manager
            new_plot_outline = PlotOutline(**result)
            sm.set_plot_outline(new_plot_outline)

            self.update_novel_props_cache()

            sm.completed_plot_points = set(
                await plot_queries.get_completed_plot_points()
            )
        else:
            logger.error("Failed to refresh plot outline from DB: %s", result)

    async def refresh_knowledge_cache(self) -> None:
        """Reload character and world knowledge from the database into StateManagementService."""
        sm = self.orchestrator.state_manager # Get StateManagementService
        logger.info("Refreshing knowledge cache from Neo4j into StateManagementService...")

        # Get data
        character_profiles_data = await character_queries.get_character_profiles_from_db()
        world_building_data = await world_queries.get_world_building_from_db()

        # Update StateManagementService's knowledge_cache directly
        # Assuming sm.knowledge_cache is the correct place to store this.
        # StateManagementService has `self.knowledge_cache = KnowledgeCache()`
        sm.knowledge_cache.characters = character_profiles_data
        sm.knowledge_cache.world = world_building_data

        logger.info(
            "Knowledge cache in StateManagementService refreshed: %d characters, %d world categories.",
            len(sm.knowledge_cache.characters),
            len(sm.knowledge_cache.world),
        )

    async def generate_plot_points_from_kg(self, count: int) -> None:
        """Create additional plot points using the planner agent."""
        o = self.orchestrator
        sm = o.state_manager # Get StateManagementService instance

        if count <= 0:
            return

        summaries: list[str] = []
        current_chapter_count = sm.get_chapter_count() # Use chapter_count from StateManagementService
        start = max(1, current_chapter_count - settings.CONTEXT_CHAPTER_COUNT + 1)
        for i in range(start, current_chapter_count + 1):
            chap = await chapter_repository.get_chapter_data(i)
            if chap and (chap.get("summary") or chap.get("text")):
                summaries.append((chap.get("summary") or chap.get("text", "")).strip())

        combined_summary = "\n".join(summaries)
        if not combined_summary.strip():
            logger.warning("No summaries available for continuation planning.")
            return

        new_points, usage = await o.planner_agent.plan_continuation(
            combined_summary, count
        )
        o._accumulate_tokens(Stage.PLAN_CONTINUATION.value, usage) # Orchestrator method
        if not new_points:
            logger.error("Failed to generate continuation plot points.")
            return

        current_plot_outline = sm.get_plot_outline() # Get plot_outline from StateManagementService
        plot_points_list = current_plot_outline.setdefault("plot_points", [])

        for desc in new_points:
            if await plot_queries.plot_point_exists(desc):
                logger.info("Plot point already exists, skipping: %s", desc)
                continue
            prev_id = await plot_queries.get_last_plot_point_id()
            await o.kg_maintainer_agent.add_plot_point(desc, prev_id or "") # Orchestrator's agent
            plot_points_list.append(desc)

        sm.set_plot_outline(current_plot_outline) # Set modified plot_outline back to StateManagementService

        self.update_novel_props_cache() # This will use the new plot_outline from state_manager

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
        """Refresh novel property cache from the current plot outline."""
        o = self.orchestrator
        o.novel_props_cache = {
            "title": o.plot_outline.get("title", settings.DEFAULT_PLOT_OUTLINE_TITLE),
            "genre": o.plot_outline.get("genre", settings.CONFIGURED_GENRE),
            "theme": o.plot_outline.get("theme", settings.CONFIGURED_THEME),
            "protagonist_name": o.plot_outline.get(
                "protagonist_name", settings.DEFAULT_PROTAGONIST_NAME
            ),
            "character_arc": o.plot_outline.get("character_arc", "N/A"),
            "logline": o.plot_outline.get("logline", "N/A"),
            "setting": o.plot_outline.get(
                "setting", settings.CONFIGURED_SETTING_DESCRIPTION
            ),
            "narrative_style": o.plot_outline.get("narrative_style", "N/A"),
            "tone": o.plot_outline.get("tone", "N/A"),
            "pacing": o.plot_outline.get("pacing", "N/A"),
            "plot_points": o.plot_outline.get("plot_points", []),
            "plot_outline_full": o.plot_outline,
        }
        o._update_rich_display()

    async def refresh_plot_outline(self) -> None:
        """Reload the plot outline from the database."""
        result = await plot_queries.get_plot_outline_from_db()
        if isinstance(result, dict):
            self.orchestrator.plot_outline = PlotOutline(**result)
            self.update_novel_props_cache()
            self.orchestrator.completed_plot_points = set(
                await plot_queries.get_completed_plot_points()
            )
        else:
            logger.error("Failed to refresh plot outline from DB: %s", result)

    async def refresh_knowledge_cache(self) -> None:
        """Reload character and world knowledge from the database."""
        o = self.orchestrator
        logger.info("Refreshing knowledge cache from Neo4j...")
        o.knowledge_cache.characters = (
            await character_queries.get_character_profiles_from_db()
        )
        o.knowledge_cache.world = await world_queries.get_world_building_from_db()
        logger.info(
            "Knowledge cache refreshed: %d characters, %d world categories.",
            len(o.knowledge_cache.characters),
            len(o.knowledge_cache.world),
        )

    async def generate_plot_points_from_kg(self, count: int) -> None:
        """Create additional plot points using the planner agent."""
        o = self.orchestrator
        if count <= 0:
            return

        summaries: list[str] = []
        start = max(1, o.chapter_count - settings.CONTEXT_CHAPTER_COUNT + 1)
        for i in range(start, o.chapter_count + 1):
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
        o._accumulate_tokens(Stage.PLAN_CONTINUATION.value, usage)
        if not new_points:
            logger.error("Failed to generate continuation plot points.")
            return

        for desc in new_points:
            if await plot_queries.plot_point_exists(desc):
                logger.info("Plot point already exists, skipping: %s", desc)
                continue
            prev_id = await plot_queries.get_last_plot_point_id()
            await o.kg_maintainer_agent.add_plot_point(desc, prev_id or "")
            o.plot_outline.setdefault("plot_points", []).append(desc)
        self.update_novel_props_cache()

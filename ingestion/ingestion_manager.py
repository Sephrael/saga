"""Logic for ingesting existing text into the knowledge graph."""

from __future__ import annotations

from typing import Any

import structlog
from agents.finalize_agent import FinalizeAgent
from agents.kg_maintainer_agent import KGMaintainerAgent
from agents.planner_agent import PlannerAgent
from config import settings
from core.db_manager import neo4j_manager
from data_access import plot_queries
from kg_maintainer.models import CharacterProfile, WorldItem
from storage.file_manager import FileManager
from utils.ingestion_utils import split_text_into_chapters

logger = structlog.get_logger(__name__)


class IngestionManager:
    """Handle ingestion of plain text into the SAGA knowledge graph."""

    def __init__(
        self,
        finalize_agent: FinalizeAgent,
        planner_agent: PlannerAgent,
        kg_maintainer: KGMaintainerAgent,
        file_manager: FileManager | None = None,
    ) -> None:
        self.finalize_agent = finalize_agent
        self.planner_agent = planner_agent
        self.kg_maintainer_agent = kg_maintainer
        self.file_manager = file_manager or FileManager()

    async def ingest(self, text_file: str) -> tuple[dict[str, Any], int]:
        """Ingest text and return plot outline and chapter count."""
        logger.info("--- NANA: Starting Ingestion Process ---")

        async with neo4j_manager:
            await neo4j_manager.create_db_schema()
            if neo4j_manager.driver is not None:
                await plot_queries.ensure_novel_info()
            else:  # pragma: no cover - network issues
                logger.warning(
                    "Neo4j driver not initialized. Skipping NovelInfo setup."
                )
            await self.kg_maintainer_agent.load_schema_from_db()

        raw_text = await self.file_manager.read_text(text_file)

        chapters = split_text_into_chapters(raw_text)
        plot_outline: dict[str, Any] = {
            "title": "Ingested Narrative",
            "plot_points": [],
        }
        character_profiles: dict[str, CharacterProfile] = {}
        world_building: dict[str, dict[str, WorldItem]] = {}
        summaries: list[str] = []

        for idx, chunk in enumerate(chapters, 1):
            result = await self.finalize_agent.ingest_and_finalize_chunk(
                plot_outline,
                character_profiles,
                world_building,
                idx,
                chunk,
            )
            if result.get("summary"):
                summaries.append(str(result["summary"]))
                plot_outline["plot_points"].append(result["summary"])

            if idx % settings.KG_HEALING_INTERVAL == 0:
                logger.info(
                    "--- NANA: Triggering KG Healing/Enrichment after Ingestion Chunk %s ---",
                    idx,
                )
                await self.kg_maintainer_agent.heal_and_enrich_kg()

        await self.kg_maintainer_agent.heal_and_enrich_kg()
        combined_summary = "\n".join(summaries)
        continuation, _ = await self.planner_agent.plan_continuation(combined_summary)
        if continuation:
            plot_outline["plot_points"].extend(continuation)

        await plot_queries.save_plot_outline_to_db(plot_outline)
        logger.info("NANA: Ingestion process completed.")
        return plot_outline, len(chapters)

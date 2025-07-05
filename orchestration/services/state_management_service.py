# orchestration/services/state_management_service.py
"""Service for managing NANA Orchestrator state."""

from typing import Any

import structlog

from chapter_generation import ContextProfileName, PrerequisiteData
from core.db_manager import neo4j_manager
from data_access import chapter_repository, plot_queries
from initialization.models import PlotOutline
from models.agent_models import ChapterEndState
from orchestration.knowledge_service import KnowledgeService
from orchestration.models import KnowledgeCache
from utils.plot import get_plot_point_info
from config import settings


logger = structlog.get_logger(__name__)


class StateManagementService:
    def __init__(self, orchestrator: "NANA_Orchestrator"):
        """
        Initializes the StateManagementService.

        Args:
            orchestrator: The main NANA_Orchestrator instance.
        """
        self._orchestrator = orchestrator
        self.plot_outline: PlotOutline = PlotOutline()
        self.chapter_count: int = 0
        self.novel_props_cache: dict[str, Any] = {}
        self.knowledge_cache = KnowledgeCache()
        self.completed_plot_points: set[str] = set()
        self.next_chapter_context: str | None = None
        self.last_chapter_end_state: ChapterEndState | None = None
        self.pending_fill_ins: list[str] = []
        self.chapter_zero_end_state: ChapterEndState | None = None
        self.missing_references: dict[str, set[str]] = {
            "characters": set(),
            "locations": set(),
        }
        # self.context_service is part of the orchestrator, but used here
        # self.knowledge_service is part of the orchestrator, but used here

    async def async_init_state(self):
        """Asynchronously initializes state from the database."""
        logger.info("StateManagementService: async_init_state started...")
        self.chapter_count = await chapter_repository.load_chapter_count()
        logger.info(f"Loaded chapter count from Neo4j: {self.chapter_count}")

        await plot_queries.ensure_novel_info()
        result = await plot_queries.get_plot_outline_from_db()
        if isinstance(result, Exception):
            logger.error(
                "Error loading plot outline during state init: %s",
                result,
                exc_info=result,
            )
            self.plot_outline = PlotOutline()
        else:
            self.plot_outline = (
                PlotOutline(**result) if isinstance(result, dict) else PlotOutline()
            )

        if not self.plot_outline.get("plot_points"):
            logger.warning(
                "State init: Plot outline loaded from DB has no plot points. Initial setup might be needed or DB is empty/corrupt."
            )
        else:
            logger.info(
                f"State init: Loaded {len(self.plot_outline.get('plot_points', []))} plot points from DB."
            )

        self._update_novel_props_cache() # Uses orchestrator's knowledge_service
        self.completed_plot_points = set(await plot_queries.get_completed_plot_points())
        await self.refresh_knowledge_cache() # Uses orchestrator's knowledge_service
        logger.info("StateManagementService: async_init_state complete.")

    def _update_novel_props_cache(self) -> None:
        """Delegate to orchestrator's KnowledgeService to refresh property cache."""
        self._orchestrator.knowledge_service.update_novel_props_cache()

    async def refresh_plot_outline(self) -> None:
        """Reload plot outline from the database via orchestrator's KnowledgeService."""
        await self._orchestrator.knowledge_service.refresh_plot_outline()
        # Update local plot_outline after refresh
        result = await plot_queries.get_plot_outline_from_db()
        if isinstance(result, Exception):
            logger.error(
                "Error reloading plot outline: %s",
                result,
                exc_info=result,
            )
            # Potentially keep old outline or clear it
        else:
            self.plot_outline = (
                PlotOutline(**result) if isinstance(result, dict) else PlotOutline()
            )


    async def refresh_knowledge_cache(self) -> None:
        """Trigger KnowledgeService to refresh data, which will update this service's cache."""
        # KnowledgeService.refresh_knowledge_cache now directly updates
        # this StateManagementService instance's knowledge_cache via the orchestrator reference.
        await self._orchestrator.knowledge_service.refresh_knowledge_cache()
        # No further action needed here as KnowledgeService updates sm.knowledge_cache internally.


    async def load_previous_end_state(
        self, chapter_number: int
    ) -> ChapterEndState | None:
        """Return the ChapterEndState for ``chapter_number`` if available."""
        if chapter_number <= 0:
            # Ensure chapter_zero_end_state is loaded if not already
            if self.chapter_zero_end_state is None:
                try:
                    data = await chapter_repository.get_chapter_data(0)
                    if data and data.get("end_state_json"):
                        self.chapter_zero_end_state = ChapterEndState.model_validate_json(
                            data["end_state_json"]
                        )
                except Exception as exc:
                    logger.error("Failed to load chapter 0 end state: %s", exc, exc_info=True)
            return self.chapter_zero_end_state
        try:
            data = await chapter_repository.get_chapter_data(chapter_number)
        except Exception as exc:  # pragma: no cover - log and skip
            logger.error(
                "Failed to load chapter data for end state",
                chapter=chapter_number,
                error=exc,
                exc_info=True,
            )
            return None
        if data and data.get("end_state_json"):
            try:
                return ChapterEndState.model_validate_json(data["end_state_json"])
            except Exception as exc:  # pragma: no cover - log and skip
                logger.error(
                    "Failed to parse end state JSON",
                    chapter=chapter_number,
                    error=exc,
                    exc_info=True,
                )
        return None

    async def update_state_after_chapter_finalization(
        self, novel_chapter_number: int, end_state: ChapterEndState | None
    ) -> None:
        """Updates caches and context after a chapter is finalized."""
        self.last_chapter_end_state = end_state

        if novel_chapter_number % settings.PLOT_POINT_CHAPTER_SPAN == 0:
            pp_focus, pp_index = get_plot_point_info(
                self.plot_outline, novel_chapter_number
            )
            if pp_focus is not None and pp_index >= 0:
                await plot_queries.mark_plot_point_completed(pp_index)
                self.completed_plot_points.add(pp_focus)

        await self.refresh_plot_outline()
        if neo4j_manager.driver is not None:
            await self.refresh_knowledge_cache()
        else:
            logger.warning(
                "Neo4j driver not initialized. Skipping knowledge cache refresh."
            )

        next_hints = {"previous_chapter_end_state": end_state} if end_state else None
        # Access context_service via orchestrator
        self.next_chapter_context = await self._orchestrator.context_service.build_hybrid_context(
            self._orchestrator, # Pass orchestrator instance
            novel_chapter_number + 1,
            None,
            next_hints,
            profile_name=ContextProfileName.DEFAULT,
        )
        self._store_pending_fill_ins() # Uses orchestrator's context_service

    def _store_pending_fill_ins(self) -> None:
        """Stores pending fill-in chunks from the context service."""
        # Access context_service via orchestrator
        self.pending_fill_ins = [
            c.text for c in self._orchestrator.context_service.llm_fill_chunks if c.text
        ]

    def get_plot_outline(self) -> PlotOutline:
        return self.plot_outline

    def get_knowledge_cache(self) -> KnowledgeCache:
        return self.knowledge_cache

    def get_chapter_zero_end_state(self) -> ChapterEndState | None:
        return self.chapter_zero_end_state

    def set_chapter_zero_end_state(self, end_state: ChapterEndState | None):
        self.chapter_zero_end_state = end_state

    def get_next_chapter_context(self) -> str | None:
        return self.next_chapter_context

    def set_next_chapter_context(self, context: str | None):
        self.next_chapter_context = context

    def get_last_chapter_end_state(self) -> ChapterEndState | None:
        return self.last_chapter_end_state

    def get_pending_fill_ins(self) -> list[str]:
        return self.pending_fill_ins

    def clear_pending_fill_ins(self) -> None:
        self.pending_fill_ins = []

    def add_to_pending_fill_ins(self, fill_in: str):
        self.pending_fill_ins.append(fill_in)

    def get_chapter_count(self) -> int:
        return self.chapter_count

    def increment_chapter_count(self):
        self.chapter_count +=1

    def set_chapter_count(self, count: int):
        self.chapter_count = count

    def get_completed_plot_points(self) -> set[str]:
        return self.completed_plot_points

    def set_plot_outline(self, plot_outline: PlotOutline):
        self.plot_outline = plot_outline

    def get_novel_props_cache(self) -> dict[str, Any]:
        return self.novel_props_cache

    def set_novel_props_cache(self, cache: dict[str, Any]):
        self.novel_props_cache = cache

    def get_missing_references(self) -> dict[str, set[str]]:
        return self.missing_references

    def add_missing_reference(self, ref_type: str, ref_name: str):
        if ref_type in self.missing_references:
            self.missing_references[ref_type].add(ref_name)
        else:
            logger.warning(f"Unknown missing reference type: {ref_type}")

    def clear_missing_references(self):
        self.missing_references = {"characters": set(), "locations": set()}

    # Note: load_state_from_user_model is closely tied to plot_outline and convert_model_to_objects
    # It might fit here or in an initialization service. For now, placing it here.
    def load_state_from_user_model(self, model: "UserStoryInputModel") -> None:
        """Populate orchestrator state from a user-provided model."""
        # Assuming convert_model_to_objects is accessible (e.g. imported)
        from initialization.data_loader import convert_model_to_objects
        plot_outline, _, _ = convert_model_to_objects(model)
        self.plot_outline = plot_outline

# Add "NANA_Orchestrator" to TYPE_CHECKING to avoid circular import for type hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from orchestration.nana_orchestrator import NANA_Orchestrator
    from models.user_input_models import UserStoryInputModel

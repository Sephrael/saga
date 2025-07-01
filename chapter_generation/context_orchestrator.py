# chapter_generation/context_orchestrator.py
"""Orchestrates context generation from multiple providers."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from core.db_manager import DBManager
from core.llm_interface import LLMInterface
from data_access.character_queries import CharacterQueries
from data_access.plot_queries import PlotQueries
from data_access.world_queries import WorldQueries
from models.agent_models import ChapterInfo
from models.user_input_models import UserInput
from processing.repetition_tracker import RepetitionTracker
from prompt_renderer import PromptRenderer
from utils.logging import get_logger

from .context_providers import (
    BaseContextProvider,
    CanonicalContextProvider,
    CharacterContextProvider,
    PlotContextProvider,
    PreviousSceneContextProvider,
    RepetitionContextProvider,
    WorldContextProvider,
)

logger = get_logger(__name__)


class ContextOrchestrator:
    """Orchestrates the gathering of context from various providers."""

    def __init__(
        self,
        db_manager: DBManager,
        user_input: UserInput,
        repetition_tracker: RepetitionTracker,
    ):
        """
        Initializes the ContextOrchestrator.

        Args:
            db_manager: An instance of DBManager.
            user_input: An instance of UserInput.
            repetition_tracker: An instance of RepetitionTracker.
        """
        self.db_manager = db_manager
        self.user_input = user_input
        self.repetition_tracker = repetition_tracker
        self.prompt_renderer = PromptRenderer()
        self.plot_queries = PlotQueries(db_manager)
        self.character_queries = CharacterQueries(db_manager)
        self.world_queries = WorldQueries(db_manager)

    async def get_context_for_drafting(
        self, chapter_info: ChapterInfo, previous_chapter_text: Optional[str] = None
    ) -> str:
        """Orchestrates fetching all context needed for drafting."""
        logger.info("Orchestrating context for drafting...")

        providers = self._get_context_providers(chapter_info, previous_chapter_text)
        context_data = {}

        logger.debug(f"Using {len(providers)} context providers.")

        tasks = [provider.get_context() for provider in providers]
        results = await asyncio.gather(*tasks)

        for provider, result in zip(providers, results):
            if result:
                context_data[provider.name] = result

        hybrid_context_for_draft = self._generate_hybrid_context(context_data)
        log_file_path = (
            self.user_input.story_file_path.parent
            / "logs"
            / f"hybrid_context_for_draft_ch_{chapter_info.chapter_number}.txt"
        )
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file_path, "w") as f:
            f.write(hybrid_context_for_draft)
        logger.info(f"Hybrid context for draft saved to {log_file_path}")

        return hybrid_context_for_draft

    def _get_context_providers(
        self, chapter_info: ChapterInfo, previous_chapter_text: Optional[str] = None
    ) -> List[BaseContextProvider]:
        """Initializes and returns a list of context providers."""
        logger.debug("Initializing context providers...")
        providers = [
            CanonicalContextProvider(self.prompt_renderer, self.user_input),
            PreviousSceneContextProvider(
                self.prompt_renderer, previous_chapter_text
            ),
        ]

        if chapter_info:
            providers.extend(
                [
                    PlotContextProvider(
                        self.prompt_renderer,
                        self.db_manager,
                        chapter_info,
                        self.plot_queries,
                    ),
                    CharacterContextProvider(
                        self.prompt_renderer,
                        self.db_manager,
                        chapter_info,
                        self.character_queries,
                    ),
                    WorldContextProvider(
                        self.prompt_renderer,
                        self.db_manager,
                        chapter_info,
                        self.world_queries,
                    ),
                ]
            )

        providers.append(
            RepetitionContextProvider(
                self.prompt_renderer, self.repetition_tracker
            )
        )

        logger.debug(f"Initialized {len(providers)} context providers.")
        return providers

    def _generate_hybrid_context(self, context_data: Dict[str, str]) -> str:
        """Generates a hybrid context string from the collected data."""
        # This is a simplified approach. You might want a more sophisticated
        # template-based approach in the future.
        context_parts = []
        # Define the desired order of context sections
        order = [
            "canon",
            "previous_scene",
            "plot",
            "characters",
            "world",
            "repetition",
        ]

        for key in order:
            if key in context_data:
                context_parts.append(context_data[key])

        return "\n\n".join(context_parts)


# orchestration/services/initialization_service.py
"""Service for handling the initial setup and run initialization of the NANA Orchestrator."""

import time
from typing import TYPE_CHECKING

import structlog
from chapter_generation import ContextProfileName
from config import settings
from core.db_manager import neo4j_manager
from initialization.genesis import run_genesis_phase

if TYPE_CHECKING:
    from orchestration.nana_orchestrator import NANA_Orchestrator

logger = structlog.get_logger(__name__)


class InitializationService:
    def __init__(self, orchestrator: "NANA_Orchestrator"):
        self._orchestrator = orchestrator
        # Access to other services/managers via orchestrator if needed
        self._state_manager = orchestrator.state_manager
        self._token_manager = orchestrator.token_manager
        self._display = (
            orchestrator.display
        )  # For direct basic updates on critical errors
        self._context_service = orchestrator.context_service
        self._kg_maintainer_agent = orchestrator.kg_maintainer_agent

    def _validate_critical_configs(self) -> bool:
        """Moved from NANA_Orchestrator._validate_critical_configs"""
        critical_str_configs = {
            "OLLAMA_EMBED_URL": settings.OLLAMA_EMBED_URL,
            "OPENAI_API_BASE": settings.OPENAI_API_BASE,
            "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
            "NEO4J_URI": settings.NEO4J_URI,
            "LARGE_MODEL": settings.LARGE_MODEL,
            "MEDIUM_MODEL": settings.MEDIUM_MODEL,
            "SMALL_MODEL": settings.SMALL_MODEL,
            "NARRATOR_MODEL": settings.NARRATOR_MODEL,
        }
        missing_or_empty_configs = []
        for name, value in critical_str_configs.items():
            if not value or not isinstance(value, str) or not value.strip():
                missing_or_empty_configs.append(name)

        if missing_or_empty_configs:
            logger.critical(
                f"InitializationService CRITICAL CONFIG ERROR: Missing/empty: {', '.join(missing_or_empty_configs)}."
            )
            return False

        if settings.EXPECTED_EMBEDDING_DIM <= 0:
            logger.critical(
                f"InitializationService CRITICAL CONFIG ERROR: EXPECTED_EMBEDDING_DIM must be > 0, is {settings.EXPECTED_EMBEDDING_DIM}."
            )
            return False
        logger.info(
            "Critical configurations validated successfully by InitializationService."
        )
        return True

    async def _setup_db_and_kg_schema(self) -> bool:
        """Moved from NANA_Orchestrator._setup_db_and_kg_schema"""
        try:
            async with (
                neo4j_manager
            ):  # Assumes neo4j_manager is globally available or passed
                await neo4j_manager.create_db_schema()
                logger.info(
                    "InitializationService: Neo4j connection and schema verified."
                )

                await self._kg_maintainer_agent.load_schema_from_db()
                logger.info(
                    "InitializationService: KG schema loaded into maintainer agent."
                )
                return True
        except Exception as exc:
            logger.critical(
                "InitializationService: Database or KG schema setup failed: %s",
                exc,
                exc_info=True,
            )
            return False

    async def initialize_run_environment(self) -> bool:
        """
        Combined logic from NANA_Orchestrator._initialize_run.
        Validates configs, sets up DB, initializes token/time, and basic orchestrator state.
        """
        if not self._validate_critical_configs():
            self._display.update_basic(step="Critical Config Error - Halting")
            # Orchestrator should handle display.stop() if this returns False
            return False

        self._token_manager.reset_tokens_for_new_run()
        self._token_manager.set_run_start_time(time.time())
        # Orchestrator should handle display.start()
        self._token_manager._update_rich_display(step="Run Initialized by InitService")

        if not await self._setup_db_and_kg_schema():
            # Error already logged by _setup_db_and_kg_schema
            return False

        try:
            # Formerly self.async_init_orchestrator()
            await self._state_manager.async_init_state()
        except Exception as exc:
            logger.critical(
                "InitializationService: Orchestrator state init (async_init_state) failed: %s",
                exc,
                exc_info=True,
            )
            return False
        return True

    async def perform_genesis_setup_if_needed(self) -> bool:
        """
        Combined logic from NANA_Orchestrator._ensure_initial_setup and perform_initial_setup.
        Checks if genesis (initial content generation) is needed and runs it.
        """
        current_plot_outline = self._state_manager.get_plot_outline()
        plot_points_exist = (
            current_plot_outline
            and current_plot_outline.get("plot_points")
            and len(
                [
                    pp
                    for pp in current_plot_outline.get("plot_points", [])
                    if not utils.is_fill_in(
                        pp
                    )  # Assuming utils.is_fill_in is accessible
                ]
            )
            > 0
        )

        if (
            not plot_points_exist
            or not current_plot_outline.get("title")
            or utils.is_fill_in(current_plot_outline.get("title"))
        ):
            logger.info(
                "InitializationService: Core plot data missing. Performing genesis setup..."
            )
            self._token_manager._update_rich_display(step="Performing Genesis Setup")

            # Logic from NANA_Orchestrator.perform_initial_setup
            try:
                # run_genesis_phase returns a new plot_outline, characters, world, usage
                (
                    plot_outline_from_genesis,
                    character_profiles,
                    world_building,
                    usage,
                ) = (
                    await run_genesis_phase()
                )  # run_genesis_phase needs to be importable

                # Orchestrator's _accumulate_tokens needs to be called
                self._orchestrator._accumulate_tokens(
                    Stage.GENESIS_PHASE,
                    usage,
                    current_step_for_display="Genesis State Bootstrapped",
                )

                self._state_manager.set_plot_outline(plot_outline_from_genesis)

                plot_source = plot_outline_from_genesis.get("source", "unknown")
                logger.info(
                    f"   Genesis: Plot Outline initialized (source: {plot_source}). "
                    f"Title: '{plot_outline_from_genesis.get('title', 'N/A')}'. "
                    f"Plot Points: {len(plot_outline_from_genesis.get('plot_points', []))}"
                )
                world_source = world_building.get("source", "unknown")
                logger.info(
                    f"   Genesis: World Building initialized (source: {world_source})."
                )
                # Display update is covered by token accumulation

                current_kc = self._state_manager.get_knowledge_cache()
                current_kc.characters = character_profiles
                current_kc.world = world_building
                await (
                    self._state_manager.refresh_knowledge_cache()
                )  # Refreshes from knowledge_service
                self._state_manager._update_novel_props_cache()
                logger.info(
                    "   Genesis: Initial plot, char, world data saved to Neo4j via services."
                )
                self._token_manager._update_rich_display(step="Genesis State Saved")

                await self._state_manager.refresh_plot_outline()
                if neo4j_manager.driver is not None:  # Check DB connection
                    await self._state_manager.refresh_knowledge_cache()

                chapter_zero_end_state = (
                    await self._state_manager.load_previous_end_state(0)
                )
                self._state_manager.set_chapter_zero_end_state(chapter_zero_end_state)

                next_chapter_context = await self._context_service.build_hybrid_context(
                    self._orchestrator,  # Context service needs orchestrator reference
                    1,
                    None,
                    (
                        {"chapter_zero_end_state": chapter_zero_end_state}
                        if chapter_zero_end_state
                        else None
                    ),
                    profile_name=ContextProfileName.DEFAULT,
                )
                self._state_manager.set_next_chapter_context(next_chapter_context)
                logger.info("InitializationService: Genesis setup complete.")
                return True
            except Exception as e:
                logger.critical(
                    f"InitializationService: Genesis setup failed: {e}", exc_info=True
                )
                self._token_manager._update_rich_display(
                    step="Genesis Setup Failed - Halting"
                )
                return False
        else:
            logger.info(
                "InitializationService: Core plot data exists. Skipping genesis setup."
            )
            # Ensure novel props cache is updated even if genesis is skipped
            self._state_manager._update_novel_props_cache()
            return True

    async def setup_and_prepare_run(self) -> bool:
        """
        Main entry point for this service from the orchestrator.
        Combines environment initialization and genesis setup if needed.
        Corresponds to NANA_Orchestrator._setup_and_prepare_run.
        """
        if not await self.initialize_run_environment():
            # Errors logged, display updated by initialize_run_environment
            return False
        if not await self.perform_genesis_setup_if_needed():
            # Errors logged, display updated by perform_genesis_setup_if_needed
            return False
        return True


# Helper import for utils.is_fill_in if not directly available
# This might require moving is_fill_in to a more common location if it's used widely
# For now, assuming it can be imported or accessed.
from orchestration.token_accountant import Stage  # For GENESIS_PHASE

import utils


# Potential future location for is_fill_in if it needs to be shared
# in, for example, utils/text_utils.py
# def is_fill_in(value: any) -> bool:
#    if isinstance(value, str):
#        return value.strip().lower() in ["[fill in]", "[to be determined]", "tbd"]
#    return False

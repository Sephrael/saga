# chapter_generation/drafting_service.py
"""Service for drafting initial chapter text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.drafting_agent import DraftingAgent
from agents.planner_agent import PlannerAgent
from core.db_manager import DBManager
from core.llm_interface import LLMInterface
from models.agent_models import Draft
from orchestration.token_accountant import TokenAccountant
from prompt_renderer import PromptRenderer
from storage.file_manager import FileManager
from utils.logging import get_logger

from .context_orchestrator import ContextOrchestrator

if TYPE_CHECKING:
    from orchestration.chapter_flow import ChapterFlow


logger = get_logger(__name__)


class DraftingService:
    """A service for drafting a chapter."""

    def __init__(
        self,
        db_manager: DBManager,
        llm: LLMInterface,
        file_manager: FileManager,
        token_accountant: TokenAccountant,
        chapter_flow: "ChapterFlow",
    ):
        """
        Initializes the DraftingService.

        Args:
            db_manager: An instance of DBManager.
            llm: An instance of LLMInterface.
            file_manager: An instance of FileManager.
            token_accountant: An instance of TokenAccountant.
            chapter_flow: A back-reference to the main ChapterFlow.
        """
        self.db_manager = db_manager
        self.llm = llm
        self.file_manager = file_manager
        self.token_accountant = token_accountant
        self.chapter_flow = chapter_flow
        self.context_orchestrator = ContextOrchestrator(
            db_manager=self.db_manager,
            user_input=self.chapter_flow.user_input,
            repetition_tracker=self.chapter_flow.repetition_tracker,
        )
        self.planner_agent = PlannerAgent(self.llm, self.token_accountant)
        self.drafting_agent = DraftingAgent(self.llm, self.token_accountant)
        self.prompt_renderer = PromptRenderer()

    async def run(self) -> Draft:
        """
        Runs the drafting process.

        This method orchestrates the generation of a chapter draft, including
        planning and the actual writing.

        Returns:
            A Draft object containing the drafted text.
        """
        logger.info(
            f"Starting draft for chapter {self.chapter_flow.chapter_info.chapter_number}..."
        )

        hybrid_context_for_draft = (
            await self.context_orchestrator.get_context_for_drafting(
                self.chapter_flow.chapter_info, self.chapter_flow.previous_chapter_text
            )
        )
        self.chapter_flow.set_hybrid_context_for_draft(hybrid_context_for_draft)

        if not self.chapter_flow.scene_plan:
            logger.info("No scene plan found, generating one now.")
            scene_plan_result = await self.planner_agent.run(
                self.chapter_flow.user_input,
                self.chapter_flow.chapter_info,
                self.chapter_flow.get_full_context_for_planning(),
                self.chapter_flow.get_previous_chapter_summary(),
            )
            if scene_plan_result.scene_plan:
                self.chapter_flow.set_scene_plan(scene_plan_result.scene_plan)
            else:
                logger.warning("Planner agent did not return a scene plan.")
                # Fallback or error handling can be added here.
                # For now, we proceed with an empty plan, and the drafter will have to work harder.
                self.chapter_flow.set_scene_plan("No plan generated.")
        else:
            logger.info("Using existing scene plan.")

        draft_result = await self.drafting_agent.run(
            self.chapter_flow.user_input,
            self.chapter_flow.chapter_info,
            self.chapter_flow.scene_plan,
            hybrid_context_for_draft,
            self.chapter_flow.get_previous_chapter_summary(),
        )

        draft = Draft(
            chapter_number=self.chapter_flow.chapter_info.chapter_number,
            text=draft_result.draft_text,
            title=self.chapter_flow.chapter_info.title,
            summary="",  # Will be generated later
        )

        self.chapter_flow.add_draft(draft)
        logger.info(
            f"Completed draft for chapter {self.chapter_flow.chapter_info.chapter_number}."
        )
        return draft

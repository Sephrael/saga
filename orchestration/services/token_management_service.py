# orchestration/services/token_management_service.py
"""Service for managing token usage and display updates related to tokens."""

import time
import structlog

from core.usage import TokenUsage
from orchestration.token_accountant import TokenAccountant, Stage
# Assuming RichDisplayManager is accessible or passed if needed for direct updates
# from ui.rich_display import RichDisplayManager

logger = structlog.get_logger(__name__)

class TokenManagementService:
    def __init__(self, display_manager: "RichDisplayManager", orchestrator_ref: "NANA_Orchestrator"):
        """
        Initializes the TokenManagementService.

        Args:
            display_manager: The RichDisplayManager instance for UI updates.
            orchestrator_ref: Reference to the main NANA_Orchestrator instance for run_start_time.
        """
        self.token_accountant = TokenAccountant()
        self.total_tokens_generated_this_run: int = 0
        self._display_manager = display_manager
        self._orchestrator_ref = orchestrator_ref # To access run_start_time and plot_outline

    def _update_rich_display(
        self, chapter_num: int | None = None, step: str | None = None
    ) -> None:
        """Updates the Rich display with the current token count and other relevant info."""
        # This method might need access to plot_outline and run_start_time from the orchestrator
        # or these could be passed in if they change, or the display manager handles them.
        # For now, assuming orchestrator_ref provides them.
        self._display_manager.update(
            plot_outline=self._orchestrator_ref.state_manager.get_plot_outline(),
            chapter_num=chapter_num,
            step=step,
            total_tokens=self.total_tokens_generated_this_run,
            run_start_time=self._orchestrator_ref.run_start_time, # Access via orchestrator_ref
        )

    def accumulate_tokens(
        self, stage: str | Stage, usage_data: dict[str, int] | TokenUsage | None,
        chapter_num: int | None = None, current_step_for_display: str | None = None
    ) -> None:
        """
        Records token usage for a given stage and updates the total.
        Also triggers a display update.
        """
        stage_value = stage.value if isinstance(stage, Stage) else stage
        self.token_accountant.record_usage(stage_value, usage_data)
        self.total_tokens_generated_this_run = self.token_accountant.total
        # Pass chapter_num and step to display if available
        self._update_rich_display(chapter_num=chapter_num, step=current_step_for_display)

    def get_total_tokens_generated_this_run(self) -> int:
        return self.total_tokens_generated_this_run

    def reset_tokens_for_new_run(self) -> None:
        """Resets token counts for a new generation run."""
        self.token_accountant = TokenAccountant() # Re-initialize
        self.total_tokens_generated_this_run = 0
        # Potentially update display to reflect reset state
        self._update_rich_display(step="New Run Initialized - Tokens Reset")

    def get_run_start_time(self) -> float:
        # Delegate to orchestrator reference as it's more global state
        return self._orchestrator_ref.run_start_time

    def set_run_start_time(self, start_time: float) -> None:
        # Delegate to orchestrator reference
        self._orchestrator_ref.run_start_time = start_time


# Add "RichDisplayManager" and "NANA_Orchestrator" to TYPE_CHECKING for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ui.rich_display import RichDisplayManager
    from orchestration.nana_orchestrator import NANA_Orchestrator

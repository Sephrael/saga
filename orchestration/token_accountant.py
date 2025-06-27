from __future__ import annotations

import logging
from enum import Enum

from core.usage import TokenUsage

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    """Stages for token accounting."""

    GENESIS_PHASE = "Genesis-Phase"
    PLAN_CONTINUATION = "PlanContinuation"
    CHAPTER_PLANNING = "Chapter-Planning"
    PLAN_CONSISTENCY = "PlanConsistency"
    DRAFTING = "Drafting"
    EVALUATION = "Evaluation"
    CONTINUITY_CHECK = "ContinuityCheck"
    REVISION = "Revision"
    SUMMARIZATION = "Summarization"
    KG_EXTRACTION_MERGE = "KGExtractionMerge"


class TokenAccountant:
    """Accumulate and log token usage across stages."""

    def __init__(self) -> None:
        self.total: int = 0
        self.stage_totals: dict[str, int] = {}

    def record_usage(
        self, stage: Stage | str, usage: dict[str, int] | TokenUsage | None
    ) -> None:
        """Record token usage for a stage."""
        stage_name = stage.value if isinstance(stage, Stage) else stage

        usage_dict: dict[str, int]
        if isinstance(usage, TokenUsage):
            usage_dict = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        else:
            usage_dict = usage or {}

        if usage_dict and isinstance(usage_dict.get("completion_tokens"), int):
            completed_tokens = usage_dict["completion_tokens"]
            self.total += completed_tokens
            self.stage_totals[stage_name] = (
                self.stage_totals.get(stage_name, 0) + completed_tokens
            )
            logger.info(
                "NANA Activity: Tokens from '%s': %s. Total generated this run: %s",
                stage_name,
                completed_tokens,
                self.total,
            )
        elif (
            usage_dict
            and isinstance(usage_dict.get("total_tokens"), int)
            and not isinstance(usage_dict.get("completion_tokens"), int)
        ):
            logger.info(
                "NANA Activity: Total tokens from '%s': %s. (Completion tokens not specifically available). Total generated this run (completion focused): %s",
                stage_name,
                usage_dict["total_tokens"],
                self.total,
            )
        elif usage_dict:
            logger.warning(
                "NANA Activity: '%s' - 'completion_tokens' missing or not int in usage_data. Tokens not added. Usage: %s",
                stage_name,
                usage_dict,
            )

    def get_stage_total(self, stage: Stage | str) -> int:
        """Return accumulated tokens for a stage."""
        stage_name = stage.value if isinstance(stage, Stage) else stage
        return self.stage_totals.get(stage_name, 0)

# agents/patch_validation_agent.py

"""Patch validation agent for SAGA."""

from __future__ import annotations

import structlog
from config import settings
from core.llm_interface import llm_service
from prompt_renderer import render_prompt

from models import PatchInstruction, ProblemDetail

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "with",
    "without",
    "to",
    "from",
    "by",
    "of",
    "in",
    "on",
    "at",
    "include",
    "mention",
    "use",
    "add",
    "make",
    "this",
    "that",
    "these",
    "those",
}

logger = structlog.get_logger(__name__)


class PatchValidationAgent:
    """Validates patch instructions using an LLM."""

    def __init__(self, model_name: str = settings.EVALUATION_MODEL) -> None:
        self.model_name = model_name
        logger.info("PatchValidationAgent initialized with model: %s", self.model_name)

    async def validate_patch(
        self,
        context_snippet: str,
        patch: PatchInstruction,
        problems: list[ProblemDetail],
    ) -> tuple[bool, str | None, dict[str, int] | None]:
        """Validate a single patch via an LLM call.

        Args:
            context_snippet: Unused snippet of surrounding text.
            patch: Proposed patch instruction.
            problems: List of problems the patch should solve. The first problem
                is used for validation.

        Returns:
            Tuple ``(is_valid, failure_reason, usage)`` where ``is_valid`` is
            ``True`` if the LLM affirms the patch fixes the problem.
        """

        problem: ProblemDetail | None = problems[0] if problems else None
        prompt = render_prompt(
            "patch_validation_agent/validate_patch.j2",
            {
                "problem": problem,
                "patch": patch,
            },
        )

        response_text, usage = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=settings.TEMPERATURE_EVALUATION,
            max_tokens=256,
            allow_fallback=True,
            stream_to_disk=False,
            auto_clean_response=True,
        )

        lines = response_text.splitlines()
        first_line = lines[0].strip().upper() if lines else ""
        is_pass = first_line.startswith("YES")
        failure_reason: str | None = None
        if not is_pass and len(lines) > 1:
            failure_reason = lines[1].strip()

        return is_pass, failure_reason, usage


class NoOpPatchValidator(PatchValidationAgent):
    """Bypass patch validation and always approve patches."""

    def __init__(self) -> None:  # noqa: D401 - no behavior to document
        """Initialize without calling the parent constructor."""
        # Intentionally skip PatchValidationAgent initialization

    async def validate_patch(
        self,
        context_snippet: str,
        patch: PatchInstruction,
        problems: list[ProblemDetail],
    ) -> tuple[bool, None, None]:
        """Approve patches without validation.

        Args:
            context_snippet: Unused text snippet from the chapter.
            patch: Patch instruction being "validated".
            problems: Original list of problems for context.

        Returns:
            ``True`` with ``None`` for usage to mimic the real validator's
            return type.

        Side Effects:
            None.
        """
        return True, None, None

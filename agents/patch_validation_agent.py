# agents/patch_validation_agent.py

"""Patch validation agent for SAGA."""

from __future__ import annotations

import string

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
    ) -> tuple[bool, dict[str, int] | None]:
        """Validate a single patch against reported problems.

        Args:
            context_snippet: Portion of chapter text surrounding the problem.
            patch: Proposed patch instruction to validate.
            problems: Collection of issues the patch should address.

        Returns:
            A tuple where the first element indicates whether the patch passed
            validation and the second contains optional token usage data from the
            LLM call.

        Side Effects:
            Sends a prompt to the LLM service and logs diagnostic information.
        """

        issues_list = "\n".join(
            f"- {p.get('problem_description', '') if isinstance(p, dict) else p.problem_description}"
            for p in problems
        )
        prompt = render_prompt(
            "patch_validation_agent/validate_patch.j2",
            {
                "context_snippet": context_snippet,
                "patch_text": patch.get("replace_with", ""),
                "issues_list": issues_list,
            },
        )

        response_text, usage = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=settings.TEMPERATURE_EVALUATION,
            max_tokens=1024,
            allow_fallback=True,
            stream_to_disk=False,
            auto_clean_response=True,
        )

        first_line = response_text.splitlines()[0].strip().lower()
        score = 0
        for token in first_line.split():
            if token.isdigit():
                score = int(token)
                break
        is_pass = score >= settings.PATCH_VALIDATION_THRESHOLD

        failure_reason: str | None = None
        if not is_pass:
            score_is_ok = score >= settings.PATCH_VALIDATION_THRESHOLD
            if not score_is_ok:
                failure_reason = f"Score {score} below threshold {settings.PATCH_VALIDATION_THRESHOLD}"
                logger.info(
                    "Patch validation FAILED. Score %d is below threshold %d.",
                    score,
                    settings.PATCH_VALIDATION_THRESHOLD,
                )
            else:
                failure_reason = (
                    first_line.split(" ", 1)[1]
                    if " " in first_line
                    else "Validation failed"
                )
        patch_text_lower = patch.get("replace_with", "").lower()

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

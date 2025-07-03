"""Patch generation utilities."""

from typing import Any

from agents.patch_validation_agent import PatchValidationAgent
from core.usage import TokenUsage

from models import ProblemDetail, SceneDetail

from . import apply, context, instructions

_get_context_window_for_patch_llm = context._get_context_window_for_patch_llm
_get_formatted_scene_plan_from_agent_or_fallback = (
    context._get_formatted_scene_plan_from_agent_or_fallback
)
_generate_single_patch_instruction_llm = (
    instructions._generate_single_patch_instruction_llm
)
_deduplicate_problems = instructions._deduplicate_problems
_group_problems_for_patch_generation = instructions._group_problems_for_patch_generation
_generate_patch_instructions_logic = instructions._generate_patch_instructions_logic
_apply_patches_to_text = apply._apply_patches_to_text
_get_sentence_embeddings = apply._get_sentence_embeddings
locate_patch_targets = apply.locate_patch_targets


class PatchGenerator:
    """Generate and apply patch instructions to chapter text."""

    async def generate_and_apply(
        self,
        plot_outline: dict[str, Any],
        original_text: str,
        problems_to_fix: list[ProblemDetail],
        chapter_number: int,
        hybrid_context_for_revision: str,
        chapter_plan: list[SceneDetail] | None,
        already_patched_spans: list[tuple[int, int]] | None,
        validator: PatchValidationAgent,
    ) -> tuple[str, list[tuple[int, int]], TokenUsage | None]:
        """Generate patch instructions and apply them.

        Args:
            plot_outline: Outline of the overall plot for context.
            original_text: The raw chapter text to be revised.
            problems_to_fix: Issues detected during evaluation.
            chapter_number: Current chapter index.
            hybrid_context_for_revision: Context window from previous chapters.
            chapter_plan: Optional list of scene details for planning.
            already_patched_spans: Ranges that should be preserved.
            validator: Agent responsible for validating proposed patches.

        Returns:
            A tuple containing the patched text, the spans in the revised text
            that correspond to applied patches, and optional token usage data.

        Side Effects:
            Calls out to the LLM service for embeddings and patch generation and
            logs progress via ``structlog``.
        """
        sentence_embeddings = await _get_sentence_embeddings(original_text)
        patch_instructions, usage = await _generate_patch_instructions_logic(
            plot_outline,
            original_text,
            problems_to_fix,
            chapter_number,
            hybrid_context_for_revision,
            chapter_plan,
            validator,
        )
        patched_text, spans = await _apply_patches_to_text(
            original_text,
            patch_instructions,
            already_patched_spans,
            sentence_embeddings,
        )
        return patched_text, spans, usage

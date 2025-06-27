"""Patch generation utilities."""

from typing import Any

from agents.patch_validation_agent import PatchValidationAgent

from models import ProblemDetail, SceneDetail

from . import apply, context, instructions

_get_context_window_for_patch_llm = context._get_context_window_for_patch_llm
_get_formatted_scene_plan_from_agent_or_fallback = (
    context._get_formatted_scene_plan_from_agent_or_fallback
)
_generate_single_patch_instruction_llm = (
    instructions._generate_single_patch_instruction_llm
)
_consolidate_overlapping_problems = instructions._consolidate_overlapping_problems
_deduplicate_problems = instructions._deduplicate_problems
_group_problems_for_patch_generation = instructions._group_problems_for_patch_generation
_generate_patch_instructions_logic = instructions._generate_patch_instructions_logic
_apply_patches_to_text = apply._apply_patches_to_text
_get_sentence_embeddings = apply._get_sentence_embeddings


class PatchGenerator:
    """Generate patch instructions and apply them to text."""

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
    ) -> tuple[str, list[tuple[int, int]]]:
        """Return revised text and updated spans after applying patches."""
        sentence_embeddings = await _get_sentence_embeddings(original_text)
        patch_instructions, _ = await _generate_patch_instructions_logic(
            plot_outline,
            original_text,
            problems_to_fix,
            chapter_number,
            hybrid_context_for_revision,
            chapter_plan,
            validator,
        )
        return await _apply_patches_to_text(
            original_text,
            patch_instructions,
            already_patched_spans,
            sentence_embeddings,
        )

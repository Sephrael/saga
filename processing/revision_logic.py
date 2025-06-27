# chapter_revision_logic.py
"""
Handles the revision of chapter drafts based on evaluation feedback for the SAGA system.
Supports both full rewrite and targeted patch-based revisions.
Context data for prompts is now formatted as plain text.
"""

from typing import Any

import structlog
import utils  # For numpy_cosine_similarity, find_semantically_closest_segment, AND find_quote_and_sentence_offsets_with_spacy, format_scene_plan_for_prompt
from utils.plot import get_plot_point_info

from models import (
    CharacterProfile,
    EvaluationResult,
    SceneDetail,
    WorldItem,
)

from . import patch_generator

_get_formatted_scene_plan_from_agent_or_fallback = (
    patch_generator._get_formatted_scene_plan_from_agent_or_fallback
)
_get_plot_point_info = get_plot_point_info
_get_context_window_for_patch_llm = patch_generator._get_context_window_for_patch_llm
_get_sentence_embeddings = patch_generator._get_sentence_embeddings
_find_sentence_via_embeddings = patch_generator._find_sentence_via_embeddings
_generate_single_patch_instruction_llm = (
    patch_generator._generate_single_patch_instruction_llm
)
_consolidate_overlapping_problems = patch_generator._consolidate_overlapping_problems
_deduplicate_problems = patch_generator._deduplicate_problems
_group_problems_for_patch_generation = (
    patch_generator._group_problems_for_patch_generation
)
_generate_patch_instructions_logic = patch_generator._generate_patch_instructions_logic
_apply_patches_to_text = patch_generator._apply_patches_to_text

logger = structlog.get_logger(__name__)
utils.load_spacy_model_if_needed()  # Ensure spaCy model is loaded when this module is imported


async def revise_chapter_draft_logic(
    plot_outline: dict[str, Any],
    character_profiles: dict[str, CharacterProfile],
    world_building: dict[str, dict[str, WorldItem]],
    original_text: str,
    chapter_number: int,
    evaluation_result: EvaluationResult,
    hybrid_context_for_revision: str,
    chapter_plan: list[SceneDetail] | None,
    is_from_flawed_source: bool = False,
    already_patched_spans: list[tuple[int, int]] | None = None,
) -> tuple[tuple[str, str | None, list[tuple[int, int]]] | None, dict[str, int] | None]:
    """Wrapper for backward compatibility calling ``RevisionManager``."""
    from .revision_manager import RevisionManager

    manager = RevisionManager()
    return await manager.revise_chapter(
        plot_outline,
        character_profiles,
        world_building,
        original_text,
        chapter_number,
        evaluation_result,
        hybrid_context_for_revision,
        chapter_plan,
        is_from_flawed_source=is_from_flawed_source,
        already_patched_spans=already_patched_spans,
    )

# utils/__init__.py
"""General utility functions for the SAGA Novel Generation system."""

from __future__ import annotations

import asyncio
import logging as std_logging
import re
from typing import TYPE_CHECKING

import numpy as np
from config import settings
from core.llm_interface import count_tokens, llm_service

from .logging import setup_logging_nana
from .plot import get_plot_point_info, get_scoped_plot_outline
from .similarity import find_semantically_closest_segment, numpy_cosine_similarity
from .text_processing import (
    SpaCyModelManager,
    _normalize_for_id,
    _normalize_text_for_matching,
    find_quote_and_sentence_offsets_with_spacy,
    get_text_segments,
    load_spacy_model_if_needed,
    normalize_trait_name,
    spacy_manager,
)
from .text_utils import _is_fill_in

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from kg_maintainer.models import SceneDetail

logger = std_logging.getLogger(__name__)


def format_scene_plan_for_prompt(
    chapter_plan: list[SceneDetail],
    model_name_for_tokens: str,
    max_tokens_budget: int,
) -> str:
    """Format a chapter plan into plain text for LLM prompts respecting token limits."""
    if not chapter_plan:
        return "No detailed scene plan available."

    plan_lines = ["**Detailed Scene Plan (MUST BE FOLLOWED CLOSELY):**"]
    current_plan_parts = [plan_lines[0]]

    for scene_idx, scene in enumerate(chapter_plan):
        scene_lines = [
            f"Scene Number: {scene.get('scene_number', 'N/A')}",
            f"  Summary: {scene.get('summary', 'N/A')}",
            f"  Characters Involved: {', '.join(scene.get('characters_involved', [])) if scene.get('characters_involved') else 'None'}",
            "  Key Dialogue Points:",
        ]
        for point in scene.get("key_dialogue_points", []):
            scene_lines.append(f"    - {point}")
        scene_lines.append(f"  Setting Details: {scene.get('setting_details', 'N/A')}")
        scene_lines.append("  Scene Focus Elements:")
        for focus_el in scene.get("scene_focus_elements", []):
            scene_lines.append(f"    - {focus_el}")
        scene_lines.append(f"  Contribution: {scene.get('contribution', 'N/A')}")

        if scene_idx < len(chapter_plan) - 1:
            scene_lines.append("-" * 20)

        scene_segment = "\n".join(scene_lines)
        prospective_plan = "\n".join(current_plan_parts + [scene_segment])

        if count_tokens(prospective_plan, model_name_for_tokens) > max_tokens_budget:
            current_plan_parts.append(
                "... (plan truncated in prompt due to token limit)"
            )
            logger.warning(
                "Chapter plan was token-truncated for the prompt. Max tokens for plan: %d. Stopped before scene %s.",
                max_tokens_budget,
                scene.get("scene_number", "N/A"),
            )
            break

        current_plan_parts.append(scene_segment)

    if len(current_plan_parts) <= 1:
        return "No detailed scene plan available or plan was too long to include any scenes."

    return "\n".join(current_plan_parts)


async def deduplicate_text_segments(
    original_text: str,
    segment_level: str = "paragraph",
    similarity_threshold: float = settings.DEDUPLICATION_SEMANTIC_THRESHOLD,
    use_semantic_comparison: bool = settings.DEDUPLICATION_USE_SEMANTIC,
    min_segment_length_chars: int = settings.DEDUPLICATION_MIN_SEGMENT_LENGTH,
    prefer_newer: bool = False,
) -> tuple[str, int]:
    """Remove near-duplicate segments from text."""
    if not original_text.strip():
        return original_text, 0

    segments_with_offsets = get_text_segments(original_text, segment_level)
    if not segments_with_offsets:
        return original_text, 0

    indices = await _find_duplicate_indices(
        segments_with_offsets,
        similarity_threshold,
        use_semantic_comparison,
        min_segment_length_chars,
        prefer_newer,
    )

    if not indices:
        return original_text, 0

    cleaned_text = remove_spans_from_text(
        original_text, [segments_with_offsets[i][1:] for i in sorted(indices)]
    )
    cleaned_text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    return cleaned_text, len(original_text) - len(cleaned_text)


async def _prepare_segment_representations(
    segments: list[tuple[str, int, int]], use_semantic: bool
) -> tuple[list[np.ndarray | None], list[str]]:
    """Return embeddings or normalized texts for ``segments``."""
    if use_semantic:
        tasks = [llm_service.async_get_embedding(seg[0]) for seg in segments]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        embeddings = [
            res if not isinstance(res, Exception) else None for res in results
        ]
        return embeddings, []
    normalized = [_normalize_text_for_matching(seg[0]) for seg in segments]
    return [], normalized


def _is_duplicate_segment(
    i: int,
    j: int,
    segments: list[tuple[str, int, int]],
    embeddings: list[np.ndarray | None],
    normalized: list[str],
    threshold: float,
    use_semantic: bool,
) -> bool:
    """Return ``True`` if segments ``i`` and ``j`` are duplicates."""
    text_i = segments[i][0]
    text_j = segments[j][0]
    if use_semantic:
        emb_i = embeddings[i] if embeddings else None
        emb_j = embeddings[j] if embeddings else None
        if emb_i is not None and emb_j is not None:
            similarity = numpy_cosine_similarity(emb_i, emb_j)
            if similarity > threshold:
                return True
        return _normalize_text_for_matching(text_i) == _normalize_text_for_matching(
            text_j
        )
    return normalized[i] == normalized[j]


async def _find_duplicate_indices(
    segments: list[tuple[str, int, int]],
    threshold: float,
    use_semantic: bool,
    min_length: int,
    prefer_newer: bool,
) -> set[int]:
    """Return set of indices for segments that should be removed."""
    embeddings, normalized = await _prepare_segment_representations(
        segments, use_semantic
    )
    num_segments = len(segments)
    indices_to_remove: set[int] = set()

    outer_range = (
        range(num_segments - 1, -1, -1) if prefer_newer else range(num_segments)
    )

    for i in outer_range:
        if i in indices_to_remove:
            continue
        if len(segments[i][0]) < min_length:
            continue
        inner_range = (
            range(i - 1, -1, -1) if prefer_newer else range(i + 1, num_segments)
        )
        for j in inner_range:
            if j in indices_to_remove:
                continue
            if len(segments[j][0]) < min_length:
                continue
            if _is_duplicate_segment(
                i, j, segments, embeddings, normalized, threshold, use_semantic
            ):
                indices_to_remove.add(j)
                method_used = "semantic" if use_semantic else "normalized string"
                logger.info(
                    "De-duplication: Marking segment (idx %d, chars %d-%d) for removal as duplicate of (idx %d, chars %d-%d). Method: %s.",
                    j,
                    segments[j][1],
                    segments[j][2],
                    i,
                    segments[i][1],
                    segments[i][2],
                    method_used,
                )
    return indices_to_remove


def remove_spans_from_text(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove character spans from ``text``."""
    if not spans:
        return text

    spans_sorted = sorted(spans, key=lambda x: x[0])
    result_parts: list[str] = []
    last_end = 0
    for start, end in spans_sorted:
        if start > last_end:
            result_parts.append(text[last_end:start])
        last_end = max(last_end, end)
    result_parts.append(text[last_end:])
    return "".join(result_parts)


__all__ = [
    "_normalize_for_id",
    "normalize_trait_name",
    "SpaCyModelManager",
    "spacy_manager",
    "_is_fill_in",
    "load_spacy_model_if_needed",
    "_normalize_text_for_matching",
    "find_quote_and_sentence_offsets_with_spacy",
    "find_semantically_closest_segment",
    "numpy_cosine_similarity",
    "get_text_segments",
    "format_scene_plan_for_prompt",
    "get_plot_point_info",
    "get_scoped_plot_outline",
    "deduplicate_text_segments",
    "remove_spans_from_text",
    "setup_logging_nana",
]

"""Context utilities for patch generation."""

import structlog
import utils

from models import ProblemDetail, SceneDetail

logger = structlog.get_logger(__name__)
utils.load_spacy_model_if_needed()


def _get_formatted_scene_plan_from_agent_or_fallback(
    chapter_plan: list[SceneDetail],
    model_name_for_tokens: str,
    max_tokens_budget: int,
) -> str:
    """Formats a chapter plan into plain text for LLM prompts."""
    return utils.format_scene_plan_for_prompt(
        chapter_plan, model_name_for_tokens, max_tokens_budget
    )


async def _get_context_window_for_patch_llm(
    original_doc_text: str, problem: ProblemDetail, window_size_chars: int
) -> str:
    """Return a context window around the problem quote."""
    if not original_doc_text:
        return ""

    quote_text_from_llm = problem["quote_from_original_text"]
    focus_start = problem.get("sentence_char_start")
    focus_end = problem.get("sentence_char_end")

    if focus_start is None or focus_end is None:
        focus_start = problem.get("quote_char_start")
        focus_end = problem.get("quote_char_end")
        if focus_start is not None:
            logger.debug(
                "Context window for patch: Using quote offsets %s-%s as sentence offsets were not available for '%s...'.",
                focus_start,
                focus_end,
                quote_text_from_llm[:30],
            )
        elif (
            "N/A - General Issue" not in quote_text_from_llm
            and quote_text_from_llm.strip()
        ):
            offsets = await utils.find_quote_and_sentence_offsets_with_spacy(
                original_doc_text, quote_text_from_llm
            )
            if offsets:
                _, _, focus_start, focus_end = offsets

    if (
        "N/A - General Issue" in quote_text_from_llm
        or focus_start is None
        or focus_end is None
    ):
        if "N/A - General Issue" not in quote_text_from_llm:
            logger.warning(
                "Context window for patch: No valid offsets for quote '%s...'. Using general snippet logic.",
                quote_text_from_llm[:30],
            )

        if len(original_doc_text) <= window_size_chars:
            return original_doc_text
        start_snippet_len = min(window_size_chars // 2, len(original_doc_text))
        remaining_chars_for_end = window_size_chars - start_snippet_len
        end_snippet_len = min(
            remaining_chars_for_end, len(original_doc_text) - start_snippet_len
        )
        start_snippet = original_doc_text[:start_snippet_len]
        end_snippet = (
            original_doc_text[-end_snippet_len:] if end_snippet_len > 0 else ""
        )
        if start_snippet_len + end_snippet_len < len(original_doc_text):
            return f"{start_snippet}\n...\n{end_snippet}"
        return original_doc_text

    focus_len = focus_end - focus_start
    half_window_around_focus = (window_size_chars - focus_len) // 2

    context_start = max(0, focus_start - half_window_around_focus)
    context_end = min(len(original_doc_text), focus_end + half_window_around_focus)

    current_window_len = context_end - context_start
    if current_window_len < window_size_chars:
        if context_start == 0:
            context_end = min(len(original_doc_text), context_start + window_size_chars)
        elif context_end == len(original_doc_text):
            context_start = max(0, context_end - window_size_chars)

    prefix = "..." if context_start > 0 else ""
    suffix = "..." if context_end < len(original_doc_text) else ""
    snippet = original_doc_text[context_start:context_end]
    return f"{prefix}{snippet}{suffix}"

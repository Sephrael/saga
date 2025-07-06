"""Context utilities for patch generation."""

# processing/patch/context.py

import structlog

import utils
from models import ProblemDetail, SceneDetail

logger = structlog.get_logger(__name__)


def _get_formatted_scene_plan_from_agent_or_fallback(
    chapter_plan: list[SceneDetail],
    model_name_for_tokens: str,
    max_tokens_budget: int,
) -> str:
    """Formats a chapter plan into plain text for LLM prompts."""
    return utils.format_scene_plan_for_prompt(
        chapter_plan, model_name_for_tokens, max_tokens_budget
    )


async def _resolve_focus_offsets(
    problem: ProblemDetail, original_doc_text: str
) -> tuple[int | None, int | None]:
    """Return start and end offsets for the problem focus."""
    quote = problem.quote_from_original_text
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
                quote[:30],
            )
        elif "N/A - General Issue" not in quote and quote.strip():
            offsets = await utils.find_quote_and_sentence_offsets_with_spacy(
                original_doc_text, quote
            )
            if offsets:
                _, _, focus_start, focus_end = offsets
    return focus_start, focus_end


def _build_general_context_window(text: str, window_size: int) -> str:
    """Return a snippet when no focus offsets are available."""
    if len(text) <= window_size:
        return text
    start_len = min(window_size // 2, len(text))
    remaining = window_size - start_len
    end_len = min(remaining, len(text) - start_len)
    start_snippet = text[:start_len]
    end_snippet = text[-end_len:] if end_len > 0 else ""
    if start_len + end_len < len(text):
        return f"{start_snippet}\n...\n{end_snippet}"
    return text


def _build_focus_context_window(
    text: str, focus_start: int, focus_end: int, window_size: int
) -> str:
    """Return a snippet centered around the focus offsets."""
    focus_len = focus_end - focus_start
    half_window = (window_size - focus_len) // 2
    context_start = max(0, focus_start - half_window)
    context_end = min(len(text), focus_end + half_window)

    if context_end - context_start < window_size:
        if context_start == 0:
            context_end = min(len(text), context_start + window_size)
        elif context_end == len(text):
            context_start = max(0, context_end - window_size)

    prefix = "..." if context_start > 0 else ""
    suffix = "..." if context_end < len(text) else ""
    snippet = text[context_start:context_end]
    return f"{prefix}{snippet}{suffix}"


async def _get_context_window_for_patch_llm(
    original_doc_text: str, problem: ProblemDetail, window_size_chars: int
) -> str:
    """Return a context window around the problem quote."""
    if not original_doc_text:
        return ""

    quote_text_from_llm = problem.quote_from_original_text
    focus_start, focus_end = await _resolve_focus_offsets(problem, original_doc_text)

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
        return _build_general_context_window(original_doc_text, window_size_chars)

    return _build_focus_context_window(
        original_doc_text, focus_start, focus_end, window_size_chars
    )

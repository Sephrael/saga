# chapter_revision_logic.py
"""
Handles the revision of chapter drafts based on evaluation feedback for the SAGA system.
Supports both full rewrite and targeted patch-based revisions.
Context data for prompts is now formatted as plain text.
"""

import asyncio
import hashlib
from typing import Any

import structlog
import utils  # For numpy_cosine_similarity, find_semantically_closest_segment, AND find_quote_and_sentence_offsets_with_spacy, format_scene_plan_for_prompt
from agents.patch_validation_agent import PatchValidationAgent
from config import settings
from core.llm_interface import count_tokens, llm_service
from kg_maintainer.models import (
    CharacterProfile,
    EvaluationResult,
    PatchInstruction,
    ProblemDetail,
    SceneDetail,
    WorldItem,
)

logger = structlog.get_logger(__name__)
utils.load_spacy_model_if_needed()  # Ensure spaCy model is loaded when this module is imported


def _get_formatted_scene_plan_from_agent_or_fallback(
    chapter_plan: list[SceneDetail],
    model_name_for_tokens: str,
    max_tokens_budget: int,
) -> str:
    """Formats a chapter plan into plain text for LLM prompts using the central utility."""
    return utils.format_scene_plan_for_prompt(
        chapter_plan, model_name_for_tokens, max_tokens_budget
    )


def _get_plot_point_info(
    plot_outline: dict[str, Any], chapter_number: int
) -> tuple[str | None, int]:
    """
    Gets the plot point information for a given chapter number.
    This corrected version handles index out of bounds errors gracefully and
    can extract descriptions from dictionary-based plot points.
    """
    plot_points = plot_outline.get("plot_points", [])
    if not isinstance(plot_points, list) or not plot_points or chapter_number <= 0:
        logger.error(
            f"No plot points available or invalid chapter number ({chapter_number})."
        )
        return None, -1

    plot_point_index = chapter_number - 1

    if 0 <= plot_point_index < len(plot_points):
        plot_point_item = plot_points[plot_point_index]
        # Handle both simple string plot points and dictionary-based ones
        plot_point_text = (
            plot_point_item.get("description")
            if isinstance(plot_point_item, dict)
            else str(plot_point_item)
        )
        if isinstance(plot_point_text, str) and plot_point_text.strip():
            return plot_point_text, plot_point_index
        else:
            logger.warning(
                f"Plot point at index {plot_point_index} for chapter {chapter_number} is empty or invalid. Using placeholder."
            )
            return settings.FILL_IN, plot_point_index
    else:
        logger.error(
            f"Plot point index {plot_point_index} is out of bounds for plot_points list (len: {len(plot_points)}) for chapter {chapter_number}."
        )
        return None, -1


async def _get_context_window_for_patch_llm(
    original_doc_text: str, problem: ProblemDetail, window_size_chars: int
) -> str:
    """
    Gets a context window around the problem's quote using precise offsets if available.
    The window is centered around the *sentence* containing the quote.
    """
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
                f"Context window for patch: Using quote offsets {focus_start}-{focus_end} as sentence offsets were not available for '{quote_text_from_llm[:30]}...'."
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
                f"Context window for patch: No valid offsets for quote '{quote_text_from_llm[:30]}...'. Using general snippet logic."
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
        else:
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


_sentence_embedding_cache: dict[str, list[tuple[int, int, Any]]] = {}


async def _get_sentence_embeddings(
    text: str, cache: dict[str, list[tuple[int, int, Any]]] | None | None = None
) -> list[tuple[int, int, Any]]:
    """Return a list of (start, end, embedding) for each sentence."""
    if cache is None:
        cache = _sentence_embedding_cache
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_hash in cache:
        return cache[text_hash]

    segments = utils.get_text_segments(text, "sentence")
    if not segments:
        return []
    tasks = [llm_service.async_get_embedding(seg[0]) for seg in segments]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    embeddings: list[tuple[int, int, Any]] = []
    for (_seg_text, start, end), res in zip(segments, results, strict=False):
        if isinstance(res, Exception) or res is None:
            continue
        embeddings.append((start, end, res))
    cache[text_hash] = embeddings
    return embeddings


async def _find_sentence_via_embeddings(
    quote_text: str, embeddings: list[tuple[int, int, Any]]
) -> tuple[int, int] | None:
    if not embeddings or not quote_text.strip():
        return None
    q_emb = await llm_service.async_get_embedding(quote_text)
    if q_emb is None:
        return None
    best_sim = -1.0
    best_span: tuple[int, int] | None = None
    for start, end, emb in embeddings:
        sim = utils.numpy_cosine_similarity(q_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_span = (start, end)
    return best_span


async def _generate_single_patch_instruction_llm(
    plot_outline: dict[str, Any],
    original_chapter_text_snippet_for_llm: str,
    problem: ProblemDetail,
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: list[SceneDetail] | None,
) -> tuple[PatchInstruction | None, dict[str, int] | None]:
    """
    Generates a single patch instruction. The PatchInstruction will store target_char_start/end
    referring to the SENTENCE containing the problem quote if available.
    """
    plan_focus_section_parts: list[str] = []
    plot_point_focus, _ = _get_plot_point_info(plot_outline, chapter_number)
    max_plan_tokens_for_patch_prompt = settings.MAX_CONTEXT_TOKENS // 2

    if settings.ENABLE_AGENTIC_PLANNING and chapter_plan:
        formatted_plan = _get_formatted_scene_plan_from_agent_or_fallback(
            chapter_plan,
            settings.PATCH_GENERATION_MODEL,
            max_plan_tokens_for_patch_prompt,
        )
        plan_focus_section_parts.append(formatted_plan)
        if "plan truncated" in formatted_plan:
            logger.warning(
                f"Scene plan token-truncated for Ch {chapter_number} patch generation prompt."
            )
    else:
        plan_focus_section_parts.append(
            f"**Original Chapter Focus (Reference for overall chapter direction):**\n{plot_point_focus or 'Not specified.'}\n"
        )
    plan_focus_section_str = "".join(plan_focus_section_parts)

    is_general_expansion_task = False
    length_expansion_instruction_header_parts: list[str] = []
    original_quote_text_from_problem = problem["quote_from_original_text"]

    if problem["issue_category"] == "narrative_depth_and_length" and (
        "short" in problem["problem_description"].lower()
        or "length" in problem["problem_description"].lower()
        or "expand" in problem["suggested_fix_focus"].lower()
        or "depth" in problem["problem_description"].lower()
        or original_quote_text_from_problem == "N/A - General Issue"
    ):
        length_expansion_instruction_header_parts.append(
            "\n**Critical: SUBSTANTIAL EXPANSION REQUIRED FOR THIS SEGMENT/PASSAGE.** "
        )
        length_expansion_instruction_header_parts.append(
            "The 'replace_with' text MUST be significantly longer and more detailed. "
        )
        length_expansion_instruction_header_parts.append(
            "Add descriptive details, character thoughts, dialogue, actions, and sensory information. "
        )
        if original_quote_text_from_problem == "N/A - General Issue":
            is_general_expansion_task = True
            length_expansion_instruction_header_parts.append(
                "Since the original quote is 'N/A - General Issue', your 'replace_with' text should be a **new, expanded passage** "
                "that addresses the 'Problem Description' and 'Suggested Fix Focus' within the broader 'Text Snippet' context. "
                "This generated text is intended as a candidate for insertion or to inform a broader rewrite of a section."
            )
        else:
            length_expansion_instruction_header_parts.append(
                "Aim for a notable increase in length and detail for the conceptual segment related to the original quote."
            )
    length_expansion_instruction_header_str = "".join(
        length_expansion_instruction_header_parts
    )

    prompt_instruction_for_replacement_scope_parts: list[str] = []
    max_patch_output_tokens = 0

    if is_general_expansion_task:
        prompt_instruction_for_replacement_scope_parts.append(
            "    - The 'Original Quote Illustrating Problem' is \"N/A - General Issue\". Therefore, your `replace_with` text should be a **new, self-contained, and substantially expanded passage** "
            'that addresses the "Problem Description" and "Suggested Fix Focus" as guided by the `length_expansion_instruction_header_str`. '
            "This new passage is intended for potential insertion into the chapter, not to replace a specific quote."
        )
        max_patch_output_tokens = settings.MAX_GENERATION_TOKENS // 2
        max_patch_output_tokens = max(max_patch_output_tokens, 750)
        logger.info(
            f"Patch (Ch {chapter_number}, general expansion): Max output tokens set to {max_patch_output_tokens}."
        )
    else:
        prompt_instruction_for_replacement_scope_parts.append(
            "    - The 'Original Quote Illustrating Problem' is specific. Your `replace_with` text should be a revised version "
            "of the **entire conceptual sentence or short paragraph** within the 'ORIGINAL TEXT SNIPPET' that best corresponds to that quote. Your output will replace that whole segment.\n"
            "    - **Crucially, for this specific fix, your replacement text should primarily focus on correcting the identified issue. "
        )
        if length_expansion_instruction_header_str:
            prompt_instruction_for_replacement_scope_parts.append(
                "If `length_expansion_instruction_header_str` is present, apply its guidance to *this specific segment*. "
            )
        prompt_instruction_for_replacement_scope_parts.append(
            "Otherwise, aim for a length comparable to the original segment, plus necessary additions for the fix. "
            "Avoid excessive unrelated expansion beyond the scope of the problem for this segment.**"
        )
        original_snippet_tokens = count_tokens(
            original_chapter_text_snippet_for_llm,
            settings.PATCH_GENERATION_MODEL,
        )
        expansion_factor = 2.5 if length_expansion_instruction_header_str else 1.5
        max_patch_output_tokens = int(original_snippet_tokens * expansion_factor)
        max_patch_output_tokens = min(
            max_patch_output_tokens, settings.MAX_GENERATION_TOKENS // 2
        )
        max_patch_output_tokens = max(max_patch_output_tokens, 200)
        logger.info(
            f"Patch (Ch {chapter_number}, specific fix): Original snippet tokens: {original_snippet_tokens}. Max output tokens set to {max_patch_output_tokens}."
        )
    prompt_instruction_for_replacement_scope_str = "".join(
        prompt_instruction_for_replacement_scope_parts
    )

    protagonist_name = plot_outline.get(
        "protagonist_name", settings.DEFAULT_PROTAGONIST_NAME
    )

    few_shot_patch_example_str = """
--- Example of how to provide `replace_with` text (content is illustrative only) ---
IF THE PROBLEM WAS:
  - Issue Category: narrative_depth
  - Problem Description: The reaction of Elara to seeing the ghost felt understated.
  - Original Quote Illustrating Problem: "Elara saw the ghost and gasped."
  - Suggested Fix Focus: Expand on Elara's internal emotional reaction and physical response.
  A suitable `replace_with` text might look like this (just the text, no extra explanation):
A chill traced Elara's spine, not from the crypt's cold, but from the translucent figure coalescing before her. Her breath hitched, a silent scream trapped in her throat as the ghostly visage turned its empty sockets towards her. Every instinct screamed to flee, but her feet felt rooted to the stone floor, a terrifying paralysis gripping her.
--- End of Example ---
"""

    prompt_lines = []
    if settings.ENABLE_LLM_NO_THINK_DIRECTIVE:
        prompt_lines.append("/no_think")

    prompt_lines.extend(
        [
            f'You are a surgical revision expert generating replacement text for Chapter {chapter_number} of a novel titled "{plot_outline.get("title", "Untitled Novel")}" about {protagonist_name}.',
            "**Novel Context:**",
            f"  - Genre: {plot_outline.get('genre', 'N/A')}",
            f"  - Theme: {plot_outline.get('theme', 'N/A')}",
            f"  - Protagonist: {protagonist_name} ({plot_outline.get('character_arc', 'N/A')})",
            "",
            plan_focus_section_str,
            "**Hybrid Context from Previous Chapters (for consistency with established canon and narrative flow):**",
            "--- BEGIN HYBRID CONTEXT ---",
            hybrid_context_for_revision
            if hybrid_context_for_revision.strip()
            else "No previous context.",
            "--- END HYBRID CONTEXT ---",
            "",
            "**Specific Problem to Address in the Chapter:**",
            f"  - Issue Category: {problem['issue_category']}",
            f"  - Problem Description: {problem['problem_description']}",
            f'  - Original Quote Illustrating Problem: "{original_quote_text_from_problem}"',
            f"  - Suggested Fix Focus: {problem['suggested_fix_focus']}",
            "",
            "**Text Snippet from Original Chapter (This is the broader context around the problem. If the quote is 'N/A - General Issue', this is general chapter context to inform your new passage):**",
            "--- BEGIN ORIGINAL TEXT SNIPPET ---",
            original_chapter_text_snippet_for_llm,
            "--- END ORIGINAL TEXT SNIPPET ---",
            length_expansion_instruction_header_str,
            "```plaintext",
            few_shot_patch_example_str.strip(),
            "```",
            "**Instructions for Generating Replacement Text:**",
            "1.  Focus EXCLUSIVELY on the problem described, particularly relating to the conceptual area highlighted by: `{original_quote_text_from_problem}` within the 'ORIGINAL TEXT SNIPPET'.",
            "2.  Generate a `replace_with` text according to the following:",
            prompt_instruction_for_replacement_scope_str,
            '3.  The `replace_with` text MUST address the "Problem Description" and "Suggested Fix Focus".',
            "   Follow the 'Suggested Fix Focus' EXACTLY, e.g., 'Rewrite this paragraph so the protagonist thinks before acting, but without using any verbs that imply a physical body.'",
            # MODIFICATION START: Added instruction for deletion via empty string.
            "4.  If the best way to fix the problem is to **completely remove** the 'Original Quote' segment (e.g., it is redundant or unnecessary), then you **MUST output an empty string**. Do not write a justification; simply provide no text as the `replace_with` output.",
            # MODIFICATION END
            "5.  Maintain the novel's style, tone, and consistency with all provided context (Novel Context, Plan, Hybrid Context).",
            "6.  Convey thematic elements through subtext, character actions, and metaphorical imagery rather than direct exposition or deus ex machina fixes.",
            "7.  If `length_expansion_instruction_header_str` is present, ensure substantial expansion as guided for the targeted segment or new passage.",
            '8.  **Output ONLY the `replace_with` text.** Do NOT include JSON, markdown, explanations, or any "Replace with:" prefixes. Just the raw text intended for replacement/insertion. (See example above for how to format the text).',
            "",
            f'--- BEGIN REPLACE_WITH TEXT (for the segment related to "{original_quote_text_from_problem}" or as a new passage if quote is "N/A - General Issue") ---',
        ]
    )
    prompt = "\n".join(prompt_lines)

    logger.info(
        f"Calling LLM ({settings.PATCH_GENERATION_MODEL}) for patch in Ch {chapter_number}. Problem: '{problem['problem_description'][:60].replace(chr(10), ' ')}...' Quote Text: '{original_quote_text_from_problem[:50].replace(chr(10), ' ')}...' Max Output Tokens: {max_patch_output_tokens}"
    )

    (
        replace_with_text_cleaned,
        usage_data,
    ) = await llm_service.async_call_llm(
        model_name=settings.PATCH_GENERATION_MODEL,
        prompt=prompt,
        temperature=settings.TEMPERATURE_PATCH,
        max_tokens=max_patch_output_tokens,
        allow_fallback=True,
        stream_to_disk=False,
        frequency_penalty=settings.FREQUENCY_PENALTY_PATCH,
        presence_penalty=settings.PRESENCE_PENALTY_PATCH,
        auto_clean_response=True,
    )

    # MODIFICATION: No longer check if the cleaned text is empty here, as an empty string is now a valid "deletion" instruction.
    # The check for a failed LLM call (returning None) is implicitly handled by the structure below.
    if replace_with_text_cleaned is None:
        logger.error(
            f"Patch LLM call failed and returned None for Ch {chapter_number} problem: {problem['problem_description']}"
        )
        return None, usage_data

    # Log if a deletion is being suggested
    if not replace_with_text_cleaned.strip():
        logger.info(
            f"Patch LLM suggested DELETION (empty output) for Ch {chapter_number} problem: {problem['problem_description']}"
        )

    if length_expansion_instruction_header_str:
        if not is_general_expansion_task:
            if (
                len(original_chapter_text_snippet_for_llm) > 100
                and len(replace_with_text_cleaned)
                < len(original_chapter_text_snippet_for_llm) * 1.2
            ):
                logger.warning(
                    f"Patch for Ch {chapter_number} (specific quote, segment expansion requested) output length ({len(replace_with_text_cleaned)}) "
                    f"is not significantly larger than context snippet ({len(original_chapter_text_snippet_for_llm)}). "
                    f"Problem: {problem['problem_description'][:60]}"
                )
        elif is_general_expansion_task and len(replace_with_text_cleaned) < 500:
            logger.warning(
                f"Patch for Ch {chapter_number} ('N/A - General Issue' expansion) produced a relatively short new passage (len: {len(replace_with_text_cleaned)}). "
                f"Problem: {problem['problem_description'][:60]}"
            )

    target_start_for_patch: int | None = problem.get("sentence_char_start")
    target_end_for_patch: int | None = problem.get("sentence_char_end")

    if (
        original_quote_text_from_problem != "N/A - General Issue"
        and (target_start_for_patch is None or target_end_for_patch is None)
        and (
            problem.get("quote_char_start") is not None
            and problem.get("quote_char_end") is not None
        )
    ):
        logger.warning(
            f"Patch for Ch {chapter_number}: Problem '{original_quote_text_from_problem[:50]}' had specific text but no sentence offsets. "
            f"PatchInstruction will use quote offsets ({problem.get('quote_char_start')}-{problem.get('quote_char_end')}). Application will use semantic search."
        )
        target_start_for_patch = problem.get("quote_char_start")
        target_end_for_patch = problem.get("quote_char_end")
    elif original_quote_text_from_problem != "N/A - General Issue" and (
        target_start_for_patch is None or target_end_for_patch is None
    ):
        logger.error(
            f"Patch for Ch {chapter_number}: Problem '{original_quote_text_from_problem[:50]}' specific text but NO OFFSETS (sentence or quote). Patch will likely fail to apply precisely."
        )

    patch_instruction: PatchInstruction = {
        "original_problem_quote_text": original_quote_text_from_problem,
        "target_char_start": target_start_for_patch,
        "target_char_end": target_end_for_patch,
        "replace_with": replace_with_text_cleaned,  # This can now be ""
        "reason_for_change": f"Fixing '{problem['issue_category']}': {problem['problem_description']}",
    }
    return patch_instruction, usage_data


def _consolidate_overlapping_problems(
    problems: list[ProblemDetail],
) -> list[ProblemDetail]:
    """
    Groups problems by their overlapping text spans and consolidates them.
    This prevents generating multiple patches for the same or overlapping sentences.
    """
    if not problems:
        return []

    # Separate problems that have a specific sentence span from those that are general.
    span_problems = [
        p
        for p in problems
        if p.get("sentence_char_start") is not None
        and p.get("sentence_char_end") is not None
    ]
    general_problems = [
        p
        for p in problems
        if p.get("sentence_char_start") is None or p.get("sentence_char_end") is None
    ]

    if not span_problems:
        return general_problems

    # Sort problems by their start offset to enable linear merging
    span_problems.sort(key=lambda p: p["sentence_char_start"])  # type: ignore

    merged_groups: list[list[ProblemDetail]] = []
    if span_problems:
        current_group = [span_problems[0]]
        current_group_end = span_problems[0]["sentence_char_end"]

        for i in range(1, len(span_problems)):
            next_problem = span_problems[i]
            next_start = next_problem["sentence_char_start"]
            next_end = next_problem["sentence_char_end"]

            # If the next problem starts before the current group's span ends, it overlaps.
            if next_start < current_group_end:  # type: ignore
                current_group.append(next_problem)
                current_group_end = max(current_group_end, next_end)  # type: ignore
            else:
                # The next problem does not overlap, so finalize the current group and start a new one.
                merged_groups.append(current_group)
                current_group = [next_problem]
                current_group_end = next_end

        merged_groups.append(current_group)  # Add the last group

    consolidated_problems: list[ProblemDetail] = []
    for group in merged_groups:
        if len(group) == 1:
            consolidated_problems.append(group[0])
            continue

        # Consolidate the group into a single new ProblemDetail
        first_problem = group[0]
        # Calculate the union of all spans in the group
        group_start_offset = min(p["sentence_char_start"] for p in group)  # type: ignore
        group_end_offset = max(p["sentence_char_end"] for p in group)  # type: ignore

        # Combine all details from the problems in the group
        all_categories = sorted(list(set(p["issue_category"] for p in group)))
        all_descriptions = "; ".join(
            f"({p['issue_category']}) {p['problem_description']}" for p in group
        )
        all_fix_foci = "; ".join(
            f"({p['issue_category']}) {p['suggested_fix_focus']}" for p in group
        )
        # Use the quote from the first problem in the span as a representative
        representative_quote = first_problem["quote_from_original_text"]

        consolidated_problem: ProblemDetail = {
            "issue_category": ", ".join(all_categories),
            "problem_description": f"Multiple issues in one segment: {all_descriptions}",
            "quote_from_original_text": representative_quote,
            "quote_char_start": first_problem.get("quote_char_start"),
            "quote_char_end": first_problem.get("quote_char_end"),
            "sentence_char_start": group_start_offset,
            "sentence_char_end": group_end_offset,
            "suggested_fix_focus": f"Holistically revise the segment to address all points: {all_fix_foci}",
        }
        consolidated_problems.append(consolidated_problem)
        logger.info(
            f"Consolidated {len(group)} overlapping problems into one targeting span {group_start_offset}-{group_end_offset}."
        )

    consolidated_problems.extend(general_problems)
    return consolidated_problems


def _deduplicate_problems(problems: list[ProblemDetail]) -> list[ProblemDetail]:
    """Remove exact duplicates based on span and quote text."""
    unique: list[ProblemDetail] = []
    seen: set[tuple[int | None, int | None, str]] = set()
    for prob in problems:
        key = (
            prob.get("sentence_char_start"),
            prob.get("sentence_char_end"),
            prob.get("quote_from_original_text", ""),
        )
        if key in seen:
            logger.info(
                "Deduplicating problem at span %s-%s with quote '%s'.",
                prob.get("sentence_char_start"),
                prob.get("sentence_char_end"),
                prob.get("quote_from_original_text", "")[:30],
            )
            continue
        seen.add(key)
        unique.append(prob)
    return unique


def _group_problems_for_patch_generation(
    problems: list[ProblemDetail],
) -> list[tuple[ProblemDetail, list[ProblemDetail]]]:
    """Return consolidated problem with list of original problems."""
    if not problems:
        return []

    span_problems = [
        p
        for p in problems
        if p.get("sentence_char_start") is not None
        and p.get("sentence_char_end") is not None
    ]
    general_problems = [
        p
        for p in problems
        if p.get("sentence_char_start") is None or p.get("sentence_char_end") is None
    ]

    span_problems.sort(key=lambda p: p["sentence_char_start"])
    merged_groups: list[list[ProblemDetail]] = []
    if span_problems:
        current_group = [span_problems[0]]
        current_end = span_problems[0]["sentence_char_end"]
        for prob in span_problems[1:]:
            start = prob["sentence_char_start"]
            end = prob["sentence_char_end"]
            if start < current_end:
                current_group.append(prob)
                current_end = max(current_end, end)
            else:
                merged_groups.append(current_group)
                current_group = [prob]
                current_end = end
        merged_groups.append(current_group)

    result: list[tuple[ProblemDetail, list[ProblemDetail]]] = []

    for group in merged_groups:
        first = group[0]
        group_start = min(p["sentence_char_start"] for p in group)  # type: ignore
        group_end = max(p["sentence_char_end"] for p in group)  # type: ignore
        all_cats = sorted(list(set(p["issue_category"] for p in group)))
        all_desc = "; ".join(
            f"({p['issue_category']}) {p['problem_description']}" for p in group
        )
        all_fix = "; ".join(
            f"({p['issue_category']}) {p['suggested_fix_focus']}" for p in group
        )
        rep_quote = first["quote_from_original_text"]
        consolidated: ProblemDetail = {
            "issue_category": ", ".join(all_cats),
            "problem_description": f"Multiple issues in one segment: {all_desc}",
            "quote_from_original_text": rep_quote,
            "quote_char_start": first.get("quote_char_start"),
            "quote_char_end": first.get("quote_char_end"),
            "sentence_char_start": group_start,
            "sentence_char_end": group_end,
            "suggested_fix_focus": f"Holistically revise the segment to address all points: {all_fix}",
        }
        result.append((consolidated, group))

    for p in general_problems:
        result.append((p, [p]))

    return result


async def _generate_patch_instructions_logic(
    plot_outline: dict[str, Any],
    original_text: str,
    problems_to_fix: list[ProblemDetail],
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: list[SceneDetail] | None,
    validator: PatchValidationAgent,
) -> tuple[list[PatchInstruction], dict[str, int] | None]:
    patch_instructions: list[PatchInstruction] = []
    total_usage: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    grouped = _group_problems_for_patch_generation(problems_to_fix)

    groups_to_process = grouped[: settings.MAX_PATCH_INSTRUCTIONS_TO_GENERATE]
    if len(grouped) > len(groups_to_process):
        logger.warning(
            f"Found {len(grouped)} patch groups for Ch {chapter_number}. "
            f"Processing only the first {len(groups_to_process)} groups."
        )
    if not groups_to_process:
        return [], None

    async def _process_group(
        group_idx: int, group_problem: ProblemDetail, group_members: list[ProblemDetail]
    ) -> tuple[PatchInstruction | None, dict[str, int]]:
        context_snippet = await _get_context_window_for_patch_llm(
            original_text,
            group_problem,
            settings.MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW,
        )

        patch_instr: PatchInstruction | None = None
        usage_acc: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for _ in range(settings.PATCH_GENERATION_ATTEMPTS):
            patch_instr_tmp, usage = await _generate_single_patch_instruction_llm(
                plot_outline,
                context_snippet,
                group_problem,
                chapter_number,
                hybrid_context_for_revision,
                chapter_plan,
            )
            if usage:
                for k, v in usage.items():
                    usage_acc[k] += v
            if not patch_instr_tmp:
                continue
            if not settings.AGENT_ENABLE_PATCH_VALIDATION:
                patch_instr = patch_instr_tmp
                break
            valid, val_usage = await validator.validate_patch(
                context_snippet, patch_instr_tmp, group_members
            )
            if val_usage:
                for k, v in val_usage.items():
                    usage_acc[k] += v
            if valid:
                patch_instr = patch_instr_tmp
                break

        if not patch_instr:
            logger.warning(
                f"Failed to generate valid patch for group {group_idx} in Ch {chapter_number}."
            )
        return patch_instr, usage_acc

    tasks = [
        _process_group(idx, gp, gm)
        for idx, (gp, gm) in enumerate(groups_to_process, start=1)
    ]

    results = await asyncio.gather(*tasks)
    for patch_instr, usage_acc in results:
        if patch_instr:
            patch_instructions.append(patch_instr)
        for k, v in usage_acc.items():
            total_usage[k] += v

    logger.info(
        f"Generated {len(patch_instructions)} patch instructions for Ch {chapter_number}."
    )
    return patch_instructions, total_usage if total_usage["total_tokens"] > 0 else None


async def _apply_patches_to_text(
    original_text: str,
    patch_instructions: list[PatchInstruction],
    already_patched_spans: list[tuple[int, int]] | None | None = None,
    sentence_embeddings: list[tuple[int, int, Any]] | None | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """
    Applies patch instructions to the original text and returns the new text and a
    comprehensive, re-mapped list of all patched spans (old and new).
    """
    if already_patched_spans is None:
        already_patched_spans = []

    if not patch_instructions:
        return original_text, already_patched_spans

    # 1. Prepare new replacements, filtering out overlaps with existing patched spans.
    replacements: list[tuple[int, int, str]] = []
    for patch_idx, patch in enumerate(patch_instructions):
        # MODIFICATION START: Handle empty replace_with as a valid deletion instruction.
        # An empty or whitespace-only replace_with string is now a valid patch.
        replacement_text = patch.get("replace_with", "")
        if replacement_text is None:  # handle case where key is missing
            replacement_text = ""
        # MODIFICATION END

        segment_start: int | None = patch.get("target_char_start")
        segment_end: int | None = patch.get("target_char_end")
        method_used = "direct offsets"

        if segment_start is None or segment_end is None:
            quote_text = patch["original_problem_quote_text"]
            if quote_text != "N/A - General Issue" and quote_text.strip():
                logger.info(
                    f"Patch {patch_idx + 1}: Missing direct offsets for '{quote_text[:50]}...'. Using semantic search."
                )
                method_used = "semantic search"
                if sentence_embeddings:
                    found = await _find_sentence_via_embeddings(
                        quote_text, sentence_embeddings
                    )
                    if found:
                        segment_start, segment_end = found
                if segment_start is None or segment_end is None:
                    match = await utils.find_semantically_closest_segment(
                        original_text, quote_text, "sentence"
                    )
                    if match:
                        segment_start, segment_end, _ = match
            else:
                logger.warning(
                    f"Patch {patch_idx + 1}: Cannot apply, no quote text for search and no offsets."
                )
                continue

        if segment_start is None or segment_end is None:
            logger.warning(
                f"Patch {patch_idx + 1}: Failed to find target segment via {method_used}."
            )
            continue

        # Check for overlaps with already patched spans and other new patches in this batch
        is_overlapping = any(
            max(segment_start, old_start) < min(segment_end, old_end)
            for old_start, old_end in already_patched_spans
        ) or any(
            max(segment_start, r_start) < min(segment_end, r_end)
            for r_start, r_end, _ in replacements
        )

        if is_overlapping:
            logger.warning(
                f"Patch {patch_idx + 1} for segment {segment_start}-{segment_end} overlaps with a previously patched area or another new patch. Skipping."
            )
            continue

        original_segment = original_text[segment_start:segment_end]
        if replacement_text.strip() == original_segment.strip():
            logger.info(
                f"Patch {patch_idx + 1}: replacement identical to original segment {segment_start}-{segment_end}. Skipping."
            )
            continue
        if replacement_text.strip() and original_segment.strip():
            orig_emb, repl_emb = await asyncio.gather(
                llm_service.async_get_embedding(original_segment),
                llm_service.async_get_embedding(replacement_text),
            )
            if (
                orig_emb is not None
                and repl_emb is not None
                and utils.numpy_cosine_similarity(orig_emb, repl_emb)
                >= settings.REVISION_SIMILARITY_ACCEPTANCE
            ):
                logger.info(
                    f"Patch {patch_idx + 1}: replacement highly similar to original segment {segment_start}-{segment_end}. Skipping."
                )
                continue

        replacements.append((segment_start, segment_end, replacement_text))
        log_action = "DELETION" if not replacement_text.strip() else "REPLACEMENT"
        logger.info(
            f"Patch {patch_idx + 1}: Queued {log_action} for {segment_start}-{segment_end} via {method_used}."
        )

    if not replacements:
        logger.info("No non-overlapping patches to apply in this cycle.")
        return original_text, already_patched_spans

    # 2. Build the new text and remap all spans in a single pass.
    # Create a unified list of all operations (old spans to copy, new spans to insert).
    all_ops: list[dict[str, Any]] = []
    for start, end in already_patched_spans:
        all_ops.append(
            {
                "type": "old",
                "start": start,
                "end": end,
                "text": original_text[start:end],
            }
        )
    for start, end, text in replacements:
        all_ops.append({"type": "new", "start": start, "end": end, "text": text})

    all_ops.sort(key=lambda x: x["start"])

    result_parts = []
    all_spans_in_new_text = []
    last_original_end = 0

    for op in all_ops:
        # Copy the text from the end of the last operation to the start of this one
        result_parts.append(original_text[last_original_end : op["start"]])

        # Calculate the starting position of the new span in the constructed text
        new_span_start = len("".join(result_parts))

        # Append the operation's text (either old text or new replacement text)
        result_parts.append(op["text"])

        # Calculate the end position and add the span to our list
        new_span_end = len("".join(result_parts))

        # MODIFICATION: Only add a protected span if the replacement was not a deletion.
        # A deleted segment should not be protected from future patches.
        if new_span_end > new_span_start:
            all_spans_in_new_text.append((new_span_start, new_span_end))

        # Update the pointer for the next iteration
        last_original_end = op["end"]

    # Append any remaining text after the last operation
    result_parts.append(original_text[last_original_end:])

    patched_text = "".join(result_parts)
    final_spans = sorted(all_spans_in_new_text)

    num_deletions = sum(1 for _, _, txt in replacements if not txt.strip())
    num_replacements = len(replacements) - num_deletions
    logger.info(
        f"Applied {num_replacements} replacements and {num_deletions} deletions. Total protected spans in new text: {len(final_spans)}."
    )

    return patched_text, final_spans


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

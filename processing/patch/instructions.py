# processing/patch/instructions.py
"""Patch instruction generation utilities."""

import asyncio
from typing import Any

import structlog
from agents.patch_validation_agent import PatchValidationAgent
from config import settings
from core.llm_interface import count_tokens, llm_service
from core.usage import TokenUsage
from prompt_renderer import render_prompt

from models import PatchInstruction, ProblemDetail, SceneDetail
from utils.plot import get_plot_point_info

from .context import (
    _get_context_window_for_patch_llm,
    _get_formatted_scene_plan_from_agent_or_fallback,
)

logger = structlog.get_logger(__name__)


def _build_plan_section(
    chapter_plan: list[SceneDetail] | None,
    plot_point_focus: str | None,
    chapter_number: int,
) -> str:
    """Return plan focus section for the patch prompt."""
    parts: list[str] = []
    max_plan_tokens = settings.MAX_CONTEXT_TOKENS // 2
    if settings.ENABLE_AGENTIC_PLANNING and chapter_plan:
        formatted = _get_formatted_scene_plan_from_agent_or_fallback(
            chapter_plan,
            settings.PATCH_GENERATION_MODEL,
            max_plan_tokens,
        )
        parts.append(formatted)
        if "plan truncated" in formatted:
            logger.warning(
                "Scene plan token-truncated for Ch %s patch generation prompt.",
                chapter_number,
            )
    else:
        parts.append(
            f"**Original Chapter Focus (Reference for overall chapter direction):**\n{plot_point_focus or 'Not specified.'}\n"
        )
    return "".join(parts)


def _build_length_instructions(
    problem: ProblemDetail,
    original_quote_text: str,
    chapter_number: int,
    original_snippet_tokens: int,
) -> tuple[bool, str, str, int]:
    """Return length expansion flags and instructions."""
    header_parts: list[str] = []
    is_general = False
    if problem.issue_category == "narrative_depth_and_length" and (
        "short" in problem.problem_description.lower()
        or "length" in problem.problem_description.lower()
        or "expand" in problem.suggested_fix_focus.lower()
        or "depth" in problem.problem_description.lower()
        or original_quote_text == "N/A - General Issue"
    ):
        header_parts.append(
            "\n**Critical: SUBSTANTIAL EXPANSION REQUIRED FOR THIS SEGMENT/PASSAGE.** "
        )
        header_parts.append(
            "The 'replace_with' text MUST be significantly longer and more detailed. "
        )
        header_parts.append(
            "Add descriptive details, character thoughts, dialogue, actions, and sensory information. "
        )
        if original_quote_text == "N/A - General Issue":
            is_general = True
            header_parts.append(
                "Since the original quote is 'N/A - General Issue', your 'replace_with' text should be a **new, expanded passage** "
                "that addresses the 'Problem Description' and 'Suggested Fix Focus' within the broader 'Text Snippet' context. "
                "This generated text is intended as a candidate for insertion or to inform a broader rewrite of a section."
            )
        else:
            header_parts.append(
                "Aim for a notable increase in length and detail for the conceptual segment related to the original quote."
            )
    header_str = "".join(header_parts)

    scope_parts: list[str] = []
    max_tokens = 0
    if is_general:
        scope_parts.append(
            "    - The 'Original Quote Illustrating Problem' is \"N/A - General Issue\". Therefore, your `replace_with` text should be a **new, self-contained, and substantially expanded passage** "
            'that addresses the "Problem Description" and "Suggested Fix Focus" as guided by the `length_expansion_instruction_header_str`. '
            "This new passage is intended for potential insertion into the chapter, not to replace a specific quote."
        )
        max_tokens = max(settings.MAX_GENERATION_TOKENS // 2, 750)
        logger.info(
            "Patch (Ch %s, general expansion): Max output tokens set to %s.",
            chapter_number,
            max_tokens,
        )
    else:
        scope_parts.append(
            "    - The 'Original Quote Illustrating Problem' is specific. Your `replace_with` text should be a revised version "
            "of the **entire conceptual sentence or short paragraph** within the 'ORIGINAL TEXT SNIPPET' that best corresponds to that quote. Your output will replace that whole segment.\n"
            "    - **Crucially, for this specific fix, your replacement text should primarily focus on correcting the identified issue. "
        )
        if header_str:
            scope_parts.append(
                "If `length_expansion_instruction_header_str` is present, apply its guidance to *this specific segment*. "
            )
        scope_parts.append(
            "Otherwise, aim for a length comparable to the original segment, plus necessary additions for the fix. "
            "Avoid excessive unrelated expansion beyond the scope of the problem for this segment.**"
        )
        expansion_factor = 2.5 if header_str else 1.5
        max_tokens = int(original_snippet_tokens * expansion_factor)
        max_tokens = min(max_tokens, settings.MAX_GENERATION_TOKENS // 2)
        max_tokens = max(max_tokens, 200)
        logger.info(
            "Patch (Ch %s, specific fix): Original snippet tokens: %s. Max output tokens set to %s.",
            chapter_number,
            original_snippet_tokens,
            max_tokens,
        )
    return is_general, header_str, "".join(scope_parts), max_tokens


async def _generate_single_patch_instruction_llm(
    plot_outline: dict[str, Any],
    original_chapter_text_snippet_for_llm: str,
    problem: ProblemDetail,
    rewrite_instruction: str | None,
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: list[SceneDetail] | None,
    plot_point_focus: str | None,
    validation_failure_reason: str | None = None,
) -> tuple[PatchInstruction | None, dict[str, int] | None]:
    """Generate a single patch instruction using the LLM.

    Args:
        plot_outline: Dictionary containing overall plot information.
        original_chapter_text_snippet_for_llm: Context window from the original
            chapter.
        problem: Structured problem detail from the evaluator.
        rewrite_instruction: Additional evaluator guidance for the rewrite.
        chapter_number: Current chapter being revised.
        hybrid_context_for_revision: Hybrid context string for continuity.
        chapter_plan: Optional scene plan for the chapter.
        plot_point_focus: Focus of the current plot point for fallback context.
        validation_failure_reason: Feedback describing why the previous patch
            attempt failed validation.

    Returns:
        A ``PatchInstruction`` with replacement text and optional token usage
        statistics, or ``None`` if generation failed.
    """
    plan_focus_section_str = _build_plan_section(
        chapter_plan, plot_point_focus, chapter_number
    )

    original_quote_text_from_problem = problem.quote_from_original_text
    original_snippet_tokens = count_tokens(
        original_chapter_text_snippet_for_llm,
        settings.PATCH_GENERATION_MODEL,
    )
    (
        is_general_expansion_task,
        length_expansion_instruction_header_str,
        prompt_instruction_for_replacement_scope_str,
        max_patch_output_tokens,
    ) = _build_length_instructions(
        problem,
        original_quote_text_from_problem,
        chapter_number,
        original_snippet_tokens,
    )

    protagonist_name = plot_outline.get(
        "protagonist_name", settings.DEFAULT_PROTAGONIST_NAME
    )

    few_shot_patch_example_str = """--- Example of how to provide `replace_with` text (content is illustrative only) ---
IF THE PROBLEM WAS:
  - Issue Category: narrative_depth
  - Problem Description: The reaction of Elara to seeing the ghost felt understated.
  - Original Quote Illustrating Problem: "Elara saw the ghost and gasped."
  - Suggested Fix Focus: Expand on Elara's internal emotional reaction and physical response.
  A suitable `replace_with` text might look like this (just the text, no extra explanation):
A chill traced Elara's spine, not from the crypt's cold, but from the translucent figure coalescing before her. Her breath hitched, a silent scream trapped in her throat as the ghostly visage turned its empty sockets towards her. Every instinct screamed to flee, but her feet felt rooted to the stone floor, a terrifying paralysis gripping her.
--- End of Example ---"""

    prompt = render_prompt(
        "patch_generation.j2",
        {
            "enable_no_think": True,
            "chapter_number": chapter_number,
            "novel_title": plot_outline.get("title", "Untitled Novel"),
            "genre": plot_outline.get("genre", "N/A"),
            "theme": plot_outline.get("theme", "N/A"),
            "protagonist_name": protagonist_name,
            "character_arc": plot_outline.get("character_arc", "N/A"),
            "plan_focus_section_str": plan_focus_section_str,
            "hybrid_context_for_revision": hybrid_context_for_revision,
            "issue_category": problem.issue_category,
            "problem_description": problem.problem_description,
            "original_quote_text_from_problem": original_quote_text_from_problem,
            "suggested_fix_focus": problem.suggested_fix_focus,
            "rewrite_instruction": rewrite_instruction or "",
            "original_chapter_text_snippet_for_llm": original_chapter_text_snippet_for_llm,
            "length_expansion_instruction_header_str": length_expansion_instruction_header_str,
            "few_shot_patch_example_str": few_shot_patch_example_str.strip(),
            "prompt_instruction_for_replacement_scope_str": prompt_instruction_for_replacement_scope_str,
            "validation_failure_reason": validation_failure_reason,
        },
    )

    logger.info(
        "Calling LLM (%s) for patch in Ch %s. Problem: '%s...' Quote Text: '%s...' Max Output Tokens: %s",
        settings.PATCH_GENERATION_MODEL,
        chapter_number,
        problem.problem_description[:60].replace(chr(10), " "),
        original_quote_text_from_problem[:50].replace(chr(10), " "),
        max_patch_output_tokens,
    )

    replace_with_text_cleaned, usage_data = await llm_service.async_call_llm(
        model_name=settings.PATCH_GENERATION_MODEL,
        prompt=prompt,
        temperature=settings.TEMPERATURE_PATCH,
        max_tokens=max_patch_output_tokens,
        allow_fallback=True,
        stream_to_disk=False,
        auto_clean_response=True,
    )

    if replace_with_text_cleaned is None:
        logger.error(
            "Patch LLM call failed and returned None for Ch %s problem: %s",
            chapter_number,
            problem.problem_description,
        )
        return None, usage_data

    if not replace_with_text_cleaned.strip():
        logger.info(
            "Patch LLM suggested DELETION (empty output) for Ch %s problem: %s",
            chapter_number,
            problem.problem_description,
        )

    if length_expansion_instruction_header_str:
        if not is_general_expansion_task:
            if (
                len(original_chapter_text_snippet_for_llm) > 100
                and len(replace_with_text_cleaned)
                < len(original_chapter_text_snippet_for_llm) * 1.2
            ):
                logger.warning(
                    "Patch for Ch %s (specific quote, segment expansion requested) output length (%s) is not significantly larger than context snippet (%s). Problem: %s",
                    chapter_number,
                    len(replace_with_text_cleaned),
                    len(original_chapter_text_snippet_for_llm),
                    problem.problem_description[:60],
                )
        elif is_general_expansion_task and len(replace_with_text_cleaned) < 500:
            logger.warning(
                "Patch for Ch %s ('N/A - General Issue' expansion) produced a relatively short new passage (len: %s). Problem: %s",
                chapter_number,
                len(replace_with_text_cleaned),
                problem.problem_description[:60],
            )

    target_start_for_patch: int | None = problem.sentence_char_start
    target_end_for_patch: int | None = problem.sentence_char_end

    if (
        original_quote_text_from_problem != "N/A - General Issue"
        and (target_start_for_patch is None or target_end_for_patch is None)
        and (
            problem.quote_char_start is not None and problem.quote_char_end is not None
        )
    ):
        logger.warning(
            "Patch for Ch %s: Problem '%s' had specific text but no sentence offsets. PatchInstruction will use quote offsets (%s-%s). Application will use semantic search.",
            chapter_number,
            original_quote_text_from_problem[:50],
            problem.quote_char_start,
            problem.quote_char_end,
        )
        target_start_for_patch = problem.quote_char_start
        target_end_for_patch = problem.quote_char_end
    elif original_quote_text_from_problem != "N/A - General Issue" and (
        target_start_for_patch is None or target_end_for_patch is None
    ):
        logger.error(
            "Patch for Ch %s: Problem '%s' specific text but NO OFFSETS (sentence or quote). Patch will likely fail to apply precisely.",
            chapter_number,
            original_quote_text_from_problem[:50],
        )

    patch_instruction: PatchInstruction = {
        "original_problem_quote_text": original_quote_text_from_problem,
        "target_char_start": target_start_for_patch,
        "target_char_end": target_end_for_patch,
        "replace_with": replace_with_text_cleaned,
        "reason_for_change": f"Fixing '{problem.issue_category}': {problem.problem_description}",
    }
    return patch_instruction, usage_data


def _deduplicate_problems(problems: list[ProblemDetail]) -> list[ProblemDetail]:
    """Remove exact duplicate problems."""
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
    """Return consolidated problem with list of originals."""
    if not problems:
        return []

    def _get(problem: ProblemDetail, field: str) -> Any:
        return getattr(problem, field, problem.get(field))

    span_problems = [
        p
        for p in problems
        if _get(p, "sentence_char_start") is not None
        and _get(p, "sentence_char_end") is not None
    ]
    general_problems = [
        p
        for p in problems
        if _get(p, "sentence_char_start") is None
        or _get(p, "sentence_char_end") is None
    ]

    span_problems.sort(key=lambda p: _get(p, "sentence_char_start"))
    merged_groups: list[list[ProblemDetail]] = []
    if span_problems:
        current_group = [span_problems[0]]
        current_end = _get(span_problems[0], "sentence_char_end")
        for prob in span_problems[1:]:
            start = _get(prob, "sentence_char_start")
            end = _get(prob, "sentence_char_end")
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
        group_start = min(_get(p, "sentence_char_start") for p in group)
        group_end = max(_get(p, "sentence_char_end") for p in group)
        all_cats = sorted(list(set(_get(p, "issue_category") for p in group)))
        all_desc = "; ".join(
            f"({_get(p, 'issue_category')}) {_get(p, 'problem_description')}"
            for p in group
        )
        all_fix = "; ".join(
            f"({_get(p, 'issue_category')}) {_get(p, 'suggested_fix_focus')}"
            for p in group
        )
        rep_quote = _get(first, "quote_from_original_text")
        consolidated = ProblemDetail(
            issue_category=", ".join(all_cats),
            problem_description=f"Multiple issues in one segment: {all_desc}",
            quote_from_original_text=rep_quote,
            quote_char_start=_get(first, "quote_char_start"),
            quote_char_end=_get(first, "quote_char_end"),
            sentence_char_start=group_start,
            sentence_char_end=group_end,
            suggested_fix_focus=f"Holistically revise the segment to address all points: {all_fix}",
        )
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
) -> tuple[list[PatchInstruction], TokenUsage | None]:
    patch_instructions: list[PatchInstruction] = []
    total_usage = TokenUsage()

    plot_point_focus, _ = get_plot_point_info(plot_outline, chapter_number)

    grouped = _group_problems_for_patch_generation(problems_to_fix)

    groups_to_process = grouped[: settings.MAX_PATCH_INSTRUCTIONS_TO_GENERATE]
    if len(grouped) > len(groups_to_process):
        logger.warning(
            "Found %s patch groups for Ch %s. Processing only the first %s groups.",
            len(grouped),
            chapter_number,
            len(groups_to_process),
        )
    if not groups_to_process:
        return [], None

    async def _process_group(
        group_idx: int, group_problem: ProblemDetail, group_members: list[ProblemDetail]
    ) -> tuple[PatchInstruction | None, TokenUsage]:
        context_snippet = await _get_context_window_for_patch_llm(
            original_text,
            group_problem,
            settings.MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW,
        )

        patch_instr: PatchInstruction | None = None
        usage_acc = TokenUsage()
        validation_reason: str | None = None

        combined_fix_text = (
            f"{group_problem.suggested_fix_focus} {group_problem.rewrite_instruction or ''}"
        ).lower()
        deletion_keywords = ("delete", "remove", "cut", "omit")
        if any(kw in combined_fix_text for kw in deletion_keywords):
            target_start_for_patch = group_problem.sentence_char_start
            target_end_for_patch = group_problem.sentence_char_end
            original_quote = group_problem.quote_from_original_text
            if (
                original_quote != "N/A - General Issue"
                and (target_start_for_patch is None or target_end_for_patch is None)
                and (
                    group_problem.quote_char_start is not None
                    and group_problem.quote_char_end is not None
                )
            ):
                logger.warning(
                    "Patch for Ch %s: Problem '%s' had specific text but no sentence offsets. PatchInstruction will use quote offsets (%s-%s). Application will use semantic search.",
                    chapter_number,
                    original_quote[:50],
                    group_problem.quote_char_start,
                    group_problem.quote_char_end,
                )
                target_start_for_patch = group_problem.quote_char_start
                target_end_for_patch = group_problem.quote_char_end
            elif original_quote != "N/A - General Issue" and (
                target_start_for_patch is None or target_end_for_patch is None
            ):
                logger.error(
                    "Patch for Ch %s: Problem '%s' specific text but NO OFFSETS (sentence or quote). Patch will likely fail to apply precisely.",
                    chapter_number,
                    original_quote[:50],
                )

            patch_instr = PatchInstruction(
                original_problem_quote_text=original_quote,
                target_char_start=target_start_for_patch,
                target_char_end=target_end_for_patch,
                replace_with="",
                reason_for_change=f"Fixing '{group_problem.issue_category}': {group_problem.problem_description}",
            )
            return patch_instr, usage_acc

        for _ in range(settings.PATCH_GENERATION_ATTEMPTS):
            patch_instr_tmp, usage = await _generate_single_patch_instruction_llm(
                plot_outline,
                context_snippet,
                group_problem,
                group_problem.rewrite_instruction,
                chapter_number,
                hybrid_context_for_revision,
                chapter_plan,
                plot_point_focus,
                validation_failure_reason=validation_reason,
            )
            if usage:
                usage_acc.add(usage)
            if not patch_instr_tmp:
                continue
            if not settings.AGENT_ENABLE_PATCH_VALIDATION:
                patch_instr = patch_instr_tmp
                break
            valid, validation_reason, val_usage = await validator.validate_patch(
                context_snippet, patch_instr_tmp, group_members
            )
            if val_usage:
                usage_acc.add(val_usage)
            if valid:
                patch_instr = patch_instr_tmp
                break

        if not patch_instr:
            logger.warning(
                "Failed to generate valid patch for group %s in Ch %s.",
                group_idx,
                chapter_number,
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
        total_usage.add(usage_acc)

    logger.info(
        "Generated %s patch instructions for Ch %s.",
        len(patch_instructions),
        chapter_number,
    )
    return patch_instructions, total_usage if total_usage.total_tokens > 0 else None

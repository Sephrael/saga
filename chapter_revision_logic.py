# chapter_revision_logic.py
"""
Handles the revision of chapter drafts based on evaluation feedback for the SAGA system.
Supports both full rewrite and targeted patch-based revisions.
Context data for prompts is now formatted as plain text.
"""
import logging
import asyncio
import re
from typing import Tuple, Optional, List, Dict, Any

import config
import llm_interface
# import drafting_agent # No longer directly instantiating DraftingAgent here.
import utils # For numpy_cosine_similarity, find_semantically_closest_segment, AND find_quote_and_sentence_offsets_with_spacy
from type import SceneDetail, ProblemDetail, PatchInstruction, EvaluationResult

logger = logging.getLogger(__name__)
utils.load_spacy_model_if_needed() # Ensure spaCy model is loaded when this module is imported

def _get_formatted_scene_plan_from_agent_or_fallback(
    agent: Any, # This 'agent' is the orchestrator
    chapter_plan: List[SceneDetail],
    model_name_for_tokens: str,
    max_tokens_budget: int
) -> str:
    """Attempts to get formatted scene plan from agent (orchestrator, which might hold a drafting_agent instance or its methods) or uses a fallback."""
    if hasattr(agent, 'drafting_agent') and hasattr(agent.drafting_agent, '_format_scene_plan_for_prompt'):
        try:
            return agent.drafting_agent._format_scene_plan_for_prompt(chapter_plan, model_name_for_tokens, max_tokens_budget)
        except Exception as e:
            logger.error(f"Error calling _format_scene_plan_for_prompt via agent.drafting_agent: {e}. Using fallback.")
    
    logger.warning("_get_formatted_scene_plan_from_agent_or_fallback: Using fallback scene plan formatting logic.")
    if not chapter_plan: return "Scene plan formatting unavailable or plan empty (stub)."
    
    plan_text_parts_list = [] 
    current_tokens = 0
    header = "**Detailed Scene Plan (Stubbed - MUST BE FOLLOWED CLOSELY):**\n"
    header_tokens = llm_interface.count_tokens(header, model_name_for_tokens)

    if header_tokens > max_tokens_budget: return "... (plan header too long for budget)"
    plan_text_parts_list.append(header)
    current_tokens += header_tokens

    for scene_idx, scene in enumerate(chapter_plan):
        scene_text_parts_inner = [
            f"Scene Number: {scene.get('scene_number', 'N/A')}",
            f"  Summary: {scene.get('summary', 'No summary')}"
        ]
        if scene_idx < len(chapter_plan) -1 : scene_text_parts_inner.append("-" * 10)
        scene_text = "\n".join(scene_text_parts_inner) + "\n"

        scene_tokens = llm_interface.count_tokens(scene_text, model_name_for_tokens)
        if current_tokens + scene_tokens > max_tokens_budget:
            plan_text_parts_list.append("... (plan truncated in prompt due to token limit)\n")
            break
        plan_text_parts_list.append(scene_text)
        current_tokens += scene_tokens
    return "".join(plan_text_parts_list)


def _get_prop_from_agent(agent: Any, key: str, default: Any = None) -> Any:
    return getattr(agent, key, agent.novel_props_cache.get(key, default) if hasattr(agent, 'novel_props_cache') else default)

def _get_nested_prop_from_agent(agent: Any, primary_key: str, secondary_key: str, default: Any = None) -> Any:
    primary_data = _get_prop_from_agent(agent, primary_key, {})
    if isinstance(primary_data, dict):
        return primary_data.get(secondary_key, default)
    return default

def _get_plot_point_info_from_agent(agent: Any, chapter_number: int) -> Tuple[Optional[str], int]:
    plot_outline_data = _get_prop_from_agent(agent, 'plot_outline', {}) # 'plot_outline' on orchestrator is the full dict
    plot_points = plot_outline_data.get("plot_points", [])
    if not isinstance(plot_points, list) or not plot_points: return None, -1
    if chapter_number <= 0: return None, -1
    plot_point_index = min(chapter_number - 1, len(plot_points) - 1)
    if 0 <= plot_point_index < len(plot_points):
        plot_point = plot_points[plot_point_index]
        return str(plot_point) if plot_point is not None else None, plot_point_index
    return None, -1


def _get_context_window_for_patch_llm(
    original_doc_text: str,
    problem: ProblemDetail,
    window_size_chars: int
) -> str:
    """
    Gets a context window around the problem's quote using precise offsets if available.
    The window is centered around the *sentence* containing the quote.
    """
    if not original_doc_text: return ""

    quote_text_from_llm = problem["quote_from_original_text"]
    focus_start = problem.get("sentence_char_start")
    focus_end = problem.get("sentence_char_end")

    if focus_start is None or focus_end is None:
        focus_start = problem.get("quote_char_start")
        focus_end = problem.get("quote_char_end")
        if focus_start is not None:
            logger.debug(f"Context window for patch: Using quote offsets {focus_start}-{focus_end} as sentence offsets were not available for '{quote_text_from_llm[:30]}...'.")


    if "N/A - General Issue" in quote_text_from_llm or focus_start is None or focus_end is None:
        if "N/A - General Issue" not in quote_text_from_llm:
             logger.warning(f"Context window for patch: No valid offsets for quote '{quote_text_from_llm[:30]}...'. Using general snippet logic.")

        if len(original_doc_text) <= window_size_chars:
            return original_doc_text
        start_snippet_len = min(window_size_chars // 2, len(original_doc_text))
        remaining_chars_for_end = window_size_chars - start_snippet_len
        end_snippet_len = min(remaining_chars_for_end, len(original_doc_text) - start_snippet_len)
        start_snippet = original_doc_text[:start_snippet_len]
        end_snippet = original_doc_text[-end_snippet_len:] if end_snippet_len > 0 else ""
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


async def _generate_single_patch_instruction_llm(
    agent: Any,
    original_chapter_text_snippet_for_llm: str, 
    problem: ProblemDetail, 
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> Tuple[Optional[PatchInstruction], Optional[Dict[str, int]]]:
    """
    Generates a single patch instruction. The PatchInstruction will store target_char_start/end
    referring to the SENTENCE containing the problem quote if available.
    """
    plan_focus_section_parts: List[str] = []
    plot_point_focus, _ = _get_plot_point_info_from_agent(agent, chapter_number)
    max_plan_tokens_for_patch_prompt = config.MAX_CONTEXT_TOKENS // 2

    if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
        formatted_plan = _get_formatted_scene_plan_from_agent_or_fallback( 
            agent, chapter_plan, config.PATCH_GENERATION_MODEL, max_plan_tokens_for_patch_prompt
        )
        plan_focus_section_parts.append(formatted_plan)
        if "plan truncated" in formatted_plan:
             logger.warning(f"Scene plan token-truncated for Ch {chapter_number} patch generation prompt.")
    else:
        plan_focus_section_parts.append(f"**Original Chapter Focus (Reference for overall chapter direction):**\n{plot_point_focus or 'Not specified.'}\n")
    plan_focus_section_str = "".join(plan_focus_section_parts)


    is_general_expansion_task = False
    length_expansion_instruction_header_parts: List[str] = []
    original_quote_text_from_problem = problem['quote_from_original_text']

    if problem['issue_category'] == "narrative_depth" and \
       ("short" in problem['problem_description'].lower() or \
        "length" in problem['problem_description'].lower() or \
        "expand" in problem['suggested_fix_focus'].lower() or \
        "depth" in problem['problem_description'].lower() or \
        original_quote_text_from_problem == "N/A - General Issue"):
        length_expansion_instruction_header_parts.append(f"\n**Critical: SUBSTANTIAL EXPANSION REQUIRED FOR THIS SEGMENT/PASSAGE.** ")
        length_expansion_instruction_header_parts.append(f"The 'replace_with' text MUST be significantly longer and more detailed. ")
        length_expansion_instruction_header_parts.append(f"Add descriptive details, character thoughts, dialogue, actions, and sensory information. ")
        if original_quote_text_from_problem == "N/A - General Issue":
            is_general_expansion_task = True
            length_expansion_instruction_header_parts.append(
                f"Since the original quote is 'N/A - General Issue', your 'replace_with' text should be a **new, expanded passage** "
                f"that addresses the 'Problem Description' and 'Suggested Fix Focus' within the broader 'Text Snippet' context. "
                f"This generated text is intended as a candidate for insertion or to inform a broader rewrite of a section."
            )
        else:
             length_expansion_instruction_header_parts.append(
                f"Aim for a notable increase in length and detail for the conceptual segment related to the original quote."
             )
    length_expansion_instruction_header_str = "".join(length_expansion_instruction_header_parts)

    prompt_instruction_for_replacement_scope_parts: List[str] = []
    max_patch_output_tokens = 0

    if is_general_expansion_task:
        prompt_instruction_for_replacement_scope_parts.append(
            "    - The 'Original Quote Illustrating Problem' is \"N/A - General Issue\". Therefore, your `replace_with` text should be a **new, self-contained, and substantially expanded passage** "
            "that addresses the \"Problem Description\" and \"Suggested Fix Focus\" as guided by the `length_expansion_instruction_header_str`. "
            "This new passage is intended for potential insertion into the chapter, not to replace a specific quote."
        )
        max_patch_output_tokens = config.MAX_GENERATION_TOKENS // 2 
        max_patch_output_tokens = max(max_patch_output_tokens, 750) 
        logger.info(f"Patch (Ch {chapter_number}, general expansion): Max output tokens set to {max_patch_output_tokens}.")
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
        original_snippet_tokens = llm_interface.count_tokens(original_chapter_text_snippet_for_llm, config.PATCH_GENERATION_MODEL)
        expansion_factor = 2.5 if length_expansion_instruction_header_str else 1.5
        max_patch_output_tokens = int(original_snippet_tokens * expansion_factor)
        max_patch_output_tokens = min(max_patch_output_tokens, config.MAX_GENERATION_TOKENS // 2) 
        max_patch_output_tokens = max(max_patch_output_tokens, 200) 
        logger.info(f"Patch (Ch {chapter_number}, specific fix): Original snippet tokens: {original_snippet_tokens}. Max output tokens set to {max_patch_output_tokens}.")
    prompt_instruction_for_replacement_scope_str = "".join(prompt_instruction_for_replacement_scope_parts)

    plot_outline_data = _get_prop_from_agent(agent, 'plot_outline', {})
    protagonist_name = _get_nested_prop_from_agent(agent, 'plot_outline', 'protagonist_name', config.DEFAULT_PROTAGONIST_NAME)

    few_shot_patch_example_str = f"""
--- Example of how to provide 'replace_with' text (this is an example, NOT part of current task) ---
IF THE PROBLEM WAS:
  - Issue Category: narrative_depth
  - Problem Description: The reaction of Elara to seeing the ghost felt understated.
  - Original Quote Illustrating Problem: "Elara saw the ghost and gasped."
  - Suggested Fix Focus: Expand on Elara's internal emotional reaction and physical response.
THEN YOUR 'replace_with' TEXT MIGHT BE (just the text, no other explanation):
A chill traced Elara's spine, not from the crypt's cold, but from the translucent figure coalescing before her. Her breath hitched, a silent scream trapped in her throat as the ghostly visage turned its empty sockets towards her. Every instinct screamed to flee, but her feet felt rooted to the stone floor, a terrifying paralysis gripping her.
--- End of Example ---
"""

    prompt_lines = [
        "/no_think",
        f"You are a surgical revision expert generating replacement text for Chapter {chapter_number} of a novel titled \"{_get_nested_prop_from_agent(agent, 'plot_outline', 'title', 'Untitled Novel')}\" about {protagonist_name}.",
        "**Novel Context:**",
        f"  - Genre: {_get_nested_prop_from_agent(agent, 'plot_outline', 'genre', 'N/A')}",
        f"  - Theme: {_get_nested_prop_from_agent(agent, 'plot_outline', 'theme', 'N/A')}",
        f"  - Protagonist: {protagonist_name} ({_get_nested_prop_from_agent(agent, 'plot_outline', 'character_arc', 'N/A')})",
        "",
        plan_focus_section_str,
        "**Hybrid Context from Previous Chapters (for consistency with established canon and narrative flow):**",
        "--- BEGIN HYBRID CONTEXT ---",
        hybrid_context_for_revision if hybrid_context_for_revision.strip() else "No previous context.",
        "--- END HYBRID CONTEXT ---",
        "",
        "**Specific Problem to Address in the Chapter:**",
        f"  - Issue Category: {problem['issue_category']}",
        f"  - Problem Description: {problem['problem_description']}",
        f"  - Original Quote Illustrating Problem: \"{original_quote_text_from_problem}\"",
        f"  - Suggested Fix Focus: {problem['suggested_fix_focus']}",
        "",
        "**Text Snippet from Original Chapter (This is the broader context around the problem. If the quote is 'N/A - General Issue', this is general chapter context to inform your new passage):**",
        "--- BEGIN ORIGINAL TEXT SNIPPET ---",
        original_chapter_text_snippet_for_llm,
        "--- END ORIGINAL TEXT SNIPPET ---",
        length_expansion_instruction_header_str,
        few_shot_patch_example_str, 
        "**Instructions for Generating Replacement Text:**",
        "1.  Focus EXCLUSIVELY on the problem described, particularly relating to the conceptual area highlighted by: `{original_quote_text_from_problem}` within the 'ORIGINAL TEXT SNIPPET'.",
        "2.  Generate a `replace_with` text according to the following:",
        prompt_instruction_for_replacement_scope_str,
        "3.  The `replace_with` text MUST address the \"Problem Description\" and \"Suggested Fix Focus\".",
        "4.  Maintain the novel's style, tone, and consistency with all provided context (Novel Context, Plan, Hybrid Context).",
        "5.  If `length_expansion_instruction_header_str` is present, ensure substantial expansion as guided for the targeted segment or new passage.",
        "6.  **Output ONLY the `replace_with` text.** Do NOT include JSON, markdown, explanations, or any \"Replace with:\" prefixes. Just the raw text intended for replacement/insertion. (See example above for how to format the text).",
        "",
        f"--- BEGIN REPLACE_WITH TEXT (for the segment related to \"{original_quote_text_from_problem}\" or as a new passage if quote is \"N/A - General Issue\") ---"
    ]
    prompt = "\n".join(prompt_lines)

    logger.info(f"Calling LLM ({config.PATCH_GENERATION_MODEL}) for patch in Ch {chapter_number}. Problem: '{problem['problem_description'][:60].replace(chr(10),' ')}...' Quote Text: '{original_quote_text_from_problem[:50].replace(chr(10),' ')}...' Max Output Tokens: {max_patch_output_tokens}")

    replace_with_text_raw, usage_data = await llm_interface.async_call_llm(
        model_name=config.PATCH_GENERATION_MODEL,
        prompt=prompt,
        temperature=config.TEMPERATURE_PATCH, 
        max_tokens=max_patch_output_tokens,
        allow_fallback=True,
        stream_to_disk=False,
        frequency_penalty=config.FREQUENCY_PENALTY_PATCH,
        presence_penalty=config.PRESENCE_PENALTY_PATCH
    )

    if not replace_with_text_raw:
        logger.error(f"Patch LLM returned no content for Ch {chapter_number} problem: {problem['problem_description']}")
        return None, usage_data

    replace_with_text_cleaned = llm_interface.clean_model_response(replace_with_text_raw)
    if not replace_with_text_cleaned.strip():
        logger.warning(f"Patch LLM returned empty after cleaning for Ch {chapter_number} problem: {problem['problem_description']}")
        return None, usage_data

    if length_expansion_instruction_header_str:
        if not is_general_expansion_task:
            if len(original_chapter_text_snippet_for_llm) > 100 and len(replace_with_text_cleaned) < len(original_chapter_text_snippet_for_llm) * 1.2:
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

    target_start_for_patch: Optional[int] = problem.get("sentence_char_start")
    target_end_for_patch: Optional[int] = problem.get("sentence_char_end")

    if original_quote_text_from_problem != "N/A - General Issue" and \
       (target_start_for_patch is None or target_end_for_patch is None) and \
       (problem.get("quote_char_start") is not None and problem.get("quote_char_end") is not None):
        logger.warning(f"Patch for Ch {chapter_number}: Problem '{original_quote_text_from_problem[:50]}' had specific text but no sentence offsets. "
                       f"PatchInstruction will use quote offsets ({problem.get('quote_char_start')}-{problem.get('quote_char_end')}). Application will use semantic search.")
        target_start_for_patch = problem.get("quote_char_start")
        target_end_for_patch = problem.get("quote_char_end")
    elif original_quote_text_from_problem != "N/A - General Issue" and \
         (target_start_for_patch is None or target_end_for_patch is None):
        logger.error(f"Patch for Ch {chapter_number}: Problem '{original_quote_text_from_problem[:50]}' specific text but NO OFFSETS (sentence or quote). Patch will likely fail to apply precisely.")

    patch_instruction: PatchInstruction = {
        "original_problem_quote_text": original_quote_text_from_problem,
        "target_char_start": target_start_for_patch,
        "target_char_end": target_end_for_patch,
        "replace_with": replace_with_text_cleaned,
        "reason_for_change": f"Fixing '{problem['issue_category']}': {problem['problem_description']}"
    }
    return patch_instruction, usage_data


async def _generate_patch_instructions_logic(
    agent: Any,
    original_text: str, 
    problems_to_fix: List[ProblemDetail],
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> Tuple[List[PatchInstruction], Optional[Dict[str, int]]]:
    patch_instructions: List[PatchInstruction] = []
    total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    actionable_problems_for_patch_generation = []
    for p_idx, p_item in enumerate(problems_to_fix):
        is_specific_and_located = (
            p_item["quote_from_original_text"] != "N/A - General Issue" and
            p_item["quote_from_original_text"].strip() and
            (p_item.get("sentence_char_start") is not None or p_item.get("quote_char_start") is not None)
        )
        is_expansion_depth_issue_general = (
            p_item["quote_from_original_text"] == "N/A - General Issue" and
            p_item["issue_category"] == "narrative_depth" and
            ("short" in p_item['problem_description'].lower() or
             "length" in p_item['problem_description'].lower() or
             "expand" in p_item['suggested_fix_focus'].lower() or
             "depth" in p_item['problem_description'].lower())
        )
        if is_specific_and_located or is_expansion_depth_issue_general:
            actionable_problems_for_patch_generation.append(p_item)
        else:
            reason_skip = "not specific quote text OR not an expansion-type general issue"
            if p_item["quote_from_original_text"] != "N/A - General Issue" and p_item["quote_from_original_text"].strip() and \
               p_item.get("sentence_char_start") is None and p_item.get("quote_char_start") is None:
                reason_skip = "specific quote text present, but no offsets found by spaCy utils"
            logger.info(f"Skipping patch generation for Ch {chapter_number} problem {p_idx+1} ({reason_skip}): '{p_item['problem_description'][:60]}'")

    problems_to_process = actionable_problems_for_patch_generation[:config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE]
    if len(problems_to_fix) > len(problems_to_process): 
        logger.warning(
            f"Found {len(problems_to_fix)} problems for Ch {chapter_number}. "
            f"{len(actionable_problems_for_patch_generation)} were actionable for patch generation. "
            f"Attempting to generate patches for the first {len(problems_to_process)} of these."
        )
    elif not problems_to_process:
        logger.info(f"No problems suitable for patch instruction generation in Ch {chapter_number}.")
        return [], None


    patch_generation_tasks = []
    for problem in problems_to_process:
        context_snippet_for_llm = _get_context_window_for_patch_llm(
            original_text, problem, config.MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW
        )
        task = _generate_single_patch_instruction_llm(
            agent, context_snippet_for_llm, problem, chapter_number, hybrid_context_for_revision, chapter_plan
        )
        patch_generation_tasks.append(task)

    if not patch_generation_tasks:
        logger.info(f"No patch generation tasks created for Ch {chapter_number}, though problems were identified.")
        return [], None

    results = await asyncio.gather(*patch_generation_tasks, return_exceptions=True)
    for i, res_or_exc in enumerate(results):
        problem_ref = problems_to_process[i]
        if isinstance(res_or_exc, Exception):
            logger.error(f"Error generating patch for Ch {chapter_number} problem '{problem_ref['problem_description'][:50].replace(chr(10),' ')}': {res_or_exc}", exc_info=res_or_exc)
        elif res_or_exc is not None:
            patch_instr, usage = res_or_exc
            if patch_instr:
                patch_instructions.append(patch_instr)
            if usage:
                total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                total_usage["total_tokens"] += usage.get("total_tokens", 0)
        else:
            logger.warning(f"Patch generation returned None for Ch {chapter_number} problem: '{problem_ref['problem_description'][:50].replace(chr(10),' ')}'.")

    logger.info(f"Generated {len(patch_instructions)} patch instructions for Ch {chapter_number}.")
    return patch_instructions, total_usage if total_usage["total_tokens"] > 0 else None


async def _apply_patches_to_text(original_text: str, patch_instructions: List[PatchInstruction]) -> str:
    """
    Applies patch instructions to the original text.
    Prioritizes precise target_char_start/end from PatchInstruction (which should be sentence boundaries).
    Falls back to semantic paragraph search if precise offsets are missing.
    """
    if not patch_instructions:
        return original_text

    applicable_patches: List[PatchInstruction] = []
    for p_idx, p_item in enumerate(patch_instructions):
        has_content = p_item["replace_with"].strip()
        has_precise_offsets = p_item.get("target_char_start") is not None and p_item.get("target_char_end") is not None
        has_specific_quote_text = p_item["original_problem_quote_text"] != "N/A - General Issue" and p_item["original_problem_quote_text"].strip()

        if has_content and (has_precise_offsets or has_specific_quote_text):
            applicable_patches.append(p_item)
        elif p_item["original_problem_quote_text"] == "N/A - General Issue" and has_content:
             logger.info(f"Patch {p_idx+1} for 'N/A - General Issue' (expansion) generated new content. This patch type is not auto-inserted here.")
        else:
            logger.debug(f"Skipping patch application for (Problem {p_idx+1}): '{p_item['original_problem_quote_text'][:50]}' (no target info or no replacement content).")


    if not applicable_patches:
        logger.info("No patches with specific target information or content to apply for segment replacement.")
        return original_text

    replacements: List[Tuple[int, int, str]] = []
    failed_patches_target_not_found = 0

    for patch_idx, patch in enumerate(applicable_patches):
        segment_to_replace_start: Optional[int] = patch.get("target_char_start")
        segment_to_replace_end: Optional[int] = patch.get("target_char_end")
        method_used = "spaCy-derived sentence/quote offsets"

        if segment_to_replace_start is None or segment_to_replace_end is None:
            quote_text_for_semantic_search = patch["original_problem_quote_text"]
            if quote_text_for_semantic_search != "N/A - General Issue" and quote_text_for_semantic_search.strip():
                logger.info(f"Patch {patch_idx+1}: Missing precise offsets for problem '{quote_text_for_semantic_search[:50]}'. Attempting semantic paragraph search.")
                method_used = "Semantic paragraph search (fallback)"
                semantic_match_info = await utils.find_semantically_closest_segment(
                    original_text,
                    quote_text_for_semantic_search,
                    segment_type="paragraph",
                    min_similarity_threshold=0.60
                )
                if semantic_match_info:
                    segment_to_replace_start, segment_to_replace_end, score = semantic_match_info
                    logger.info(f"Patch {patch_idx+1} ({method_used}): Identified target paragraph "
                                f"at original chars {segment_to_replace_start}-{segment_to_replace_end} (Score: {score:.2f}).")
                else:
                    logger.warning(f"Patching (Patch {patch_idx+1}): Fallback semantic search failed for '{quote_text_for_semantic_search[:100]}'. Skipping this patch.")
                    failed_patches_target_not_found += 1
                    continue
            else:
                logger.warning(f"Patch {patch_idx+1}: Skipping as it has no precise offsets and no specific quote text for semantic search. Quote: '{quote_text_for_semantic_search}'")
                failed_patches_target_not_found +=1
                continue 

        if segment_to_replace_start is None or segment_to_replace_end is None : 
            logger.error(f"Patch {patch_idx+1}: Logic error, segment_to_replace offsets are still None after checks. Skipping. Patch: {patch}")
            failed_patches_target_not_found +=1
            continue
        
        has_overlap = False
        for r_start, r_end, _ in replacements:
            if max(segment_to_replace_start, r_start) < min(segment_to_replace_end, r_end):
                has_overlap = True
                logger.warning(f"Patch {patch_idx+1} for problem '{patch['original_problem_quote_text'][:50]}' "
                               f"(targets {segment_to_replace_start}-{segment_to_replace_end} via {method_used}) "
                               f"overlaps with a previously determined patch for segment {r_start}-{r_end}. Skipping.")
                break
        
        if not has_overlap:
            replacements.append((segment_to_replace_start, segment_to_replace_end, patch["replace_with"]))
            logger.info(f"Patch {patch_idx+1} ({method_used}): Queued replacement for segment {segment_to_replace_start}-{segment_to_replace_end}.")


    if failed_patches_target_not_found > 0:
         logger.warning(f"{failed_patches_target_not_found}/{len(applicable_patches)} applicable patches failed (target segment not found).")

    if not replacements:
        logger.info("No patches could be confidently mapped to text segments for replacement.")
        return original_text

    replacements.sort(key=lambda x: x[0], reverse=True) 
    current_text_list = list(original_text)
    applied_count = 0

    for start_index, end_index, replace_with_text in replacements:
        current_text_list[start_index:end_index] = list(replace_with_text)
        applied_count += 1
        logger.info(f"Applied patch: Replaced original segment from char {start_index} to {end_index} "
                    f"(length {end_index-start_index}) with new text (length {len(replace_with_text)}).")

    num_patches_attempted = len(applicable_patches) - failed_patches_target_not_found
    logger.info(f"Applied {applied_count} out of {num_patches_attempted if num_patches_attempted >=0 else len(applicable_patches)} patches that targeted specific segments.")
    return "".join(current_text_list)


async def revise_chapter_draft_logic(
    agent: Any,
    original_text: str,
    chapter_number: int,
    evaluation_result: EvaluationResult,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]],
    is_from_flawed_source: bool = False # ADDED: To indicate if original_text might have gaps from de-dup
) -> Tuple[Optional[Tuple[str, str]], Optional[Dict[str, int]]]:
    cumulative_usage_data: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _add_usage(usage: Optional[Dict[str, int]]):
        if usage:
            cumulative_usage_data["prompt_tokens"] += usage.get("prompt_tokens", 0)
            cumulative_usage_data["completion_tokens"] += usage.get("completion_tokens", 0)
            cumulative_usage_data["total_tokens"] += usage.get("total_tokens", 0)

    if not original_text:
        logger.error(f"Revision for ch {chapter_number} aborted: missing original text.")
        return None, None

    problems_to_fix: List[ProblemDetail] = evaluation_result.get("problems_found", [])
    if not problems_to_fix and evaluation_result.get("needs_revision"):
        logger.warning(f"Revision for ch {chapter_number} explicitly requested, but no specific problems were itemized. This might lead to a full rewrite attempt if general reasons exist.")
    elif not problems_to_fix:
        logger.info(f"No specific problems found for ch {chapter_number}, and not marked for revision. No revision performed.")
        return None, None

    revision_reason_str_list = evaluation_result.get("reasons", [])
    revision_reason_str = "\n- ".join(revision_reason_str_list) if revision_reason_str_list else "General unspecified issues."
    logger.info(f"Attempting revision for chapter {chapter_number}. Reason(s):\n- {revision_reason_str}")

    patched_text: Optional[str] = None
    raw_patch_llm_outputs_combined_parts: List[str] = []

    actionable_problems_for_patch_gen_check = [
        p for p in problems_to_fix if
        (p["quote_from_original_text"] != "N/A - General Issue" and p["quote_from_original_text"].strip() and \
         (p.get("sentence_char_start") is not None or p.get("quote_char_start") is not None) 
        ) or
        (p["quote_from_original_text"] == "N/A - General Issue" and p["issue_category"] == "narrative_depth" and
         ("short" in p['problem_description'].lower() or "length" in p['problem_description'].lower() or
          "expand" in p['suggested_fix_focus'].lower() or "depth" in p['problem_description'].lower()))
    ]

    if config.ENABLE_PATCH_BASED_REVISION and actionable_problems_for_patch_gen_check:
        logger.info(f"Attempting patch-based revision for Ch {chapter_number} with {len(actionable_problems_for_patch_gen_check)} potentially actionable problem(s).")
        patch_instructions, patch_usage = await _generate_patch_instructions_logic(
            agent, original_text, problems_to_fix,
            chapter_number, hybrid_context_for_revision, chapter_plan
        )
        _add_usage(patch_usage)
        if patch_instructions:
            patched_text = await _apply_patches_to_text(original_text, patch_instructions)
            raw_patch_llm_outputs_combined_parts.append(
                f"Chapter revised using {len(patch_instructions)} generated patch instructions. "
                f"(Note: Not all generated patches may have been auto-applied if they were for 'N/A - General Issue' or target segment not found.)\n"
            )
            logger.info(
                f"Patch process for Ch {chapter_number}: Generated {len(patch_instructions)} patch instructions. "
                f"Original len: {len(original_text)}, Patched text len (after auto-application): {len(patched_text if patched_text else '')}."
            )
        else:
            logger.warning(f"Patch-based revision for Ch {chapter_number}: No valid patch instructions were generated. Will consider full rewrite if needed.")
    
    raw_patch_llm_outputs_combined_str = "".join(raw_patch_llm_outputs_combined_parts)

    final_revised_text: Optional[str] = None
    final_raw_llm_output: Optional[str] = None
    use_patched_text_as_final = False

    if patched_text is not None and patched_text != original_text:
        if len(patched_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH * 0.7:
            logger.warning(f"Patched draft for ch {chapter_number} is quite short ({len(patched_text)} chars). May still fall back to full rewrite if major issues remain.")
        sim_original_embedding, sim_patched_embedding = await asyncio.gather(
            llm_interface.async_get_embedding(original_text), llm_interface.async_get_embedding(patched_text)
        )
        if sim_original_embedding is not None and sim_patched_embedding is not None:
            similarity_score = utils.numpy_cosine_similarity(sim_original_embedding, sim_patched_embedding)
            logger.info(f"Patched text similarity with original: {similarity_score:.4f}")
            if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Patched text for ch {chapter_number} is very similar to original (Score: {similarity_score:.4f}). Patches might have been ineffective or minor. May consider full rewrite if problems persist.")
            else:
                use_patched_text_as_final = True
        else:
            logger.warning(f"Could not get embeddings for patched text similarity check of ch {chapter_number}. Assuming patched text is different enough if it exists and changed.")
            use_patched_text_as_final = True
        if use_patched_text_as_final:
            final_revised_text = patched_text
            final_raw_llm_output = raw_patch_llm_outputs_combined_str
            logger.info(f"Ch {chapter_number}: Tentatively using patched text as the revised version. Final decision after re-evaluation (if any problems necessitate full rewrite).")

    if not use_patched_text_as_final and evaluation_result.get("needs_revision"):
        if config.ENABLE_PATCH_BASED_REVISION and actionable_problems_for_patch_gen_check and patched_text is None:
             logger.warning(f"Patching attempted for Ch {chapter_number} but produced no usable text. Falling back to full rewrite.")
        elif not actionable_problems_for_patch_gen_check and evaluation_result.get("needs_revision"):
             logger.info(f"No problems suitable for patching in Ch {chapter_number}, but revision needed. Proceeding with full rewrite.")
        elif not config.ENABLE_PATCH_BASED_REVISION and evaluation_result.get("needs_revision"):
             logger.info(f"Patching disabled, and revision needed. Proceeding with full rewrite for Ch {chapter_number}.")

        logger.info(f"Proceeding with full chapter rewrite for Ch {chapter_number}.")
        max_original_snippet_tokens = config.MAX_CONTEXT_TOKENS // 3
        original_snippet = llm_interface.truncate_text_by_tokens(
            original_text, config.REVISION_MODEL, max_original_snippet_tokens,
            truncation_marker="\n... (original draft snippet truncated for brevity in rewrite prompt)"
        )
        plan_focus_section_full_rewrite_parts: List[str] = []
        plot_point_focus_full_rewrite, _ = _get_plot_point_info_from_agent(agent, chapter_number)
        max_plan_tokens_for_full_rewrite = config.MAX_CONTEXT_TOKENS // 2
        if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
            formatted_plan_fr = _get_formatted_scene_plan_from_agent_or_fallback(
                agent, chapter_plan, config.REVISION_MODEL, max_plan_tokens_for_full_rewrite
            )
            plan_focus_section_full_rewrite_parts.append(formatted_plan_fr)
            if "plan truncated" in formatted_plan_fr:
                 logger.warning(f"Scene plan token-truncated for Ch {chapter_number} full rewrite prompt.")
        else:
            plan_focus_section_full_rewrite_parts.append(f"**Original Chapter Focus (Target):**\n{plot_point_focus_full_rewrite or 'Not specified.'}\n")
        plan_focus_section_full_rewrite_str = "".join(plan_focus_section_full_rewrite_parts)

        length_issue_explicit_instruction_full_rewrite_parts: List[str] = []
        needs_expansion_from_problems = any(
            (p['issue_category'] == 'narrative_depth' and
             ("short" in p['problem_description'].lower() or "length" in p['problem_description'].lower() or
              "expand" in p['suggested_fix_focus'].lower() or "depth" in p['problem_description'].lower()))
            for p in problems_to_fix
        )
        needs_expansion_from_reasons = any(
            kw in revision_reason_str.lower() for kw in ["too short", "lacking in depth", "brief", "expand", "length", "narrative depth", "detail", "insufficient"]
        )
        if needs_expansion_from_problems or needs_expansion_from_reasons:
            length_issue_explicit_instruction_full_rewrite_parts.extend([
                f"\n**Specific Focus on Expansion:** A key critique involves insufficient length and/or narrative depth. ",
                f"Your revision MUST substantially expand the narrative by incorporating more descriptive details, character thoughts/introspection, dialogue, actions, and sensory information. ",
                f"Aim for a chapter length of at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters."
            ])
        length_issue_explicit_instruction_full_rewrite_str = "".join(length_issue_explicit_instruction_full_rewrite_parts)

        plot_outline_data_full_rewrite = _get_prop_from_agent(agent, 'plot_outline', {})
        protagonist_name_full_rewrite = _get_nested_prop_from_agent(agent, 'plot_outline', "protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        
        all_problem_descriptions_parts: List[str] = []
        if problems_to_fix:
            all_problem_descriptions_parts.append("**Detailed Issues to Address (from evaluation):**\n")
            for prob_idx, prob_item in enumerate(problems_to_fix):
                all_problem_descriptions_parts.extend([
                    f"  {prob_idx+1}. Category: {prob_item['issue_category']}",
                    f"     Description: {prob_item['problem_description']}",
                    f"     Quote Ref: \"{prob_item['quote_from_original_text'][:100].replace(chr(10),' ')}...\"",
                    f"     Fix Focus: {prob_item['suggested_fix_focus']}\n"
                ])
            all_problem_descriptions_parts.append("---\n")
        all_problem_descriptions_str = "".join(all_problem_descriptions_parts)

        # MODIFIED: Add note about de-duplication if is_from_flawed_source is True
        deduplication_note = ""
        if is_from_flawed_source: # This flag is passed from orchestrator, true if de-dup happened
            deduplication_note = (
                "\n**(Note: The 'Original Draft Snippet' below may have had repetitive content removed "
                "prior to evaluation, or other flaws were present. Ensure your rewrite is cohesive "
                "and addresses any resulting narrative gaps or inconsistencies.)**\n"
            )
        
        prompt_full_rewrite_lines = [
            "/no_think",
            f"You are an expert novelist rewriting Chapter {chapter_number} featuring protagonist {protagonist_name_full_rewrite}.",
            "**Critique/Reason(s) for Revision (MUST be addressed comprehensively):**",
            "--- FEEDBACK START ---",
            llm_interface.clean_model_response(revision_reason_str).strip(),
            "--- FEEDBACK END ---",
            all_problem_descriptions_str,
            deduplication_note, # ADDED
            length_issue_explicit_instruction_full_rewrite_str,
            plan_focus_section_full_rewrite_str,
            "**Hybrid Context from Previous Chapters (for consistency with established canon and narrative flow):**",
            "--- BEGIN HYBRID CONTEXT ---",
            hybrid_context_for_revision if hybrid_context_for_revision.strip() else "No previous context.",
            "--- END HYBRID CONTEXT ---",
            "**Original Draft Snippet (for reference of what went wrong - DO NOT COPY VERBATIM. Your goal is a fresh rewrite addressing all critique and aligning with the plan/focus):**",
            "--- BEGIN ORIGINAL DRAFT SNIPPET ---",
            original_snippet,
            "--- END ORIGINAL DRAFT SNIPPET ---",
            "",
            "**Revision Instructions:**",
            "1.  **ABSOLUTE PRIORITY:** Thoroughly address ALL issues listed in **Critique/Reason(s) for Revision** and **Detailed Issues to Address**. "
            "If the original text had content removed (e.g., due to de-duplication) or other flaws as noted, pay special attention to ensuring a smooth, coherent narrative flow and filling any gaps logically.", # MODIFIED to be more general
            "2.  **Rewrite the ENTIRE chapter.** Produce a fresh, coherent, and engaging narrative.",
            "3.  If a Detailed Scene Plan is provided in `plan_focus_section_full_rewrite_str`, follow it closely. Otherwise, align with the `Original Chapter Focus`.",
            "4.  Ensure seamless narrative flow with the **Hybrid Context**. Pay close attention to any `KEY RELIABLE KG FACTS` mentioned.",
            f"5.  Maintain the novel's established tone, style, and genre ('{_get_nested_prop_from_agent(agent, 'plot_outline', 'genre', 'story')}').",
            f"6.  Target a substantial chapter length, aiming for at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters of narrative text.",
            "7.  Output ONLY the rewritten chapter text.** Do NOT include \"Chapter X\" headers, titles, author commentary, or any meta-discussion.",
            "",
            f"--- BEGIN REVISED CHAPTER {chapter_number} TEXT ---"
        ]
        prompt_full_rewrite = "\n".join(prompt_full_rewrite_lines)

        logger.info(f"Calling LLM ({config.REVISION_MODEL}) for Ch {chapter_number} full rewrite. Min length: {config.MIN_ACCEPTABLE_DRAFT_LENGTH} chars.")
        revised_raw_llm_output_full, full_rewrite_usage = await llm_interface.async_call_llm(
            model_name=config.REVISION_MODEL,
            prompt=prompt_full_rewrite,
            temperature=config.TEMPERATURE_REVISION, 
            max_tokens=None,
            allow_fallback=True,
            stream_to_disk=True,
            frequency_penalty=config.FREQUENCY_PENALTY_REVISION,
            presence_penalty=config.PRESENCE_PENALTY_REVISION
        )
        _add_usage(full_rewrite_usage)
        if not revised_raw_llm_output_full:
            logger.error(f"Full rewrite LLM failed for ch {chapter_number} (returned empty). Original text will be kept if patching also failed.")
            return None, cumulative_usage_data if cumulative_usage_data["total_tokens"] > 0 else None
        final_revised_text = llm_interface.clean_model_response(revised_raw_llm_output_full)
        final_raw_llm_output = revised_raw_llm_output_full
        logger.info(f"Full rewrite for Ch {chapter_number} generated text of length {len(final_revised_text)}.")
    elif not use_patched_text_as_final and not evaluation_result.get("needs_revision"):
        logger.info(f"No revision performed for Ch {chapter_number} (original deemed acceptable or patching ineffective but not critical as no revision was strictly needed).")
        return None, cumulative_usage_data if cumulative_usage_data["total_tokens"] > 0 else None

    if not final_revised_text or len(final_revised_text) < 50 :
        logger.error(f"Revision process for ch {chapter_number} resulted in no usable content. Original len: {len(original_text)}")
        return None, cumulative_usage_data if cumulative_usage_data["total_tokens"] > 0 else None
    if len(final_revised_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        logger.warning(f"Final revised draft for ch {chapter_number} is short ({len(final_revised_text)} chars). Min target: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")
    if final_revised_text is not patched_text: # Only check similarity if it wasn't the patched text
        original_embedding_full_final, revised_embedding_full_final = await asyncio.gather(
            llm_interface.async_get_embedding(original_text), llm_interface.async_get_embedding(final_revised_text)
        )
        if original_embedding_full_final is not None and revised_embedding_full_final is not None:
            similarity_score_full_final = utils.numpy_cosine_similarity(original_embedding_full_final, revised_embedding_full_final)
            logger.info(f"Full rewrite similarity with original text (final check): {similarity_score_full_final:.4f}")
            if similarity_score_full_final >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(
                    f"Full rewrite for ch {chapter_number} is very similar to original (Score: {similarity_score_full_final:.4f}). "
                    f"The LLM may not have made sufficient changes despite instructions."
                )
        else:
            logger.warning(f"Could not get embeddings for full rewrite similarity check (final check) of ch {chapter_number}.")

    logger.info(f"Revision process for ch {chapter_number} produced a candidate text (Length: {len(final_revised_text)} chars).")
    return (final_revised_text, final_raw_llm_output), cumulative_usage_data if cumulative_usage_data["total_tokens"] > 0 else None
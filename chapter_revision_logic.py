# chapter_revision_logic.py
"""
Handles the revision of chapter drafts based on evaluation feedback for the SAGA system.
Supports both full rewrite and targeted patch-based revisions.
Context data for prompts is now formatted as plain text.
MODIFIED: Resolve _format_scene_plan_for_prompt import, adapt agent access.
MODIFIED: Enhanced patch generation logic, especially for "N/A - General Issue" and length expansion.
          Improved application of patches and skipping non-actionable ones.
MODIFIED: Implemented semantic targeting for patch application.
"""
import logging
import asyncio
import re
from typing import Tuple, Optional, List, Dict, Any

import config
import llm_interface
import drafting_agent 
import utils # For numpy_cosine_similarity and find_semantically_closest_segment
from type import SceneDetail, ProblemDetail, PatchInstruction, EvaluationResult
# No direct state_manager import needed here

logger = logging.getLogger(__name__)

# Attempt to get _format_scene_plan_for_prompt from drafting_agent
# This remains a bit of a workaround for direct import. Consider refactoring
# _format_scene_plan_for_prompt to a utility module if it's widely used.
try:
    _temp_drafting_agent_for_format = drafting_agent.DraftingAgent()
    _format_scene_plan_for_prompt_func = _temp_drafting_agent_for_format._format_scene_plan_for_prompt
    logger.debug("Successfully imported _format_scene_plan_for_prompt from drafting_agent.")
except ImportError:
    logger.error("Could not import _format_scene_plan_for_prompt from drafting_agent for chapter_revision_logic. Revision planning might be affected.")
    def _format_scene_plan_for_prompt_func(chapter_plan: List[SceneDetail], model_name_for_tokens: str, max_tokens_budget: int) -> str:
        logger.warning("_format_scene_plan_for_prompt_func is a fallback stub!")
        if not chapter_plan: return "Scene plan formatting unavailable or plan empty (stub)."
        plan_text_parts = []
        current_tokens = 0
        header = "**Detailed Scene Plan (Stubbed - MUST BE FOLLOWED CLOSELY):**\n"
        header_tokens = llm_interface.count_tokens(header, model_name_for_tokens)
        if header_tokens > max_tokens_budget: return "... (plan header too long for budget)"
        plan_text_parts.append(header)
        current_tokens += header_tokens

        for scene_idx, scene in enumerate(chapter_plan):
            scene_text = (
                f"Scene Number: {scene.get('scene_number', 'N/A')}\n"
                f"  Summary: {scene.get('summary', 'No summary')}\n"
                # Add other key fields if space allows and critical for stub
            )
            if scene_idx < len(chapter_plan) -1 : scene_text += ("-" * 10) + "\n"

            scene_tokens = llm_interface.count_tokens(scene_text, model_name_for_tokens)
            if current_tokens + scene_tokens > max_tokens_budget:
                plan_text_parts.append("... (plan truncated in prompt due to token limit)\n")
                break
            plan_text_parts.append(scene_text)
            current_tokens += scene_tokens
        return "".join(plan_text_parts)


def _get_prop_from_agent(agent: Any, key: str, default: Any = None) -> Any:
    # Accessing attributes from the NANA_Orchestrator instance
    return getattr(agent, key, default)

def _get_nested_prop_from_agent(agent: Any, primary_key: str, secondary_key: str, default: Any = None) -> Any:
    primary_data = _get_prop_from_agent(agent, primary_key, {})
    if isinstance(primary_data, dict):
        return primary_data.get(secondary_key, default)
    return default

def _get_plot_point_info_from_agent(agent: Any, chapter_number: int) -> Tuple[Optional[str], int]:
    plot_outline_data = _get_prop_from_agent(agent, 'plot_outline', {})
    plot_points = plot_outline_data.get("plot_points", [])
    if not isinstance(plot_points, list) or not plot_points: return None, -1
    if chapter_number <= 0: return None, -1
    plot_point_index = min(chapter_number - 1, len(plot_points) - 1)
    if 0 <= plot_point_index < len(plot_points):
        plot_point = plot_points[plot_point_index]
        return str(plot_point) if plot_point is not None else None, plot_point_index
    return None, -1


def _get_context_window(text: str, quote: str, window_size_chars: int) -> str:
    if not text: return ""
    if quote == "N/A - General Issue":
        # For general issues, provide start and end snippets of the whole text
        if len(text) <= window_size_chars:
            return text
        
        # Calculate how much to take from start and end
        # Aim for roughly half from each, but adjust if text is short
        start_snippet_len = min(window_size_chars // 2, len(text))
        remaining_chars_for_end = window_size_chars - start_snippet_len
        end_snippet_len = min(remaining_chars_for_end, len(text) - start_snippet_len)
        
        start_snippet = text[:start_snippet_len]
        # Ensure end_snippet_len is positive and doesn't cause negative indexing
        end_snippet = text[-end_snippet_len:] if end_snippet_len > 0 else ""
        
        if start_snippet_len + end_snippet_len < len(text):
             # Only add ellipsis if there's actually a gap
             return f"{start_snippet}\n...\n{end_snippet}"
        else: # The snippets cover the whole text or overlap
             return text

    try:
        # Find the first occurrence of the quote
        start_index = text.index(quote)
        end_index = start_index + len(quote)
        
        # Calculate context window boundaries
        half_window_chars = window_size_chars // 2
        context_start = max(0, start_index - half_window_chars)
        context_end = min(len(text), end_index + half_window_chars)
        
        # Adjust if the window is cut off at the beginning or end of the text
        if context_start == 0: # Window is at the start of the text
            context_end = min(len(text), window_size_chars)
        if context_end == len(text): # Window is at the end of the text
            context_start = max(0, len(text) - window_size_chars)
            
        prefix = "..." if context_start > 0 else ""
        suffix = "..." if context_end < len(text) else ""
        
        return f"{prefix}{text[context_start:context_end]}{suffix}"
    except ValueError:
        logger.warning(f"Quote for context window not found: '{quote[:50].replace(chr(10),' ')}...'. Providing general text snippet fallback.")
        # Fallback: return start of the text if quote not found
        return text[:window_size_chars] + ("..." if len(text) > window_size_chars else "")


async def _generate_single_patch_instruction_llm(
    agent: Any,
    original_chapter_text_snippet: str, # This is the CONTEXT shown to the LLM
    problem: ProblemDetail,
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> Tuple[Optional[PatchInstruction], Optional[Dict[str, int]]]:
    """
    Generates a single patch instruction using an LLM.
    Differentiates instructions and max_tokens for general expansion vs. specific corrective patches.
    Returns patch instruction and LLM usage data.
    """
    plan_focus_section = ""
    plot_point_focus, _ = _get_plot_point_info_from_agent(agent, chapter_number)
    max_plan_tokens_for_patch_prompt = config.MAX_CONTEXT_TOKENS // 4 # Reserve space

    if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
        plan_focus_section = _format_scene_plan_for_prompt_func(chapter_plan, config.PATCH_GENERATION_MODEL, max_plan_tokens_for_patch_prompt)
        if "plan truncated" in plan_focus_section:
             logger.warning(f"Scene plan token-truncated for Ch {chapter_number} patch generation prompt.")
    else:
        plan_focus_section = f"**Original Chapter Focus (Reference for overall chapter direction):**\n{plot_point_focus or 'Not specified.'}\n"

    # Determine if this is a general expansion task based on the problem
    is_general_expansion_task = False
    length_expansion_instruction_header = "" # Only used if expansion is explicitly needed

    if problem['issue_category'] == "narrative_depth" and \
       ("short" in problem['problem_description'].lower() or \
        "length" in problem['problem_description'].lower() or \
        "expand" in problem['suggested_fix_focus'].lower() or \
        "depth" in problem['problem_description'].lower() or \
        problem['quote_from_original'] == "N/A - General Issue"): # "N/A" quote for depth often means overall lack
        
        length_expansion_instruction_header = (
            f"\n**Critical: SUBSTANTIAL EXPANSION REQUIRED FOR THIS SEGMENT/PASSAGE.** "
            f"The 'replace_with' text MUST be significantly longer and more detailed. "
            f"Add descriptive details, character thoughts, dialogue, actions, and sensory information. "
        )
        if problem['quote_from_original'] == "N/A - General Issue":
            is_general_expansion_task = True
            length_expansion_instruction_header += (
                f"Since the original quote is 'N/A - General Issue', your 'replace_with' text should be a **new, expanded passage** "
                f"that addresses the 'Problem Description' and 'Suggested Fix Focus' within the broader 'Text Snippet' context. "
                f"This generated text is intended as a candidate for insertion or to inform a broader rewrite of a section."
            )
        else: # Specific quote, but problem still implies expansion for this segment
             length_expansion_instruction_header += (
                f"Aim for a notable increase in length and detail for the conceptual segment related to the original quote."
             )


    # Dynamically set instructions for replacement scope and max_tokens
    prompt_instruction_for_replacement_scope = ""
    max_patch_output_tokens = 0

    if is_general_expansion_task: # Handles "N/A - General Issue" for length/depth needing significant new text
        prompt_instruction_for_replacement_scope = (
            "    - The 'Original Quote Illustrating Problem' is \"N/A - General Issue\". Therefore, your `replace_with` text should be a **new, self-contained, and substantially expanded passage** "
            "that addresses the \"Problem Description\" and \"Suggested Fix Focus\" as guided by the `length_expansion_instruction_header`. "
            "This new passage is intended for potential insertion into the chapter, not to replace a specific quote."
        )
        # Allow a larger token budget for generating a new, substantial passage
        max_patch_output_tokens = config.MAX_GENERATION_TOKENS // 2 # Example: half of max generation
        max_patch_output_tokens = max(max_patch_output_tokens, 750) # Ensure a decent minimum for a new passage
        logger.info(f"Patch (Ch {chapter_number}, general expansion): Max output tokens set to {max_patch_output_tokens}.")
    else: # Specific quote, corrective patch (may or may not need minor expansion for the fix)
        prompt_instruction_for_replacement_scope = (
            "    - The 'Original Quote Illustrating Problem' is specific. Your `replace_with` text should be a revised version "
            "of the **entire conceptual segment or paragraph** within the 'ORIGINAL TEXT SNIPPET' that best corresponds to that quote. Your output will replace that whole segment.\n"
            "    - **Crucially, for this specific fix, your replacement text should primarily focus on correcting the identified issue. "
            "If `length_expansion_instruction_header` is present, apply its guidance to *this specific segment*. Otherwise, aim for a length comparable to the original segment, plus necessary additions for the fix. "
            "Avoid excessive unrelated expansion beyond the scope of the problem for this segment.**"
        )
        # Estimate original snippet tokens and allow some expansion (e.g., 1.5x to 2.5x depending on need)
        original_snippet_tokens = llm_interface.count_tokens(original_chapter_text_snippet, config.PATCH_GENERATION_MODEL)
        expansion_factor = 2.5 if length_expansion_instruction_header else 1.5 # More expansion if explicitly asked for this segment
        max_patch_output_tokens = int(original_snippet_tokens * expansion_factor) 
        max_patch_output_tokens = min(max_patch_output_tokens, config.MAX_GENERATION_TOKENS // 3) # Cap it
        max_patch_output_tokens = max(max_patch_output_tokens, 200) # Ensure at least a decent minimum for a fix
        logger.info(f"Patch (Ch {chapter_number}, specific fix): Original snippet tokens: {original_snippet_tokens}. Max output tokens set to {max_patch_output_tokens}.")


    plot_outline_data = _get_prop_from_agent(agent, 'plot_outline', {})
    protagonist_name = _get_nested_prop_from_agent(agent, 'plot_outline', 'protagonist_name', config.DEFAULT_PROTAGONIST_NAME) # Corrected access

    prompt = f"""/no_think
You are a surgical revision expert generating replacement text for Chapter {chapter_number} of a novel titled "{_get_nested_prop_from_agent(agent, 'plot_outline', 'title', 'Untitled Novel')}" about {protagonist_name}.
**Novel Context:**
  - Genre: {_get_nested_prop_from_agent(agent, 'plot_outline', 'genre', 'N/A')}
  - Theme: {_get_nested_prop_from_agent(agent, 'plot_outline', 'theme', 'N/A')}
  - Protagonist: {protagonist_name} ({_get_nested_prop_from_agent(agent, 'plot_outline', 'character_arc', 'N/A')})

{plan_focus_section}
**Hybrid Context from Previous Chapters (for consistency with established canon and narrative flow):**
--- BEGIN HYBRID CONTEXT ---
{hybrid_context_for_revision if hybrid_context_for_revision.strip() else "No previous context."}
--- END HYBRID CONTEXT ---

**Specific Problem to Address in the Chapter:**
  - Issue Category: {problem['issue_category']}
  - Problem Description: {problem['problem_description']}
  - Original Quote Illustrating Problem: "{problem['quote_from_original']}"
  - Suggested Fix Focus: {problem['suggested_fix_focus']}

**Text Snippet from Original Chapter (This is the broader context around the problem. If the quote is 'N/A - General Issue', this is general chapter context to inform your new passage):**
--- BEGIN ORIGINAL TEXT SNIPPET ---
{original_chapter_text_snippet}
--- END ORIGINAL TEXT SNIPPET ---
{length_expansion_instruction_header} 
**Instructions for Generating Replacement Text:**
1.  Focus EXCLUSIVELY on the problem described, particularly relating to the conceptual area highlighted by: `{problem['quote_from_original']}` within the 'ORIGINAL TEXT SNIPPET'.
2.  Generate a `replace_with` text according to the following:
{prompt_instruction_for_replacement_scope}
3.  The `replace_with` text MUST address the "Problem Description" and "Suggested Fix Focus".
4.  Maintain the novel's style, tone, and consistency with all provided context (Novel Context, Plan, Hybrid Context).
5.  If `length_expansion_instruction_header` is present, ensure substantial expansion as guided for the targeted segment or new passage.
6.  **Output ONLY the `replace_with` text.** Do NOT include JSON, markdown, explanations, or any "Replace with:" prefixes. Just the raw text intended for replacement/insertion.

--- BEGIN REPLACE_WITH TEXT (for the segment related to "{problem['quote_from_original']}" or as a new passage if quote is "N/A - General Issue") ---
"""
    logger.info(f"Calling LLM ({config.PATCH_GENERATION_MODEL}) for patch in Ch {chapter_number}. Problem: '{problem['problem_description'][:60].replace(chr(10),' ')}...' Quote: '{problem['quote_from_original'][:50].replace(chr(10),' ')}...' Max Output Tokens: {max_patch_output_tokens}")
    
    replace_with_text_raw, usage_data = await llm_interface.async_call_llm(
        model_name=config.PATCH_GENERATION_MODEL, 
        prompt=prompt, 
        temperature=0.6, 
        max_tokens=max_patch_output_tokens, 
        allow_fallback=True, 
        stream_to_disk=False 
    )

    if not replace_with_text_raw:
        logger.error(f"Patch LLM returned no content for Ch {chapter_number} problem: {problem['problem_description']}")
        return None, usage_data

    replace_with_text_cleaned = llm_interface.clean_model_response(replace_with_text_raw)
    if not replace_with_text_cleaned.strip():
        logger.warning(f"Patch LLM returned empty after cleaning for Ch {chapter_number} problem: {problem['problem_description']}")
        return None, usage_data

    # Qualitative check for expansion if it was instructed
    if length_expansion_instruction_header:
        if not is_general_expansion_task: # Specific quote that needed segment expansion
            # This check is tricky as original_chapter_text_snippet is a WINDOW, not the exact segment being replaced.
            # A more accurate check would need the length of the actual segment identified by semantic search.
            # For now, a simple check against the snippet length if it's not too short.
            if len(original_chapter_text_snippet) > 100 and len(replace_with_text_cleaned) < len(original_chapter_text_snippet) * 1.2:
                logger.warning(
                    f"Patch for Ch {chapter_number} (specific quote, segment expansion requested) output length ({len(replace_with_text_cleaned)}) "
                    f"is not significantly larger than context snippet ({len(original_chapter_text_snippet)}). "
                    f"Problem: {problem['problem_description'][:60]}"
                )
        elif is_general_expansion_task and len(replace_with_text_cleaned) < 500: # Arbitrary threshold for a "substantial new passage"
             logger.warning(
                    f"Patch for Ch {chapter_number} ('N/A - General Issue' expansion) produced a relatively short new passage (len: {len(replace_with_text_cleaned)}). "
                    f"Problem: {problem['problem_description'][:60]}"
                )

    patch_instruction: PatchInstruction = {
        "search_text": problem['quote_from_original'], # This is still used by _apply_patches_to_text to find the target
        "replace_with": replace_with_text_cleaned,
        "original_quote_ref": problem['quote_from_original'], 
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
    """Generates patch instructions. Returns list of instructions and summed LLM usage."""
    patch_instructions: List[PatchInstruction] = []
    total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Filter problems:
    # Include problems with specific quotes OR narrative_depth issues that imply expansion,
    # even if their quote is "N/A - General Issue".
    actionable_problems_for_patch_generation = []
    for p in problems_to_fix:
        is_specific_quote = p["quote_from_original"] != "N/A - General Issue" and p["quote_from_original"].strip()
        is_expansion_depth_issue = (
            p["issue_category"] == "narrative_depth" and
            ("short" in p['problem_description'].lower() or
             "length" in p['problem_description'].lower() or
             "expand" in p['suggested_fix_focus'].lower() or
             "depth" in p['problem_description'].lower())
        )
        if is_specific_quote or (p["quote_from_original"] == "N/A - General Issue" and is_expansion_depth_issue):
            actionable_problems_for_patch_generation.append(p)
        else:
            logger.info(f"Skipping patch generation for Ch {chapter_number} problem (not specific quote & not expansion depth): '{p['problem_description'][:60]}'")

    # Limit the number of patches to generate per cycle
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
        # _get_context_window provides the snippet around the problem.quote_from_original
        # or general chapter context if quote is "N/A - General Issue".
        context_snippet = _get_context_window(original_text, problem["quote_from_original"], config.MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW)
        
        task = _generate_single_patch_instruction_llm(
            agent, context_snippet, problem, chapter_number, hybrid_context_for_revision, chapter_plan
        )
        patch_generation_tasks.append(task)

    if not patch_generation_tasks: # Should not happen if problems_to_process was non-empty
        logger.info(f"No patch generation tasks created for Ch {chapter_number}, though problems were identified.")
        return [], None

    results = await asyncio.gather(*patch_generation_tasks, return_exceptions=True)
    
    for i, res_or_exc in enumerate(results):
        # problem_ref's index needs to align with problems_to_process
        problem_ref = problems_to_process[i] 
        if isinstance(res_or_exc, Exception):
            logger.error(f"Error generating patch for Ch {chapter_number} problem '{problem_ref['problem_description'][:50].replace(chr(10),' ')}': {res_or_exc}", exc_info=res_or_exc)
        elif res_or_exc is not None:
            patch_instr, usage = res_or_exc
            if patch_instr:
                patch_instructions.append(patch_instr)
            if usage: # Aggregate usage data
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
    Uses semantic search to locate the target segment if the patch's search_text isn't "N/A - General Issue".
    Patches with "N/A - General Issue" as search_text are currently logged and their `replace_with` content
    is not automatically inserted, as their placement is ambiguous.
    """
    if not patch_instructions:
        return original_text

    # Patches that can be applied by finding a segment
    applicable_patches_for_replacement: List[PatchInstruction] = []
    # Patches that generated new text for general issues (e.g., length expansion without specific quote)
    new_content_patches: List[PatchInstruction] = []

    for p in patch_instructions:
        if p["search_text"] != "N/A - General Issue" and p["search_text"].strip():
            applicable_patches_for_replacement.append(p)
        elif p["search_text"] == "N/A - General Issue" and p["replace_with"].strip():
            new_content_patches.append(p)
            logger.info(f"Patch for 'N/A - General Issue' generated new content: '{p['replace_with'][:100].replace(chr(10),' ')}...'. "
                        "This content will NOT be automatically inserted by this function. Orchestrator may need to handle its placement or use it to inform full rewrite.")
        else:
            logger.debug(f"Skipping patch with empty search_text or empty replace_with: {p['original_quote_ref']}")


    if not applicable_patches_for_replacement:
        logger.info("No applicable patches with specific search_text to apply for segment replacement.")
        # If there's new_content_patches, the orchestrator might decide what to do.
        # For now, this function only modifies based on specific search_text.
        return original_text
        
    # Store (original_start, original_end, replacement_text)
    # These will be character indices into the *original_text*
    replacements: List[Tuple[int, int, str]] = []
    
    failed_patches_search_text_not_found = 0

    # Process patches one by one to find their target segments
    for patch_idx, patch in enumerate(applicable_patches_for_replacement):
        target_start_char, target_end_char = -1, -1
        
        # Tier 1: Try exact match for the problem's original_quote_ref
        # This is a quick check if the evaluator's quote is still perfectly valid.
        # However, the patch LLM was instructed to replace the *segment containing* this quote.
        exact_quote_match_start = -1
        try:
            exact_quote_match_start = original_text.index(patch["original_quote_ref"])
        except ValueError:
            pass # Quote not found exactly

        # Tier 2: Semantic search for the paragraph most related to the problem's original_quote_ref.
        # This is the primary method for finding the segment to replace.
        # The patch['search_text'] is the original problem['quote_from_original'].
        semantic_query = patch["search_text"] 
        
        # We use "paragraph" segmentation because the patch LLM was instructed to revise a segment/paragraph.
        semantic_match_info = await utils.find_semantically_closest_segment(
            original_text, 
            semantic_query, 
            segment_type="paragraph", 
            min_similarity_threshold=0.60 # Threshold for identifying the target paragraph
        )
        
        if semantic_match_info:
            target_start_char, target_end_char, score = semantic_match_info
            # Log which method found the target segment
            log_method = "Semantic search (paragraph)"
            if exact_quote_match_start != -1:
                # If both exact quote and semantic segment are found, ensure semantic segment contains the exact quote
                if not (target_start_char <= exact_quote_match_start < target_end_char):
                    logger.warning(f"Patch {patch_idx+1}: Exact quote '{patch['original_quote_ref'][:50]}' found, but semantically closest paragraph "
                                   f"({target_start_char}-{target_end_char}) does not contain it. Prioritizing semantic segment for replacement.")
                else:
                    log_method += " (consistent with exact quote location)"

            logger.info(f"Patch {patch_idx+1}: {log_method} identified target segment for '{patch['original_quote_ref'][:50].replace(chr(10),' ')}...' "
                        f"at original chars {target_start_char}-{target_end_char} (Score: {score:.2f}).")
        elif exact_quote_match_start != -1:
            # Fallback: If semantic search fails but exact quote was found, replace only the exact quote.
            # This is less ideal as the LLM was told to revise a segment.
            target_start_char = exact_quote_match_start
            target_end_char = exact_quote_match_start + len(patch["original_quote_ref"])
            logger.warning(f"Patch {patch_idx+1}: Semantic search failed for '{patch['original_quote_ref'][:50].replace(chr(10),' ')}...'. "
                           f"Falling back to replacing ONLY the exact quote at {target_start_char}-{target_end_char}. "
                           f"This might be suboptimal as LLM revised a broader segment.")
        else:
            logger.warning(f"Patching (Patch {patch_idx+1}): `search_text` (original quote) '{patch['search_text'][:100].replace(chr(10),' ')}...' "
                           f"not found by exact or semantic paragraph search. Skipping this patch.")
            failed_patches_search_text_not_found += 1
            continue
        
        # Check for overlapping patches based on the identified target_start_char and target_end_char
        # current_segment_tuple = (target_start_char, target_end_char) # Not used directly
        has_overlap = False
        for r_start, r_end, _ in replacements:
            if max(target_start_char, r_start) < min(target_end_char, r_end): # Check for overlap
                has_overlap = True
                logger.warning(f"Patch {patch_idx+1} for '{patch['original_quote_ref'][:50].replace(chr(10),' ')}...' (targets segment {target_start_char}-{target_end_char}) "
                               f"overlaps with a previously determined patch for segment {r_start}-{r_end}. Skipping this patch to avoid conflict.")
                break
        
        if not has_overlap:
            replacements.append((target_start_char, target_end_char, patch["replace_with"]))
        
    if failed_patches_search_text_not_found > 0:
         logger.warning(f"{failed_patches_search_text_not_found}/{len(applicable_patches_for_replacement)} applicable patches failed (target segment not found).")

    if not replacements:
        logger.info("No patches could be confidently mapped to text segments for replacement.")
        return original_text

    # Sort replacements by start index in reverse order to apply from end to start
    replacements.sort(key=lambda x: x[0], reverse=True)

    current_text_list = list(original_text)
    applied_count = 0
    for start_index, end_index, replace_with_text in replacements:
        current_text_list[start_index:end_index] = list(replace_with_text)
        applied_count += 1
        logger.info(f"Applied patch: Replaced original segment from char {start_index} to {end_index} "
                    f"(length {end_index-start_index}) with new text (length {len(replace_with_text)}). "
                    f"Replacement snippet: '{replace_with_text[:60].replace(chr(10),' ')}...'")

    logger.info(f"Applied {applied_count} out of {len(applicable_patches_for_replacement)} patches that targeted specific segments.")
    return "".join(current_text_list)


async def revise_chapter_draft_logic(
    agent: Any, 
    original_text: str,
    chapter_number: int,
    evaluation_result: EvaluationResult,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> Tuple[Optional[Tuple[str, str]], Optional[Dict[str, int]]]:
    """
    Main logic for revising a chapter draft.
    Returns (revised_text, raw_llm_output_tuple) and summed LLM usage data.
    """
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
    elif not problems_to_fix: # No problems and not needs_revision
        logger.info(f"No specific problems found for ch {chapter_number}, and not marked for revision. No revision performed.")
        return None, None # No revision needed

    revision_reason_str_list = evaluation_result.get("reasons", [])
    revision_reason_str = "\n- ".join(revision_reason_str_list) if revision_reason_str_list else "General unspecified issues."
    logger.info(f"Attempting revision for chapter {chapter_number}. Reason(s):\n- {revision_reason_str}")

    patched_text: Optional[str] = None
    raw_patch_llm_outputs_combined: str = "" # To store combined raw outputs if multiple patches
    
    # Determine if any problems are suitable for patch generation (includes specific quotes OR general expansion depth issues)
    actionable_problems_for_patch_gen_check = [
        p for p in problems_to_fix if 
        (p["quote_from_original"] != "N/A - General Issue" and p["quote_from_original"].strip()) or
        (p["quote_from_original"] == "N/A - General Issue" and p["issue_category"] == "narrative_depth" and
         ("short" in p['problem_description'].lower() or "length" in p['problem_description'].lower() or
          "expand" in p['suggested_fix_focus'].lower() or "depth" in p['problem_description'].lower()))
    ]

    if config.ENABLE_PATCH_BASED_REVISION and actionable_problems_for_patch_gen_check:
        logger.info(f"Attempting patch-based revision for Ch {chapter_number} with {len(actionable_problems_for_patch_gen_check)} potentially actionable problem(s).")
        
        # _generate_patch_instructions_logic itself filters for what it can generate
        patch_instructions, patch_usage = await _generate_patch_instructions_logic(
            agent, original_text, problems_to_fix, # Pass all problems, let it filter
            chapter_number, hybrid_context_for_revision, chapter_plan
        )
        _add_usage(patch_usage)
        
        if patch_instructions:
            # _apply_patches_to_text is now async and handles filtering for applicable patches internally
            patched_text = await _apply_patches_to_text(original_text, patch_instructions)
            # The raw_patch_llm_outputs_combined could be a concatenation of individual patch LLM raw outputs if needed for debugging
            # For now, a summary message is sufficient.
            raw_patch_llm_outputs_combined = (
                f"Chapter revised using {len(patch_instructions)} generated patch instructions. "
                f"(Note: Not all generated patches may have been auto-applied if they were for 'N/A - General Issue' or target segment not found.)\n"
            )
            logger.info(
                f"Patch process for Ch {chapter_number}: Generated {len(patch_instructions)} patch instructions. "
                f"Original len: {len(original_text)}, Patched text len (after auto-application): {len(patched_text if patched_text else '')}."
            )
        else:
            logger.warning(f"Patch-based revision for Ch {chapter_number}: No valid patch instructions were generated. Will consider full rewrite if needed.")
    
    final_revised_text: Optional[str] = None
    final_raw_llm_output: Optional[str] = None
    use_patched_text_as_final = False

    if patched_text is not None and patched_text != original_text: # Check if patching actually changed the text
        if len(patched_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH * 0.7: # More lenient if it's an improvement
            logger.warning(f"Patched draft for ch {chapter_number} is quite short ({len(patched_text)} chars). May still fall back to full rewrite if major issues remain.")
        
        # Semantic similarity check between original and patched
        sim_original_embedding, sim_patched_embedding = await asyncio.gather(
            llm_interface.async_get_embedding(original_text), llm_interface.async_get_embedding(patched_text)
        )
        if sim_original_embedding is not None and sim_patched_embedding is not None:
            similarity_score = utils.numpy_cosine_similarity(sim_original_embedding, sim_patched_embedding)
            logger.info(f"Patched text similarity with original: {similarity_score:.4f}")
            if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE: # If too similar
                logger.warning(f"Patched text for ch {chapter_number} is very similar to original (Score: {similarity_score:.4f}). Patches might have been ineffective or minor. May consider full rewrite if problems persist.")
                # We don't set use_patched_text_as_final = False yet, depends on if *any* problems remain for a full rewrite
            else: # Different enough
                use_patched_text_as_final = True 
        else: # Could not get embeddings
            logger.warning(f"Could not get embeddings for patched text similarity check of ch {chapter_number}. Assuming patched text is different enough if it exists and changed.")
            use_patched_text_as_final = True
        
        if use_patched_text_as_final:
            final_revised_text = patched_text
            final_raw_llm_output = raw_patch_llm_outputs_combined
            logger.info(f"Ch {chapter_number}: Tentatively using patched text as the revised version. Final decision after re-evaluation (if any problems necessitate full rewrite).")
            # Note: The decision to *actually* use patched_text happens in the orchestrator's loop after re-evaluation.
            # This function just provides the patched text as a candidate. If *major* issues like overall length persist,
            # the orchestrator might still decide a full rewrite is needed even if patching made some changes.
            # For this function, if patching produces *something*, we return it.
            # The "fallback to full rewrite" below handles cases where patching was skipped or produced nothing.

    # If patching was skipped, or produced no usable text, OR if evaluation *still* demands revision after patching (handled by orchestrator),
    # then a full rewrite might be attempted. This function handles the "patching produced nothing" or "patching skipped" case for full rewrite.
    if not use_patched_text_as_final and evaluation_result.get("needs_revision"):
        # Log why we are proceeding to full rewrite
        if config.ENABLE_PATCH_BASED_REVISION and actionable_problems_for_patch_gen_check and patched_text is None:
             logger.warning(f"Patching attempted for Ch {chapter_number} but produced no usable text. Falling back to full rewrite.")
        elif not actionable_problems_for_patch_gen_check and evaluation_result.get("needs_revision"):
             logger.info(f"No problems suitable for patching in Ch {chapter_number}, but revision needed. Proceeding with full rewrite.")
        elif not config.ENABLE_PATCH_BASED_REVISION and evaluation_result.get("needs_revision"):
             logger.info(f"Patching disabled, and revision needed. Proceeding with full rewrite for Ch {chapter_number}.")
        # Else: Patching might have occurred but wasn't deemed "final" (e.g., too similar or other logic). If `needs_revision` is still true, proceed.
        
        # --- Full Rewrite Logic ---
        logger.info(f"Proceeding with full chapter rewrite for Ch {chapter_number}.")
        max_original_snippet_tokens = config.MAX_CONTEXT_TOKENS // 3 # Allow more of original for full rewrite context
        original_snippet = llm_interface.truncate_text_by_tokens(
            original_text, config.REVISION_MODEL, max_original_snippet_tokens,
            truncation_marker="\n... (original draft snippet truncated for brevity in rewrite prompt)"
        )
        
        plan_focus_section_full_rewrite = ""
        plot_point_focus_full_rewrite, _ = _get_plot_point_info_from_agent(agent, chapter_number)
        max_plan_tokens_for_full_rewrite = config.MAX_CONTEXT_TOKENS // 3
        if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
            plan_focus_section_full_rewrite = _format_scene_plan_for_prompt_func(chapter_plan, config.REVISION_MODEL, max_plan_tokens_for_full_rewrite)
            if "plan truncated" in plan_focus_section_full_rewrite:
                 logger.warning(f"Scene plan token-truncated for Ch {chapter_number} full rewrite prompt.")
        else:
            plan_focus_section_full_rewrite = f"**Original Chapter Focus (Target):**\n{plot_point_focus_full_rewrite or 'Not specified.'}\n"

        length_issue_explicit_instruction_full_rewrite = ""
        # Check if any problem or reason implies a length/depth issue
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
            length_issue_explicit_instruction_full_rewrite = (
                f"\n**Specific Focus on Expansion:** A key critique involves insufficient length and/or narrative depth. "
                f"Your revision MUST substantially expand the narrative by incorporating more descriptive details, character thoughts/introspection, dialogue, actions, and sensory information. "
                f"Aim for a chapter length of at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters."
            )
        
        plot_outline_data_full_rewrite = _get_prop_from_agent(agent, 'plot_outline', {})
        protagonist_name_full_rewrite = _get_nested_prop_from_agent(agent, 'plot_outline', "protagonist_name", config.DEFAULT_PROTAGONIST_NAME) # Corrected access
        
        all_problem_descriptions_str = ""
        if problems_to_fix: # Ensure there are problems to list
            all_problem_descriptions_str = "**Detailed Issues to Address (from evaluation):**\n"
            for prob_idx, prob_item in enumerate(problems_to_fix):
                all_problem_descriptions_str += (
                    f"  {prob_idx+1}. Category: {prob_item['issue_category']}\n"
                    f"     Description: {prob_item['problem_description']}\n"
                    f"     Quote Ref: \"{prob_item['quote_from_original'][:100].replace(chr(10),' ')}...\"\n" # Show quote for context
                    f"     Fix Focus: {prob_item['suggested_fix_focus']}\n"
                )
            all_problem_descriptions_str += "---\n"

        prompt_full_rewrite = f"""/no_think
You are an expert novelist rewriting Chapter {chapter_number} featuring protagonist {protagonist_name_full_rewrite}.
**Critique/Reason(s) for Revision (MUST be addressed comprehensively):**
--- FEEDBACK START ---
{llm_interface.clean_model_response(revision_reason_str).strip()}
--- FEEDBACK END ---
{all_problem_descriptions_str}
{length_issue_explicit_instruction_full_rewrite}
{plan_focus_section_full_rewrite}
**Hybrid Context from Previous Chapters (Semantic Context & KG Facts - for consistency):**
--- BEGIN HYBRID CONTEXT ---
{hybrid_context_for_revision if hybrid_context_for_revision.strip() else "No previous context."}
--- END HYBRID CONTEXT ---
**Original Draft Snippet (for reference of what went wrong - DO NOT COPY VERBATIM. Your goal is a fresh rewrite addressing all critique and aligning with the plan/focus):**
--- BEGIN ORIGINAL DRAFT SNIPPET ---
{original_snippet}
--- END ORIGINAL DRAFT SNIPPET ---

**Revision Instructions:**
1.  **ABSOLUTE PRIORITY:** Thoroughly address ALL issues listed in **Critique/Reason(s) for Revision** and **Detailed Issues to Address**.
2.  **Rewrite the ENTIRE chapter.** Produce a fresh, coherent, and engaging narrative.
3.  If a Detailed Scene Plan is provided in `plan_focus_section_full_rewrite`, follow it closely. Otherwise, align with the `Original Chapter Focus`.
4.  Ensure seamless narrative flow with the **Hybrid Context**. Pay close attention to any `KEY RELIABLE KG FACTS` mentioned.
5.  Maintain the novel's established tone, style, and genre ('{_get_nested_prop_from_agent(agent, 'plot_outline', 'genre', 'story')}').
6.  Target a substantial chapter length, aiming for at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters of narrative text.
7.  **Output ONLY the rewritten chapter text.** Do NOT include "Chapter X" headers, titles, author commentary, or any meta-discussion.

--- BEGIN REVISED CHAPTER {chapter_number} TEXT ---
"""
        logger.info(f"Calling LLM ({config.REVISION_MODEL}) for Ch {chapter_number} full rewrite. Min length: {config.MIN_ACCEPTABLE_DRAFT_LENGTH} chars.")
        revised_raw_llm_output_full, full_rewrite_usage = await llm_interface.async_call_llm(
            model_name=config.REVISION_MODEL, 
            prompt=prompt_full_rewrite, 
            temperature=0.6, # Standard creative temp
            max_tokens=None, # Allow LLM to generate full chapter
            allow_fallback=True, 
            stream_to_disk=True # Full chapter rewrites can be long
        )
        _add_usage(full_rewrite_usage)

        if not revised_raw_llm_output_full:
            logger.error(f"Full rewrite LLM failed for ch {chapter_number} (returned empty). Original text will be kept if patching also failed.")
            # Return original text and its raw output if available from a previous stage, or None
            # For now, returning None as this indicates a failure in this revision step.
            return None, cumulative_usage_data if cumulative_usage_data["total_tokens"] > 0 else None
        
        final_revised_text = llm_interface.clean_model_response(revised_raw_llm_output_full)
        final_raw_llm_output = revised_raw_llm_output_full
        logger.info(f"Full rewrite for Ch {chapter_number} generated text of length {len(final_revised_text)}.")

    elif not use_patched_text_as_final and not evaluation_result.get("needs_revision"):
        # This case means: patching was not effective (e.g. too similar or no patches applied)
        # BUT the evaluation_result says no revision is needed anyway.
        # This can happen if the initial draft was "good enough" and patching minor things didn't change it much.
        logger.info(f"No revision performed for Ch {chapter_number} (original deemed acceptable or patching ineffective but not critical as no revision was strictly needed).")
        return None, cumulative_usage_data if cumulative_usage_data["total_tokens"] > 0 else None


    # Final checks on the chosen revised text (either from patch or full rewrite)
    if not final_revised_text or len(final_revised_text) < 50 : # Arbitrary short threshold
        logger.error(f"Revision process for ch {chapter_number} resulted in no usable content. Original len: {len(original_text)}")
        return None, cumulative_usage_data if cumulative_usage_data["total_tokens"] > 0 else None

    if len(final_revised_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        logger.warning(f"Final revised draft for ch {chapter_number} is short ({len(final_revised_text)} chars). Min target: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")

    # If a full rewrite occurred, do a final similarity check
    if final_revised_text is not patched_text: # Indicates a full rewrite was the source of final_revised_text
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
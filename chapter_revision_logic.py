# chapter_revision_logic.py
"""
Handles the revision of chapter drafts based on evaluation feedback for the SAGA system.
Supports both full rewrite and targeted patch-based revisions.
Context data for prompts is now formatted as plain text.
MODIFIED: Resolve _format_scene_plan_for_prompt import, adapt agent access.
"""
import logging
import asyncio
import re
from typing import Tuple, Optional, List, Dict, Any

import config
import llm_interface
import drafting_agent
import utils
from type import SceneDetail, ProblemDetail, PatchInstruction, EvaluationResult

logger = logging.getLogger(__name__)

# Import the plain text formatter for scene plan
# If chapter_drafting_logic.py is removed, this function needs a new home or to be passed.
# For now, let's assume it's moved to parsing_utils.py or similar if chapter_drafting_logic.py is deleted.
# Option 1: Keep import as is for now. Orchestrator must ensure chapter_drafting_logic.py (or its replacement) is available.
# Option 2: Define it locally or import from a central util.
# For this exercise, I'll assume the orchestrator might pass it, or it's moved.
# Let's try to import it from drafting_agent as a temporary measure, assuming that file exists.
# This is not ideal for standalone use of this module but works for the current refactor.
try:
    from drafting_agent import DraftingAgent # To access its _format_scene_plan_for_prompt
    _temp_drafting_agent_for_format = DraftingAgent()
    _format_scene_plan_for_prompt_func = _temp_drafting_agent_for_format._format_scene_plan_for_prompt
except ImportError:
    logger.error("Could not import _format_scene_plan_for_prompt from drafting_agent for chapter_revision_logic. Revision planning might be affected.")
    def _format_scene_plan_for_prompt_func(chapter_plan: List[SceneDetail], model_name_for_tokens: str, max_tokens_budget: int) -> str:
        logger.warning("_format_scene_plan_for_prompt_func is a fallback stub!")
        return "Scene plan formatting unavailable."

def _get_prop_from_agent(agent: Any, key: str, default: Any = None) -> Any:
    """Helper to get a property from an agent-like object (orchestrator)."""
    return getattr(agent, key, default)

def _get_nested_prop_from_agent(agent: Any, primary_key: str, secondary_key: str, default: Any = None) -> Any:
    """Helper to get a nested property from an agent-like object."""
    primary_data = _get_prop_from_agent(agent, primary_key, {})
    if isinstance(primary_data, dict):
        return primary_data.get(secondary_key, default)
    return default

def _get_plot_point_info_from_agent(agent: Any, chapter_number: int) -> Tuple[Optional[str], int]:
    """Replicates the _get_plot_point_info from NovelWriterAgent/NANA_Orchestrator."""
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
    if quote == "N/A - General Issue":
        return text[:window_size_chars//2] + "\n...\n" + text[-window_size_chars//2:]
    try:
        start_index = text.index(quote)
        end_index = start_index + len(quote)
        context_start = max(0, start_index - window_size_chars // 2)
        context_end = min(len(text), end_index + window_size_chars // 2)
        prefix = "..." if context_start > 0 else ""
        suffix = "..." if context_end < len(text) else ""
        return f"{prefix}{text[context_start:context_end]}{suffix}"
    except ValueError:
        logger.warning(f"Quote for context window not found: '{quote[:50]}...'. Full text snippet fallback.")
        return text[:window_size_chars] + "..." if len(text) > window_size_chars else text


async def _generate_single_patch_instruction_llm(
    agent: Any, # NANA_Orchestrator instance
    original_chapter_text_snippet: str,
    problem: ProblemDetail,
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> Optional[PatchInstruction]:
    plan_focus_section = ""
    plot_point_focus, _ = _get_plot_point_info_from_agent(agent, chapter_number)
    max_plan_tokens_for_patch_prompt = config.MAX_CONTEXT_TOKENS // 4

    if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
        plan_focus_section = _format_scene_plan_for_prompt_func(chapter_plan, config.PATCH_GENERATION_MODEL, max_plan_tokens_for_patch_prompt)
        if "plan truncated" in plan_focus_section:
             logger.warning(f"Scene plan token-truncated for Ch {chapter_number} patch generation.")
    else:
        plan_focus_section = f"**Original Chapter Focus (Reference):**\n{plot_point_focus or 'Not specified.'}\n"

    length_expansion_instruction = ""
    if problem['issue_category'] == "narrative_depth" and \
       ("short" in problem['problem_description'].lower() or \
        "length" in problem['problem_description'].lower() or \
        "expand" in problem['suggested_fix_focus'].lower() or \
        "depth" in problem['problem_description'].lower() or \
        problem['quote_from_original'] == "N/A - General Issue"):
        length_expansion_instruction = (
            f"\n**Critical: SUBSTANTIAL EXPANSION REQUIRED.** "
            f"The 'replace_with' text MUST be significantly longer and more detailed. "
            f"Add descriptive details, thoughts, dialogue, actions. Aim for 50-100%+ increase for this segment."
        )

    plot_outline_data = _get_prop_from_agent(agent, 'plot_outline', {})
    prompt = f"""/no_think
You are a surgical revision expert generating replacement text for Chapter {chapter_number}.
**Novel Context:**
  - Genre: {_get_prop_from_agent(plot_outline_data, 'genre', 'N/A')}
  - Theme: {_get_prop_from_agent(plot_outline_data, 'theme', 'N/A')}
  - Protagonist: {_get_prop_from_agent(plot_outline_data, 'protagonist_name', 'N/A')} ({_get_prop_from_agent(plot_outline_data, 'character_arc', 'N/A')})

{plan_focus_section}
**Hybrid Context from Previous Chapters (for consistency):**
--- BEGIN HYBRID CONTEXT ---
{hybrid_context_for_revision if hybrid_context_for_revision.strip() else "No previous context."}
--- END HYBRID CONTEXT ---

**Specific Problem to Address:**
  - Issue Category: {problem['issue_category']}
  - Problem Description: {problem['problem_description']}
  - Original Quote Illustrating Problem: "{problem['quote_from_original']}"
  - Suggested Fix Focus: {problem['suggested_fix_focus']}

**Text Snippet from Original Chapter (around problem quote):**
--- BEGIN ORIGINAL TEXT SNIPPET ---
{original_chapter_text_snippet}
--- END ORIGINAL TEXT SNIPPET ---
{length_expansion_instruction}
**Instructions for Generating Replacement Text:**
1.  Focus EXCLUSIVELY on: `{problem['quote_from_original']}`.
2.  Generate a `replace_with` text to substitute this *exact* original quote.
3.  `replace_with` text must address "Problem Description" and "Suggested Fix Focus".
4.  Maintain novel's style, tone, and context consistency.
5.  If `length_expansion_instruction` present, ensure substantial expansion.
6.  **Output ONLY the `replace_with` text.** No JSON, markdown, or explanation.

--- BEGIN REPLACE_WITH TEXT (for "{problem['quote_from_original']}") ---
"""
    logger.info(f"Calling LLM ({config.PATCH_GENERATION_MODEL}) for patch in Ch {chapter_number}. Problem: {problem['problem_description'][:60]}...")
    max_patch_output_tokens = config.MAX_GENERATION_TOKENS // 4
    replace_with_text_raw = await llm_interface.async_call_llm(
        config.PATCH_GENERATION_MODEL, prompt, 0.6, max_patch_output_tokens, True, False
    )
    if not replace_with_text_raw:
        logger.error(f"Patch LLM returned no content for Ch {chapter_number}: {problem['problem_description']}")
        return None
    replace_with_text_cleaned = llm_interface.clean_model_response(replace_with_text_raw)
    if not replace_with_text_cleaned.strip():
        logger.warning(f"Patch LLM returned empty after cleaning for Ch {chapter_number}: {problem['problem_description']}")
        return None
    if length_expansion_instruction and problem['quote_from_original'] != "N/A - General Issue":
        if len(replace_with_text_cleaned) < len(problem['quote_from_original']) * 1.3:
            logger.warning(f"Patch for Ch {chapter_number} did not sufficiently expand text. Original: {len(problem['quote_from_original'])}, New: {len(replace_with_text_cleaned)}.")
    return {
        "search_text": problem['quote_from_original'], "replace_with": replace_with_text_cleaned,
        "original_quote_ref": problem['quote_from_original'],
        "reason_for_change": f"Fixing '{problem['issue_category']}': {problem['problem_description']}"
    }


async def _generate_patch_instructions_logic(
    agent: Any, # NANA_Orchestrator instance
    original_text: str,
    problems_to_fix: List[ProblemDetail],
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> List[PatchInstruction]:
    patch_instructions: List[PatchInstruction] = []
    problems_to_process = problems_to_fix[:config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE]
    if len(problems_to_fix) > config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE:
        logger.warning(f"Found {len(problems_to_fix)} problems for Ch {chapter_number}, patching first {config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE}.")

    patch_generation_tasks = []
    for problem in problems_to_process:
        if problem["quote_from_original"] == "N/A - General Issue" and problem["issue_category"] != "narrative_depth":
            logger.info(f"Skipping patch for Ch {chapter_number} problem '{problem['problem_description']}' (N/A quote, not depth).")
            continue
        if problem["quote_from_original"] == "N/A - General Issue" and "length" not in problem['problem_description'].lower() and "depth" not in problem['problem_description'].lower() and "expand" not in problem['suggested_fix_focus'].lower():
            logger.info(f"Skipping patch for N/A quote problem in Ch {chapter_number}: '{problem['problem_description']}' (not about expansion).")
            continue
        context_snippet = _get_context_window(original_text, problem["quote_from_original"], config.MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW)
        task = _generate_single_patch_instruction_llm(
            agent, context_snippet, problem, chapter_number, hybrid_context_for_revision, chapter_plan
        )
        patch_generation_tasks.append(task)
    if not patch_generation_tasks:
        logger.info(f"No actionable problems for patch generation in Ch {chapter_number}.")
        return []
    results = await asyncio.gather(*patch_generation_tasks, return_exceptions=True)
    for i, res_or_exc in enumerate(results):
        problem_ref = problems_to_process[i]
        if isinstance(res_or_exc, Exception):
            logger.error(f"Error generating patch for Ch {chapter_number} problem '{problem_ref['problem_description'][:50]}': {res_or_exc}", exc_info=res_or_exc)
        elif res_or_exc is not None: patch_instructions.append(res_or_exc)
        else: logger.warning(f"Patch generation None for Ch {chapter_number} problem: {problem_ref['problem_description'][:50]}.")
    logger.info(f"Generated {len(patch_instructions)} patch instructions for Ch {chapter_number}.")
    return patch_instructions

def _apply_patches_to_text(original_text: str, patch_instructions: List[PatchInstruction]) -> str:
    if not patch_instructions: return original_text
    replacements: List[Tuple[int, int, str]] = []
    failed_patches_search_text_not_found = 0
    applicable_patches = [p for p in patch_instructions if p["search_text"] != "N/A - General Issue"]
    for patch in applicable_patches:
        try:
            match_indices = [m.start() for m in re.finditer(re.escape(patch["search_text"]), original_text)]
            if not match_indices:
                logger.warning(f"Patching: `search_text` not found. Skipping. Search: '{patch['search_text'][:100]}...'")
                failed_patches_search_text_not_found += 1
                continue
            start_index = match_indices[0]
            end_index = start_index + len(patch["search_text"])
            replacements.append((start_index, end_index, patch["replace_with"]))
        except Exception as e:
            logger.error(f"Error preparing patch for '{patch['search_text'][:50]}': {e}", exc_info=True)
            failed_patches_search_text_not_found +=1
    if failed_patches_search_text_not_found > 0 and applicable_patches:
         logger.warning(f"{failed_patches_search_text_not_found}/{len(applicable_patches)} patches failed (search_text not found).")
    if not replacements: return original_text
    replacements.sort(key=lambda x: x[0], reverse=True)
    current_text_list = list(original_text)
    for start_index, end_index, replace_with_text in replacements:
        current_text_list[start_index:end_index] = list(replace_with_text)
        logger.info(f"Applied patch: Replaced {start_index}-{end_index} (len {end_index-start_index}) with new (len {len(replace_with_text)}). Snippet: '{replace_with_text[:50]}...'")
    return "".join(current_text_list)


async def revise_chapter_draft_logic(
    agent: Any, # NANA_Orchestrator instance
    original_text: str,
    chapter_number: int,
    evaluation_result: EvaluationResult,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> Optional[Tuple[str, str]]:
    if not original_text:
        logger.error(f"Revision for ch {chapter_number} aborted: missing original text.")
        return None

    problems_to_fix: List[ProblemDetail] = evaluation_result.get("problems_found", [])
    if not problems_to_fix and evaluation_result.get("needs_revision"):
        logger.warning(f"Revision for ch {chapter_number} requested, but no specific problems. Full rewrite incoming.")
    elif not problems_to_fix:
        logger.info(f"No specific problems for ch {chapter_number}. No revision needed based on problem list.")
        return None # If no problems, no patch-based revision. Full rewrite is for `needs_revision` flag.

    revision_reason_str = "\n- ".join(evaluation_result["reasons"])
    if not revision_reason_str.strip(): revision_reason_str = "General unspecified issues."
    logger.warning(f"Attempting revision for chapter {chapter_number}. Reason(s):\n- {revision_reason_str}")

    patched_text: Optional[str] = None
    raw_patch_llm_outputs_combined: str = ""
    actionable_problems_for_patching = [p for p in problems_to_fix if p["quote_from_original"] != "N/A - General Issue" or p["issue_category"] == "narrative_depth"]

    if config.ENABLE_PATCH_BASED_REVISION and actionable_problems_for_patching:
        logger.info(f"Patch-based revision for Ch {chapter_number} ({len(actionable_problems_for_patching)} actionable problems).")
        patch_instructions = await _generate_patch_instructions_logic(
            agent, original_text, actionable_problems_for_patching, chapter_number, hybrid_context_for_revision, chapter_plan
        )
        if patch_instructions:
            patched_text = _apply_patches_to_text(original_text, patch_instructions)
            raw_patch_llm_outputs_combined = f"Chapter revised using {len(patch_instructions)} patches.\n"
            logger.info(f"Applied {len(patch_instructions)} patches to Ch {chapter_number}. Orig len: {len(original_text)}, Patched: {len(patched_text)}.")
        else: logger.warning(f"Patch-based revision for Ch {chapter_number}: No valid patches generated. Fallback to full rewrite if configured.")

    final_revised_text: Optional[str] = None
    final_raw_llm_output: Optional[str] = None

    if patched_text is not None:
        if len(patched_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.warning(f"Patched draft for ch {chapter_number} too short ({len(patched_text)} chars).")
        sim_original_embedding, sim_patched_embedding = await asyncio.gather(
            llm_interface.async_get_embedding(original_text), llm_interface.async_get_embedding(patched_text)
        )
        if sim_original_embedding is not None and sim_patched_embedding is not None:
            similarity_score = utils.numpy_cosine_similarity(sim_original_embedding, sim_patched_embedding)
            logger.info(f"Patched text similarity with original: {similarity_score:.4f}")
            if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Patched text for ch {chapter_number} too similar to original (Score: {similarity_score:.4f}). Patches ineffective?.")
        else: logger.warning(f"Could not get embeddings for patched text similarity check of ch {chapter_number}.")
        final_revised_text = patched_text
        final_raw_llm_output = raw_patch_llm_outputs_combined
    else:
        if config.ENABLE_PATCH_BASED_REVISION and actionable_problems_for_patching :
             logger.warning(f"Patching failed for Ch {chapter_number}. Falling back to full rewrite.")
        elif not actionable_problems_for_patching and evaluation_result.get("needs_revision"):
             logger.info(f"No actionable problems for patching in Ch {chapter_number}, but revision needed. Full rewrite.")
        elif not config.ENABLE_PATCH_BASED_REVISION and evaluation_result.get("needs_revision"):
             logger.info(f"Patching disabled and revision needed. Full rewrite for Ch {chapter_number}.")
        else:
            logger.info(f"No revision for Ch {chapter_number} (no patches, no explicit full rewrite trigger here).")
            return None

        max_original_snippet_tokens = config.MAX_CONTEXT_TOKENS // 3
        original_snippet = llm_interface.truncate_text_by_tokens(
            original_text, config.REVISION_MODEL, max_original_snippet_tokens,
            truncation_marker="\n... (original draft snippet truncated)"
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
        if any(kw in revision_reason_str.lower() for kw in ["too short", "lacking in depth", "brief", "expand", "length", "narrative depth", "detail"]):
            length_issue_explicit_instruction_full_rewrite = (
                "\n**Specific Focus on Expansion:** Key critique involves insufficient length/depth. "
                "Your revision MUST substantially expand narrative (descriptions, thoughts, dialogue, sensory details). "
                f"Aim for at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} chars."
            )
        plot_outline_data = _get_prop_from_agent(agent, 'plot_outline', {})
        protagonist_name_full_rewrite = _get_prop_from_agent(plot_outline_data, "protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        prompt_full_rewrite = f"""/no_think
You are rewriting Chapter {chapter_number} for protagonist {protagonist_name_full_rewrite}.
**Critique/Reason(s) for Revision (MUST be addressed comprehensively):**
--- FEEDBACK START ---
{llm_interface.clean_model_response(revision_reason_str).strip()}
--- FEEDBACK END ---
{length_issue_explicit_instruction_full_rewrite}
{plan_focus_section_full_rewrite}
**Hybrid Context from Previous Chapters (Semantic Context & KG Facts):**
--- BEGIN HYBRID CONTEXT ---
{hybrid_context_for_revision if hybrid_context_for_revision.strip() else "No previous context."}
--- END HYBRID CONTEXT ---
**Original Draft Snippet (for reference ONLY - main goal is critique & plan alignment):**
--- BEGIN ORIGINAL DRAFT SNIPPET ---
{original_snippet}
--- END ORIGINAL DRAFT SNIPPET ---
**Revision Instructions:**
1. **ABSOLUTE PRIORITY:** Thoroughly address ALL issues in **Critique/Reason(s) for Revision**.
2. **Rewrite ENTIRE chapter.** Produce fresh, coherent narrative.
3. Align with **Original Detailed Scene Plan** or **Original Chapter Focus**.
4. Ensure flow with **Hybrid Context**. Respect `KEY RELIABLE KG FACTS`.
5. Maintain tone, style, genre ('{_get_prop_from_agent(plot_outline_data, 'genre', 'story')}').
6. Target at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters.
7. **Output ONLY rewritten chapter text.** No headers, titles, meta-commentary.
--- BEGIN REVISED CHAPTER {chapter_number} TEXT ---
"""
        logger.info(f"Calling LLM ({config.REVISION_MODEL}) for Ch {chapter_number} full rewrite. Min length: {config.MIN_ACCEPTABLE_DRAFT_LENGTH} chars.")
        revised_raw_llm_output_full = await llm_interface.async_call_llm(
            config.REVISION_MODEL, prompt_full_rewrite, 0.6, None, True, True
        )
        if not revised_raw_llm_output_full:
            logger.error(f"Full rewrite LLM failed for ch {chapter_number} (empty).")
            return None
        final_revised_text = llm_interface.clean_model_response(revised_raw_llm_output_full)
        final_raw_llm_output = revised_raw_llm_output_full

    if not final_revised_text or len(final_revised_text) < 50 :
        logger.error(f"Revision for ch {chapter_number} resulted in no content. Original len: {len(original_text)}")
        # Orchestrator should handle debug saving: await agent._save_debug_output(...)
        return None
    if len(final_revised_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        logger.warning(f"Revised draft for ch {chapter_number} short ({len(final_revised_text)} chars). Min: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")
    if final_revised_text is not patched_text: # i.e., full rewrite was done
        original_embedding_full, revised_embedding_full = await asyncio.gather(
            llm_interface.async_get_embedding(original_text), llm_interface.async_get_embedding(final_revised_text)
        )
        if original_embedding_full is not None and revised_embedding_full is not None:
            similarity_score_full = utils.numpy_cosine_similarity(original_embedding_full, revised_embedding_full)
            logger.info(f"Full rewrite similarity with original: {similarity_score_full:.4f}")
            if similarity_score_full >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Full rewrite for ch {chapter_number} too similar to original (Score: {similarity_score_full:.4f}). LLM may not have made sufficient changes.")
        else: logger.warning(f"Could not get embeddings for full rewrite similarity check of ch {chapter_number}.")
    logger.info(f"Revision for ch {chapter_number} produced candidate text (Length: {len(final_revised_text)} chars).")
    return final_revised_text, final_raw_llm_output
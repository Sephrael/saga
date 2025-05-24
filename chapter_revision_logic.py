# chapter_revision_logic.py
"""
Handles the revision of chapter drafts based on evaluation feedback for the SAGA system.
Supports both full rewrite and targeted patch-based revisions.
Context data for prompts is now formatted as plain text.
MODIFIED: Resolve _format_scene_plan_for_prompt import, adapt agent access.
MODIFIED: Enhanced patch generation logic, especially for "N/A - General Issue" and length expansion.
          Improved application of patches and skipping non-actionable ones.
"""
import logging
import asyncio
import re
from typing import Tuple, Optional, List, Dict, Any

import config
import llm_interface
import drafting_agent # For _format_scene_plan_for_prompt if needed
import utils
from type import SceneDetail, ProblemDetail, PatchInstruction, EvaluationResult

logger = logging.getLogger(__name__)

# Import the plain text formatter for scene plan
# Temporary: Borrow from DraftingAgent instance.
try:
    _temp_drafting_agent_for_format = drafting_agent.DraftingAgent()
    _format_scene_plan_for_prompt_func = _temp_drafting_agent_for_format._format_scene_plan_for_prompt
except ImportError:
    logger.error("Could not import _format_scene_plan_for_prompt from drafting_agent for chapter_revision_logic. Revision planning might be affected.")
    # Fallback stub if drafting_agent or its method is unavailable
    def _format_scene_plan_for_prompt_func(chapter_plan: List[SceneDetail], model_name_for_tokens: str, max_tokens_budget: int) -> str:
        logger.warning("_format_scene_plan_for_prompt_func is a fallback stub!")
        plan_text_parts = []
        current_tokens = 0
        for scene in chapter_plan:
            scene_text = f"Scene {scene.get('scene_number', 'N/A')}: {scene.get('summary', 'No summary')}\n"
            scene_tokens = llm_interface.count_tokens(scene_text, model_name_for_tokens)
            if current_tokens + scene_tokens > max_tokens_budget:
                plan_text_parts.append("... (plan truncated in prompt due to token limit)\n")
                break
            plan_text_parts.append(scene_text)
            current_tokens += scene_tokens
        return "".join(plan_text_parts) if plan_text_parts else "Scene plan formatting unavailable or plan empty."


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
    """
    Extracts a context window around a quote. If quote is "N/A - General Issue",
    returns a snippet from the beginning and end of the text.
    """
    if not text: return ""
    if quote == "N/A - General Issue":
        # For general issues, provide a generic context window (e.g., start and end of chapter)
        if len(text) <= window_size_chars:
            return text
        half_window = window_size_chars // 2
        # Ensure half_window doesn't exceed text length if text is very short
        start_snippet_len = min(half_window, len(text))
        end_snippet_len = min(half_window, len(text) - start_snippet_len)
        
        start_snippet = text[:start_snippet_len]
        end_snippet = text[-end_snippet_len:] if end_snippet_len > 0 else ""
        
        if start_snippet_len + end_snippet_len < len(text):
             return f"{start_snippet}\n...\n{end_snippet}"
        else: # text is short enough that start and end snippets would overlap or cover it
             return text

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
        # Fallback if quote is not found (should be rare if quote is verbatim)
        return text[:window_size_chars] + "..." if len(text) > window_size_chars else text


async def _generate_single_patch_instruction_llm(
    agent: Any, # NANA_Orchestrator instance
    original_chapter_text_snippet: str,
    problem: ProblemDetail,
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> Optional[PatchInstruction]:
    """
    Generates a single patch instruction using an LLM.
    If quote_from_original is "N/A - General Issue", the generated patch text
    is intended for expansion and won't be auto-applied by search_text.
    """
    plan_focus_section = ""
    plot_point_focus, _ = _get_plot_point_info_from_agent(agent, chapter_number)
    max_plan_tokens_for_patch_prompt = config.MAX_CONTEXT_TOKENS // 4 # Quarter of total for plan in patch prompt

    if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
        plan_focus_section = _format_scene_plan_for_prompt_func(chapter_plan, config.PATCH_GENERATION_MODEL, max_plan_tokens_for_patch_prompt)
        if "plan truncated" in plan_focus_section:
             logger.warning(f"Scene plan token-truncated for Ch {chapter_number} patch generation.")
    else:
        plan_focus_section = f"**Original Chapter Focus (Reference):**\n{plot_point_focus or 'Not specified.'}\n"

    length_expansion_instruction = ""
    is_general_expansion_task = False
    if problem['issue_category'] == "narrative_depth" and \
       ("short" in problem['problem_description'].lower() or \
        "length" in problem['problem_description'].lower() or \
        "expand" in problem['suggested_fix_focus'].lower() or \
        "depth" in problem['problem_description'].lower() or \
        problem['quote_from_original'] == "N/A - General Issue"): # If quote is N/A and it's depth, assume expansion
        
        length_expansion_instruction = (
            f"\n**Critical: SUBSTANTIAL EXPANSION REQUIRED.** "
            f"The 'replace_with' text MUST be significantly longer and more detailed than the original context related to the problem. "
            f"Add descriptive details, character thoughts, dialogue, actions, and sensory information. "
            f"Aim for a 50-100%+ increase in length for the conceptual segment being addressed."
        )
        if problem['quote_from_original'] == "N/A - General Issue":
            is_general_expansion_task = True
            length_expansion_instruction += (
                f"\nSince the original quote is 'N/A - General Issue', your 'replace_with' text should be a new, expanded passage "
                f"that addresses the 'Problem Description' and 'Suggested Fix Focus' within the broader 'Text Snippet' context. "
                f"This generated text is intended as a candidate for insertion or to inform a broader rewrite."
            )

    plot_outline_data = _get_prop_from_agent(agent, 'plot_outline', {})
    protagonist_name = _get_prop_from_agent(plot_outline_data, 'protagonist_name', config.DEFAULT_PROTAGONIST_NAME)

    prompt = f"""/no_think
You are a surgical revision expert generating replacement text for Chapter {chapter_number} of a novel about {protagonist_name}.
**Novel Context:**
  - Genre: {_get_nested_prop_from_agent(agent, 'plot_outline', 'genre', 'N/A')}
  - Theme: {_get_nested_prop_from_agent(agent, 'plot_outline', 'theme', 'N/A')}
  - Protagonist: {protagonist_name} ({_get_nested_prop_from_agent(agent, 'plot_outline', 'character_arc', 'N/A')})

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

**Text Snippet from Original Chapter (around problem quote, or general context if quote is N/A):**
--- BEGIN ORIGINAL TEXT SNIPPET ---
{original_chapter_text_snippet}
--- END ORIGINAL TEXT SNIPPET ---
{length_expansion_instruction}
**Instructions for Generating Replacement Text:**
1.  Focus EXCLUSIVELY on the problem described, particularly relating to: `{problem['quote_from_original']}`.
2.  Generate a `replace_with` text. If the original quote is specific, this text should substitute it. If the quote is "N/A - General Issue", generate new text to address the problem as described in the `length_expansion_instruction`.
3.  The `replace_with` text must address the "Problem Description" and "Suggested Fix Focus".
4.  Maintain the novel's style, tone, and consistency with all provided context.
5.  If `length_expansion_instruction` is present, ensure substantial expansion as guided.
6.  **Output ONLY the `replace_with` text.** Do NOT include JSON, markdown, explanations, or any "Replace with:" prefixes. Just the raw text.

--- BEGIN REPLACE_WITH TEXT (for "{problem['quote_from_original']}") ---
"""
    logger.info(f"Calling LLM ({config.PATCH_GENERATION_MODEL}) for patch in Ch {chapter_number}. Problem: {problem['problem_description'][:60]}... Quote: '{problem['quote_from_original'][:50]}...'")
    max_patch_output_tokens = config.MAX_GENERATION_TOKENS // 4 # Allow generous output for expansion patches
    if is_general_expansion_task: max_patch_output_tokens = config.MAX_GENERATION_TOKENS // 2 # Even more for general expansion

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

    # Check expansion for specific quotes only if expansion was instructed
    if length_expansion_instruction and problem['quote_from_original'] != "N/A - General Issue":
        # A simple length check; more sophisticated checks could be semantic.
        if len(replace_with_text_cleaned) < len(problem['quote_from_original']) * 1.3: # Aim for at least 30% longer
            logger.warning(
                f"Patch for Ch {chapter_number} (specific quote) did not sufficiently expand text as instructed. "
                f"Original len: {len(problem['quote_from_original'])}, New len: {len(replace_with_text_cleaned)}."
            )

    return {
        "search_text": problem['quote_from_original'], # This will be "N/A - General Issue" for those cases
        "replace_with": replace_with_text_cleaned,
        "original_quote_ref": problem['quote_from_original'], # For reference
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
    
    # Limit the number of patches to generate in one go
    problems_to_process = problems_to_fix[:config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE]
    if len(problems_to_fix) > config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE:
        logger.warning(f"Found {len(problems_to_fix)} problems for Ch {chapter_number}, attempting to generate patches for the first {config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE}.")

    patch_generation_tasks = []
    for problem in problems_to_process:
        # Skip patch generation for "N/A - General Issue" if not specifically about length/depth expansion
        if problem["quote_from_original"] == "N/A - General Issue":
            is_depth_issue = problem["issue_category"] == "narrative_depth"
            is_expansion_related = (
                "short" in problem['problem_description'].lower() or
                "length" in problem['problem_description'].lower() or
                "expand" in problem['suggested_fix_focus'].lower() or
                "depth" in problem['problem_description'].lower() # Broader "depth" can imply expansion
            )
            if not (is_depth_issue and is_expansion_related):
                logger.info(f"Skipping patch generation for Ch {chapter_number} problem '{problem['problem_description'][:60]}' (N/A quote, not an expansion-focused depth issue).")
                continue
        
        # Get context window. For "N/A - General Issue", this provides a general chapter snippet.
        context_snippet = _get_context_window(original_text, problem["quote_from_original"], config.MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW)
        
        task = _generate_single_patch_instruction_llm(
            agent, context_snippet, problem, chapter_number, hybrid_context_for_revision, chapter_plan
        )
        patch_generation_tasks.append(task)

    if not patch_generation_tasks:
        logger.info(f"No actionable problems for patch instruction generation in Ch {chapter_number}.")
        return []

    results = await asyncio.gather(*patch_generation_tasks, return_exceptions=True)
    
    for i, res_or_exc in enumerate(results):
        problem_ref = problems_to_process[i] # Relies on order preservation by asyncio.gather
        if isinstance(res_or_exc, Exception):
            logger.error(f"Error generating patch for Ch {chapter_number} problem '{problem_ref['problem_description'][:50]}': {res_or_exc}", exc_info=res_or_exc)
        elif res_or_exc is not None:
            patch_instructions.append(res_or_exc)
        else:
            # _generate_single_patch_instruction_llm returning None means LLM call failed or no content
            logger.warning(f"Patch generation returned None for Ch {chapter_number} problem: {problem_ref['problem_description'][:50]}.")
            
    logger.info(f"Generated {len(patch_instructions)} patch instructions for Ch {chapter_number}.")
    return patch_instructions

def _apply_patches_to_text(original_text: str, patch_instructions: List[PatchInstruction]) -> str:
    if not patch_instructions:
        return original_text

    # Filter out patches that cannot be automatically applied (e.g., "N/A - General Issue" search text)
    applicable_patches = [p for p in patch_instructions if p["search_text"] != "N/A - General Issue"]
    if not applicable_patches:
        logger.info("No applicable patches with specific search_text to apply.")
        return original_text
        
    # Store replacements as (start_index, end_index, replacement_text)
    replacements: List[Tuple[int, int, str]] = []
    failed_patches_search_text_not_found = 0

    for patch in applicable_patches:
        try:
            # Find all occurrences of the search_text
            # For simplicity, we'll replace only the first one found.
            # More sophisticated logic could allow choosing an occurrence or handling multiple.
            match_indices = [m.start() for m in re.finditer(re.escape(patch["search_text"]), original_text)]
            if not match_indices:
                logger.warning(f"Patching: `search_text` not found in original text. Skipping this patch. Search: '{patch['search_text'][:100]}...'")
                failed_patches_search_text_not_found += 1
                continue
            
            start_index = match_indices[0] # Apply to the first occurrence
            end_index = start_index + len(patch["search_text"])
            replacements.append((start_index, end_index, patch["replace_with"]))
        except Exception as e:
            logger.error(f"Error preparing patch for '{patch['search_text'][:50]}': {e}", exc_info=True)
            failed_patches_search_text_not_found +=1 # Count as failed if error during prep

    if failed_patches_search_text_not_found > 0 and applicable_patches:
         logger.warning(f"{failed_patches_search_text_not_found}/{len(applicable_patches)} applicable patches failed (search_text not found or error during prep).")

    if not replacements:
        return original_text

    # Sort replacements by start_index in reverse order to apply them without affecting subsequent indices
    replacements.sort(key=lambda x: x[0], reverse=True)

    current_text_list = list(original_text)
    for start_index, end_index, replace_with_text in replacements:
        current_text_list[start_index:end_index] = list(replace_with_text)
        logger.info(f"Applied patch: Replaced original text from index {start_index} to {end_index} (length {end_index-start_index}) with new text (length {len(replace_with_text)}). Snippet of replacement: '{replace_with_text[:50]}...'")

    return "".join(current_text_list)


async def revise_chapter_draft_logic(
    agent: Any, # NANA_Orchestrator instance
    original_text: str,
    chapter_number: int,
    evaluation_result: EvaluationResult,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> Optional[Tuple[str, str]]:
    """
    Main logic for revising a chapter draft.
    Attempts patch-based revision first, then falls back to full rewrite if needed.
    """
    if not original_text:
        logger.error(f"Revision for ch {chapter_number} aborted: missing original text.")
        return None

    problems_to_fix: List[ProblemDetail] = evaluation_result.get("problems_found", [])
    if not problems_to_fix and evaluation_result.get("needs_revision"):
        # This case means revision is needed for general reasons not tied to specific problems
        # (e.g., overall too short, but no specific quotes were given, or evaluator just said "rewrite")
        logger.warning(f"Revision for ch {chapter_number} explicitly requested, but no specific problems were itemized. Proceeding to full rewrite.")
    elif not problems_to_fix:
        logger.info(f"No specific problems found for ch {chapter_number}. No revision needed based on problem list.")
        # If `needs_revision` is False and no problems, this is correct.
        # If `needs_revision` is True but no problems, previous log handles it.
        return None 

    revision_reason_str_list = evaluation_result.get("reasons", [])
    revision_reason_str = "\n- ".join(revision_reason_str_list) if revision_reason_str_list else "General unspecified issues."
    logger.warning(f"Attempting revision for chapter {chapter_number}. Reason(s):\n- {revision_reason_str}")

    patched_text: Optional[str] = None
    raw_patch_llm_outputs_combined: str = ""

    # Determine which problems are actionable for patching attempts
    # Actionable = has a specific quote OR is a narrative_depth issue potentially suitable for expansion generation
    actionable_problems_for_patching = [
        p for p in problems_to_fix if p["quote_from_original"] != "N/A - General Issue" or \
        (p["issue_category"] == "narrative_depth" and \
         ("short" in p['problem_description'].lower() or \
          "length" in p['problem_description'].lower() or \
          "expand" in p['suggested_fix_focus'].lower() or \
          "depth" in p['problem_description'].lower()))
    ]

    if config.ENABLE_PATCH_BASED_REVISION and actionable_problems_for_patching:
        logger.info(f"Attempting patch-based revision for Ch {chapter_number} with {len(actionable_problems_for_patching)} actionable problem(s).")
        patch_instructions = await _generate_patch_instructions_logic(
            agent, original_text, actionable_problems_for_patching, chapter_number, hybrid_context_for_revision, chapter_plan
        )
        
        if patch_instructions:
            # Filter instructions again for those that are actually applicable by _apply_patches_to_text
            applicable_instructions_for_apply = [pi for pi in patch_instructions if pi["search_text"] != "N/A - General Issue"]
            if applicable_instructions_for_apply:
                patched_text = _apply_patches_to_text(original_text, applicable_instructions_for_apply)
                raw_patch_llm_outputs_combined = f"Chapter revised using {len(applicable_instructions_for_apply)} applied patches.\n"
                # Log details about generated vs. applied patches
                num_generated_patches = len(patch_instructions)
                num_applied_patches = len(applicable_instructions_for_apply)
                logger.info(
                    f"Patch process for Ch {chapter_number}: Generated {num_generated_patches} patch instructions. "
                    f"Applied {num_applied_patches} patches. Orig len: {len(original_text)}, Patched len: {len(patched_text if patched_text else '')}."
                )
                if num_generated_patches > num_applied_patches:
                    logger.info(f"{num_generated_patches - num_applied_patches} generated patches were not auto-applied (e.g., 'N/A - General Issue' or search text not found).")
            else:
                 logger.info(f"Patch-based revision for Ch {chapter_number}: Patches were generated, but none had specific 'search_text' for auto-application. Proceeding as if no patches applied.")
        else:
            logger.warning(f"Patch-based revision for Ch {chapter_number}: No valid patch instructions were generated. Fallback to full rewrite if configured and needed.")
    
    final_revised_text: Optional[str] = None
    final_raw_llm_output: Optional[str] = None

    # Condition for using patched text:
    # 1. Patched text exists (i.e., patches were applicable and applied).
    # 2. Patched text is significantly different from original OR meets other quality checks.
    use_patched_text = False
    if patched_text is not None:
        if len(patched_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH * 0.8: # Allow somewhat shorter if patches reduced, but not drastically
            logger.warning(f"Patched draft for ch {chapter_number} is quite short ({len(patched_text)} chars). May still fall back to full rewrite.")
        
        # Check similarity
        sim_original_embedding, sim_patched_embedding = await asyncio.gather(
            llm_interface.async_get_embedding(original_text), llm_interface.async_get_embedding(patched_text)
        )
        if sim_original_embedding is not None and sim_patched_embedding is not None:
            similarity_score = utils.numpy_cosine_similarity(sim_original_embedding, sim_patched_embedding)
            logger.info(f"Patched text similarity with original: {similarity_score:.4f}")
            if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Patched text for ch {chapter_number} is very similar to original (Score: {similarity_score:.4f}). Patches might have been ineffective or minor. Considering full rewrite.")
            else:
                # Patched text is sufficiently different and exists
                use_patched_text = True
        else:
            logger.warning(f"Could not get embeddings for patched text similarity check of ch {chapter_number}. Assuming patched text is different enough if it exists.")
            use_patched_text = True # If embeddings fail, and patched text exists, cautiously proceed with it.
        
        if use_patched_text:
            final_revised_text = patched_text
            final_raw_llm_output = raw_patch_llm_outputs_combined
            logger.info(f"Ch {chapter_number}: Using patched text as the revised version.")

    # Fallback to full rewrite if:
    # - Patched text was not used (e.g., too similar, too short after patching, or no applicable patches)
    # - AND revision is still marked as needed by the evaluation result.
    if not use_patched_text and evaluation_result.get("needs_revision"):
        if config.ENABLE_PATCH_BASED_REVISION and actionable_problems_for_patching:
             logger.warning(f"Patching did not result in a usable revision for Ch {chapter_number}. Falling back to full rewrite.")
        elif not actionable_problems_for_patching and evaluation_result.get("needs_revision"):
             logger.info(f"No actionable problems for patching in Ch {chapter_number}, but revision needed. Proceeding with full rewrite.")
        elif not config.ENABLE_PATCH_BASED_REVISION and evaluation_result.get("needs_revision"):
             logger.info(f"Patching disabled, and revision needed. Proceeding with full rewrite for Ch {chapter_number}.")
        # If none of the above, it means no revision was needed in the first place or patching was attempted but failed to produce a usable output, AND needs_revision is true.

        max_original_snippet_tokens = config.MAX_CONTEXT_TOKENS // 3
        original_snippet = llm_interface.truncate_text_by_tokens(
            original_text, config.REVISION_MODEL, max_original_snippet_tokens,
            truncation_marker="\n... (original draft snippet truncated for brevity)"
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
        # Check for length/depth issues in problem descriptions or reasons for revision
        needs_expansion_from_problems = any(
            (p['issue_category'] == 'narrative_depth' and 
             ("short" in p['problem_description'].lower() or "length" in p['problem_description'].lower() or
              "expand" in p['suggested_fix_focus'].lower() or "depth" in p['problem_description'].lower()))
            for p in problems_to_fix
        )
        needs_expansion_from_reasons = any(
            kw in revision_reason_str.lower() for kw in ["too short", "lacking in depth", "brief", "expand", "length", "narrative depth", "detail"]
        )

        if needs_expansion_from_problems or needs_expansion_from_reasons:
            length_issue_explicit_instruction_full_rewrite = (
                "\n**Specific Focus on Expansion:** A key critique involves insufficient length and/or narrative depth. "
                "Your revision MUST substantially expand the narrative by incorporating more descriptive details, character thoughts/introspection, dialogue, actions, and sensory information. "
                f"Aim for a chapter length of at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters."
            )
        
        plot_outline_data = _get_prop_from_agent(agent, 'plot_outline', {})
        protagonist_name_full_rewrite = _get_prop_from_agent(plot_outline_data, "protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        
        # Compile all problem descriptions for the full rewrite prompt
        all_problem_descriptions_str = ""
        if problems_to_fix:
            all_problem_descriptions_str = "**Detailed Issues to Address (from evaluation):**\n"
            for prob_idx, prob_item in enumerate(problems_to_fix):
                all_problem_descriptions_str += (
                    f"  {prob_idx+1}. Category: {prob_item['issue_category']}\n"
                    f"     Description: {prob_item['problem_description']}\n"
                    f"     Quote Ref: \"{prob_item['quote_from_original'][:100]}...\"\n"
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
**Original Draft Snippet (for reference of what went wrong - DO NOT COPY VERBATIM. Your goal is a fresh rewrite addressing all critique and aligning with the plan):**
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
        revised_raw_llm_output_full = await llm_interface.async_call_llm(
            config.REVISION_MODEL, prompt_full_rewrite, 0.6, None, True, True # Allow full token generation, stream to disk
        )
        if not revised_raw_llm_output_full:
            logger.error(f"Full rewrite LLM failed for ch {chapter_number} (returned empty).")
            # Orchestrator should handle debug saving if it passes `self` as `agent` to this function
            # and this function then calls agent._save_debug_output(...)
            return None
        
        final_revised_text = llm_interface.clean_model_response(revised_raw_llm_output_full)
        final_raw_llm_output = revised_raw_llm_output_full
    elif not use_patched_text and not evaluation_result.get("needs_revision"):
        # This case means: patching was attempted (or not), no usable patched text resulted,
        # AND the evaluation said no revision is needed.
        # This implies the original text was fine, or patching was not necessary/effective but the result is still acceptable.
        logger.info(f"No revision performed for Ch {chapter_number} (original deemed acceptable or patching ineffective but not critical).")
        return None # Return None to indicate no *new* revision was made. Orchestrator will use original.

    # Post-revision checks for the final chosen text (either patched or full rewrite)
    if not final_revised_text or len(final_revised_text) < 50 : # Arbitrary small number to catch effectively empty outputs
        logger.error(f"Revision process for ch {chapter_number} resulted in no usable content. Original len: {len(original_text)}")
        # Orchestrator can save debug: await agent._save_debug_output(chapter_number, "revision_empty_final_output", final_raw_llm_output or original_text)
        return None

    if len(final_revised_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        logger.warning(f"Final revised draft for ch {chapter_number} is short ({len(final_revised_text)} chars). Min target: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")

    # If a full rewrite was done (final_revised_text is not the patched_text we might have generated earlier)
    if final_revised_text is not patched_text:
        original_embedding_full, revised_embedding_full = await asyncio.gather(
            llm_interface.async_get_embedding(original_text), llm_interface.async_get_embedding(final_revised_text)
        )
        if original_embedding_full is not None and revised_embedding_full is not None:
            similarity_score_full = utils.numpy_cosine_similarity(original_embedding_full, revised_embedding_full)
            logger.info(f"Full rewrite similarity with original text: {similarity_score_full:.4f}")
            if similarity_score_full >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(
                    f"Full rewrite for ch {chapter_number} is very similar to original (Score: {similarity_score_full:.4f}). "
                    f"The LLM may not have made sufficient changes despite instructions."
                )
        else:
            logger.warning(f"Could not get embeddings for full rewrite similarity check of ch {chapter_number}.")

    logger.info(f"Revision process for ch {chapter_number} produced a candidate text (Length: {len(final_revised_text)} chars).")
    return final_revised_text, final_raw_llm_output
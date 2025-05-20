# chapter_revision_logic.py
"""
Handles the revision of chapter drafts based on evaluation feedback for the SAGA system.
Supports both full rewrite and targeted patch-based revisions using line numbers.
"""
import logging
import json
import asyncio
import re
from typing import Tuple, Optional, List, Dict, Any

import config
import llm_interface
import utils 
from type import SceneDetail, ProblemDetail, PatchInstruction, EvaluationResult

logger = logging.getLogger(__name__)

def _get_line_based_context_snippet(
    full_text_lines: List[str],
    problem_line_num: Optional[int], # 1-indexed
    char_window_size: int,
    line_context_radius: int # Number of lines before/after target line
) -> str:
    """
    Extracts a text snippet for the LLM to process for patch generation.
    It prioritizes a few lines around problem_line_num, then expands character-wise if needed.
    """
    if problem_line_num is None or not (1 <= problem_line_num <= len(full_text_lines)):
        # Fallback if line number is invalid or not provided: use start of text
        logger.debug("Invalid or no line number for patch context, using start of text.")
        return "".join(full_text_lines)[:char_window_size] + "..." if len("".join(full_text_lines)) > char_window_size else "".join(full_text_lines)

    # Get line-based context
    start_line_idx = max(0, problem_line_num - 1 - line_context_radius)
    end_line_idx = min(len(full_text_lines), problem_line_num - 1 + line_context_radius + 1)
    
    context_lines = full_text_lines[start_line_idx:end_line_idx]
    line_based_snippet = "\n".join(context_lines)

    # If the line-based snippet is too small, try to expand character-wise around the central line
    if len(line_based_snippet) < char_window_size // 2 and len(line_based_snippet) < char_window_size : # Ensure it's not already huge
        # Find char index of the start of the problem_line_num
        chars_before_problem_line = len("\n".join(full_text_lines[:problem_line_num-1]))
        if problem_line_num > 1 : chars_before_problem_line +=1 # for the newline
        
        # Center the character window around the start of the problem line
        char_context_start = max(0, chars_before_problem_line - char_window_size // 2)
        char_context_end = min(len("".join(full_text_lines)), chars_before_problem_line + char_window_size // 2)
        
        full_text_str = "\n".join(full_text_lines) # Rejoin for char slicing
        char_based_snippet = full_text_str[char_context_start:char_context_end]
        
        prefix = "..." if char_context_start > 0 else ""
        suffix = "..." if char_context_end < len(full_text_str) else ""
        final_snippet = f"{prefix}{char_based_snippet}{suffix}"
        logger.debug(f"Used char-expanded context for patch LLM. Line-based snippet was {len(line_based_snippet)} chars. Char-based: {len(final_snippet)} chars.")
        return final_snippet
    
    logger.debug(f"Used line-based context for patch LLM. Snippet length: {len(line_based_snippet)} chars.")
    return line_based_snippet


async def _generate_single_patch_instruction_llm(
    agent,
    original_chapter_text_lines: List[str], # Full chapter text split into lines
    problem: ProblemDetail,
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> Optional[PatchInstruction]:
    """Generates a single patch instruction using an LLM, with line-number awareness."""
    
    # Use the new helper to get a context snippet for the LLM prompt
    context_snippet_for_llm_prompt = _get_line_based_context_snippet(
        original_chapter_text_lines,
        problem['line_number'],
        config.MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW,
        config.LINE_CONTEXT_WINDOW_FOR_PATCH_GENERATION_LLM
    )

    plan_focus_section = ""
    plot_point_focus, _ = agent._get_plot_point_info(chapter_number) 

    if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
        try:
            plan_json_str = json.dumps(chapter_plan, indent=2, ensure_ascii=False)
            plan_snippet_for_prompt = plan_json_str[:(config.MAX_CONTEXT_LENGTH // 6)]
            if len(plan_json_str) > len(plan_snippet_for_prompt):
                plan_snippet_for_prompt += "\n... (plan truncated in prompt)"
            plan_focus_section = f"**Original Detailed Scene Plan (Reference for context):**\n```json\n{plan_snippet_for_prompt}\n```\n"
        except TypeError: 
             plan_focus_section = f"**Original Chapter Focus (Reference for context):**\n{plot_point_focus or 'Not specified.'}\n"
    else: 
        plan_focus_section = f"**Original Chapter Focus (Reference for context):**\n{plot_point_focus or 'Not specified.'}\n"

    length_expansion_instruction = ""
    if problem['issue_category'] == "narrative_depth" and \
       ("short" in problem['problem_description'].lower() or \
        "length" in problem['problem_description'].lower() or \
        "expand" in problem['suggested_fix_focus'].lower() or \
        "depth" in problem['problem_description'].lower() or \
        problem['quote_from_original'] == "N/A - General Issue"):
        length_expansion_instruction = (
            f"\n**Critical for this Patch: SUBSTANTIAL EXPANSION REQUIRED.** "
            f"The 'replace_with' text MUST be significantly longer and more detailed than the original quote ('{problem['quote_from_original'] if problem['quote_from_original'] != 'N/A - General Issue' else 'relevant section'}'). "
            f"Add descriptive details, character thoughts/emotions, extend dialogue, or elaborate on actions to achieve this. Aim to increase text volume for this segment by at least 50-100% or more if needed to address the critique about depth/length."
        )

    prompt = f"""/no_think
You are a surgical revision expert. You are tasked with generating a replacement text snippet for a specific problematic part of Chapter {chapter_number}.
**Novel Context:**
  - Genre: {agent.plot_outline.get('genre', 'N/A')}
  - Theme: {agent.plot_outline.get('theme', 'N/A')}
  - Protagonist: {agent.plot_outline.get('protagonist_name', 'N/A')} ({agent.plot_outline.get('character_arc', 'N/A')})

{plan_focus_section}
**Hybrid Context from Previous Chapters (for overall consistency):**
--- BEGIN HYBRID CONTEXT ---
{hybrid_context_for_revision if hybrid_context_for_revision.strip() else "No previous context."}
--- END HYBRID CONTEXT ---

**Specific Problem to Address:**
  - Issue Category: {problem['issue_category']}
  - Problem Description: {problem['problem_description']}
  - Original Quote Illustrating Problem: "{problem['quote_from_original']}" 
  - Approx. Starting Line of Quote (1-indexed, in full chapter): {problem['line_number'] if problem['line_number'] is not None else "N/A or general issue"}
  - Suggested Fix Focus: {problem['suggested_fix_focus']}

**Text Snippet from Original Chapter (around the problem quote/line):**
--- BEGIN ORIGINAL TEXT SNIPPET ---
{context_snippet_for_llm_prompt}
--- END ORIGINAL TEXT SNIPPET ---
{length_expansion_instruction}
**Instructions for Generating Replacement Text:**
1.  Focus EXCLUSIVELY on the "Original Quote Illustrating Problem": `{problem['quote_from_original']}`.
2.  Your goal is to generate a `replace_with` text that will substitute this *exact* original quote.
3.  The `replace_with` text must directly address the "Problem Description" and incorporate the "Suggested Fix Focus".
4.  Maintain the novel's style, tone, and consistency with all provided context (Novel, Plan, Hybrid).
5.  If `length_expansion_instruction` is present, ensure your `replace_with` text is substantially longer and more detailed.
6.  **Output ONLY the `replace_with` text.** Do NOT output JSON, markdown, or any explanation. Just the raw text for replacement.

--- BEGIN REPLACE_WITH TEXT (for "{problem['quote_from_original']}") ---
"""
    
    logger.info(f"Calling LLM ({config.PATCH_GENERATION_MODEL}) for single patch in Ch {chapter_number}. Problem: {problem['problem_description'][:60]}... Quote: {problem['quote_from_original'][:60]}... Line: {problem['line_number']}")
    
    replace_with_text_raw = await llm_interface.async_call_llm(
        model_name=config.PATCH_GENERATION_MODEL,
        prompt=prompt,
        temperature=0.6, 
        max_tokens=config.MAX_GENERATION_TOKENS // 4, 
        allow_fallback=True, 
        stream_to_disk=False 
    )

    if not replace_with_text_raw:
        logger.error(f"Patch generation LLM returned no content for problem in Ch {chapter_number}: {problem['problem_description']}")
        return None
        
    replace_with_text_cleaned = llm_interface.clean_model_response(replace_with_text_raw)

    if not replace_with_text_cleaned.strip():
        logger.warning(f"Patch generation LLM returned empty or whitespace-only content after cleaning for Ch {chapter_number}: {problem['problem_description']}")
        return None

    if length_expansion_instruction and problem['quote_from_original'] != "N/A - General Issue":
        if len(replace_with_text_cleaned) < len(problem['quote_from_original']) * 1.3: 
            logger.warning(f"Patch for Ch {chapter_number} problem '{problem['problem_description'][:50]}...' did not sufficiently expand text. Original len: {len(problem['quote_from_original'])}, New len: {len(replace_with_text_cleaned)}.")

    return {
        "search_text": problem['quote_from_original'],
        "replace_with": replace_with_text_cleaned,
        "original_quote_ref": problem['quote_from_original'], 
        "reason_for_change": f"Fixing '{problem['issue_category']}' issue: {problem['problem_description']}",
        "line_number_ref": problem['line_number'] # Pass line number for localized search
    }


async def _generate_patch_instructions_logic(
    agent,
    original_text_lines: List[str], # Pass full chapter text as lines
    problems_to_fix: List[ProblemDetail],
    chapter_number: int,
    hybrid_context_for_revision: str,
    chapter_plan: Optional[List[SceneDetail]]
) -> List[PatchInstruction]:
    """Generates a list of PatchInstruction objects by calling LLM for each problem."""
    patch_instructions: List[PatchInstruction] = []
    
    problems_to_process = problems_to_fix[:config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE]
    if len(problems_to_fix) > config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE:
        logger.warning(f"Found {len(problems_to_fix)} problems for Ch {chapter_number}, but will only attempt to patch the first {config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE}.")

    patch_generation_tasks = []
    for problem in problems_to_process:
        if problem["quote_from_original"] == "N/A - General Issue" and \
           (problem["line_number"] is None or problem['issue_category'] != "narrative_depth"):
            logger.info(f"Skipping patch generation for Ch {chapter_number} problem: '{problem['problem_description']}' as it's general, not narrative depth, or lacks a line number for targeted expansion context.")
            continue
        
        # For "N/A - General Issue" narrative depth problems WITH a line number,
        # the LLM is guided by suggested_fix_focus and the line number implies a region.
        # The 'quote_from_original' will still be "N/A - General Issue", so the patch application will need to handle this.
        # This type of patch is inherently less precise.
        # For now, we'll assume such general depth issues on specific lines are rare or better handled by full rewrite.
        # The main target here is specific quotes.
        if problem["quote_from_original"] == "N/A - General Issue":
            logger.info(f"Skipping patch generation for 'N/A - General Issue' problem in Ch {chapter_number}: '{problem['problem_description']}'. This general issue type is hard to target with precise quote-based patching, even with a line number, and is better assessed by overall length evaluation after other specific patches.")
            continue


        task = _generate_single_patch_instruction_llm(
            agent, original_text_lines, problem, chapter_number, hybrid_context_for_revision, chapter_plan
        )
        patch_generation_tasks.append(task)

    if not patch_generation_tasks:
        logger.info(f"No actionable problems with specific quotes/line numbers found for patch generation in Ch {chapter_number}.")
        return []

    results = await asyncio.gather(*patch_generation_tasks, return_exceptions=True)
    
    for i, res_or_exc in enumerate(results):
        # Find the corresponding problem (this assumes tasks were added in order of problems_to_process)
        # A more robust way would be to pass problem ID or use a dict for tasks.
        original_problem_index = -1
        current_task_idx = 0
        for k_prob_idx, k_prob in enumerate(problems_to_process):
            if not (k_prob["quote_from_original"] == "N/A - General Issue" and \
                    (k_prob["line_number"] is None or k_prob['issue_category'] != "narrative_depth")):
                if current_task_idx == i:
                    original_problem_index = k_prob_idx
                    break
                current_task_idx +=1
        
        problem_ref_desc = "Unknown Problem"
        if original_problem_index != -1 and original_problem_index < len(problems_to_process):
            problem_ref_desc = problems_to_process[original_problem_index]['problem_description'][:50]


        if isinstance(res_or_exc, Exception):
            logger.error(f"Error generating patch for Ch {chapter_number} problem '{problem_ref_desc}...': {res_or_exc}", exc_info=res_or_exc)
        elif res_or_exc is not None:
            patch_instructions.append(res_or_exc)
        else:
            logger.warning(f"Patch generation returned None for Ch {chapter_number} problem: {problem_ref_desc}...")
            
    logger.info(f"Generated {len(patch_instructions)} patch instructions for Ch {chapter_number}.")
    return patch_instructions

def _apply_patches_to_text(original_text: str, patch_instructions: List[PatchInstruction]) -> str:
    """
    Applies a list of patch instructions to the original text.
    Uses line_number_ref to localize search for search_text.
    """
    if not patch_instructions:
        return original_text

    replacements: List[Tuple[int, int, str]] = [] # (global_char_start_index, global_char_end_index, replace_with_text)
    failed_patches_search_text_not_found = 0
    original_text_lines = original_text.splitlines(True) # Keep newlines for char index calculation

    for patch_idx, patch in enumerate(patch_instructions):
        search_text = patch["search_text"]
        line_num_ref = patch["line_number_ref"] # 1-indexed

        if search_text == "N/A - General Issue":
            logger.warning(f"Skipping patch application for 'N/A - General Issue' (Patch #{patch_idx}). This should have been filtered earlier.")
            continue

        search_snippet_lines: List[str]
        snippet_global_char_offset: int = 0

        if line_num_ref is not None and 1 <= line_num_ref <= len(original_text_lines):
            # Define a window of lines around the reference line number for searching
            search_line_start_idx = max(0, line_num_ref - 1 - config.LINE_CONTEXT_WINDOW_FOR_PATCH_SEARCH)
            search_line_end_idx = min(len(original_text_lines), line_num_ref - 1 + config.LINE_CONTEXT_WINDOW_FOR_PATCH_SEARCH + 1)
            
            search_snippet_lines = original_text_lines[search_line_start_idx:search_line_end_idx]
            search_snippet_str = "".join(search_snippet_lines)
            
            # Calculate the global character offset of this snippet
            snippet_global_char_offset = len("".join(original_text_lines[:search_line_start_idx]))
            
            logger.debug(f"Patch #{patch_idx}: Localized search for '{search_text[:30]}...' around line {line_num_ref} (lines {search_line_start_idx+1}-{search_line_end_idx}). Snippet char offset: {snippet_global_char_offset}.")
        else:
            # Fallback to searching the whole text if line number is unreliable
            search_snippet_str = original_text
            snippet_global_char_offset = 0
            logger.warning(f"Patch #{patch_idx}: line_number_ref invalid ({line_num_ref}). Falling back to global search for '{search_text[:30]}...'. This may be slow or inaccurate.")

        try:
            # Find the search_text within the localized snippet
            # Use re.escape for literal matching
            # Find first occurrence only
            match_in_snippet = re.search(re.escape(search_text), search_snippet_str)

            if match_in_snippet:
                local_start_index = match_in_snippet.start()
                local_end_index = match_in_snippet.end()

                global_start_index = snippet_global_char_offset + local_start_index
                global_end_index = snippet_global_char_offset + local_end_index
                
                # Sanity check: ensure the found text in original_text matches search_text
                if original_text[global_start_index:global_end_index] == search_text:
                    replacements.append((global_start_index, global_end_index, patch["replace_with"]))
                    logger.debug(f"Patch #{patch_idx} (Line ref {line_num_ref}): Prepared. Global indices: {global_start_index}-{global_end_index}.")
                else:
                    logger.error(f"Patch #{patch_idx} (Line ref {line_num_ref}): Sanity check FAILED. Text at calculated global indices ('{original_text[global_start_index:global_end_index][:50]}...') does not match search_text ('{search_text[:50]}...'). Skipping patch.")
                    failed_patches_search_text_not_found += 1
            else:
                logger.warning(f"Patch #{patch_idx} (Line ref {line_num_ref}): `search_text` ('{search_text[:50]}...') NOT FOUND in localized snippet (Lines {search_line_start_idx+1 if line_num_ref else 'N/A'}-{search_line_end_idx if line_num_ref else 'N/A'}). Snippet: '{search_snippet_str[:100]}...'. Skipping patch.")
                failed_patches_search_text_not_found += 1
                # Log the problematic search_text and the snippet it was searched in for debugging
                debug_file_name = f"patch_search_fail_ch{patch.get('chapter_number_debug', 'N_A')}_patch{patch_idx}.txt"
                debug_content = f"Search Text:\n{search_text}\n\nLine Num Ref: {line_num_ref}\n\nSearched Snippet:\n{search_snippet_str}"
                # Consider saving this to a debug file if issues persist
                # await agent._save_debug_output(...) # Can't call async from sync; would need to pass agent or make it callable.
                # For now, rely on logger.

        except Exception as e:
            logger.error(f"Error processing patch #{patch_idx} (Line ref {line_num_ref}) for search_text '{search_text[:50]}...': {e}", exc_info=True)
            failed_patches_search_text_not_found +=1

    if failed_patches_search_text_not_found > 0:
         logger.warning(f"{failed_patches_search_text_not_found}/{len(patch_instructions) - (len([p for p in patch_instructions if p['search_text'] == 'N/A - General Issue']))} specific patches could not be applied because their search_text was not found in the designated original chapter area.")

    replacements.sort(key=lambda x: x[0], reverse=True)

    current_text_list = list(original_text)
    applied_count = 0
    for start_index, end_index, replace_with_text in replacements:
        # Basic overlap check (more robust needed for complex cases but good for now)
        # This check is less critical now because replacements are derived from original_text indices.
        # The main risk is if the LLM produces quotes that *semantically* overlap but are different strings.
        current_text_list[start_index:end_index] = list(replace_with_text)
        applied_count +=1
        logger.info(f"Applied patch: Replaced original text from {start_index}-{end_index} (len {end_index-start_index}) with new text (len {len(replace_with_text)}). Snippet: '{replace_with_text[:50]}...'")
    
    logger.info(f"Patch application: Applied {applied_count} patches successfully out of {len(replacements)} prepared.")
    return "".join(current_text_list)


async def revise_chapter_draft_logic(
    agent, 
    original_text: str, 
    chapter_number: int, 
    evaluation_result: EvaluationResult, 
    hybrid_context_for_revision: str, 
    chapter_plan: Optional[List[SceneDetail]]
) -> Optional[Tuple[str, str]]:
    """
    Attempts to revise a chapter using line-number aware patching or full rewrite.
    """
    if not original_text:
        logger.error(f"Revision for ch {chapter_number} cannot proceed: missing original text.")
        return None

    problems_to_fix: List[ProblemDetail] = evaluation_result.get("problems_found", [])
    if not problems_to_fix:
        logger.warning(f"Revision requested for ch {chapter_number}, but no specific problems_found in evaluation_result. Proceeding with full rewrite if evaluation.needs_revision is True.")
    
    revision_reason_str = "\n- ".join(evaluation_result["reasons"])
    if not revision_reason_str.strip():
         revision_reason_str = "General unspecified issues based on evaluation."
         logger.warning(f"Revision reason for ch {chapter_number} is empty. Using generic fallback for logs/prompts.")

    logger.warning(f"Attempting revision for chapter {chapter_number}. Reason(s) summary:\n- {revision_reason_str}")

    patched_text: Optional[str] = None
    raw_patch_llm_outputs_combined: str = "" 
    original_text_lines = original_text.splitlines(True) # Keep newlines for char indexing if needed later

    if config.ENABLE_PATCH_BASED_REVISION and problems_to_fix:
        logger.info(f"Patch-based revision enabled. Attempting to generate and apply patches for Ch {chapter_number} ({len(problems_to_fix)} problems identified).")
        
        patch_instructions = await _generate_patch_instructions_logic(
            agent, original_text_lines, problems_to_fix, chapter_number, hybrid_context_for_revision, chapter_plan
        )
        
        if patch_instructions:
            # Pass original_text (full string) for _apply_patches_to_text
            # It will split into lines internally if it still needs to, but the current impl uses char indices.
            patched_text = _apply_patches_to_text(original_text, patch_instructions)
            raw_patch_llm_outputs_combined = f"Chapter was revised using {len(patch_instructions)} generated patches. Individual LLM calls for patches are not combined here.\n"
            # Add details of patches for debugging
            for i, p_instr in enumerate(patch_instructions):
                raw_patch_llm_outputs_combined += (
                    f"  Patch {i+1}: Line Ref: {p_instr['line_number_ref']}, "
                    f"Search: '{p_instr['search_text'][:30]}...', Replace: '{p_instr['replace_with'][:30]}...'\n"
                )
            logger.info(f"Successfully applied {len(patch_instructions)} patches to Ch {chapter_number}. Original len: {len(original_text)}, Patched len: {len(patched_text)}.")
        else:
            logger.warning(f"Patch-based revision for Ch {chapter_number}: No valid patch instructions were generated. Will fallback to full rewrite if configured or fail.")
    
    final_revised_text: Optional[str] = None
    final_raw_llm_output: Optional[str] = None

    if patched_text is not None:
        # Further checks on patched_text (length, similarity)
        if len(patched_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
            logger.warning(f"Patched draft for ch {chapter_number} is too short ({len(patched_text)} chars). Min required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")
            # Decision to fallback or accept is made by the caller (NovelWriterAgent) after re-evaluation.

        sim_original_embedding_task = llm_interface.async_get_embedding(original_text)
        sim_patched_embedding_task = llm_interface.async_get_embedding(patched_text)
        sim_original_embedding, sim_patched_embedding = await asyncio.gather(sim_original_embedding_task, sim_patched_embedding_task)

        if sim_original_embedding is not None and sim_patched_embedding is not None:
            similarity_score = utils.numpy_cosine_similarity(sim_original_embedding, sim_patched_embedding)
            logger.info(f"Patched text similarity score with original draft: {similarity_score:.4f}")
            if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Patched text for ch {chapter_number} is too similar to original (Score: {similarity_score:.4f}). Patches may have been ineffective.")
        else:
            logger.warning(f"Could not get embeddings for patched text similarity check of ch {chapter_number}.")
        
        final_revised_text = patched_text
        final_raw_llm_output = raw_patch_llm_outputs_combined
    
    if final_revised_text is None: 
        if config.ENABLE_PATCH_BASED_REVISION:
             logger.warning(f"Patch-based revision did not produce a result for Ch {chapter_number}. Falling back to full chapter rewrite.")
        else:
             logger.info(f"Patch-based revision disabled. Proceeding with full chapter rewrite for Ch {chapter_number}.")

        original_text_limit = config.MAX_CONTEXT_LENGTH // 3 
        original_snippet = original_text[:original_text_limit].strip() + ("..." if len(original_text) > original_text_limit else "")
        
        plan_focus_section_full_rewrite = ""
        plot_point_focus_full_rewrite, _ = agent._get_plot_point_info(chapter_number) 

        if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
            try:
                plan_json_str_full = json.dumps(chapter_plan, indent=2, ensure_ascii=False)
                plan_snippet_full = plan_json_str_full[:(config.MAX_CONTEXT_LENGTH // 4)] 
                if len(plan_json_str_full) > len(plan_snippet_full):
                    plan_snippet_full += "\n... (plan truncated in prompt)"
                plan_focus_section_full_rewrite = f"**Original Detailed Scene Plan (Target - align with this while fixing issues):**\n```json\n{plan_snippet_full}\n```\n"
            except TypeError: 
                 plan_focus_section_full_rewrite = f"**Original Chapter Focus (Target):**\n{plot_point_focus_full_rewrite or 'Not specified.'}\n"
        else: 
            plan_focus_section_full_rewrite = f"**Original Chapter Focus (Target):**\n{plot_point_focus_full_rewrite or 'Not specified.'}\n"

        length_issue_explicit_instruction_full_rewrite = ""
        if any(kw in revision_reason_str.lower() for kw in ["too short", "lacking in depth", "brief", "expand", "length", "narrative depth", "detail"]):
            length_issue_explicit_instruction_full_rewrite = (
                "\n**Specific Focus on Expansion:** A key critique involves insufficient length or narrative depth. "
                "Your revision MUST substantially expand the narrative. This means: \n"
                "- Adding more detailed descriptions of settings, character appearances, and actions.\n"
                "- Fleshing out character thoughts, internal monologues, and emotional reactions.\n"
                "- Extending dialogue sequences, making them more nuanced and revealing.\n"
                "- Exploring the sensory details and emotional impact of events more thoroughly.\n"
                f"Do not just rephrase; aim to significantly increase the volume of narrative content to at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters, guided by the original scene plan and critique."
            )
            
        protagonist_name_full_rewrite = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        prompt_full_rewrite = f"""/no_think
You are a skilled revising author tasked with rewriting Chapter {chapter_number} of a novel featuring protagonist {protagonist_name_full_rewrite}.
**Critique/Reason(s) for Revision (These issues MUST be addressed comprehensively and demonstrably. Your revision should show clear changes related to these points.):**
--- FEEDBACK START ---
{llm_interface.clean_model_response(revision_reason_str).strip()}
--- FEEDBACK END ---
{length_issue_explicit_instruction_full_rewrite}
{plan_focus_section_full_rewrite}
**Hybrid Context from Previous Chapters (Semantic Context for Flow & KG Facts for Canon):**
--- BEGIN HYBRID CONTEXT ---
{hybrid_context_for_revision if hybrid_context_for_revision.strip() else "No previous context (e.g., Chapter 1)."}
--- END HYBRID CONTEXT ---

**Original Draft Snippet (for reference ONLY - your main goal is to address the critique and align with the plan/focus, NOT to make minimal changes):**
--- BEGIN ORIGINAL DRAFT SNIPPET ---
{original_snippet}
--- END ORIGINAL DRAFT SNIPPET ---

**Revision Instructions:**
1. **ABSOLUTE PRIORITY:** Thoroughly and demonstrably address ALL issues listed in the **Critique/Reason(s) for Revision**. The rewritten chapter must clearly show how these points were resolved.
2. **Rewrite the ENTIRE chapter text.** Do not just patch the original. Produce a fresh, coherent narrative that incorporates the fixes.
3. Align the rewritten chapter with the **Original Detailed Scene Plan** (if provided) or the **Original Chapter Focus**.
4. Ensure the revised chapter flows smoothly with the **Hybrid Context from Previous Chapters**.
   - Pay particular attention to the `KEY RELIABLE KG FACTS` section of the Hybrid Context for established canon.
   - Use the `SEMANTIC CONTEXT` section of the Hybrid Context for narrative flow and tone.
5. Maintain the established tone, style, and genre ('{agent.plot_outline.get('genre', 'story')}') of the novel.
6. The revised chapter should be substantial, aiming for at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters.
7. **Output ONLY the rewritten chapter text.** No "Chapter X" headers, titles, or meta-commentary.

--- BEGIN REVISED CHAPTER {chapter_number} TEXT ---
"""
        logger.info(f"Calling LLM ({config.REVISION_MODEL}) for Ch {chapter_number} full rewrite. Target minimum length: {config.MIN_ACCEPTABLE_DRAFT_LENGTH} chars.")
        revised_raw_llm_output_full = await llm_interface.async_call_llm(
            model_name=config.REVISION_MODEL,
            prompt=prompt_full_rewrite, 
            temperature=0.6,
            allow_fallback=True,
            stream_to_disk=True 
        ) 
        if not revised_raw_llm_output_full:
            logger.error(f"Full rewrite LLM call failed for ch {chapter_number} (returned empty).")
            return None
            
        revised_cleaned_text_full = llm_interface.clean_model_response(revised_raw_llm_output_full)
        
        final_revised_text = revised_cleaned_text_full
        final_raw_llm_output = revised_raw_llm_output_full

    if not final_revised_text or len(final_revised_text) < 50 :
        logger.error(f"Revision for ch {chapter_number} resulted in virtually no content after processing. Original content length: {len(original_text)}")
        await agent._save_debug_output(chapter_number, "revision_fail_empty_final_raw_llm", final_raw_llm_output or "No LLM output was generated/captured.")
        return None

    if len(final_revised_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        logger.warning(
             f"Revised draft for ch {chapter_number} is short ({len(final_revised_text)} chars) after processing. Min required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}. "
             f"This text will be returned for further evaluation by the agent."
        )
        
    if final_revised_text is not patched_text: 
        original_embedding_task_full = llm_interface.async_get_embedding(original_text)
        revised_embedding_task_full = llm_interface.async_get_embedding(final_revised_text)
        original_embedding_full, revised_embedding_full = await asyncio.gather(original_embedding_task_full, revised_embedding_task_full)

        if original_embedding_full is not None and revised_embedding_full is not None:
            similarity_score_full = utils.numpy_cosine_similarity(original_embedding_full, revised_embedding_full)
            logger.info(f"Full rewrite similarity score with original draft: {similarity_score_full:.4f}")
            if similarity_score_full >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Full rewrite for ch {chapter_number} is too similar to original (Score: {similarity_score_full:.4f}). This may indicate the LLM did not make sufficient changes.")
        else:
            logger.warning(f"Could not get embeddings for full rewrite similarity check of ch {chapter_number}.")
            
    logger.info(f"Revision process for ch {chapter_number} produced a candidate text (Length: {len(final_revised_text)} chars).")
    return final_revised_text, final_raw_llm_output
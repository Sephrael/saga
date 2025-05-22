# chapter_revision_logic.py
"""
Handles the revision of chapter drafts based on evaluation feedback for the SAGA system.
Supports both full rewrite and targeted patch-based revisions.
Context data for prompts is now formatted as plain text.
"""
import logging
import json # Retained for SceneDetail if it remains complex dict internally
import asyncio
import re
from typing import Tuple, Optional, List, Dict, Any

import config
import llm_interface 
import utils
from type import SceneDetail, ProblemDetail, PatchInstruction, EvaluationResult
# Import the plain text formatter for scene plan from chapter_drafting_logic
from chapter_drafting_logic import _format_scene_plan_for_prompt

logger = logging.getLogger(__name__)

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
        logger.warning(f"Quote for context window not found in text: '{quote[:50]}...' Returning full text snippet as fallback.")
        return text[:window_size_chars] + "..." if len(text) > window_size_chars else text


async def _generate_single_patch_instruction_llm(
    agent,
    original_chapter_text_snippet: str, 
    problem: ProblemDetail,
    chapter_number: int,
    hybrid_context_for_revision: str, 
    chapter_plan: Optional[List[SceneDetail]]
) -> Optional[PatchInstruction]:
    """Generates a single patch instruction (replace_with text) using an LLM."""

    plan_focus_section = ""
    plot_point_focus, _ = agent._get_plot_point_info(chapter_number)
    max_plan_tokens_for_patch_prompt = config.MAX_CONTEXT_TOKENS // 4

    if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
        plan_focus_section = _format_scene_plan_for_prompt(chapter_plan, config.PATCH_GENERATION_MODEL, max_plan_tokens_for_patch_prompt)
        if "plan truncated" in plan_focus_section: # Log if truncated specifically for patch prompt
             logger.warning(f"Scene plan was token-truncated for Ch {chapter_number} patch generation prompt.")
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
  - Suggested Fix Focus: {problem['suggested_fix_focus']}

**Text Snippet from Original Chapter (around the problem quote):**
--- BEGIN ORIGINAL TEXT SNIPPET ---
{original_chapter_text_snippet}
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

    logger.info(f"Calling LLM ({config.PATCH_GENERATION_MODEL}) for single patch in Ch {chapter_number}. Problem: {problem['problem_description'][:60]}... Quote: {problem['quote_from_original'][:60]}...")
    max_patch_output_tokens = config.MAX_GENERATION_TOKENS // 4

    replace_with_text_raw = await llm_interface.async_call_llm(
        model_name=config.PATCH_GENERATION_MODEL,
        prompt=prompt,
        temperature=0.6,
        max_tokens=max_patch_output_tokens,
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
            logger.warning(f"Patch for Ch {chapter_number} problem '{problem['problem_description'][:50]}...' did not sufficiently expand text (char length). Original len: {len(problem['quote_from_original'])}, New len: {len(replace_with_text_cleaned)}. May still use.")

    return {
        "search_text": problem['quote_from_original'],
        "replace_with": replace_with_text_cleaned,
        "original_quote_ref": problem['quote_from_original'],
        "reason_for_change": f"Fixing '{problem['issue_category']}' issue: {problem['problem_description']}"
    }


async def _generate_patch_instructions_logic(
    agent,
    original_text: str,
    problems_to_fix: List[ProblemDetail],
    chapter_number: int,
    hybrid_context_for_revision: str, 
    chapter_plan: Optional[List[SceneDetail]]
) -> List[PatchInstruction]:
    patch_instructions: List[PatchInstruction] = []
    problems_to_process = problems_to_fix[:config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE]
    if len(problems_to_fix) > config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE:
        logger.warning(f"Found {len(problems_to_fix)} problems for Ch {chapter_number}, but will only attempt to patch the first {config.MAX_PATCH_INSTRUCTIONS_TO_GENERATE}.")

    patch_generation_tasks = []
    for problem in problems_to_process:
        if problem["quote_from_original"] == "N/A - General Issue" and problem["issue_category"] != "narrative_depth":
            logger.info(f"Skipping patch generation for Ch {chapter_number} problem with 'N/A - General Issue' quote and category '{problem['issue_category']}' as it's not a narrative depth/length issue requiring general expansion focus.")
            continue
        if problem["quote_from_original"] == "N/A - General Issue" and "length" not in problem['problem_description'].lower() and "depth" not in problem['problem_description'].lower() and "expand" not in problem['suggested_fix_focus'].lower():
            logger.info(f"Skipping patch generation for 'N/A - General Issue' problem in Ch {chapter_number}: '{problem['problem_description']}'. This type of issue is hard to target with specific quote-based patches if not explicitly about length/depth needing expansion.")
            continue

        context_snippet = _get_context_window(original_text, problem["quote_from_original"], config.MAX_CHARS_FOR_PATCH_CONTEXT_WINDOW)
        task = _generate_single_patch_instruction_llm(
            agent, context_snippet, problem, chapter_number, hybrid_context_for_revision, chapter_plan
        )
        patch_generation_tasks.append(task)

    if not patch_generation_tasks:
        logger.info(f"No actionable problems with specific quotes found for patch generation in Ch {chapter_number}.")
        return []

    results = await asyncio.gather(*patch_generation_tasks, return_exceptions=True)

    for i, res_or_exc in enumerate(results):
        problem_ref = problems_to_process[i] 
        if isinstance(res_or_exc, Exception):
            logger.error(f"Error generating patch for Ch {chapter_number} problem '{problem_ref['problem_description'][:50]}...': {res_or_exc}", exc_info=res_or_exc)
        elif res_or_exc is not None:
            patch_instructions.append(res_or_exc)
        else:
            logger.warning(f"Patch generation returned None for Ch {chapter_number} problem: {problem_ref['problem_description'][:50]}...")

    logger.info(f"Generated {len(patch_instructions)} patch instructions for Ch {chapter_number}.")
    return patch_instructions

def _apply_patches_to_text(original_text: str, patch_instructions: List[PatchInstruction]) -> str:
    """Applies a list of patch instructions to the original text."""
    if not patch_instructions:
        return original_text

    replacements: List[Tuple[int, int, str]] = []
    failed_patches_search_text_not_found = 0

    # Filter out patches with "N/A - General Issue" as search_text, as they cannot be applied directly
    applicable_patches = [p for p in patch_instructions if p["search_text"] != "N/A - General Issue"]


    for patch in applicable_patches:
        try:
            # Use a more robust way to find all occurrences and decide which one to patch if multiple exist.
            # For now, just patch the first one found.
            # If quotes are truly verbatim and specific, first match should usually be correct.
            match_indices = [m.start() for m in re.finditer(re.escape(patch["search_text"]), original_text)]
            if not match_indices:
                logger.warning(f"Patching: `search_text` not found in original text. Skipping patch. Search: '{patch['search_text'][:100]}...'")
                failed_patches_search_text_not_found += 1
                continue

            start_index = match_indices[0] # Patch first occurrence
            end_index = start_index + len(patch["search_text"])
            replacements.append((start_index, end_index, patch["replace_with"]))
            logger.debug(f"Prepared patch: Replace original text from {start_index} to {end_index} with '{patch['replace_with'][:50]}...'")

        except Exception as e:
            logger.error(f"Error preparing patch for search_text '{patch['search_text'][:50]}...': {e}", exc_info=True)
            failed_patches_search_text_not_found +=1

    if failed_patches_search_text_not_found > 0 and applicable_patches:
         logger.warning(f"{failed_patches_search_text_not_found}/{len(applicable_patches)} applicable patches could not be applied because their search_text was not found in the original chapter.")

    if not replacements: # No applicable patches were successfully prepared
        return original_text

    replacements.sort(key=lambda x: x[0], reverse=True)

    current_text_list = list(original_text)
    for start_index, end_index, replace_with_text in replacements:
        current_text_list[start_index:end_index] = list(replace_with_text)
        logger.info(f"Applied patch: Replaced original text from {start_index}-{end_index} (len {end_index-start_index}) with new text (len {len(replace_with_text)}). Snippet: '{replace_with_text[:50]}...'")

    return "".join(current_text_list)


async def revise_chapter_draft_logic(
    agent,
    original_text: str,
    chapter_number: int,
    evaluation_result: EvaluationResult,
    hybrid_context_for_revision: str, 
    chapter_plan: Optional[List[SceneDetail]]
) -> Optional[Tuple[str, str]]:
    if not original_text:
        logger.error(f"Revision for ch {chapter_number} cannot proceed: missing original text.")
        return None

    problems_to_fix: List[ProblemDetail] = evaluation_result.get("problems_found", [])
    if not problems_to_fix and evaluation_result.get("needs_revision"): # Needs revision but no specific problems
        logger.warning(f"Revision requested for ch {chapter_number}, but no specific problems_found in evaluation_result, though needs_revision is True. Proceeding with full rewrite.")
    elif not problems_to_fix: # No revision needed based on problems
        logger.info(f"No specific problems found for ch {chapter_number}. No revision attempted based on problem list.")
        # The main loop should decide if a full rewrite is needed based on 'needs_revision' flag alone
        # For now, if no problems, we assume no patch-based revision.
        # If 'needs_revision' is true due to general reasons (e.g. length, coherence), full rewrite is the path.

    revision_reason_str = "\n- ".join(evaluation_result["reasons"])
    if not revision_reason_str.strip():
         revision_reason_str = "General unspecified issues based on evaluation."
         logger.warning(f"Revision reason for ch {chapter_number} is empty. Using generic fallback for logs/prompts.")

    logger.warning(f"Attempting revision for chapter {chapter_number}. Reason(s) summary:\n- {revision_reason_str}")

    patched_text: Optional[str] = None
    raw_patch_llm_outputs_combined: str = ""

    actionable_problems_for_patching = [p for p in problems_to_fix if p["quote_from_original"] != "N/A - General Issue" or p["issue_category"] == "narrative_depth"]

    if config.ENABLE_PATCH_BASED_REVISION and actionable_problems_for_patching:
        logger.info(f"Patch-based revision enabled. Attempting to generate and apply patches for Ch {chapter_number} ({len(actionable_problems_for_patching)} actionable problems identified).")

        patch_instructions = await _generate_patch_instructions_logic(
            agent, original_text, actionable_problems_for_patching, chapter_number, hybrid_context_for_revision, chapter_plan
        )

        if patch_instructions:
            patched_text = _apply_patches_to_text(original_text, patch_instructions)
            raw_patch_llm_outputs_combined = f"Chapter was revised using {len(patch_instructions)} generated patches. Individual LLM calls for patches are not combined here.\n"
            logger.info(f"Successfully applied {len(patch_instructions)} patches to Ch {chapter_number}. Original len: {len(original_text)}, Patched len: {len(patched_text)}.")
        else:
            logger.warning(f"Patch-based revision for Ch {chapter_number}: No valid patch instructions were generated. Will fallback to full rewrite if configured or fail.")

    final_revised_text: Optional[str] = None
    final_raw_llm_output: Optional[str] = None

    if patched_text is not None:
        if len(patched_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH: 
            logger.warning(f"Patched draft for ch {chapter_number} is too short ({len(patched_text)} chars). Min required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")

        sim_original_embedding_task = llm_interface.async_get_embedding(original_text)
        sim_patched_embedding_task = llm_interface.async_get_embedding(patched_text)
        sim_original_embedding, sim_patched_embedding = await asyncio.gather(sim_original_embedding_task, sim_patched_embedding_task)

        if sim_original_embedding is not None and sim_patched_embedding is not None:
            similarity_score = utils.numpy_cosine_similarity(sim_original_embedding, sim_patched_embedding)
            logger.info(f"Patched text similarity score with original draft: {similarity_score:.4f}")
            if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE:
                logger.warning(f"Patched text for ch {chapter_number} is too similar to original (Score: {similarity_score:.4f} >= Threshold: {config.REVISION_SIMILARITY_ACCEPTANCE}). Patches may have been ineffective.")
        else:
            logger.warning(f"Could not get embeddings for patched text similarity check of ch {chapter_number}.")

        final_revised_text = patched_text
        final_raw_llm_output = raw_patch_llm_outputs_combined
    else: 
        if config.ENABLE_PATCH_BASED_REVISION and actionable_problems_for_patching : # Only log fallback if patching was attempted
             logger.warning(f"Patch-based revision did not produce a result for Ch {chapter_number}. Falling back to full chapter rewrite.")
        elif not actionable_problems_for_patching and evaluation_result.get("needs_revision"):
             logger.info(f"No actionable problems for patching in Ch {chapter_number}, but revision needed. Proceeding with full chapter rewrite.")
        elif not config.ENABLE_PATCH_BASED_REVISION and evaluation_result.get("needs_revision"):
             logger.info(f"Patch-based revision disabled and revision needed. Proceeding with full chapter rewrite for Ch {chapter_number}.")
        else: # No revision needed, or patching disabled and no revision flag.
            logger.info(f"No revision performed for Ch {chapter_number} (no patches generated/applied and no explicit fallback to full rewrite triggered by this function's logic here).")
            return None # No revision performed


        max_original_snippet_tokens = config.MAX_CONTEXT_TOKENS // 3
        original_snippet = llm_interface.truncate_text_by_tokens(
            original_text,
            config.REVISION_MODEL,
            max_original_snippet_tokens,
            truncation_marker="\n... (original draft snippet truncated due to token limit)"
        )

        plan_focus_section_full_rewrite = ""
        plot_point_focus_full_rewrite, _ = agent._get_plot_point_info(chapter_number)
        max_plan_tokens_for_full_rewrite = config.MAX_CONTEXT_TOKENS // 3

        if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
            plan_focus_section_full_rewrite = _format_scene_plan_for_prompt(chapter_plan, config.REVISION_MODEL, max_plan_tokens_for_full_rewrite)
            if "plan truncated" in plan_focus_section_full_rewrite:
                 logger.warning(f"Scene plan was token-truncated for Ch {chapter_number} full rewrite prompt.")
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

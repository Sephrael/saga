# chapter_revision_logic.py
"""
Handles the revision of chapter drafts based on evaluation feedback for the SAGA system.
"""
import logging
import json
import asyncio
from typing import Tuple, Optional, List

import config
import llm_interface
import utils # For numpy_cosine_similarity
from types import SceneDetail # Assuming this is in types.py

logger = logging.getLogger(__name__)

async def revise_chapter_draft_logic(agent, original_text: str, chapter_number: int, revision_reason: str, context_from_previous: str, chapter_plan: Optional[List[SceneDetail]]) -> Optional[Tuple[str, str]]:
    """Attempts to revise a chapter based on evaluation feedback.
    'agent' is an instance of NovelWriterAgent.
    Returns (revised_cleaned_text, revised_raw_llm_output) or None if revision fails.
    """
    if not original_text or not revision_reason:
        logger.error(f"Revision for ch {chapter_number} cannot proceed: missing original text or revision reason.")
        return None
    
    clean_reason = llm_interface.clean_model_response(revision_reason).strip()
    if not clean_reason:
        logger.error(f"Revision reason for ch {chapter_number} is empty after cleaning. Cannot proceed with revision.")
        return None
        
    logger.warning(f"Attempting revision for chapter {chapter_number}. Reason(s):\n{clean_reason}")
    
    context_limit = config.MAX_CONTEXT_LENGTH // 4  
    original_text_limit = config.MAX_CONTEXT_LENGTH // 2
    
    context_snippet = context_from_previous[:context_limit].strip() + ("..." if len(context_from_previous) > context_limit else "")
    original_snippet = original_text[:original_text_limit].strip() + ("..." if len(original_text) > original_text_limit else "")
    
    plan_focus_section = ""
    plot_point_focus, _ = agent._get_plot_point_info(chapter_number) 

    if config.ENABLE_AGENTIC_PLANNING and chapter_plan:
        try:
            plan_json_str = json.dumps(chapter_plan, indent=2, ensure_ascii=False)
            plan_snippet_for_prompt = plan_json_str[:(config.MAX_CONTEXT_LENGTH // 4)] 
            if len(plan_json_str) > len(plan_snippet_for_prompt):
                plan_snippet_for_prompt += "\n... (plan truncated)"
            plan_focus_section = f"**Original Detailed Scene Plan (Target - align with this while fixing issues):**\n```json\n{plan_snippet_for_prompt}\n```\n"
        except TypeError: 
             plan_focus_section = f"**Original Chapter Focus (Target):**\n{plot_point_focus or 'Not specified.'}\n"
    else: 
        plan_focus_section = f"**Original Chapter Focus (Target):**\n{plot_point_focus or 'Not specified.'}\n"
        
    protagonist_name = agent.plot_outline.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
    prompt = f"""/no_think
You are a skilled revising author tasked with rewriting Chapter {chapter_number} of a novel featuring protagonist {protagonist_name}.
**Critique/Reason(s) for Revision (These issues MUST be addressed comprehensively):**
--- FEEDBACK START ---
{clean_reason}
--- FEEDBACK END ---

{plan_focus_section}
**Context from Previous Chapters (for flow and continuity):**
--- BEGIN CONTEXT ---
{context_snippet if context_snippet else "No previous context (e.g., Chapter 1)."}
--- END CONTEXT ---

**Original Draft Snippet (for reference ONLY - your main goal is to address the critique and align with the plan/focus):**
--- BEGIN ORIGINAL DRAFT SNIPPET ---
{original_snippet}
--- END ORIGINAL DRAFT SNIPPET ---

**Revision Instructions:**
1. **PRIORITY:** Thoroughly address all issues listed in the **Critique/Reason(s) for Revision**.
2. **Rewrite the ENTIRE chapter text.** Do not just patch the original.
3. Align the rewritten chapter with the **Original Detailed Scene Plan** (if provided) or the **Original Chapter Focus**.
4. Ensure the revised chapter flows smoothly with the **Context from Previous Chapters**.
5. Maintain the established tone, style, and genre ('{agent.plot_outline.get('genre', 'story')}') of the novel.
6. The revised chapter should be substantial, aiming for at least {config.MIN_ACCEPTABLE_DRAFT_LENGTH} characters.
7. **Output ONLY the rewritten chapter text.** No "Chapter X" headers, titles, or meta-commentary.

--- BEGIN REVISED CHAPTER {chapter_number} TEXT ---
"""
    revised_raw_llm_output = await llm_interface.async_call_llm(
        model_name=config.REVISION_MODEL,
        prompt=prompt, 
        temperature=0.6 
    ) 
    if not revised_raw_llm_output:
        logger.error(f"Revision LLM call failed for ch {chapter_number} (returned empty).")
        return None
        
    revised_cleaned_text = llm_interface.clean_model_response(revised_raw_llm_output)
    if not revised_cleaned_text or len(revised_cleaned_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH:
        logger.error(f"Revised draft for ch {chapter_number} is too short ({len(revised_cleaned_text or '')} chars) after cleaning. Min required: {config.MIN_ACCEPTABLE_DRAFT_LENGTH}.")
        await agent._save_debug_output(chapter_number, "revision_fail_short_raw_llm", revised_raw_llm_output)
        return None
    
    original_embedding_task = llm_interface.async_get_embedding(original_text)
    revised_embedding_task = llm_interface.async_get_embedding(revised_cleaned_text)
    original_embedding, revised_embedding = await asyncio.gather(original_embedding_task, revised_embedding_task)

    if original_embedding is not None and revised_embedding is not None:
        similarity_score = utils.numpy_cosine_similarity(original_embedding, revised_embedding)
        logger.info(f"Revision similarity score with original draft: {similarity_score:.4f}")
        if similarity_score >= config.REVISION_SIMILARITY_ACCEPTANCE:
            logger.warning(f"Revision for ch {chapter_number} rejected: Too similar to original (Score: {similarity_score:.4f} >= Threshold: {config.REVISION_SIMILARITY_ACCEPTANCE}).")
            await agent._save_debug_output(chapter_number, "revision_rejected_similar_raw_llm", revised_raw_llm_output)
            return None 
    else:
        logger.warning(f"Could not get embeddings for revision similarity check of ch {chapter_number}. Accepting revision by default.")
        
    logger.info(f"Revision for ch {chapter_number} accepted (Length: {len(revised_cleaned_text)} chars).")
    return revised_cleaned_text, revised_raw_llm_output
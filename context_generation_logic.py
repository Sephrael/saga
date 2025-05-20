# context_generation_logic.py
"""
Handles the generation of contextual information for chapter writing in the SAGA system.
Now includes a hybrid approach combining semantic context and Knowledge Graph facts.
"""
import logging
import asyncio
from typing import List, Optional

import config
import llm_interface # Now needed for token counting/truncation
import utils
from state_manager import state_manager
from prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt
from type import SceneDetail

logger = logging.getLogger(__name__)

async def _generate_semantic_chapter_context_logic(agent, current_chapter_number: int) -> str:
    """
    Constructs SEMANTIC context for the current chapter from previous summaries/text.
    'agent' is an instance of NovelWriterAgent.
    This function will now be mindful of token limits for its output.
    """
    if current_chapter_number <= 1:
        return ""
    logger.debug(f"Retrieving and constructing SEMANTIC context for Chapter {current_chapter_number}...")

    plot_point_focus, plot_point_index = agent._get_plot_point_info(current_chapter_number)
    context_query_text = plot_point_focus if plot_point_focus else f"Narrative context relevant to events leading up to chapter {current_chapter_number}."

    if plot_point_focus:
        logger.info(f"Semantic context query for ch {current_chapter_number} based on Plot Point {plot_point_index + 1}: '{context_query_text[:100]}...'")
    else:
        logger.warning(f"No specific plot point found for ch {current_chapter_number}. Using generic semantic context query.")

    query_embedding = await llm_interface.async_get_embedding(context_query_text)

    # Define max tokens for the semantic context part (e.g., 2/3 of total hybrid context budget)
    # The overall hybrid context is capped at MAX_CONTEXT_TOKENS.
    # This semantic part contributes to that, so it should be less than MAX_CONTEXT_TOKENS.
    # Let's say hybrid context can be up to MAX_CONTEXT_TOKENS, and semantic part is roughly 2/3 of that.
    # The KG facts part is usually smaller.
    max_semantic_tokens = (config.MAX_CONTEXT_TOKENS * 2) // 3


    if query_embedding is None:
        logger.warning("Failed to generate embedding for semantic context query. Falling back to sequential previous chapter summaries/text for semantic portion.")
        context_parts: List[str] = []
        total_tokens_accumulated = 0
        fallback_chapter_limit = config.CONTEXT_CHAPTER_COUNT

        for i in range(max(1, current_chapter_number - fallback_chapter_limit), current_chapter_number):
            if total_tokens_accumulated >= max_semantic_tokens: break
            chap_data = await state_manager.async_get_chapter_data_from_db(i)
            if chap_data:
                content = (chap_data.get('summary') or chap_data.get('text', '')).strip()
                is_prov = chap_data.get('is_provisional', False)
                ctype = "Provisional Summary" if chap_data.get('summary') and is_prov else \
                        "Summary" if chap_data.get('summary') else \
                        "Provisional Text Snippet" if is_prov else "Text Snippet"

                if content:
                    prefix = f"[Fallback Semantic Context from Chapter {i} ({ctype})]:\n"; suffix = "\n---\n"
                    full_content_part = f"{prefix}{content}{suffix}"
                    
                    # Check tokens for this part
                    part_tokens = llm_interface.count_tokens(full_content_part, config.DRAFTING_MODEL) # Assuming drafting model consumes this context

                    if total_tokens_accumulated + part_tokens <= max_semantic_tokens:
                        context_parts.append(full_content_part)
                        total_tokens_accumulated += part_tokens
                    else: # Truncate this part to fit remaining token budget
                        remaining_tokens = max_semantic_tokens - total_tokens_accumulated
                        if remaining_tokens > llm_interface.count_tokens(prefix + suffix, config.DRAFTING_MODEL) + 10: # Min 10 tokens for content
                            truncated_content_part = llm_interface.truncate_text_by_tokens(
                                full_content_part, config.DRAFTING_MODEL, remaining_tokens
                            )
                            context_parts.append(truncated_content_part)
                            total_tokens_accumulated += remaining_tokens # Approximate
                        break # No more space
        
        final_semantic_context = "\n".join(reversed(context_parts)).strip()
        final_tokens = llm_interface.count_tokens(final_semantic_context, config.DRAFTING_MODEL)
        logger.info(f"Constructed fallback semantic context: {final_tokens} tokens.")
        return final_semantic_context

    past_embeddings = await state_manager.async_get_all_past_embeddings(current_chapter_number)
    if not past_embeddings:
        logger.info("No past embeddings found for semantic context search.")
        return ""

    similarities = sorted(
        [(chap_num, utils.numpy_cosine_similarity(query_embedding, emb))
         for chap_num, emb in past_embeddings if emb is not None],
        key=lambda item: item[1],
        reverse=True
    )
    if not similarities:
        logger.info("No valid similarities found with past embeddings for semantic context.")
        return ""

    top_n_indices = [cs[0] for cs in similarities[:config.CONTEXT_CHAPTER_COUNT]]
    logger.info(f"Top {len(top_n_indices)} relevant chapters for SEMANTIC context (semantic search): {top_n_indices} (Scores: {[f'{s:.3f}' for _, s in similarities[:config.CONTEXT_CHAPTER_COUNT]]})")

    immediate_prev_chap_num = current_chapter_number - 1
    if immediate_prev_chap_num > 0 and immediate_prev_chap_num not in top_n_indices:
        top_n_indices.append(immediate_prev_chap_num)
        logger.debug(f"Added immediate previous chapter {immediate_prev_chap_num} to semantic context list.")

    chapters_to_fetch = sorted(list(set(top_n_indices)), reverse=True)
    logger.debug(f"Final list of chapters to fetch for SEMANTIC context: {chapters_to_fetch}")

    context_parts: List[str] = []
    total_tokens_accumulated = 0

    chap_data_tasks = {
        chap_num: state_manager.async_get_chapter_data_from_db(chap_num)
        for chap_num in chapters_to_fetch
    }
    chap_data_results_list = await asyncio.gather(*chap_data_tasks.values())
    chap_data_map = dict(zip(chap_data_tasks.keys(), chap_data_results_list))

    for chap_num in chapters_to_fetch:
        if total_tokens_accumulated >= max_semantic_tokens: break
        chap_data = chap_data_map.get(chap_num)
        if chap_data:
            content = (chap_data.get('summary') or chap_data.get('text', '')).strip()
            is_prov = chap_data.get('is_provisional', False)
            ctype = "Provisional Summary" if chap_data.get('summary') and is_prov else \
                    "Summary" if chap_data.get('summary') else \
                    "Provisional Text Snippet" if is_prov else "Text Snippet"

            if content:
                prefix = f"[Semantic Context from Chapter {chap_num} ({ctype})]:\n"; suffix = "\n---\n"
                full_content_part = f"{prefix}{content}{suffix}"
                part_tokens = llm_interface.count_tokens(full_content_part, config.DRAFTING_MODEL)

                if total_tokens_accumulated + part_tokens <= max_semantic_tokens:
                    context_parts.append(full_content_part)
                    total_tokens_accumulated += part_tokens
                else:
                    remaining_tokens = max_semantic_tokens - total_tokens_accumulated
                    if remaining_tokens > llm_interface.count_tokens(prefix + suffix, config.DRAFTING_MODEL) + 10:
                        truncated_content_part = llm_interface.truncate_text_by_tokens(
                            full_content_part, config.DRAFTING_MODEL, remaining_tokens
                        )
                        context_parts.append(truncated_content_part)
                        total_tokens_accumulated += remaining_tokens # Approximate
                    break
                logger.debug(f"Added SEMANTIC context from ch {chap_num} ({ctype}), {part_tokens} tokens. Total semantic tokens: {total_tokens_accumulated}.")
        else:
            logger.warning(f"Could not retrieve chapter data for ch {chap_num} during SEMANTIC context construction.")

    final_semantic_context = "\n".join(reversed(context_parts)).strip()
    final_tokens = llm_interface.count_tokens(final_semantic_context, config.DRAFTING_MODEL)
    logger.info(f"Constructed final SEMANTIC context: {final_tokens} tokens from chapters {chapters_to_fetch}.")
    return final_semantic_context


async def generate_hybrid_chapter_context_logic(agent, current_chapter_number: int, chapter_plan: Optional[List[SceneDetail]]) -> str:
    """
    Constructs HYBRID context for the current chapter by combining:
    1. Semantic context from previous chapter summaries/text.
    2. Reliable Knowledge Graph facts relevant to the upcoming chapter.
    The total length of this hybrid context will be capped by config.MAX_CONTEXT_TOKENS.
    'agent' is an instance of NovelWriterAgent.
    'chapter_plan' is the plan for the current_chapter_number, used by KG fact getter.
    """
    if current_chapter_number <= 0:
        return ""

    logger.info(f"Generating HYBRID context for Chapter {current_chapter_number}...")

    semantic_context_task = _generate_semantic_chapter_context_logic(agent, current_chapter_number)
    # KG facts getter needs to be aware of model for token counting if it were to also truncate.
    # For now, assume KG facts are relatively small.
    kg_facts_task = get_reliable_kg_facts_for_drafting_prompt(agent, current_chapter_number, chapter_plan)

    semantic_context_str, kg_facts_str = await asyncio.gather(
        semantic_context_task,
        kg_facts_task
    )

    hybrid_context_parts = []
    if semantic_context_str and semantic_context_str.strip():
        hybrid_context_parts.append("--- SEMANTIC CONTEXT FROM PAST CHAPTERS (FOR NARRATIVE FLOW & TONE) ---")
        hybrid_context_parts.append(semantic_context_str)
        hybrid_context_parts.append("--- END SEMANTIC CONTEXT ---")
    else:
        hybrid_context_parts.append("--- SEMANTIC CONTEXT FROM PAST CHAPTERS (FOR NARRATIVE FLOW & TONE) ---")
        hybrid_context_parts.append("No relevant semantic context could be retrieved.")
        hybrid_context_parts.append("--- END SEMANTIC CONTEXT ---")

    if kg_facts_str and kg_facts_str.strip():
        hybrid_context_parts.append("\n\n--- KEY RELIABLE KG FACTS (FOR ESTABLISHED CANON & CONTINUITY) ---")
        cleaned_kg_facts = kg_facts_str.split("\n", 1)[-1] if kg_facts_str.startswith("**Key Reliable KG Facts") else kg_facts_str
        if not cleaned_kg_facts.strip() or cleaned_kg_facts.lower().startswith("no specific reliable kg facts"):
            hybrid_context_parts.append("No specific reliable KG facts were identified as highly relevant for this chapter's focus.")
        else:
            hybrid_context_parts.append(cleaned_kg_facts)
        hybrid_context_parts.append("--- END KEY RELIABLE KG FACTS ---")
    else:
        hybrid_context_parts.append("\n\n--- KEY RELIABLE KG FACTS (FOR ESTABLISHED CANON & CONTINUITY) ---")
        hybrid_context_parts.append("Knowledge Graph fact retrieval did not yield specific results for this chapter.")
        hybrid_context_parts.append("--- END KEY RELIABLE KG FACTS ---")

    final_hybrid_context = "\n".join(hybrid_context_parts).strip()

    # Final safeguard for total context tokens
    # Assuming the context is for the DRAFTING_MODEL
    num_hybrid_tokens = llm_interface.count_tokens(final_hybrid_context, config.DRAFTING_MODEL)
    if num_hybrid_tokens > config.MAX_CONTEXT_TOKENS:
        logger.warning(
            f"Hybrid context token count ({num_hybrid_tokens}) exceeds MAX_CONTEXT_TOKENS ({config.MAX_CONTEXT_TOKENS}). Truncating."
        )
        final_hybrid_context = llm_interface.truncate_text_by_tokens(
            final_hybrid_context,
            config.DRAFTING_MODEL,
            config.MAX_CONTEXT_TOKENS,
            truncation_marker="\n... (Hybrid context truncated due to token limit)"
        )
        num_hybrid_tokens = llm_interface.count_tokens(final_hybrid_context, config.DRAFTING_MODEL) # Re-count after truncation


    logger.info(f"Generated HYBRID context for Chapter {current_chapter_number}, Tokens (est.): {num_hybrid_tokens}.")
    return final_hybrid_context
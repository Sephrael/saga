# context_generation_logic.py
"""
Handles the generation of contextual information for chapter writing in the SAGA system.
"""
import logging
import asyncio
from typing import List

import config
import llm_interface
import utils # For numpy_cosine_similarity

logger = logging.getLogger(__name__)

async def generate_chapter_context_logic(agent, current_chapter_number: int) -> str:
    """Constructs context for the current chapter from previous summaries/text and KG.
    'agent' is an instance of NovelWriterAgent.
    """
    if current_chapter_number <= 1:
        return "" 
    logger.debug(f"Retrieving and constructing context for Chapter {current_chapter_number}...")
    
    plot_point_focus, plot_point_index = agent._get_plot_point_info(current_chapter_number)
    context_query_text = plot_point_focus if plot_point_focus else f"Narrative context relevant to events leading up to chapter {current_chapter_number}."
    
    if plot_point_focus:
        logger.info(f"Context query for ch {current_chapter_number} based on Plot Point {plot_point_index + 1}: '{context_query_text[:100]}...'")
    else:
        logger.warning(f"No specific plot point found for ch {current_chapter_number}. Using generic context query.")
        
    query_embedding = await llm_interface.async_get_embedding(context_query_text)
    
    if query_embedding is None:
        logger.warning("Failed to generate embedding for context query. Falling back to sequential previous chapter summaries/text.")
        context_parts: List[str] = []
        total_chars = 0
        for i in range(max(1, current_chapter_number - config.CONTEXT_CHAPTER_COUNT), current_chapter_number):
            if total_chars >= config.MAX_CONTEXT_LENGTH: break
            chap_data = await agent.db_manager.async_get_chapter_data_from_db(i)
            if chap_data:
                content = (chap_data.get('summary') or chap_data.get('text', '')).strip()
                is_prov = chap_data.get('is_provisional', False)
                ctype = "Provisional Summary" if chap_data.get('summary') and is_prov else \
                        "Summary" if chap_data.get('summary') else \
                        "Provisional Text Snippet" if is_prov else "Text Snippet"
                
                if content:
                    prefix = f"[Fallback Context from Chapter {i} ({ctype})]:\n"; suffix = "\n---\n"
                    available_space = config.MAX_CONTEXT_LENGTH - total_chars - (len(prefix) + len(suffix))
                    if available_space <= 0: break
                    
                    truncated_content = content[:available_space]
                    context_parts.append(f"{prefix}{truncated_content}{suffix}")
                    total_chars += len(prefix) + len(truncated_content) + len(suffix)
        
        final_context = "\n".join(reversed(context_parts)).strip() 
        logger.info(f"Constructed fallback context: {len(final_context)} chars.")
        return final_context
        
    past_embeddings = await agent.db_manager.async_get_all_past_embeddings(current_chapter_number)
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
        logger.info("No valid similarities found with past embeddings.")
        return ""
    
    top_n_indices = [cs[0] for cs in similarities[:config.CONTEXT_CHAPTER_COUNT]]
    logger.info(f"Top {len(top_n_indices)} relevant chapters for context (semantic search): {top_n_indices} (Scores: {[f'{s:.3f}' for _, s in similarities[:config.CONTEXT_CHAPTER_COUNT]]})")
    
    immediate_prev_chap_num = current_chapter_number - 1
    if immediate_prev_chap_num > 0 and immediate_prev_chap_num not in top_n_indices:
        top_n_indices.append(immediate_prev_chap_num)
        logger.debug(f"Added immediate previous chapter {immediate_prev_chap_num} to context list.")
        
    chapters_to_fetch = sorted(list(set(top_n_indices)), reverse=True) 
    logger.debug(f"Final list of chapters to fetch for context: {chapters_to_fetch}")
    
    context_parts: List[str] = []
    total_chars = 0
    
    chap_data_tasks = {
        chap_num: agent.db_manager.async_get_chapter_data_from_db(chap_num) 
        for chap_num in chapters_to_fetch
    }
    chap_data_results_list = await asyncio.gather(*chap_data_tasks.values())
    chap_data_map = dict(zip(chap_data_tasks.keys(), chap_data_results_list))

    for chap_num in chapters_to_fetch: 
        if total_chars >= config.MAX_CONTEXT_LENGTH: break
        chap_data = chap_data_map.get(chap_num)
        if chap_data:
            content = (chap_data.get('summary') or chap_data.get('text', '')).strip()
            is_prov = chap_data.get('is_provisional', False)
            ctype = "Provisional Summary" if chap_data.get('summary') and is_prov else \
                    "Summary" if chap_data.get('summary') else \
                    "Provisional Text Snippet" if is_prov else "Text Snippet"
            
            if content:
                prefix = f"[Context from Chapter {chap_num} ({ctype})]:\n"; suffix = "\n---\n"
                available_space = config.MAX_CONTEXT_LENGTH - total_chars - (len(prefix) + len(suffix))
                if available_space <= 0: break 

                truncated_content = content[:available_space]
                context_parts.append(f"{prefix}{truncated_content}{suffix}")
                total_chars += len(prefix) + len(truncated_content) + len(suffix)
                logger.debug(f"Added context from ch {chap_num} ({ctype}), {len(truncated_content)} chars. Total context chars: {total_chars}.")
        else:
            logger.warning(f"Could not retrieve chapter data for ch {chap_num} during context construction.")
            
    final_context = "\n".join(reversed(context_parts)).strip() 
    logger.info(f"Constructed final semantic context: {len(final_context)} chars from chapters {chapters_to_fetch}.")
    return final_context
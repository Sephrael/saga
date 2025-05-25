# context_generation_logic.py
"""
Handles the generation of contextual information for chapter writing in the SAGA system.
Now includes a hybrid approach combining semantic context and Knowledge Graph facts.
MODIFIED: To handle 'agent_or_props' and ensure it's passed correctly.
MODIFIED: To use Neo4j vector similarity search for semantic context.
"""
import logging
import asyncio
from typing import List, Optional, Dict, Any 

import config
import llm_interface
# import utils # numpy_cosine_similarity no longer needed here
# from state_manager import state_manager # No longer directly used
from data_access import chapter_queries # For chapter data and similarity search
from prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt 
from type import SceneDetail

logger = logging.getLogger(__name__)

def _get_nested_prop_from_agent_or_props(agent_or_props: Any, primary_key: str, secondary_key: str, default: Any = None) -> Any:
    """Helper to get a nested property from an agent-like object or a dictionary."""
    primary_data = getattr(agent_or_props, primary_key, None) if not isinstance(agent_or_props, dict) else agent_or_props.get(primary_key, {})
    if isinstance(primary_data, dict):
        return primary_data.get(secondary_key, default)
    return default

async def _generate_semantic_chapter_context_logic(agent_or_props: Any, current_chapter_number: int) -> str:
    """
    Constructs SEMANTIC context for the current chapter from previous summaries/text,
    using Neo4j vector similarity search.
    'agent_or_props' can be the NANA_Orchestrator instance or a novel_props dictionary.
    This function will now be mindful of token limits for its output.
    """
    if current_chapter_number <= 1:
        return ""
    logger.debug(f"Retrieving and constructing SEMANTIC context for Chapter {current_chapter_number} via Neo4j vector search...")

    plot_outline_data = getattr(agent_or_props, 'plot_outline', None)
    if not plot_outline_data and isinstance(agent_or_props, dict): 
        plot_outline_data = agent_or_props.get('plot_outline_full', agent_or_props.get('plot_outline', {}))

    plot_points = []
    if isinstance(plot_outline_data, dict):
        plot_points = plot_outline_data.get("plot_points", [])

    plot_point_focus = None
    plot_point_index = -1
    if plot_points and isinstance(plot_points, list) and current_chapter_number > 0:
        idx = min(current_chapter_number - 1, len(plot_points) - 1)
        if 0 <= idx < len(plot_points):
            plot_point_focus = str(plot_points[idx]) if plot_points[idx] is not None else None
            plot_point_index = idx

    context_query_text = plot_point_focus if plot_point_focus else f"Narrative context relevant to events leading up to chapter {current_chapter_number}."

    if plot_point_focus:
        logger.info(f"Semantic context query for ch {current_chapter_number} based on Plot Point {plot_point_index + 1}: '{context_query_text[:100]}...'")
    else:
        logger.warning(f"No specific plot point found for ch {current_chapter_number}. Using generic semantic context query.")

    query_embedding_np = await llm_interface.async_get_embedding(context_query_text)
    max_semantic_tokens = (config.MAX_CONTEXT_TOKENS * 2) // 3 # Reserve 2/3 for semantic, 1/3 for other prompt parts

    if query_embedding_np is None:
        logger.warning("Failed to generate embedding for semantic context query. Falling back to sequential previous chapter summaries/text.")
        # Fallback logic remains the same as before
        context_parts: List[str] = []
        total_tokens_accumulated = 0
        fallback_chapter_limit = config.CONTEXT_CHAPTER_COUNT
        for i in range(max(1, current_chapter_number - fallback_chapter_limit), current_chapter_number):
            if total_tokens_accumulated >= max_semantic_tokens: break
            chap_data = await chapter_queries.get_chapter_data_from_db(i) # MODIFIED
            if chap_data:
                content = (chap_data.get('summary') or chap_data.get('text', '')).strip()
                is_prov = chap_data.get('is_provisional', False)
                ctype = "Provisional Summary" if chap_data.get('summary') and is_prov else \
                        "Summary" if chap_data.get('summary') else \
                        "Provisional Text Snippet" if is_prov else "Text Snippet"
                if content:
                    prefix = f"[Fallback Semantic Context from Chapter {i} ({ctype})]:\n"; suffix = "\n---\n"
                    full_content_part = f"{prefix}{content}{suffix}"
                    part_tokens = llm_interface.count_tokens(full_content_part, config.DRAFTING_MODEL) # Use drafting model for token count
                    if total_tokens_accumulated + part_tokens <= max_semantic_tokens:
                        context_parts.append(full_content_part)
                        total_tokens_accumulated += part_tokens
                    else:
                        remaining_tokens = max_semantic_tokens - total_tokens_accumulated
                        if remaining_tokens > llm_interface.count_tokens(prefix + suffix, config.DRAFTING_MODEL) + 10: # Ensure some content fits
                            truncated_content_part = llm_interface.truncate_text_by_tokens(
                                full_content_part, config.DRAFTING_MODEL, remaining_tokens
                            )
                            context_parts.append(truncated_content_part)
                            total_tokens_accumulated += remaining_tokens
                        break # Stop adding more parts
        final_semantic_context = "\n".join(reversed(context_parts)).strip()
        final_tokens = llm_interface.count_tokens(final_semantic_context, config.DRAFTING_MODEL)
        logger.info(f"Constructed fallback semantic context: {final_tokens} tokens.")
        return final_semantic_context

    # Use Neo4j vector search
    similar_chapters_data = await chapter_queries.find_similar_chapters_in_db( # MODIFIED
        query_embedding_np,
        config.CONTEXT_CHAPTER_COUNT,
        current_chapter_number # Exclude the current chapter if it somehow got an embedding
    )

    if not similar_chapters_data:
        logger.info("No similar past chapters found via Neo4j vector search for semantic context.")
        return ""

    # Ensure the immediate previous chapter is included if not already top N,
    # and it has data, to maintain strong sequential flow.
    immediate_prev_chap_num = current_chapter_number - 1
    if immediate_prev_chap_num > 0:
        is_immediate_prev_present = any(ch_data['chapter_number'] == immediate_prev_chap_num for ch_data in similar_chapters_data)
        if not is_immediate_prev_present:
            immediate_prev_data = await chapter_queries.get_chapter_data_from_db(immediate_prev_chap_num) # MODIFIED
            if immediate_prev_data and (immediate_prev_data.get('summary') or immediate_prev_data.get('text')):
                # Prepend it to ensure it's considered first for token budget, but it might displace a less relevant "similar" chapter
                # Or, just add it, and let token truncation handle it. Let's add it.
                similar_chapters_data.append({
                    "chapter_number": immediate_prev_chap_num,
                    "summary": immediate_prev_data.get('summary'),
                    "text": immediate_prev_data.get('text'),
                    "is_provisional": immediate_prev_data.get('is_provisional', False),
                    "score": 0.999 # Assign a high mock score to prioritize if sorting by score
                })
                logger.debug(f"Added immediate previous chapter {immediate_prev_chap_num} to semantic context list from Neo4j search.")

    # Sort by chapter number descending to process most recent first for token budget
    # (Neo4j results are already sorted by score, but we might have added immediate prev)
    # Re-sort by score (desc) if present, then by chapter_number (desc) as tie-breaker
    sorted_chapters_for_context = sorted(
        similar_chapters_data,
        key=lambda x: (x.get('score', 0.0), x.get('chapter_number', 0)),
        reverse=True
    )


    context_parts: List[str] = []
    total_tokens_accumulated = 0

    for chap_data in sorted_chapters_for_context:
        if total_tokens_accumulated >= max_semantic_tokens: break
        
        chap_num = chap_data['chapter_number']
        # Neo4j search might return chapters that have no summary/text if nodes exist but properties are null
        content = (chap_data.get('summary') or chap_data.get('text', '')).strip()
        is_prov = chap_data.get('is_provisional', False)
        score = chap_data.get('score', 'N/A')
        score_str = f"{score:.3f}" if isinstance(score, float) else str(score)


        ctype = "Provisional Summary" if chap_data.get('summary') and is_prov else \
                "Summary" if chap_data.get('summary') else \
                "Provisional Text Snippet" if is_prov else "Text Snippet"
        
        if content:
            prefix = f"[Semantic Context from Chapter {chap_num} (Similarity: {score_str}, Type: {ctype})]:\n"
            suffix = "\n---\n"
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
                    total_tokens_accumulated += remaining_tokens
                break 
            logger.debug(f"Added SEMANTIC context from ch {chap_num} ({ctype}, Sim: {score_str}), {part_tokens} tokens. Total: {total_tokens_accumulated}.")
        else:
            logger.warning(f"Chapter {chap_num} (Sim: {score_str}) from vector search had no content (summary/text). Skipping.")

    # The order for the final prompt should be chronological for reading flow,
    # so reverse the list of parts (which were added in relevance/recency order for token budget)
    final_semantic_context = "\n".join(reversed(context_parts)).strip()
    final_tokens = llm_interface.count_tokens(final_semantic_context, config.DRAFTING_MODEL)
    logger.info(f"Constructed final SEMANTIC context: {final_tokens} tokens from {len(context_parts)} chapter snippets (via Neo4j vector search).")
    return final_semantic_context


async def generate_hybrid_chapter_context_logic(agent_or_props: Any, current_chapter_number: int, chapter_plan: Optional[List[SceneDetail]]) -> str:
    """
    Constructs HYBRID context for the current chapter.
    'agent_or_props' can be the NANA_Orchestrator instance or a novel_props dictionary.
    MODIFIED: Ensures agent_or_props is passed to helpers.
    """
    if current_chapter_number <= 0:
        return ""
    logger.info(f"Generating HYBRID context for Chapter {current_chapter_number}...")

    semantic_context_task = _generate_semantic_chapter_context_logic(agent_or_props, current_chapter_number)
    kg_facts_task = get_reliable_kg_facts_for_drafting_prompt(agent_or_props, current_chapter_number, chapter_plan)

    semantic_context_str, kg_facts_str = await asyncio.gather(
        semantic_context_task, kg_facts_task
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
        else: hybrid_context_parts.append(cleaned_kg_facts)
        hybrid_context_parts.append("--- END KEY RELIABLE KG FACTS ---")
    else:
        hybrid_context_parts.append("\n\n--- KEY RELIABLE KG FACTS (FOR ESTABLISHED CANON & CONTINUITY) ---")
        hybrid_context_parts.append("Knowledge Graph fact retrieval did not yield specific results for this chapter.")
        hybrid_context_parts.append("--- END KEY RELIABLE KG FACTS ---")
    final_hybrid_context = "\n".join(hybrid_context_parts).strip()
    num_hybrid_tokens = llm_interface.count_tokens(final_hybrid_context, config.DRAFTING_MODEL) # Use drafting model here for consistency
    if num_hybrid_tokens > config.MAX_CONTEXT_TOKENS:
        logger.warning(f"Hybrid context token count ({num_hybrid_tokens}) exceeds MAX_CONTEXT_TOKENS ({config.MAX_CONTEXT_TOKENS}). Truncating.")
        final_hybrid_context = llm_interface.truncate_text_by_tokens(
            final_hybrid_context, config.DRAFTING_MODEL, config.MAX_CONTEXT_TOKENS, # Use drafting model here
            truncation_marker="\n... (Hybrid context truncated due to token limit)"
        )
        num_hybrid_tokens = llm_interface.count_tokens(final_hybrid_context, config.DRAFTING_MODEL) # Re-count
    logger.info(f"Generated HYBRID context for Chapter {current_chapter_number}, Tokens (est.): {num_hybrid_tokens}.")
    return final_hybrid_context
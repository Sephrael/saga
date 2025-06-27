from __future__ import annotations

import asyncio
from typing import Any

import structlog
from config import settings
from core.llm_interface import count_tokens, truncate_text_by_tokens

# Default modules for dependency injection
from core.llm_interface import llm_service as default_llm_service
from data_access import chapter_queries as default_chapter_queries
from prompt_data_getters import get_reliable_kg_facts_for_drafting_prompt

from models import SceneDetail

logger = structlog.get_logger(__name__)


class ContextService:
    """Generate semantic and hybrid context snippets."""

    def __init__(
        self,
        chapter_queries_module: Any = default_chapter_queries,
        llm_service_instance: Any = default_llm_service,
    ) -> None:
        self.chapter_queries = chapter_queries_module
        self.llm_service = llm_service_instance

    @staticmethod
    def _get_nested_prop(
        agent_or_props: Any, primary_key: str, secondary_key: str, default: Any = None
    ) -> Any:
        """Return a nested property from an agent-like object or dictionary."""
        primary_data = (
            getattr(agent_or_props, primary_key, None)
            if not isinstance(agent_or_props, dict)
            else agent_or_props.get(primary_key, {})
        )
        if isinstance(primary_data, dict):
            return primary_data.get(secondary_key, default)
        return getattr(primary_data, secondary_key, default)

    async def get_semantic_context(
        self, agent_or_props: Any, current_chapter_number: int
    ) -> str:
        """Retrieve semantic context using vector similarity search."""
        if current_chapter_number <= 1:
            return ""
        logger.debug(
            "Retrieving and constructing SEMANTIC context for Chapter %s via Neo4j vector search...",
            current_chapter_number,
        )

        if isinstance(agent_or_props, dict):
            plot_outline_data = agent_or_props.get(
                "plot_outline_full", agent_or_props.get("plot_outline", {})
            )
        else:
            plot_outline_data = getattr(agent_or_props, "plot_outline_full", None)
            if not plot_outline_data:
                plot_outline_data = getattr(agent_or_props, "plot_outline", {})

        plot_points = []
        if isinstance(plot_outline_data, dict):
            plot_points = plot_outline_data.get("plot_points", [])
        else:
            plot_points = getattr(plot_outline_data, "plot_points", [])

        plot_point_focus = None
        if plot_points and isinstance(plot_points, list) and current_chapter_number > 0:
            idx = current_chapter_number - 1
            if 0 <= idx < len(plot_points):
                plot_point_focus = (
                    str(plot_points[idx]) if plot_points[idx] is not None else None
                )
            else:
                logger.warning(
                    "Cannot determine plot point focus for chapter %s: index %s out of bounds for %s plot points.",
                    current_chapter_number,
                    idx,
                    len(plot_points),
                )
        else:
            logger.warning(
                "Cannot determine plot point focus for chapter %s: plot_points list is empty or invalid.",
                current_chapter_number,
            )

        context_query_text = (
            plot_point_focus
            if plot_point_focus
            else f"Narrative context relevant to events leading up to chapter {current_chapter_number}."
        )

        if plot_point_focus:
            plot_point_index_display = (current_chapter_number - 1) + 1
            logger.info(
                "Semantic context query for ch %s based on Plot Point %s: '%s...'",
                current_chapter_number,
                plot_point_index_display,
                context_query_text[:100],
            )
        else:
            logger.warning(
                "No specific plot point found for ch %s. Using generic semantic context query.",
                current_chapter_number,
            )

        max_semantic_tokens = (settings.MAX_CONTEXT_TOKENS * 2) // 3
        immediate_context_limit = 2
        immediate_start = max(1, current_chapter_number - immediate_context_limit)
        immediate_parts: list[str] = []
        total_tokens_accumulated = 0

        for i in range(immediate_start, current_chapter_number):
            if total_tokens_accumulated >= max_semantic_tokens:
                break
            chap_data = await self.chapter_queries.get_chapter_data_from_db(i)
            if chap_data:
                content = (
                    chap_data.get("summary") or chap_data.get("text", "")
                ).strip()
                is_prov = chap_data.get("is_provisional", False)
                ctype = (
                    "Provisional Summary"
                    if chap_data.get("summary") and is_prov
                    else "Summary"
                    if chap_data.get("summary")
                    else "Provisional Text Snippet"
                    if is_prov
                    else "Text Snippet"
                )
                if content:
                    prefix = f"[Immediate Context from Chapter {i} ({ctype})]:\n"
                    suffix = "\n---\n"
                    full_content_part = f"{prefix}{content}{suffix}"
                    part_tokens = count_tokens(
                        full_content_part, settings.DRAFTING_MODEL
                    )
                    if total_tokens_accumulated + part_tokens <= max_semantic_tokens:
                        immediate_parts.append(full_content_part)
                        total_tokens_accumulated += part_tokens
                    else:
                        remaining_tokens = (
                            max_semantic_tokens - total_tokens_accumulated
                        )
                        if (
                            remaining_tokens
                            > count_tokens(prefix + suffix, settings.DRAFTING_MODEL)
                            + 10
                        ):
                            truncated_content_part = truncate_text_by_tokens(
                                full_content_part,
                                settings.DRAFTING_MODEL,
                                remaining_tokens,
                            )
                            immediate_parts.append(truncated_content_part)
                            total_tokens_accumulated += remaining_tokens
                        break

        if total_tokens_accumulated >= max_semantic_tokens:
            final_semantic_context = "\n".join(immediate_parts).strip()
            final_tokens_count = count_tokens(
                final_semantic_context, settings.DRAFTING_MODEL
            )
            logger.info(
                "Constructed semantic context solely from immediate chapters: %s tokens.",
                final_tokens_count,
            )
            return final_semantic_context

        query_embedding_np = await self.llm_service.async_get_embedding(
            context_query_text
        )
        if query_embedding_np is None:
            logger.warning(
                "Failed to generate embedding for semantic context query. Falling back to sequential previous chapter summaries/text."
            )
            context_parts_list: list[str] = []
            fallback_chapter_limit = settings.CONTEXT_CHAPTER_COUNT
            for i in range(
                max(1, current_chapter_number - fallback_chapter_limit), immediate_start
            ):
                if total_tokens_accumulated >= max_semantic_tokens:
                    break
                chap_data = await self.chapter_queries.get_chapter_data_from_db(i)
                if chap_data:
                    content = (
                        chap_data.get("summary") or chap_data.get("text", "")
                    ).strip()
                    is_prov = chap_data.get("is_provisional", False)
                    ctype = (
                        "Provisional Summary"
                        if chap_data.get("summary") and is_prov
                        else "Summary"
                        if chap_data.get("summary")
                        else "Provisional Text Snippet"
                        if is_prov
                        else "Text Snippet"
                    )
                    if content:
                        prefix = (
                            f"[Fallback Semantic Context from Chapter {i} ({ctype})]:\n"
                        )
                        suffix = "\n---\n"
                        full_content_part = f"{prefix}{content}{suffix}"
                        part_tokens = count_tokens(
                            full_content_part, settings.DRAFTING_MODEL
                        )
                        if (
                            total_tokens_accumulated + part_tokens
                            <= max_semantic_tokens
                        ):
                            context_parts_list.append(full_content_part)
                            total_tokens_accumulated += part_tokens
                        else:
                            remaining_tokens = (
                                max_semantic_tokens - total_tokens_accumulated
                            )
                            if (
                                remaining_tokens
                                > count_tokens(prefix + suffix, settings.DRAFTING_MODEL)
                                + 10
                            ):
                                truncated_content_part = truncate_text_by_tokens(
                                    full_content_part,
                                    settings.DRAFTING_MODEL,
                                    remaining_tokens,
                                )
                                context_parts_list.append(truncated_content_part)
                                total_tokens_accumulated += remaining_tokens
                            break
            final_semantic_context = "\n".join(
                immediate_parts + context_parts_list
            ).strip()
            final_tokens_count = count_tokens(
                final_semantic_context, settings.DRAFTING_MODEL
            )
            logger.info(
                "Constructed fallback semantic context: %s tokens.",
                final_tokens_count,
            )
            return final_semantic_context

        similar_chapters_data = await self.chapter_queries.find_similar_chapters_in_db(
            query_embedding_np,
            settings.CONTEXT_CHAPTER_COUNT,
            current_chapter_number,
        )

        if not similar_chapters_data:
            logger.info(
                "No similar past chapters found via Neo4j vector search for semantic context."
            )
            return ""

        excluded_chapters = set(range(immediate_start, current_chapter_number))
        filtered_chapters: list[dict[str, Any]] = []
        for ch in similar_chapters_data:
            if ch.get("chapter_number") in excluded_chapters:
                continue
            score = float(ch.get("score", 0.0))
            distance = current_chapter_number - int(ch.get("chapter_number", 0))
            adjusted = score * (0.95 ** max(distance, 0))
            ch["adjusted_score"] = adjusted
            filtered_chapters.append(ch)

        sorted_chapters_for_context = sorted(
            filtered_chapters, key=lambda x: x.get("adjusted_score", 0.0), reverse=True
        )

        context_parts_list: list[str] = []

        for chap_data in sorted_chapters_for_context:
            if total_tokens_accumulated >= max_semantic_tokens:
                break
            chap_num = chap_data["chapter_number"]
            content = (chap_data.get("summary") or chap_data.get("text", "")).strip()
            is_prov = chap_data.get("is_provisional", False)
            score = chap_data.get("score", "N/A")
            score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
            ctype = (
                "Provisional Summary"
                if chap_data.get("summary") and is_prov
                else "Summary"
                if chap_data.get("summary")
                else "Provisional Text Snippet"
                if is_prov
                else "Text Snippet"
            )
            if content:
                prefix = f"[Semantic Context from Chapter {chap_num} (Similarity: {score_str}, Type: {ctype})]:\n"
                suffix = "\n---\n"
                full_content_part = f"{prefix}{content}{suffix}"
                part_tokens = count_tokens(full_content_part, settings.DRAFTING_MODEL)
                if total_tokens_accumulated + part_tokens <= max_semantic_tokens:
                    context_parts_list.append(full_content_part)
                    total_tokens_accumulated += part_tokens
                else:
                    remaining_tokens = max_semantic_tokens - total_tokens_accumulated
                    if (
                        remaining_tokens
                        > count_tokens(prefix + suffix, settings.DRAFTING_MODEL) + 10
                    ):
                        truncated_content_part = truncate_text_by_tokens(
                            full_content_part,
                            settings.DRAFTING_MODEL,
                            remaining_tokens,
                        )
                        context_parts_list.append(truncated_content_part)
                        total_tokens_accumulated += remaining_tokens
                    break
                logger.debug(
                    "Added SEMANTIC context from ch %s (%s, Sim: %s), %s tokens. Total: %s.",
                    chap_num,
                    ctype,
                    score_str,
                    part_tokens,
                    total_tokens_accumulated,
                )
            else:
                logger.warning(
                    "Chapter %s (Sim: %s) from vector search had no content (summary/text). Skipping.",
                    chap_num,
                    score_str,
                )

        final_semantic_context = "\n".join(immediate_parts + context_parts_list).strip()
        final_tokens_count = count_tokens(
            final_semantic_context, settings.DRAFTING_MODEL
        )
        logger.info(
            "Constructed final SEMANTIC context: %s tokens from %s chapter snippets (via Neo4j vector search).",
            final_tokens_count,
            len(immediate_parts) + len(context_parts_list),
        )
        return final_semantic_context

    async def build_hybrid_context(
        self,
        agent_or_props: Any,
        current_chapter_number: int,
        chapter_plan: list[SceneDetail] | None,
    ) -> str:
        """Combine semantic context and KG facts for drafting."""
        if current_chapter_number <= 0:
            return ""
        logger.info(
            "Generating HYBRID context for Chapter %s...", current_chapter_number
        )

        semantic_context_task = self.get_semantic_context(
            agent_or_props, current_chapter_number
        )
        if isinstance(agent_or_props, dict):
            plot_outline_data = agent_or_props.get(
                "plot_outline_full", agent_or_props.get("plot_outline", {})
            )
        else:
            plot_outline_data = getattr(agent_or_props, "plot_outline_full", None)
            if not plot_outline_data:
                plot_outline_data = getattr(agent_or_props, "plot_outline", {})

        kg_facts_task = get_reliable_kg_facts_for_drafting_prompt(
            plot_outline_data, current_chapter_number, chapter_plan
        )

        semantic_context_str, kg_facts_str = await asyncio.gather(
            semantic_context_task, kg_facts_task
        )

        hybrid_context_parts: list[str] = []
        if semantic_context_str and semantic_context_str.strip():
            hybrid_context_parts.append(
                "--- SEMANTIC CONTEXT FROM PAST CHAPTERS (FOR NARRATIVE FLOW & TONE) ---"
            )
            hybrid_context_parts.append(semantic_context_str)
            hybrid_context_parts.append("--- END SEMANTIC CONTEXT ---")
        else:
            hybrid_context_parts.append(
                "--- SEMANTIC CONTEXT FROM PAST CHAPTERS (FOR NARRATIVE FLOW & TONE) ---"
            )
            hybrid_context_parts.append(
                "No relevant semantic context could be retrieved."
            )
            hybrid_context_parts.append("--- END SEMANTIC CONTEXT ---")
        if kg_facts_str and kg_facts_str.strip():
            hybrid_context_parts.append(
                "\n\n--- KEY RELIABLE KG FACTS (FOR ESTABLISHED CANON & CONTINUITY) ---"
            )
            cleaned_kg_facts = (
                kg_facts_str.split("\n", 1)[-1]
                if kg_facts_str.startswith("**Key Reliable KG Facts")
                else kg_facts_str
            )
            if not cleaned_kg_facts.strip() or cleaned_kg_facts.lower().startswith(
                "no specific reliable kg facts"
            ):
                hybrid_context_parts.append(
                    "No specific reliable KG facts were identified as highly relevant for this chapter's focus."
                )
            else:
                hybrid_context_parts.append(cleaned_kg_facts)
            hybrid_context_parts.append("--- END KEY RELIABLE KG FACTS ---")
        else:
            hybrid_context_parts.append(
                "\n\n--- KEY RELIABLE KG FACTS (FOR ESTABLISHED CANON & CONTINUITY) ---"
            )
            hybrid_context_parts.append(
                "Knowledge Graph fact retrieval did not yield specific results for this chapter."
            )
            hybrid_context_parts.append("--- END KEY RELIABLE KG FACTS ---")

        final_hybrid_context = "\n".join(hybrid_context_parts).strip()
        num_hybrid_tokens = count_tokens(final_hybrid_context, settings.DRAFTING_MODEL)
        if num_hybrid_tokens > settings.MAX_CONTEXT_TOKENS:
            logger.warning(
                "Hybrid context token count (%s) exceeds MAX_CONTEXT_TOKENS (%s). Truncating.",
                num_hybrid_tokens,
                settings.MAX_CONTEXT_TOKENS,
            )
            final_hybrid_context = truncate_text_by_tokens(
                final_hybrid_context,
                settings.DRAFTING_MODEL,
                settings.MAX_CONTEXT_TOKENS,
                truncation_marker="\n... (Hybrid context truncated due to token limit)",
            )
            num_hybrid_tokens = count_tokens(
                final_hybrid_context, settings.DRAFTING_MODEL
            )
        logger.info(
            "Generated HYBRID context for Chapter %s, Tokens(est.): %s.",
            current_chapter_number,
            num_hybrid_tokens,
        )
        return final_hybrid_context

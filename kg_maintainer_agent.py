# kg_maintainer_agent.py
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from async_lru import alru_cache  # type: ignore
from llm_interface import llm_service
from prompt_renderer import render_prompt

import config
from core_db.base_db_manager import neo4j_manager
from data_access import kg_queries
from parsing_utils import (
    parse_rdf_triples_with_rdflib,
)  # Will be modified to custom parser

# Assuming a package structure for kg_maintainer components
from kg_maintainer import models, parsing, merge, cypher_generation


logger = logging.getLogger(__name__)


@alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
async def _llm_summarize_full_chapter_text(
    chapter_text: str, chapter_number: int
) -> Tuple[str, Optional[Dict[str, int]]]:
    """Summarize full chapter text via the configured LLM."""
    prompt = render_prompt(
        "kg_maintainer_agent/chapter_summary.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "chapter_number": chapter_number,
            "chapter_text": chapter_text,
        },
    )
    summary, usage_data = await llm_service.async_call_llm(
        model_name=config.Models.SMALL,  # Using SMALL_MODEL for summarization
        prompt=prompt,
        temperature=config.Temperatures.SUMMARY,
        max_tokens=config.MAX_SUMMARY_TOKENS,  # Should be small for 1-3 sentences
        stream_to_disk=False,
        frequency_penalty=config.FREQUENCY_PENALTY_SUMMARY,
        presence_penalty=config.PRESENCE_PENALTY_SUMMARY,
        auto_clean_response=True,
    )
    return summary.strip(), usage_data


class KGMaintainerAgent:
    """High level interface for KG parsing and persistence."""

    def __init__(self, model_name: str = config.KNOWLEDGE_UPDATE_MODEL):
        self.model_name = model_name
        logger.info(
            "KGMaintainerAgent initialized with model for extraction: %s",
            self.model_name,
        )

    def parse_character_updates(
        self, text: str, chapter_number: int
    ) -> Dict[str, models.CharacterProfile]:
        """Parse character update text into structured profiles."""
        return parsing.parse_unified_character_updates(text, chapter_number)

    def parse_world_updates(
        self, text: str, chapter_number: int
    ) -> Dict[str, Dict[str, models.WorldItem]]:
        """Parse world update text into structured items."""
        return parsing.parse_unified_world_updates(text, chapter_number)

    def merge_updates(
        self,
        current_profiles: Dict[str, models.CharacterProfile],
        current_world: Dict[str, Dict[str, models.WorldItem]],
        char_updates_parsed: Dict[str, models.CharacterProfile],
        world_updates_parsed: Dict[str, Dict[str, models.WorldItem]],
        chapter_number: int,
        from_flawed_draft: bool = False,
    ) -> None:
        """Merge parsed updates into existing state (Python objects)."""
        merge.merge_character_profile_updates(
            current_profiles, char_updates_parsed, chapter_number, from_flawed_draft
        )
        merge.merge_world_item_updates(
            current_world, world_updates_parsed, chapter_number, from_flawed_draft
        )

    async def persist_profiles(
        self,
        profiles_to_persist: Dict[str, models.CharacterProfile],
        chapter_number_for_delta: int,
    ) -> None:
        """Persist character profiles (delta from a chapter) to Neo4j using cypher_generation."""
        statements: List[Tuple[str, Dict[str, Any]]] = []
        for profile_obj in profiles_to_persist.values():
            # Pass chapter_number_for_delta for context if needed by cypher_generation
            statements.extend(
                cypher_generation.generate_character_node_cypher(
                    profile_obj, chapter_number_for_delta
                )
            )
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
            logger.info(
                f"Persisted {len(profiles_to_persist)} character profile updates from chapter {chapter_number_for_delta} delta to Neo4j."
            )

    async def persist_world(
        self,
        world_items_to_persist: Dict[str, Dict[str, models.WorldItem]],
        chapter_number_for_delta: int,
    ) -> None:
        """Persist world elements (delta from a chapter) to Neo4j using cypher_generation."""
        statements: List[Tuple[str, Dict[str, Any]]] = []
        count = 0
        for category_items in world_items_to_persist.values():
            for item_obj in category_items.values():
                # Pass chapter_number_for_delta for context
                statements.extend(
                    cypher_generation.generate_world_element_node_cypher(
                        item_obj, chapter_number_for_delta
                    )
                )
                count += 1
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
            logger.info(
                f"Persisted {count} world element updates from chapter {chapter_number_for_delta} delta to Neo4j."
            )

    async def summarize_chapter(
        self, chapter_text: Optional[str], chapter_number: int
    ) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        if (
            not chapter_text
            or len(chapter_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH // 2
        ):
            logger.warning(
                "Chapter %s text too short for summarization (%d chars, min_req for meaningful summary: %d).",
                chapter_number,
                len(chapter_text or ""),
                config.MIN_ACCEPTABLE_DRAFT_LENGTH // 2,
            )
            return None, None

        try:
            cleaned_summary, usage = await _llm_summarize_full_chapter_text(
                chapter_text, chapter_number
            )
            if cleaned_summary:
                logger.info(
                    "Generated summary for ch %d: '%s...'",
                    chapter_number,
                    cleaned_summary[:100].strip(),
                )
                return cleaned_summary, usage
            logger.warning("LLM returned empty summary for ch %d.", chapter_number)
            return None, usage
        except Exception as e:
            logger.error(
                f"Error during chapter summarization for ch {chapter_number}: {e}",
                exc_info=True,
            )
            return None, None

    async def _llm_extract_updates(
        self,
        plot_outline: Dict[str, Any],
        chapter_text: str,
        chapter_number: int,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        """Call the LLM to extract structured updates from chapter text, including typed entities in triples."""
        protagonist = plot_outline.get(
            "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
        )

        prompt = render_prompt(
            "kg_maintainer_agent/extract_updates.j2",
            {
                "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "protagonist": protagonist,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled Novel"),
                "novel_genre": plot_outline.get("genre", "Unknown"),
                "chapter_text": chapter_text,
            },
        )

        try:
            text, usage = await llm_service.async_call_llm(
                model_name=self.model_name,
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                max_tokens=config.MAX_KG_TRIPLE_TOKENS,
                allow_fallback=True,
                stream_to_disk=False,
                frequency_penalty=config.FREQUENCY_PENALTY_KG_EXTRACTION,
                presence_penalty=config.PRESENCE_PENALTY_KG_EXTRACTION,
                auto_clean_response=True,
            )
            return text, usage
        except Exception as e:
            logger.error(f"LLM call for KG extraction failed: {e}", exc_info=True)
            return "", None

    async def extract_and_merge_knowledge(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, models.CharacterProfile],
        world_building: Dict[str, Dict[str, models.WorldItem]],
        chapter_number: int,
        chapter_text: str,
        is_from_flawed_draft: bool = False,
    ) -> Optional[Dict[str, int]]:
        if (
            not chapter_text
            or len(chapter_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH // 2
        ):
            logger.warning(
                "Skipping knowledge extraction for chapter %s: text too short (%d chars, min_req: %d).",
                chapter_number,
                len(chapter_text or ""),
                config.MIN_ACCEPTABLE_DRAFT_LENGTH // 2,
            )
            return None

        logger.info(
            "KGMaintainerAgent: Starting knowledge extraction for chapter %d. Flawed draft: %s",
            chapter_number,
            is_from_flawed_draft,
        )

        raw_extracted_text, usage_data = await self._llm_extract_updates(
            plot_outline, chapter_text, chapter_number
        )

        if not raw_extracted_text.strip():
            logger.warning(
                "LLM extraction returned no text for chapter %d.", chapter_number
            )
            return usage_data

        # Corrected section parsing logic:
        sections_parts = re.split(
            r"(^\s*###\s*[\w\s]+?\s*###\s*$)",
            raw_extracted_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        parsed_sections: Dict[str, str] = {}
        current_section_key = None
        # sections_parts will be like [text_before_first_header, header1, content1, header2, content2, ...]
        # So we iterate in steps of 2, starting from index 1 (first header)
        for i in range(1, len(sections_parts), 2):
            header_text_raw = sections_parts[i].strip()
            content_text_raw = (
                sections_parts[i + 1].strip() if (i + 1) < len(sections_parts) else ""
            )

            header_text = header_text_raw.lower()  # Normalize header for matching
            if "character updates" in header_text:
                current_section_key = "character_updates"
            elif "world updates" in header_text:
                current_section_key = "world_updates"
            elif "kg triples" in header_text:
                current_section_key = "kg_triples"
            else:
                logger.warning(
                    f"Unknown section header found in LLM output: '{header_text_raw}'"
                )
                current_section_key = None  # Reset if unknown header

            if current_section_key:
                # Remove potential ```json ... ``` markdown if LLM adds it for JSON sections
                if current_section_key in ["character_updates", "world_updates"]:
                    content_text_raw = re.sub(
                        r"^\s*```json\s*\n?", "", content_text_raw, flags=re.MULTILINE
                    )
                    content_text_raw = re.sub(
                        r"\n?\s*```\s*$", "", content_text_raw, flags=re.MULTILINE
                    )
                parsed_sections[current_section_key] = content_text_raw.strip()
                current_section_key = None  # Reset for next potential header block

        char_updates_from_llm = self.parse_character_updates(
            parsed_sections.get("character_updates", ""), chapter_number
        )
        world_updates_from_llm = self.parse_world_updates(
            parsed_sections.get("world_updates", ""), chapter_number
        )

        # Use the corrected function name for structured triple parsing
        parsed_triples_structured = parse_rdf_triples_with_rdflib(
            parsed_sections.get("kg_triples", "")
        )

        logger.info(
            f"Chapter {chapter_number} LLM Extraction: "
            f"{len(char_updates_from_llm)} char updates, "
            f"{sum(len(items) for items in world_updates_from_llm.values())} world item updates, "
            f"{len(parsed_triples_structured)} KG triples."
        )

        current_char_profiles_models = character_profiles
        current_world_models = world_building

        self.merge_updates(
            current_char_profiles_models,  # Pass model instances
            current_world_models,  # Pass model instances
            char_updates_from_llm,  # Already model instances
            world_updates_from_llm,  # Already model instances
            chapter_number,
            is_from_flawed_draft,
        )
        logger.info(
            f"Merged LLM updates into in-memory state for chapter {chapter_number}."
        )

        # Persist the DELTA of updates (char_updates_from_llm, world_updates_from_llm)
        # These functions expect model instances and the chapter number for delta context.
        if char_updates_from_llm:
            await self.persist_profiles(char_updates_from_llm, chapter_number)
        if world_updates_from_llm:
            await self.persist_world(world_updates_from_llm, chapter_number)

        if parsed_triples_structured:
            try:
                await kg_queries.add_kg_triples_batch_to_db(
                    parsed_triples_structured, chapter_number, is_from_flawed_draft
                )
                logger.info(
                    f"Persisted {len(parsed_triples_structured)} KG triples for chapter {chapter_number} to Neo4j."
                )
            except Exception as e:
                logger.error(
                    f"Failed to persist KG triples for chapter {chapter_number}: {e}",
                    exc_info=True,
                )

        logger.info(
            "Knowledge extraction, in-memory merge, and delta persistence complete for chapter %d.",
            chapter_number,
        )
        return usage_data


__all__ = ["KGMaintainerAgent"]

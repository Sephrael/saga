"""Knowledge graph maintenance agent.

This module provides a lightweight wrapper around parsing and merge helpers for
updating the Saga knowledge graph. The project is licensed under the Apache 2.0
license; see the LICENSE file for details.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from async_lru import alru_cache
from llm_interface import llm_service

import config
from core_db.base_db_manager import neo4j_manager
from data_access import kg_queries
from parsing_utils import parse_kg_triples_from_text
from kg_maintainer import (
    CharacterProfile,
    WorldItem,
    parse_unified_character_updates,
    parse_unified_world_updates,
    merge_character_profile_updates,
    merge_world_item_updates,
    generate_character_node_cypher,
    generate_world_element_node_cypher,
)

logger = logging.getLogger(__name__)


@alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
async def _llm_summarize_full_chapter_text(
    chapter_text: str, chapter_number: int
) -> Tuple[str, Optional[Dict[str, int]]]:
    """Summarize full chapter text via the configured LLM."""
    prompt_lines = []
    if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
        prompt_lines.append("/no_think")
    prompt_lines.extend(
        [
            f"You are a concise summarizer. Summarize the key events, character developments, and plot advancements from the following Chapter {chapter_number} text.",
            "The summary should be 1-3 sentences long and capture the most crucial information.",
            "Focus on what changed or was revealed.",
            "",
            "Full Chapter Text:",
            "--- BEGIN TEXT ---",
            chapter_text,
            "--- END TEXT ---",
            "",
            "Output ONLY the summary text. No extra commentary or \"Summary:\" prefix.",
        ]
    )
    prompt = "\n".join(prompt_lines)
    summary, usage_data = await llm_service.async_call_llm(
        model_name=config.SMALL_MODEL,
        prompt=prompt,
        temperature=config.TEMPERATURE_SUMMARY,
        max_tokens=config.MAX_SUMMARY_TOKENS,
        stream_to_disk=False,
        frequency_penalty=config.FREQUENCY_PENALTY_SUMMARY,
        presence_penalty=config.PRESENCE_PENALTY_SUMMARY,
        auto_clean_response=True,
    )
    return summary.strip(), usage_data


class KGMaintainerAgent:
    """High level interface for KG parsing and persistence."""

    def __init__(self, model_name: str = config.NARRATOR_MODEL):
        self.model_name = model_name
        logger.info("KGMaintainerAgent initialized with model: %s", model_name)

    def parse_character_updates(self, text: str, chapter_number: int) -> Dict[str, CharacterProfile]:
        """Parse character update text into structured profiles."""
        return parse_unified_character_updates(text, chapter_number)

    def parse_world_updates(self, text: str, chapter_number: int) -> Dict[str, Dict[str, WorldItem]]:
        """Parse world update text into structured items."""
        return parse_unified_world_updates(text, chapter_number)

    def merge_updates(
        self,
        current_profiles: Dict[str, CharacterProfile],
        current_world: Dict[str, Dict[str, WorldItem]],
        char_updates: Dict[str, CharacterProfile],
        world_updates: Dict[str, Dict[str, WorldItem]],
        chapter_number: int,
        from_flawed_draft: bool = False,
    ) -> None:
        """Merge parsed updates into existing state."""
        merge_character_profile_updates(current_profiles, char_updates, chapter_number, from_flawed_draft)
        merge_world_item_updates(current_world, world_updates, chapter_number, from_flawed_draft)

    async def persist_profiles(self, profiles: Dict[str, CharacterProfile]) -> None:
        """Persist character profiles to Neo4j."""
        statements = []
        for profile in profiles.values():
            statements.extend(generate_character_node_cypher(profile))
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)

    async def persist_world(self, world: Dict[str, Dict[str, WorldItem]]) -> None:
        """Persist world elements to Neo4j."""
        statements = []
        for cat_items in world.values():
            for item in cat_items.values():
                statements.append(generate_world_element_node_cypher(item))
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)

    async def summarize_chapter(
        self, chapter_text: Optional[str], chapter_number: int
    ) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        """Summarize the provided chapter text using an LLM."""
        if not chapter_text or len(chapter_text) < 50:
            logger.warning(
                "Chapter %s text too short for summarization (%d chars).",
                chapter_number,
                len(chapter_text or ""),
            )
            return None, None
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
        logger.warning("Failed to generate a valid summary for ch %d via LLM.", chapter_number)
        return None, usage

    async def _llm_extract_updates(
        self, novel_props: Dict[str, Any], chapter_text: str, chapter_number: int
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        """Call the LLM to extract structured updates from chapter text."""
        prompt_lines: List[str] = []
        if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
            prompt_lines.append("/no_think")

        protagonist = novel_props.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME)
        prompt_lines.extend(
            [
                "You analyze the following chapter text and extract updates for the knowledge graph.",
                f"Protagonist: {protagonist}",
                "Output plain text in three sections using these headers exactly:",
                "### CHARACTER UPDATES ###",
                "### WORLD UPDATES ###",
                "### KG TRIPLES ###",
                "Provide character updates using the format 'Character: Name' followed by key/value lines.",
                "World updates are grouped by 'Category: <name>' then 'Item: <item name>' blocks.",
                "List KG triples one per line using 'Subject | Predicate | Object'.",
                "--- BEGIN CHAPTER TEXT ---",
                chapter_text,
                "--- END CHAPTER TEXT ---",
            ]
        )
        prompt = "\n".join(prompt_lines)
        text, usage = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=config.TEMPERATURE_KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=config.FREQUENCY_PENALTY_KG_EXTRACTION,
            presence_penalty=config.PRESENCE_PENALTY_KG_EXTRACTION,
            auto_clean_response=True,
        )
        return text, usage

    async def extract_and_merge_knowledge(
        self,
        novel_props: Dict[str, Any],
        chapter_number: int,
        chapter_text: str,
        is_from_flawed_draft: bool = False,
    ) -> Optional[Dict[str, int]]:
        """Extract knowledge from chapter text, merge into state, and persist."""
        if not chapter_text:
            logger.warning(
                "Skipping knowledge extraction for chapter %s: no text provided.",
                chapter_number,
            )
            return None

        logger.info(
            "KGMaintainerAgent extracting knowledge for chapter %d", chapter_number
        )

        raw_text, usage = await self._llm_extract_updates(
            novel_props, chapter_text, chapter_number
        )

        sections = re.split(r"^\s*###\s*([\w\s]+?)\s*###\s*$", raw_text, flags=re.IGNORECASE | re.MULTILINE)
        parsed: Dict[str, str] = {}
        current = None
        for i in range(1, len(sections)):
            if i % 2 == 1:
                header = sections[i].strip().lower()
                if "character" in header:
                    current = "character_updates"
                elif "world" in header:
                    current = "world_updates"
                elif "kg" in header:
                    current = "kg_triples"
                else:
                    current = None
            elif current:
                parsed[current] = sections[i].strip()
                current = None

        char_updates = self.parse_character_updates(
            parsed.get("character_updates", ""), chapter_number
        )
        world_updates = self.parse_world_updates(
            parsed.get("world_updates", ""), chapter_number
        )
        kg_triples = parse_kg_triples_from_text(parsed.get("kg_triples", ""))

        # Convert current novel state into dataclasses
        current_profiles: Dict[str, CharacterProfile] = {}
        for name, data in novel_props.get("character_profiles", {}).items():
            if isinstance(data, CharacterProfile):
                current_profiles[name] = data
            elif isinstance(data, dict):
                current_profiles[name] = CharacterProfile.from_dict(name, data)

        current_world: Dict[str, Dict[str, WorldItem]] = {}
        for cat, items in novel_props.get("world_building", {}).items():
            if not isinstance(items, dict):
                continue
            cat_dict: Dict[str, WorldItem] = {}
            for item_name, item_data in items.items():
                if isinstance(item_data, WorldItem):
                    cat_dict[item_name] = item_data
                elif isinstance(item_data, dict):
                    cat_dict[item_name] = WorldItem.from_dict(cat, item_name, item_data)
            if cat_dict:
                current_world[cat] = cat_dict

        self.merge_updates(
            current_profiles,
            current_world,
            char_updates,
            world_updates,
            chapter_number,
            is_from_flawed_draft,
        )

        await self.persist_profiles(char_updates)
        await self.persist_world(world_updates)

        if kg_triples:
            triples_data = [
                (s, p, o, chapter_number, 1.0, is_from_flawed_draft)
                for s, p, o in kg_triples
            ]
            await kg_queries.add_kg_triples_batch_to_db(triples_data)

        novel_props["character_profiles"] = {
            name: prof.to_dict() for name, prof in current_profiles.items()
        }
        novel_props["world_building"] = {
            cat: {n: item.to_dict() for n, item in items.items()}
            for cat, items in current_world.items()
        }

        logger.info(
            "Knowledge extraction and merge complete for chapter %d", chapter_number
        )
        return usage


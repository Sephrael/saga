# kg_maintainer_agent.py
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from async_lru import alru_cache  # type: ignore

import config
from core_db.base_db_manager import neo4j_manager
from data_access import (
    character_queries,
    kg_queries,
    world_queries,
)

# Assuming a package structure for kg_maintainer components
from kg_maintainer import merge, models, parsing
from llm_interface import llm_service
from parsing_utils import (
    parse_rdf_triples_with_rdflib,
)  # Will be modified to custom parser
from prompt_renderer import render_prompt

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
        model_name=config.SMALL_MODEL,  # Using SMALL_MODEL for summarization
        prompt=prompt,
        temperature=config.Temperatures.SUMMARY,
        max_tokens=config.MAX_SUMMARY_TOKENS,  # Should be small for 1-3 sentences
        stream_to_disk=False,
        frequency_penalty=config.FREQUENCY_PENALTY_SUMMARY,
        presence_penalty=config.PRESENCE_PENALTY_SUMMARY,
        auto_clean_response=True,
    )
    summary_text = summary.strip()
    if summary_text:
        try:
            parsed = json.loads(summary_text)
            if isinstance(parsed, dict):
                summary_text = parsed.get("summary", "")
        except json.JSONDecodeError:
            logger.debug(f"Summary for chapter {chapter_number} was not a JSON object.")
    return summary_text, usage_data


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
        """Persist character profiles to Neo4j."""
        await character_queries.sync_characters(
            profiles_to_persist, chapter_number_for_delta
        )

    async def persist_world(
        self,
        world_items_to_persist: Dict[str, Dict[str, models.WorldItem]],
        chapter_number_for_delta: int,
    ) -> None:
        """Persist world elements to Neo4j."""
        await world_queries.sync_world_items(
            world_items_to_persist, chapter_number_for_delta
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
        if not chapter_text:
            logger.warning(
                "Skipping knowledge extraction for chapter %s: no text provided.",
                chapter_number,
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

        char_updates_raw = "{}"
        world_updates_raw = "{}"
        kg_triples_text = ""

        try:
            parsed_json = json.loads(raw_extracted_text)
            char_updates_raw = json.dumps(parsed_json.get("character_updates", {}))
            world_updates_raw = json.dumps(parsed_json.get("world_updates", {}))
            kg_triples_list = parsed_json.get("kg_triples", [])
            if isinstance(kg_triples_list, list):
                kg_triples_text = "\n".join([str(t) for t in kg_triples_list])
            else:
                kg_triples_text = str(kg_triples_list)

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse full extraction JSON for chapter {chapter_number}: {e}. "
                f"Attempting to extract individual sections with regex."
            )
            # Fallback to regex extraction
            char_match = re.search(
                r'"character_updates"\s*:\s*({.*?})', raw_extracted_text, re.DOTALL
            )
            if char_match:
                char_updates_raw = char_match.group(1)
                logger.info(
                    f"Regex successfully extracted character_updates block for Ch {chapter_number}."
                )
            else:
                logger.warning(
                    f"Could not find character_updates JSON block via regex for Ch {chapter_number}."
                )

            world_match = re.search(
                r'"world_updates"\s*:\s*({.*?})', raw_extracted_text, re.DOTALL
            )
            if world_match:
                world_updates_raw = world_match.group(1)
                logger.info(
                    f"Regex successfully extracted world_updates block for Ch {chapter_number}."
                )
            else:
                logger.warning(
                    f"Could not find world_updates JSON block via regex for Ch {chapter_number}."
                )

            triples_match = re.search(
                r'"kg_triples"\s*:\s*(\[.*?\])', raw_extracted_text, re.DOTALL
            )
            if triples_match:
                try:
                    triples_list_from_regex = json.loads(triples_match.group(1))
                    if isinstance(triples_list_from_regex, list):
                        kg_triples_text = "\n".join(
                            [str(t) for t in triples_list_from_regex]
                        )
                        logger.info(
                            f"Regex successfully extracted and parsed kg_triples block for Ch {chapter_number}."
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Found kg_triples block via regex for Ch {chapter_number}, but it was invalid JSON."
                    )
            else:
                logger.warning(
                    f"Could not find kg_triples JSON array via regex for Ch {chapter_number}."
                )

        char_updates_from_llm = self.parse_character_updates(
            char_updates_raw, chapter_number
        )
        world_updates_from_llm = self.parse_world_updates(
            world_updates_raw, chapter_number
        )

        parsed_triples_structured = parse_rdf_triples_with_rdflib(kg_triples_text)

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

    async def heal_and_enrich_kg(self):
        """
        Performs maintenance on the Knowledge Graph by enriching thin nodes
        and checking for inconsistencies.
        """
        logger.info("KG Healer/Enricher: Starting maintenance cycle.")

        # 1. Enrichment (which includes healing orphans/stubs)
        enrichment_cypher = await self._find_and_enrich_thin_nodes()

        if enrichment_cypher:
            logger.info(
                f"Applying {len(enrichment_cypher)} enrichment updates to the KG."
            )
            try:
                await neo4j_manager.execute_cypher_batch(enrichment_cypher)
            except Exception as e:
                logger.error(
                    f"KG Healer/Enricher: Error applying enrichment batch: {e}",
                    exc_info=True,
                )
        else:
            logger.info(
                "KG Healer/Enricher: No thin nodes found for enrichment in this cycle."
            )

        # 2. Consistency Checks
        await self._run_consistency_checks()

        logger.info("KG Healer/Enricher: Maintenance cycle complete.")

    async def _find_and_enrich_thin_nodes(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Finds thin characters and world elements and generates enrichment updates."""
        statements: List[Tuple[str, Dict[str, Any]]] = []

        # Enrich Characters
        thin_chars = await character_queries.find_thin_characters_for_enrichment()
        for char_info in thin_chars:
            char_name = char_info.get("name")
            if not char_name:
                continue

            logger.info(
                f"KG Healer: Found thin character '{char_name}' for enrichment."
            )
            context_chapters = await kg_queries.get_chapter_context_for_entity(
                entity_name=char_name
            )

            prompt = render_prompt(
                "kg_maintainer_agent/enrich_character.j2",
                {"character_name": char_name, "chapter_context": context_chapters},
            )

            enrichment_text, _ = await llm_service.async_call_llm(
                model_name=config.KNOWLEDGE_UPDATE_MODEL,
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                auto_clean_response=True,
            )

            if enrichment_text:
                try:
                    data = json.loads(enrichment_text)
                    new_description = data.get("description")
                    if new_description and isinstance(new_description, str):
                        statements.append(
                            (
                                "MATCH (c:Character {name: $name}) SET c.description = $desc",
                                {"name": char_name, "desc": new_description},
                            )
                        )
                        logger.info(
                            f"KG Healer: Generated new description for '{char_name}'."
                        )
                except json.JSONDecodeError:
                    logger.error(
                        f"KG Healer: Failed to parse enrichment JSON for character '{char_name}': {enrichment_text}"
                    )

        # Enrich World Elements
        thin_elements = await world_queries.find_thin_world_elements_for_enrichment()
        for element_info in thin_elements:
            element_id = element_info.get("id")
            if not element_id:
                continue

            logger.info(
                f"KG Healer: Found thin world element '{element_info.get('name')}' (id: {element_id}) for enrichment."
            )
            context_chapters = await kg_queries.get_chapter_context_for_entity(
                entity_id=element_id
            )

            prompt = render_prompt(
                "kg_maintainer_agent/enrich_world_element.j2",
                {"element": element_info, "chapter_context": context_chapters},
            )

            enrichment_text, _ = await llm_service.async_call_llm(
                model_name=config.KNOWLEDGE_UPDATE_MODEL,
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                auto_clean_response=True,
            )

            if enrichment_text:
                try:
                    data = json.loads(enrichment_text)
                    new_description = data.get("description")
                    if new_description and isinstance(new_description, str):
                        statements.append(
                            (
                                "MATCH (we:WorldElement {id: $id}) SET we.description = $desc",
                                {"id": element_id, "desc": new_description},
                            )
                        )
                        logger.info(
                            f"KG Healer: Generated new description for world element id '{element_id}'."
                        )
                except json.JSONDecodeError:
                    logger.error(
                        f"KG Healer: Failed to parse enrichment JSON for world element id '{element_id}': {enrichment_text}"
                    )

        return statements

    async def _run_consistency_checks(self) -> None:
        """Runs various consistency checks on the KG and logs findings."""
        logger.info("KG Healer: Running consistency checks...")

        # 1. Check for contradictory traits
        contradictory_pairs = [
            ("Brave", "Cowardly"),
            ("Honest", "Deceitful"),
            ("Kind", "Cruel"),
            ("Generous", "Selfish"),
            ("Loyal", "Treacherous"),
        ]
        trait_findings = await kg_queries.find_contradictory_trait_characters(
            contradictory_pairs
        )
        if trait_findings:
            for finding in trait_findings:
                logger.warning(
                    f"KG Consistency Alert: Character '{finding.get('character_name')}' has contradictory traits: "
                    f"'{finding.get('trait1')}' and '{finding.get('trait2')}'."
                )
        else:
            logger.info("KG Consistency Check: No contradictory traits found.")

        # 2. Check for post-mortem activity
        activity_findings = await kg_queries.find_post_mortem_activity()
        if activity_findings:
            for finding in activity_findings:
                logger.warning(
                    f"KG Consistency Alert: Character '{finding.get('character_name')}' was marked dead in chapter "
                    f"{finding.get('death_chapter')} but has later activities: {finding.get('post_mortem_activities')}."
                )
        else:
            logger.info("KG Consistency Check: No post-mortem activity found.")


__all__ = ["KGMaintainerAgent"]

"""Knowledge graph maintenance agent.

This module provides a lightweight wrapper around parsing and merge helpers for
updating the Saga knowledge graph. The project is licensed under the Apache 2.0
license; see the LICENSE file for details.
"""

import logging
from typing import Dict

import config
from core_db.base_db_manager import neo4j_manager
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


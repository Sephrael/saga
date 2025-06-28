"""Agent for KG pre-flight consistency checks."""

from __future__ import annotations

import json
from typing import Any

import structlog
from config import settings
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service
from data_access import character_queries

logger = structlog.get_logger(__name__)

CONTRADICTORY_TRAIT_PAIRS = [("Incorporeal", "Corporeal")]


class PreFlightCheckAgent:
    """Performs core contradiction checks before drafting."""

    def __init__(self, model_name: str = settings.KNOWLEDGE_UPDATE_MODEL) -> None:
        self.model_name = model_name
        logger.info("PreFlightCheckAgent initialized")

    async def _character_has_conflict(
        self, name: str, trait1: str, trait2: str
    ) -> bool:
        query = (
            "MATCH (c:Character {name: $name})-[:HAS_TRAIT]->(t1:Trait {name: $t1})"
            " MATCH (c)-[:HAS_TRAIT]->(t2:Trait {name: $t2}) RETURN c"
        )
        results = await neo4j_manager.execute_read_query(
            query, {"name": name, "t1": trait1, "t2": trait2}
        )
        return bool(results)

    async def _resolve_trait_conflict(
        self, name: str, trait1: str, trait2: str
    ) -> None:
        prompt = (
            "/no_think\n"
            f"A character named {name} has both traits '{trait1}' and '{trait2}'. "
            "Choose the single trait that should remain canonical. "
            'Respond with JSON {"trait": "chosen"}.'
        )
        text, _ = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=0.0,
            max_tokens=40,
            auto_clean_response=True,
        )
        chosen = trait1
        try:
            data = json.loads(text)
            if data.get("trait") in {trait1, trait2}:
                chosen = data["trait"]
        except json.JSONDecodeError:
            logger.warning(
                "PreFlightCheckAgent: could not parse LLM resolution for %s", name
            )
        to_remove = trait2 if chosen == trait1 else trait1
        await character_queries.remove_character_trait(name, to_remove)
        logger.info(
            "PreFlightCheckAgent resolved trait conflict for %s: kept %s removed %s",
            name,
            chosen,
            to_remove,
        )

    async def perform_core_checks(self, plot_outline: dict[str, Any]) -> None:
        protagonist = plot_outline.get("protagonist_name")
        if not protagonist:
            return
        for trait1, trait2 in CONTRADICTORY_TRAIT_PAIRS:
            if await self._character_has_conflict(protagonist, trait1, trait2):
                await self._resolve_trait_conflict(protagonist, trait1, trait2)

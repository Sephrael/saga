from unittest.mock import AsyncMock

import pytest
from agents.pre_flight_check_agent import PreFlightCheckAgent
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service
from data_access import character_queries


@pytest.mark.asyncio
async def test_preflight_resolves_trait(monkeypatch):
    agent = PreFlightCheckAgent()
    monkeypatch.setattr(
        neo4j_manager,
        "execute_read_query",
        AsyncMock(return_value=[{"c": "Saga"}]),
    )

    removed: dict[str, str] = {}

    async def fake_remove(name: str, trait: str) -> bool:
        removed["name"] = name
        removed["trait"] = trait
        return True

    monkeypatch.setattr(
        character_queries,
        "remove_character_trait",
        AsyncMock(side_effect=fake_remove),
    )
    monkeypatch.setattr(
        llm_service,
        "async_call_llm",
        AsyncMock(return_value=('{"trait": "Incorporeal"}', {})),
    )

    await agent.perform_core_checks({"protagonist_name": "Saga"})
    assert removed == {"name": "Saga", "trait": "Corporeal"}

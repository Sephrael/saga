# tests/test_agent_extract.py
import asyncio
from typing import Any

import pytest
from agents.kg_maintainer_agent import KGMaintainerAgent
from kg_maintainer.models import CharacterProfile


class DummyLLM:
    async def async_call_llm(self, *args, **kwargs):
        return (
            '{"character_updates": {"Alice": {"traits": ["brave"], "development_in_chapter_1": "Did stuff"}}, "world_updates": {}, "kg_triples": ["Alice | visited | Town"]}',
            {"total_tokens": 10},
        )


llm_service_mock = DummyLLM()


def test_extract_and_merge(monkeypatch) -> None:
    agent = KGMaintainerAgent()
    monkeypatch.setattr(
        agent,
        "_llm_extract_updates",
        lambda *a, **k: llm_service_mock.async_call_llm(),
    )
    monkeypatch.setattr(
        agent, "persist_profiles", lambda profiles, chapter: asyncio.sleep(0)
    )
    monkeypatch.setattr(agent, "persist_world", lambda world, chapter: asyncio.sleep(0))
    monkeypatch.setattr(
        "data_access.kg_queries.add_kg_triples_batch_to_db",
        lambda triples: asyncio.sleep(0),
    )

    plot_outline = {}
    character_profiles = {"Alice": CharacterProfile(name="Alice", description="Old")}
    world_building = {}

    usage = asyncio.run(
        agent.extract_and_merge_knowledge(
            plot_outline,
            character_profiles,
            world_building,
            1,
            "text",
        )
    )
    assert usage == {"total_tokens": 10}
    assert character_profiles["Alice"].traits == ["brave"]


def test_extract_with_fill_ins(monkeypatch) -> None:
    agent = KGMaintainerAgent()
    monkeypatch.setattr(
        agent,
        "_llm_extract_updates",
        lambda *a, **k: llm_service_mock.async_call_llm(),
    )
    called: dict[str, Any] = {}

    async def fake_add(triples, chapter, provisional):
        called["provisional"] = provisional

    monkeypatch.setattr("data_access.kg_queries.add_kg_triples_batch_to_db", fake_add)
    monkeypatch.setattr(agent, "persist_profiles", lambda *a, **k: asyncio.sleep(0))
    monkeypatch.setattr(agent, "persist_world", lambda *a, **k: asyncio.sleep(0))

    asyncio.run(
        agent.extract_and_merge_knowledge(
            {},
            {},
            {},
            1,
            "text",
            fill_in_context="extra",
        )
    )
    assert called.get("provisional") is True


@pytest.mark.asyncio
async def test_summarize_chapter_json(monkeypatch) -> None:
    agent = KGMaintainerAgent()

    async def _fake_llm(*args, **kwargs):
        return '{"summary": "Short"}', {"prompt_tokens": 1}

    monkeypatch.setattr(
        "agents.kg_maintainer_agent.llm_service.async_call_llm",
        _fake_llm,
    )

    summary, usage = await agent.summarize_chapter("x" * 6000, 1)
    assert summary == "Short"
    assert usage == {"prompt_tokens": 1}

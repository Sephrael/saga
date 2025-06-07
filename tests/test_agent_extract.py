import asyncio

from kg_maintainer.models import CharacterProfile
from kg_maintainer_agent import KGMaintainerAgent


class DummyLLM:
    async def async_call_llm(self, *args, **kwargs):
        return (
            """### CHARACTER UPDATES ###\n{\n    \"Alice\": {\"traits\": [\"brave\"], \"development_in_chapter_1\": \"Did stuff\"}\n}\n\n### KG TRIPLES ###\nAlice | visited | Town""",
            {"total_tokens": 10},
        )


llm_service_mock = DummyLLM()


def test_extract_and_merge(monkeypatch):
    agent = KGMaintainerAgent()
    monkeypatch.setattr(
        agent,
        "_llm_extract_updates",
        lambda props, text, num: llm_service_mock.async_call_llm(),
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

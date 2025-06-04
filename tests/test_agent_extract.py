import asyncio

from kg_maintainer_agent import KGMaintainerAgent

class DummyLLM:
    async def async_call_llm(self, *args, **kwargs):
        return (
            """### CHARACTER UPDATES ###\nCharacter: Alice\nTraits: brave\nDevelopment in Chapter 1: Did stuff\n\n### KG TRIPLES ###\nAlice | visited | Town""",
            {"total_tokens": 10},
        )

llm_service_mock = DummyLLM()

def test_extract_and_merge(monkeypatch):
    agent = KGMaintainerAgent()
    monkeypatch.setattr(
        agent, "_llm_extract_updates", lambda props, text, num: llm_service_mock.async_call_llm()
    )
    monkeypatch.setattr(
        agent, "persist_profiles", lambda profiles: asyncio.sleep(0)
    )
    monkeypatch.setattr(
        agent, "persist_world", lambda world: asyncio.sleep(0)
    )
    monkeypatch.setattr(
        "data_access.kg_queries.add_kg_triples_batch_to_db", lambda triples: asyncio.sleep(0)
    )

    props = {"character_profiles": {"Alice": {"description": "Old"}}, "world_building": {}}

    usage = asyncio.run(agent.extract_and_merge_knowledge(props, 1, "text"))
    assert usage == {"total_tokens": 10}
    assert props["character_profiles"]["Alice"]["traits"] == ["brave"]

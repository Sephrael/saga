import pytest

from agents.kg_maintainer_agent import KGMaintainerAgent
from core.llm_interface import llm_service


@pytest.mark.asyncio
async def test_extract_updates_prompt_includes_names(monkeypatch):
    agent = KGMaintainerAgent()
    captured = {}

    async def fake_llm(*args, **kwargs):
        captured["prompt"] = kwargs.get("prompt") or args[1]
        return "{}", {}

    monkeypatch.setattr(llm_service, "async_call_llm", fake_llm)
    monkeypatch.setattr(agent, "_extract_character_names_from_text", lambda text: [])

    plot_outline = {"protagonist_name": "Hero"}
    await agent._llm_extract_updates(plot_outline, "some text", 1, ["Hero", "Sidekick"])

    assert "Hero" in captured["prompt"]
    assert "Sidekick" in captured["prompt"]

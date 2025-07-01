from unittest.mock import AsyncMock

import pytest
from chapter_generation.context_providers import (
    CanonProvider,
    ContextRequest,
    KGFactProvider,
    KGReasoningProvider,
    PlanProvider,
    UserNoteProvider,
)


@pytest.mark.asyncio
async def test_plan_provider_basic():
    provider = PlanProvider()
    request = ContextRequest(1, None, {}, chapter_plan=[{"summary": "A"}])
    chunk = await provider.get_context(request)
    assert "A" in chunk.text


@pytest.mark.asyncio
async def test_user_note_provider():
    provider = UserNoteProvider()
    request = ContextRequest(1, None, {}, agent_hints={"user_notes": "note"})
    chunk = await provider.get_context(request)
    assert chunk.text == "note"


@pytest.mark.asyncio
async def test_kg_fact_provider(monkeypatch):
    async def fake_get(*args, **kwargs):
        return "fact"

    monkeypatch.setattr(
        "chapter_generation.context_kg_utils.get_reliable_kg_facts_for_drafting_prompt",
        fake_get,
    )
    provider = KGFactProvider()
    request = ContextRequest(2, None, {})
    chunk = await provider.get_context(request)
    assert chunk.text == "fact"


@pytest.mark.asyncio
async def test_kg_reasoning_provider(monkeypatch):
    async def fake_reason(*args, **kwargs):
        return "guide"

    monkeypatch.setattr(
        "chapter_generation.context_kg_utils.get_kg_reasoning_guidance_for_prompt",
        fake_reason,
    )
    provider = KGReasoningProvider()
    request = ContextRequest(2, None, {})
    chunk = await provider.get_context(request)
    assert chunk.text == "guide"


@pytest.mark.asyncio
async def test_canon_provider(monkeypatch):
    async def fake_query(*_args, **_kwargs):
        return [{"name": "S\xe1g\xe1", "trait": "Corporeal"}]

    monkeypatch.setattr(
        "core.db_manager.neo4j_manager.execute_read_query",
        AsyncMock(side_effect=fake_query),
    )

    provider = CanonProvider()
    request = ContextRequest(1, None, {})
    chunk = await provider.get_context(request)
    assert "CANONICAL TRUTHS" in chunk.text
    assert "S\xe1g\xe1" in chunk.text


@pytest.mark.asyncio
async def test_canon_provider_llm_fallback(monkeypatch):
    monkeypatch.setattr(
        "core.db_manager.neo4j_manager.execute_read_query",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "core.llm_interface.llm_service.async_call_llm",
        AsyncMock(return_value=("fallback canon", {})),
    )
    provider = CanonProvider()
    request = ContextRequest(1, None, {"title": "T"})
    chunk = await provider.get_context(request)
    assert "fallback canon" in chunk.text


@pytest.mark.asyncio
async def test_plan_provider_llm_fallback(monkeypatch):
    monkeypatch.setattr(
        "core.llm_interface.llm_service.async_call_llm",
        AsyncMock(return_value=("- a\n- b", {})),
    )
    provider = PlanProvider()
    request = ContextRequest(1, "intro", {"plot_points": ["intro"]})
    chunk = await provider.get_context(request)
    assert "a" in chunk.text

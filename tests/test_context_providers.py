# tests/test_context_providers.py
from unittest.mock import AsyncMock

import pytest
from chapter_generation.context_models import ProviderSettings
from chapter_generation.context_providers import (
    CanonProvider,
    ContextRequest,
    KGFactProvider,
    KGReasoningProvider,
    PlanProvider,
    PlotFocusProvider,
    UpcomingPlotPointsProvider,
    UserNoteProvider,
    WorldStateProvider,
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
    monkeypatch.setattr(
        "chapter_generation.context_kg_utils.get_canonical_truths_from_kg",
        AsyncMock(return_value=["- S\xe1g\xe1 is Corporeal"]),
    )

    provider = CanonProvider()
    request = ContextRequest(1, None, {})
    chunk = await provider.get_context(request)
    assert "CANONICAL TRUTHS" in chunk.text
    assert "S\xe1g\xe1" in chunk.text


@pytest.mark.asyncio
async def test_canon_provider_llm_fallback(monkeypatch):
    monkeypatch.setattr(
        "chapter_generation.context_kg_utils.get_canonical_truths_from_kg",
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
    assert chunk.from_llm_fill


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
    assert chunk.from_llm_fill


@pytest.mark.asyncio
async def test_plot_focus_provider_truncates():
    provider = PlotFocusProvider()
    request = ContextRequest(1, "A very long focus line", {})
    settings = ProviderSettings(provider, max_tokens=2)
    chunk = await provider.get_context(request, settings)
    assert len(chunk.text.split()) <= 2


@pytest.mark.asyncio
async def test_upcoming_plot_points_provider():
    provider = UpcomingPlotPointsProvider()
    outline = {"plot_points": ["a", "b", "c", "d"]}
    request = ContextRequest(1, None, outline)
    chunk = await provider.get_context(request)
    assert "b" in chunk.text and "c" in chunk.text


@pytest.mark.asyncio
async def test_world_state_provider_respects_max_tokens(monkeypatch):
    async def fake_world(*args, **kwargs):
        return {"locations": {"L1": {}, "L2": {}, "L3": {}}}

    monkeypatch.setattr(
        "data_access.world_queries.get_world_building_from_db", fake_world
    )
    provider = WorldStateProvider()
    request = ContextRequest(2, None, {})
    settings = ProviderSettings(provider, max_tokens=3)
    chunk = await provider.get_context(request, settings)
    assert chunk.tokens <= 3


@pytest.mark.asyncio
async def test_plan_provider_unresolved_entities_llm(monkeypatch):
    async def fake_llm(*args, **kwargs):
        prompt = kwargs.get("prompt", "")
        if "A1" in prompt:
            return ("desc1", {})
        return ("desc2", {})

    monkeypatch.setattr(
        "core.llm_interface.llm_service.async_call_llm",
        AsyncMock(side_effect=fake_llm),
    )

    provider = PlanProvider()
    hints = {"unresolved_entities": ["A1", "B2"]}
    request = ContextRequest(1, None, {}, agent_hints=hints)
    chunk = await provider.get_context(request)
    assert "A1" in chunk.text and "B2" in chunk.text
    assert chunk.provenance.get("llm_fill_ins") == {"A1": "desc1", "B2": "desc2"}
    assert chunk.from_llm_fill

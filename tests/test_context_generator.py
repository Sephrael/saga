# tests/test_context_generator.py
from unittest.mock import AsyncMock

import numpy as np
import pytest
from chapter_generation.context_models import (
    ContextProfileName,
    ContextRequest,
    ProfileConfiguration,
    ProviderSettings,
)
from chapter_generation.context_orchestrator import ContextOrchestrator
from chapter_generation.context_providers import SemanticHistoryProvider
from core.llm_interface import llm_service
from data_access import chapter_queries


@pytest.mark.asyncio
async def test_immediate_context_added(monkeypatch):
    async def fake_get_range(start, end):
        return [
            {"number": i, "summary": f"sum{i}", "is_provisional": False}
            for i in range(start, end)
        ]

    async def fake_embedding(_):
        return np.array([0.1], dtype=np.float32)

    async def fake_find_similar(*_args, **_kwargs):
        return [
            {
                "chapter_number": 1,
                "summary": "old",
                "is_provisional": False,
                "score": 0.9,
            }
        ]

    monkeypatch.setattr(
        chapter_queries,
        "get_chapters_data_from_db",
        AsyncMock(side_effect=fake_get_range),
    )
    monkeypatch.setattr(
        llm_service, "async_get_embedding", AsyncMock(side_effect=fake_embedding)
    )
    monkeypatch.setattr(
        chapter_queries,
        "find_similar_chapters_in_db",
        AsyncMock(side_effect=fake_find_similar),
    )

    provider = SemanticHistoryProvider(chapter_queries, llm_service)
    profiles = {
        ContextProfileName.DEFAULT: ProfileConfiguration(
            providers=[ProviderSettings(provider)],
            max_tokens=100,
        )
    }
    orchestrator = ContextOrchestrator(profiles)
    req = ContextRequest(chapter_number=4, plot_focus=None, plot_outline={})
    ctx = await orchestrator.build_context(req)
    assert ctx.startswith("[semantic_history]")
    assert ctx.index("[Immediate Context from Chapter 3") < ctx.index(
        "Semantic Context from Chapter 1"
    )


@pytest.mark.asyncio
async def test_decay_sorting(monkeypatch):
    async def fake_get_range(start, end):
        return [
            {"number": i, "summary": f"sum{i}", "is_provisional": False}
            for i in range(start, end)
        ]

    async def fake_embedding(_):
        return np.array([0.1], dtype=np.float32)

    async def fake_find_similar(*_args, **_kwargs):
        return [
            {
                "chapter_number": 1,
                "summary": "s1",
                "is_provisional": False,
                "score": 0.95,
            },
            {
                "chapter_number": 5,
                "summary": "s5",
                "is_provisional": False,
                "score": 0.95,
            },
            {
                "chapter_number": 8,
                "summary": "s8",
                "is_provisional": False,
                "score": 0.95,
            },
        ]

    monkeypatch.setattr(
        chapter_queries,
        "get_chapters_data_from_db",
        AsyncMock(side_effect=fake_get_range),
    )
    monkeypatch.setattr(
        llm_service, "async_get_embedding", AsyncMock(side_effect=fake_embedding)
    )
    monkeypatch.setattr(
        chapter_queries,
        "find_similar_chapters_in_db",
        AsyncMock(side_effect=fake_find_similar),
    )

    provider = SemanticHistoryProvider(chapter_queries, llm_service)
    profiles = {
        ContextProfileName.DEFAULT: ProfileConfiguration(
            providers=[ProviderSettings(provider)],
            max_tokens=100,
        )
    }
    orchestrator = ContextOrchestrator(profiles)
    req = ContextRequest(chapter_number=11, plot_focus=None, plot_outline={})
    ctx = await orchestrator.build_context(req)
    pos8 = ctx.index("Semantic Context from Chapter 8")
    pos5 = ctx.index("Semantic Context from Chapter 5")
    pos1 = ctx.index("Semantic Context from Chapter 1")
    assert pos8 < pos5 < pos1

from unittest.mock import AsyncMock

import numpy as np
import pytest
from core.llm_interface import llm_service
from data_access import chapter_queries
from processing import context_generator


@pytest.mark.asyncio
async def test_immediate_context_added(monkeypatch):
    async def fake_get_chapter_data(num):
        return {"summary": f"sum{num}", "is_provisional": False}

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
        "get_chapter_data_from_db",
        AsyncMock(side_effect=fake_get_chapter_data),
    )
    monkeypatch.setattr(
        llm_service, "async_get_embedding", AsyncMock(side_effect=fake_embedding)
    )
    monkeypatch.setattr(
        chapter_queries,
        "find_similar_chapters_in_db",
        AsyncMock(side_effect=fake_find_similar),
    )

    ctx = await context_generator._generate_semantic_chapter_context_logic({}, 4)
    assert ctx.startswith("[Immediate Context from Chapter 3")
    assert ctx.index("[Immediate Context from Chapter 2") < ctx.index(
        "Semantic Context from Chapter 1"
    )


@pytest.mark.asyncio
async def test_decay_sorting(monkeypatch):
    async def fake_get_chapter_data(num):
        return {"summary": f"sum{num}", "is_provisional": False}

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
        "get_chapter_data_from_db",
        AsyncMock(side_effect=fake_get_chapter_data),
    )
    monkeypatch.setattr(
        llm_service, "async_get_embedding", AsyncMock(side_effect=fake_embedding)
    )
    monkeypatch.setattr(
        chapter_queries,
        "find_similar_chapters_in_db",
        AsyncMock(side_effect=fake_find_similar),
    )

    ctx = await context_generator._generate_semantic_chapter_context_logic({}, 11)
    pos8 = ctx.index("Semantic Context from Chapter 8")
    pos5 = ctx.index("Semantic Context from Chapter 5")
    pos1 = ctx.index("Semantic Context from Chapter 1")
    assert pos8 < pos5 < pos1

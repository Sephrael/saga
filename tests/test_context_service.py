from unittest.mock import AsyncMock

import numpy as np
import pytest
from chapter_generation.context_service import ContextService
from core.llm_interface import llm_service
from data_access import chapter_queries
from initialization.models import PlotOutline


@pytest.mark.asyncio
async def test_plot_point_focus_used(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_embedding(text: str) -> np.ndarray:
        captured["text"] = text
        return np.array([0.1], dtype=np.float32)

    monkeypatch.setattr(
        llm_service, "async_get_embedding", AsyncMock(side_effect=fake_embedding)
    )
    monkeypatch.setattr(
        chapter_queries, "find_similar_chapters_in_db", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        chapter_queries, "get_chapter_data_from_db", AsyncMock(return_value=None)
    )

    service = ContextService(chapter_queries, llm_service)
    outline = PlotOutline(plot_points=["PP1", "PP2", "PP3"])
    await service.get_semantic_context(outline, 2)
    assert captured["text"] == "PP2"

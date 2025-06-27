from unittest.mock import AsyncMock

import config
import pytest
from initialization.models import PlotOutline
from orchestration.chapter_generation_runner import ChapterGenerationRunner
from orchestration.nana_orchestrator import NANA_Orchestrator


@pytest.mark.asyncio
async def test_runner_basic_flow(monkeypatch):
    monkeypatch.setattr(config.settings, "CHAPTERS_PER_RUN", 2)
    monkeypatch.setattr(config.settings, "KG_HEALING_INTERVAL", 1)

    orch = NANA_Orchestrator()
    orch.plot_outline = PlotOutline(title="T", plot_points=["p1", "p2"])
    orch.chapter_count = 0

    async def fake_run(chapter: int) -> str:
        orch.chapter_count = chapter
        return "text"

    monkeypatch.setattr(
        orch,
        "run_chapter_generation_process",
        AsyncMock(side_effect=fake_run),
    )
    monkeypatch.setattr(orch.kg_maintainer_agent, "heal_and_enrich_kg", AsyncMock())
    monkeypatch.setattr(orch, "refresh_plot_outline", AsyncMock())
    monkeypatch.setattr(orch, "refresh_knowledge_cache", AsyncMock())
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)

    runner = ChapterGenerationRunner(orch)
    await runner.run()

    assert runner.chapters_written == 2
    assert orch.chapter_count == 2
    assert orch.kg_maintainer_agent.heal_and_enrich_kg.call_count == 2

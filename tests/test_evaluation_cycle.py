from unittest.mock import AsyncMock

import pytest
from data_access import character_queries, world_queries
from orchestration.nana_orchestrator import NANA_Orchestrator

import utils


@pytest.fixture
def orchestrator(monkeypatch):
    monkeypatch.setattr(utils, "load_spacy_model_if_needed", lambda: None)
    orch = NANA_Orchestrator()
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)
    return orch


@pytest.mark.asyncio
async def test_ignore_spans_all_attempts(orchestrator, monkeypatch):
    monkeypatch.setattr(
        character_queries, "get_all_character_names", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(
        world_queries, "get_all_world_item_ids_by_category", AsyncMock(return_value={})
    )

    received = {}

    async def fake_eval(*args, ignore_spans=None, **kwargs):
        received.setdefault("eval", []).append(ignore_spans)
        return {"needs_revision": False, "problems_found": []}, {}

    monkeypatch.setattr(
        orchestrator.evaluator_agent,
        "evaluate_chapter_draft",
        AsyncMock(side_effect=fake_eval),
    )

    patched_spans = [(0, 5)]

    await orchestrator._run_evaluation_cycle(
        1,
        1,
        "draft",
        "focus",
        0,
        "ctx",
        patched_spans,
    )

    await orchestrator._run_evaluation_cycle(
        1,
        2,
        "draft",
        "focus",
        0,
        "ctx",
        patched_spans,
    )

    assert received["eval"] == [patched_spans, patched_spans]

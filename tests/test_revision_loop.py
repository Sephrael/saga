from unittest.mock import AsyncMock

import pytest
from config import settings
from data_access import character_queries, world_queries
from initialization.models import PlotOutline
from orchestration.nana_orchestrator import NANA_Orchestrator


class DummyOrchestrator(NANA_Orchestrator):
    def __init__(self):
        super().__init__()
        self.plot_outline = PlotOutline(plot_points=["a"])


@pytest.mark.asyncio
async def test_revision_loop_retries_on_failure(monkeypatch):
    orch = DummyOrchestrator()
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(orch, "_save_debug_output", _noop)
    monkeypatch.setattr(orch, "_accumulate_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        character_queries,
        "get_character_profiles_from_db",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        world_queries,
        "get_world_building_from_db",
        AsyncMock(return_value={}),
    )

    async def fake_eval(*_args, **_kwargs):
        return (
            {"needs_revision": True, "problems_found": [], "reasons": ["bad"]},
            [],
            {},
            {},
            [],
        )

    call_counter = {"count": 0}

    async def fake_revise(*_args, revision_cycle: int, **_kwargs):
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return ("start", None, []), {}
        return ("fixed" * 12, "raw", []), {}

    monkeypatch.setattr(orch, "_run_evaluation_cycle", fake_eval)
    monkeypatch.setattr(orch.revision_manager, "revise_chapter", fake_revise)

    result = await orch._run_revision_loop(
        1,
        "start",
        "raw",
        "focus",
        0,
        "ctx",
        None,
        [],
        False,
    )
    assert result[0].startswith("fixed")
    assert call_counter["count"] == 2


@pytest.mark.asyncio
async def test_revision_loop_deduplicates_each_cycle(monkeypatch):
    orch = DummyOrchestrator()
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(orch, "_save_debug_output", _noop)
    monkeypatch.setattr(orch, "_accumulate_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        character_queries,
        "get_character_profiles_from_db",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        world_queries,
        "get_world_building_from_db",
        AsyncMock(return_value={}),
    )

    eval_calls = {"count": 0}

    async def fake_eval(*_args, **_kwargs):
        eval_calls["count"] += 1
        if eval_calls["count"] == 1:
            return (
                {"needs_revision": True, "problems_found": [], "reasons": ["bad"]},
                [],
                {},
                {},
                [],
            )
        return (
            {"needs_revision": False, "problems_found": [], "reasons": []},
            [],
            {},
            {},
            [],
        )

    async def fake_revise(*_a, revision_cycle: int, **_k):
        return ("revised", None, []), {}

    dedup_calls: list[str] = []

    async def fake_dedup(text: str, _num: int) -> tuple[str, int]:
        dedup_calls.append(text)
        return text + "_d", 1

    monkeypatch.setattr(
        orch, "perform_deduplication", AsyncMock(side_effect=fake_dedup)
    )
    monkeypatch.setattr(orch, "_run_evaluation_cycle", fake_eval)
    monkeypatch.setattr(orch.revision_manager, "revise_chapter", fake_revise)

    result = await orch._run_revision_loop(
        1,
        "start",
        "raw",
        "focus",
        0,
        "ctx",
        None,
        [],
        False,
    )
    assert result[0].endswith("_d")
    assert result[2]
    assert len(dedup_calls) == 2


@pytest.mark.asyncio
async def test_revision_loop_respects_max_cycles(monkeypatch):
    orch = DummyOrchestrator()
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(orch, "_save_debug_output", _noop)
    monkeypatch.setattr(orch, "_accumulate_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        character_queries,
        "get_character_profiles_from_db",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        world_queries,
        "get_world_building_from_db",
        AsyncMock(return_value={}),
    )

    monkeypatch.setattr(
        orch,
        "perform_deduplication",
        AsyncMock(return_value=("start", 0)),
    )

    async def always_needs_revision(*_args, **_kwargs):
        return (
            {"needs_revision": True, "problems_found": [], "reasons": ["bad"]},
            [],
            {},
            {},
            [],
        )

    call_counter = {"count": 0}

    async def fake_revise(*_args, revision_cycle: int, **_kwargs):
        call_counter["count"] += 1
        return ("start", None, []), {}

    monkeypatch.setattr(orch, "_run_evaluation_cycle", always_needs_revision)
    monkeypatch.setattr(orch.revision_manager, "revise_chapter", fake_revise)
    monkeypatch.setattr(settings, "MAX_REVISION_CYCLES_PER_CHAPTER", 3)

    await orch._run_revision_loop(
        1,
        "start",
        "raw",
        "focus",
        0,
        "ctx",
        None,
        [],
        False,
    )

    assert call_counter["count"] == 3

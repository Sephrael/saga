# tests/test_kg_root_cause_trigger.py
from unittest.mock import AsyncMock

import pytest
from data_access import character_queries, world_queries
from initialization.models import PlotOutline
from orchestration.nana_orchestrator import NANA_Orchestrator


class DummyOrchestrator(NANA_Orchestrator):
    def __init__(self) -> None:
        super().__init__()
        self.plot_outline = PlotOutline(plot_points=["a"])


@pytest.mark.asyncio
async def test_kg_healing_triggered(monkeypatch: pytest.MonkeyPatch) -> None:
    orch = DummyOrchestrator()
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_save_debug_output", AsyncMock())
    monkeypatch.setattr(orch, "_accumulate_tokens", lambda *a, **k: None)
    monkeypatch.setattr(
        character_queries, "get_character_profiles_from_db", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        world_queries, "get_world_building_from_db", AsyncMock(return_value={})
    )
    heal = AsyncMock()
    monkeypatch.setattr(orch.kg_maintainer_agent, "heal_and_enrich_kg", heal)

    async def fake_eval(*_a, **_k):
        return (
            {"needs_revision": False, "problems_found": [], "reasons": []},
            [],
            {},
            {},
            [],
        )

    async def fake_revise(*_a, **_k):
        return ("text", None, [])

    monkeypatch.setattr(orch, "_run_evaluation_cycle", fake_eval)
    monkeypatch.setattr(orch.revision_manager, "revise_chapter", fake_revise)
    monkeypatch.setattr(
        orch.revision_manager,
        "identify_root_cause",
        lambda *_a,
        **_k: "The error originates from the conflicting description in Bob's character profile.",
    )

    await orch._run_revision_loop(1, "t", "raw", "focus", 0, "ctx", None, [], False)
    assert heal.call_count == 1

from unittest.mock import AsyncMock

import pytest

import utils
from nana_orchestrator import NANA_Orchestrator
from story_models import (
    KeyLocationModel,
    NovelConceptModel,
    ProtagonistModel,
    SettingModel,
    UserStoryInputModel,
)


@pytest.fixture
def orchestrator(monkeypatch):
    monkeypatch.setattr(utils, "load_spacy_model_if_needed", lambda: None)
    orch = NANA_Orchestrator()
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)
    return orch


@pytest.mark.asyncio
async def test_validate_plot_outline_missing(orchestrator):
    orchestrator.plot_outline = {}
    assert not await orchestrator._validate_plot_outline(1)


@pytest.mark.asyncio
async def test_process_prereq_result_failure(orchestrator):
    result = await orchestrator._process_prereq_result(1, (None, -1, None, None))
    assert result is None


@pytest.mark.asyncio
async def test_process_initial_draft_failure(orchestrator):
    result = await orchestrator._process_initial_draft(1, (None, None))
    assert result is None


@pytest.mark.asyncio
async def test_process_revision_result_failure(orchestrator):
    result = await orchestrator._process_revision_result(1, (None, None, False))
    assert result is None


@pytest.mark.asyncio
async def test_finalize_and_log_success(orchestrator, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_finalize_and_save_chapter",
        AsyncMock(return_value="final"),
    )
    result = await orchestrator._finalize_and_log(1, "text", None, False)
    assert result == "final"


@pytest.mark.asyncio
async def test_finalize_and_log_failure(orchestrator, monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "_finalize_and_save_chapter",
        AsyncMock(return_value=None),
    )
    result = await orchestrator._finalize_and_log(1, "text", None, True)
    assert result is None


def test_load_state_from_user_model(orchestrator):
    model = UserStoryInputModel(
        novel_concept=NovelConceptModel(title="My Tale"),
        protagonist=ProtagonistModel(name="Hero"),
        setting=SettingModel(key_locations=[KeyLocationModel(name="Town")]),
    )
    orchestrator.load_state_from_user_model(model)
    assert orchestrator.plot_outline.get("title") == "My Tale"
    assert "Hero" in orchestrator.character_profiles
    assert orchestrator.world_building.get("locations", {}).get("Town")


@pytest.mark.asyncio
async def test_prepare_prerequisites_uses_plan(orchestrator, monkeypatch):
    orchestrator.plot_outline = {"plot_points": ["Intro"]}
    monkeypatch.setattr(orchestrator, "_update_novel_props_cache", lambda: None)

    async def fake_plan(*_args, **_kwargs):
        return ([{"scene_number": 1}], {"total_tokens": 1})

    async def fake_context(_self, chapter_number: int, plan):
        assert chapter_number == 1
        assert plan == [{"scene_number": 1}]
        return "ctx"

    monkeypatch.setattr(
        orchestrator.planner_agent,
        "plan_chapter_scenes",
        AsyncMock(side_effect=fake_plan),
    )
    monkeypatch.setattr(
        "nana_orchestrator.generate_hybrid_chapter_context_logic",
        AsyncMock(side_effect=fake_context),
    )

    result = await orchestrator._prepare_chapter_prerequisites(1)
    assert result == ("Intro", 0, [{"scene_number": 1}], "ctx")

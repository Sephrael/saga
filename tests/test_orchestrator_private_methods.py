from unittest.mock import AsyncMock

import pytest
import utils
from chapter_generation.drafting_service import DraftResult
from chapter_generation.prerequisites_service import PrerequisiteData
from data_access import character_queries, world_queries
from initialization.models import PlotOutline
from orchestration.nana_orchestrator import NANA_Orchestrator, RevisionOutcome

from models.user_input_models import (
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
    monkeypatch.setattr(orch, "refresh_plot_outline", AsyncMock())
    monkeypatch.setattr(orch, "refresh_knowledge_cache", AsyncMock())
    monkeypatch.setattr(
        orch.context_service,
        "build_hybrid_context",
        AsyncMock(return_value="ctx"),
    )
    return orch


@pytest.mark.asyncio
async def test_validate_plot_outline_missing(orchestrator):
    orchestrator.plot_outline = PlotOutline()
    assert not await orchestrator._validate_plot_outline(1)


@pytest.mark.asyncio
async def test_process_prereq_result_failure(orchestrator):
    result = await orchestrator._process_prereq_result(
        1, PrerequisiteData(None, -1, None, None)
    )
    assert result is None


@pytest.mark.asyncio
async def test_process_initial_draft_failure(orchestrator):
    result = await orchestrator._process_initial_draft(1, DraftResult(None, None))
    assert result is None


@pytest.mark.asyncio
async def test_process_revision_result_failure(orchestrator):
    result = await orchestrator._process_revision_result(
        1, RevisionOutcome(None, None, False)
    )
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
    assert orchestrator.plot_outline.get("protagonist_name") == "Hero"


@pytest.mark.asyncio
async def test_prepare_prerequisites_uses_plan(orchestrator, monkeypatch):
    orchestrator.plot_outline = PlotOutline(plot_points=["Intro"])
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
        character_queries,
        "get_character_profiles_from_db",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        world_queries,
        "get_world_building_from_db",
        AsyncMock(return_value={}),
    )
    await orchestrator.refresh_knowledge_cache()
    monkeypatch.setattr(
        character_queries,
        "get_character_profiles_from_db",
        AsyncMock(side_effect=Exception("cache not used")),
    )
    monkeypatch.setattr(
        world_queries,
        "get_world_building_from_db",
        AsyncMock(side_effect=Exception("cache not used")),
    )
    monkeypatch.setattr(
        orchestrator.world_continuity_agent,
        "check_scene_plan_consistency",
        AsyncMock(return_value=([], {})),
    )
    monkeypatch.setattr(
        orchestrator.context_service,
        "build_hybrid_context",
        AsyncMock(side_effect=fake_context),
    )

    result = await orchestrator._prepare_chapter_prerequisites(1)
    assert result == PrerequisiteData(
        plot_point_focus="Intro",
        plot_point_index=0,
        chapter_plan=[{"scene_number": 1}],
        hybrid_context_for_draft="ctx",
    )

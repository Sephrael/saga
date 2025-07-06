from unittest.mock import AsyncMock

import config
import pytest
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service
from data_access import plot_queries
from orchestration.nana_orchestrator import NANA_Orchestrator

import utils


@pytest.mark.asyncio
async def test_ingestion_triggers_healing(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "load_spacy_model_if_needed", lambda: None)
    monkeypatch.setattr(config.settings, "KG_HEALING_INTERVAL", 2)

    orch = NANA_Orchestrator()
    monkeypatch.setattr(orch, "_update_rich_display", lambda *a, **k: None)

    chapters = ["a", "b", "c", "d", "e"]

    monkeypatch.setattr(
        "ingestion.ingestion_manager.split_text_into_chapters",
        lambda _t: chapters,
    )
    monkeypatch.setattr(
        orch.finalize_agent,
        "ingest_and_finalize_chunk",
        AsyncMock(return_value={"summary": "s"}),
    )
    heal = AsyncMock()
    monkeypatch.setattr(orch.kg_maintainer_agent, "heal_and_enrich_kg", heal)
    monkeypatch.setattr(
        orch.planner_agent, "plan_continuation", AsyncMock(return_value=(None, {}))
    )

    monkeypatch.setattr(orch.display, "start", lambda: None)
    monkeypatch.setattr(orch.display, "stop", AsyncMock())
    llm_close = AsyncMock()
    monkeypatch.setattr(llm_service, "aclose", llm_close)
    monkeypatch.setattr(neo4j_manager, "connect", AsyncMock())
    monkeypatch.setattr(neo4j_manager, "create_db_schema", AsyncMock())
    monkeypatch.setattr(neo4j_manager, "close", AsyncMock())
    monkeypatch.setattr(orch.kg_maintainer_agent, "load_schema_from_db", AsyncMock())
    monkeypatch.setattr(plot_queries, "save_plot_outline_to_db", AsyncMock())
    monkeypatch.setattr(
        plot_queries,
        "get_plot_outline_from_db",
        AsyncMock(return_value={"plot_points": []}),
    )

    text_file = tmp_path / "novel.txt"
    text_file.write_text("dummy")

    await orch.run_ingestion_process(str(text_file))

    assert heal.call_count == 3
    llm_close.assert_awaited_once()

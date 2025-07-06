# tests/ingestion/test_ingestion_manager.py

from unittest.mock import AsyncMock

import config
import pytest
from agents.finalize_agent import FinalizeAgent
from agents.kg_maintainer_agent import KGMaintainerAgent
from agents.planner_agent import PlannerAgent
from core.db_manager import neo4j_manager
from data_access import plot_queries
from ingestion import IngestionManager
from storage.file_manager import FileManager


@pytest.mark.asyncio
async def test_ingest_collects_plot_and_triggers_healing(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "KG_HEALING_INTERVAL", 1)

    text_file = tmp_path / "dummy.txt"
    text_file.write_text("a\n\nb")

    finalize_agent = FinalizeAgent()
    planner_agent = PlannerAgent()
    kg_agent = KGMaintainerAgent()

    fm = FileManager(
        chapters_dir=str(tmp_path / "chapters"),
        logs_dir=str(tmp_path / "logs"),
        debug_dir=str(tmp_path / "debug"),
    )
    manager = IngestionManager(finalize_agent, planner_agent, kg_agent, fm)

    monkeypatch.setattr(neo4j_manager, "connect", AsyncMock())
    monkeypatch.setattr(neo4j_manager, "create_db_schema", AsyncMock())
    monkeypatch.setattr(neo4j_manager, "close", AsyncMock())
    monkeypatch.setattr(kg_agent, "load_schema_from_db", AsyncMock())

    chapters = ["c1", "c2"]
    monkeypatch.setattr(
        "ingestion.ingestion_manager.split_text_into_chapters", lambda _t: chapters
    )

    async def fake_ingest(po, cp, wb, idx, chunk):
        return {"summary": f"sum{idx}"}

    monkeypatch.setattr(finalize_agent, "ingest_and_finalize_chunk", fake_ingest)

    heal = AsyncMock()
    monkeypatch.setattr(kg_agent, "heal_and_enrich_kg", heal)
    monkeypatch.setattr(
        planner_agent, "plan_continuation", AsyncMock(return_value=(["p"], {}))
    )
    save_outline = AsyncMock()
    monkeypatch.setattr(plot_queries, "save_plot_outline_to_db", save_outline)

    outline, count = await manager.ingest(str(text_file))

    assert count == 2
    assert outline["plot_points"] == ["sum1", "sum2", "p"]
    assert heal.call_count == 3
    save_outline.assert_awaited_once_with(outline)

# tests/data_access/services/test_world_persistence_service.py
import kg_constants as kg_keys
import pytest
from config import settings
from core.db_manager import neo4j_manager
from data_access.services import world_persistence_service as wps
from kg_maintainer.models import WorldItem


@pytest.mark.asyncio
async def test_sync_world_items_incremental(monkeypatch):
    service = wps.WorldPersistenceService()

    item = WorldItem(id="loc_town", category="Location", name="Town", properties={})
    items = {"Location": {"Town": item}}
    expected_stmts = [("CREATE (n)", {"id": "loc_town"})]

    def fake_gen(item_arg: WorldItem, chapter: int):
        assert item_arg is item
        assert chapter == 1
        return expected_stmts

    async def fake_batch(stmts):
        monkeypatch.stmts = stmts

    monkeypatch.setattr(wps, "generate_world_element_node_cypher", fake_gen)
    monkeypatch.setattr(neo4j_manager, "execute_cypher_batch", fake_batch)

    result = await service.sync_world_items_incremental(items, 1)
    assert result is True
    assert monkeypatch.stmts == expected_stmts


def test_generate_world_container_statements():
    service = wps.WorldPersistenceService()
    ov_details = {
        "description": "Overview",
        kg_keys.source_quality_key(
            settings.KG_PREPOPULATION_CHAPTER_NUM
        ): "provisional_from_unrevised_draft",
        "title": "World",
    }
    stmts = service._generate_world_container_statements(ov_details, "wc1", "novel1")

    assert len(stmts) == 2
    merge_query, merge_params = stmts[0]
    assert "MERGE (wc:Entity" in merge_query
    props = merge_params["props"]
    assert props["id"] == "wc1"
    assert props["overview_description"] == "Overview"
    assert props[kg_keys.KG_IS_PROVISIONAL] is True
    assert props["title"] == "World"

    link_query, link_params = stmts[1]
    assert "HAS_WORLD_META" in link_query
    assert link_params == {"novel_id_val": "novel1", "wc_id_val": "wc1"}

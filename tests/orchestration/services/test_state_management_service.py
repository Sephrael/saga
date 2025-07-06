# tests/orchestration/services/test_state_management_service.py
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from initialization.models import PlotOutline
from orchestration.models import KnowledgeCache
from orchestration.services.state_management_service import StateManagementService

from models.agent_models import ChapterEndState


# Mock parts of NANA_Orchestrator that StateManagementService interacts with
class MockKnowledgeService:
    update_novel_props_cache = MagicMock()
    refresh_plot_outline = AsyncMock()
    refresh_knowledge_cache = AsyncMock()


class MockContextService:
    build_hybrid_context = AsyncMock(return_value="mock_next_chapter_context")
    llm_fill_chunks = []  # Simulate property


class MockOrchestratorForState:
    def __init__(self):
        self.knowledge_service = MockKnowledgeService()
        self.context_service = MockContextService()
        # StateManager also accesses orchestrator.knowledge_cache directly for refresh
        self.knowledge_cache = KnowledgeCache(characters={}, world={})
        # Add any other attributes StateManager might access from orchestrator


class TestStateManagementService(
    unittest.IsolatedAsyncioTestCase
):  # Use IsolatedAsyncioTestCase for async tests
    def setUp(self):
        self.mock_orchestrator = MockOrchestratorForState()
        self.service = StateManagementService(self.mock_orchestrator)

    def test_initialization(self):
        self.assertIsInstance(self.service.plot_outline, PlotOutline)
        self.assertEqual(self.service.chapter_count, 0)
        self.assertIsInstance(self.service.knowledge_cache, KnowledgeCache)
        # ... test other initial states

    @patch(
        "orchestration.services.state_management_service.chapter_repository",
        new_callable=MagicMock,
    )
    @patch(
        "orchestration.services.state_management_service.plot_queries",
        new_callable=MagicMock,
    )
    async def test_async_init_state_success(self, mock_plot_queries, mock_chapter_repo):
        mock_chapter_repo.load_chapter_count = AsyncMock(return_value=5)
        mock_plot_queries.ensure_novel_info = AsyncMock()
        mock_plot_queries.get_plot_outline_from_db = AsyncMock(
            return_value={"title": "Test Novel", "plot_points": [{"summary": "PP1"}]}
        )
        mock_plot_queries.get_completed_plot_points = AsyncMock(
            return_value={"PP1_focus"}
        )

        # Mock knowledge service calls made during init
        self.mock_orchestrator.knowledge_service.update_novel_props_cache.reset_mock()
        self.mock_orchestrator.knowledge_service.refresh_knowledge_cache = AsyncMock()

        await self.service.async_init_state()

        self.assertEqual(self.service.chapter_count, 5)
        self.assertEqual(self.service.plot_outline.get("title"), "Test Novel")
        self.assertIn("PP1_focus", self.service.completed_plot_points)

        mock_chapter_repo.load_chapter_count.assert_called_once()
        mock_plot_queries.get_plot_outline_from_db.assert_called_once()
        self.mock_orchestrator.knowledge_service.update_novel_props_cache.assert_called_once()
        self.mock_orchestrator.knowledge_service.refresh_knowledge_cache.assert_called_once()

    @patch(
        "orchestration.services.state_management_service.chapter_repository",
        new_callable=MagicMock,
    )
    @patch(
        "orchestration.services.state_management_service.plot_queries",
        new_callable=MagicMock,
    )
    async def test_async_init_state_plot_outline_exception(
        self, mock_plot_queries, mock_chapter_repo
    ):
        mock_chapter_repo.load_chapter_count = AsyncMock(return_value=2)
        mock_plot_queries.ensure_novel_info = AsyncMock()
        mock_plot_queries.get_plot_outline_from_db = AsyncMock(
            side_effect=Exception("DB error")
        )
        mock_plot_queries.get_completed_plot_points = AsyncMock(return_value=set())
        self.mock_orchestrator.knowledge_service.refresh_knowledge_cache = AsyncMock()

        await self.service.async_init_state()

        self.assertEqual(self.service.chapter_count, 2)
        self.assertIsNotNone(
            self.service.plot_outline
        )  # Should be an empty PlotOutline
        self.assertEqual(len(self.service.plot_outline.get("plot_points", [])), 0)

    async def test_refresh_plot_outline(self):
        # Mock the knowledge_service call and the subsequent DB call
        self.mock_orchestrator.knowledge_service.refresh_plot_outline = AsyncMock()
        with patch(
            "orchestration.services.state_management_service.plot_queries.get_plot_outline_from_db",
            AsyncMock(return_value={"title": "Refreshed"}),
        ) as mock_get_db:
            await self.service.refresh_plot_outline()
            self.mock_orchestrator.knowledge_service.refresh_plot_outline.assert_called_once()
            mock_get_db.assert_called_once()
            self.assertEqual(self.service.plot_outline.get("title"), "Refreshed")

    async def test_refresh_knowledge_cache(self):
        self.mock_orchestrator.knowledge_service.refresh_knowledge_cache = AsyncMock()
        # Simulate knowledge_cache on orchestrator being updated by knowledge_service
        self.mock_orchestrator.knowledge_cache = KnowledgeCache(
            characters={"char1": {"name": "Char1"}}, world={"loc1": {"name": "Loc1"}}
        )

        await self.service.refresh_knowledge_cache()

        self.mock_orchestrator.knowledge_service.refresh_knowledge_cache.assert_called_once()
        self.assertEqual(
            self.service.knowledge_cache.characters, {"char1": {"name": "Char1"}}
        )
        self.assertEqual(self.service.knowledge_cache.world, {"loc1": {"name": "Loc1"}})

    @patch(
        "orchestration.services.state_management_service.chapter_repository.get_chapter_data",
        new_callable=AsyncMock,
    )
    async def test_load_previous_end_state(self, mock_get_chapter_data):
        # Test chapter > 0
        mock_end_state_json = ChapterEndState(
            summary="End state sum",
            final_prompt="Final prompt",
            characters_present=["c1"],
        ).model_dump_json()
        mock_get_chapter_data.return_value = {"end_state_json": mock_end_state_json}

        end_state = await self.service.load_previous_end_state(3)
        self.assertIsNotNone(end_state)
        self.assertEqual(end_state.summary, "End state sum")
        mock_get_chapter_data.assert_called_with(3)

        # Test chapter 0 (should use self.chapter_zero_end_state if loaded, or load it)
        self.service.chapter_zero_end_state = ChapterEndState(
            summary="Ch0 Sum", final_prompt="FP0", characters_present=[]
        )
        end_state_0_cached = await self.service.load_previous_end_state(0)
        self.assertEqual(end_state_0_cached.summary, "Ch0 Sum")
        mock_get_chapter_data.assert_called_with(
            3
        )  # Should not be called again for 0 if cached

        # Test chapter 0 (not cached, needs loading)
        self.service.chapter_zero_end_state = None
        mock_get_chapter_data.reset_mock()
        mock_get_chapter_data.return_value = {
            "end_state_json": ChapterEndState(
                summary="Loaded Ch0", final_prompt="LFP0", characters_present=[]
            ).model_dump_json()
        }
        end_state_0_loaded = await self.service.load_previous_end_state(0)
        self.assertIsNotNone(end_state_0_loaded)
        self.assertEqual(end_state_0_loaded.summary, "Loaded Ch0")
        mock_get_chapter_data.assert_called_with(0)

    @patch(
        "orchestration.services.state_management_service.plot_queries",
        new_callable=MagicMock,
    )
    @patch(
        "orchestration.services.state_management_service.neo4j_manager.driver",
        new_callable=MagicMock,
    )
    async def test_update_state_after_chapter_finalization(
        self, mock_driver, mock_plot_queries
    ):
        mock_driver.is_connected = True  # Simulate connected driver
        self.service.plot_outline = PlotOutline(
            title="Test", plot_points=[{"summary": "pp1"}, {"summary": "pp2"}]
        )
        self.service.completed_plot_points = set()

        # Mock methods called during update
        self.service.refresh_plot_outline = AsyncMock()
        self.service.refresh_knowledge_cache = AsyncMock()
        self.service._store_pending_fill_ins = MagicMock()  # It's synchronous

        mock_plot_queries.mark_plot_point_completed = AsyncMock()

        end_state_mock = ChapterEndState(
            summary="Chapter Done", final_prompt="FP", characters_present=[]
        )

        # Test when chapter number aligns with PLOT_POINT_CHAPTER_SPAN
        # Assume settings.PLOT_POINT_CHAPTER_SPAN = 1 for simplicity here, or mock get_plot_point_info
        with patch(
            "orchestration.services.state_management_service.get_plot_point_info",
            return_value=("pp1_focus", 0),
        ):
            with patch(
                "orchestration.services.state_management_service.settings.PLOT_POINT_CHAPTER_SPAN",
                1,
            ):
                await self.service.update_state_after_chapter_finalization(
                    1, end_state_mock
                )
                mock_plot_queries.mark_plot_point_completed.assert_called_once_with(0)
                self.assertIn("pp1_focus", self.service.completed_plot_points)

        self.assertEqual(self.service.last_chapter_end_state, end_state_mock)
        self.service.refresh_plot_outline.assert_called()  # Called at least once
        self.service.refresh_knowledge_cache.assert_called()  # Called at least once
        self.mock_orchestrator.context_service.build_hybrid_context.assert_called_once()
        self.service._store_pending_fill_ins.assert_called_once()
        self.assertEqual(self.service.next_chapter_context, "mock_next_chapter_context")

    def test_store_pending_fill_ins(self):
        # Mock context_service.llm_fill_chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Fill in 1"
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "Fill in 2"
        mock_chunk3 = MagicMock()  # Test with None text
        mock_chunk3.text = None
        self.mock_orchestrator.context_service.llm_fill_chunks = [
            mock_chunk1,
            mock_chunk2,
            mock_chunk3,
        ]

        self.service._store_pending_fill_ins()
        self.assertEqual(self.service.pending_fill_ins, ["Fill in 1", "Fill in 2"])

    @patch("orchestration.services.state_management_service.convert_model_to_objects")
    def test_load_state_from_user_model(self, mock_convert):
        mock_user_model = MagicMock()  # UserStoryInputModel
        expected_plot_outline = PlotOutline(title="From User")
        mock_convert.return_value = (expected_plot_outline, [], [])

        self.service.load_state_from_user_model(mock_user_model)

        mock_convert.assert_called_once_with(mock_user_model)
        self.assertEqual(self.service.plot_outline, expected_plot_outline)

    # Test getters and setters
    def test_getters_setters(self):
        po = PlotOutline(title="PO Test")
        self.service.set_plot_outline(po)
        self.assertEqual(self.service.get_plot_outline(), po)

        kc = KnowledgeCache(characters={"c": {}}, world={"w": {}})
        self.service.knowledge_cache = (
            kc  # Direct set for simplicity if no setter logic
        )
        self.assertEqual(self.service.get_knowledge_cache(), kc)

        # ... and so on for other simple getters/setters ...
        self.service.set_chapter_count(10)
        self.assertEqual(self.service.get_chapter_count(), 10)
        self.service.increment_chapter_count()
        self.assertEqual(self.service.get_chapter_count(), 11)


if __name__ == "__main__":
    asyncio.run(unittest.main())

# tests/orchestration/services/test_token_management_service.py

import asyncio
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

from orchestration.services.token_management_service import TokenManagementService
from orchestration.token_accountant import TokenAccountant, Stage
from core.usage import TokenUsage

# Mock NANA_Orchestrator and RichDisplayManager enough for the tests
class MockRichDisplayManager:
    def update(self, plot_outline, chapter_num, step, total_tokens, run_start_time):
        pass # We'll mock this specifically in tests if needed

class MockOrchestrator:
    def __init__(self):
        self.state_manager = MagicMock()
        # Mock plot_outline directly on state_manager if get_plot_outline is simple
        self.state_manager.get_plot_outline = MagicMock(return_value={"title": "Test Novel"})
        self.run_start_time = 0.0


class TestTokenManagementService(unittest.TestCase):

    def setUp(self):
        self.mock_display_manager = MockRichDisplayManager()
        self.mock_orchestrator_ref = MockOrchestrator()
        self.service = TokenManagementService(self.mock_display_manager, self.mock_orchestrator_ref)

    def test_initialization(self):
        self.assertIsNotNone(self.service.token_accountant)
        self.assertEqual(self.service.total_tokens_generated_this_run, 0)

    @patch.object(TokenAccountant, 'record_usage')
    @patch.object(TokenManagementService, '_update_rich_display')
    def test_accumulate_tokens_records_and_updates_display(self, mock_update_display, mock_record_usage):
        # Mock TokenAccountant's total property
        # One way is to patch the instance's 'total'
        # For simplicity, we'll assume record_usage updates an internal state that
        # would be reflected if we had a real TokenAccountant, and then we set 'total'
        # on the service's accountant instance if needed, or mock accountant.total.

        # Let's make the accountant a MagicMock too for easier total mocking
        self.service.token_accountant = MagicMock(spec=TokenAccountant)
        type(self.service.token_accountant).total = PropertyMock(return_value=100)

        stage = Stage.DRAFTING
        usage_data = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        chapter_num = 1
        step_display = "Drafting Chapter 1"

        self.service.accumulate_tokens(stage, usage_data, chapter_num, step_display)

        self.service.token_accountant.record_usage.assert_called_once_with(stage.value, usage_data)
        self.assertEqual(self.service.total_tokens_generated_this_run, 100)
        mock_update_display.assert_called_once_with(chapter_num=chapter_num, step=step_display)

    def test_accumulate_tokens_with_string_stage(self):
        self.service.token_accountant = MagicMock(spec=TokenAccountant)
        type(self.service.token_accountant).total = PropertyMock(return_value=50)

        stage_str = "custom_stage"
        usage_data = {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}

        with patch.object(self.service, '_update_rich_display') as mock_update_display:
            self.service.accumulate_tokens(stage_str, usage_data)
            self.service.token_accountant.record_usage.assert_called_once_with(stage_str, usage_data)
            self.assertEqual(self.service.total_tokens_generated_this_run, 50)
            mock_update_display.assert_called_once_with(chapter_num=None, step=None)


    @patch.object(MockRichDisplayManager, 'update')
    def test_update_rich_display_calls_display_manager(self, mock_display_update):
        self.mock_orchestrator_ref.run_start_time = 123.45
        expected_plot_outline = {"title": "Test Novel From Mock"}
        self.mock_orchestrator_ref.state_manager.get_plot_outline.return_value = expected_plot_outline

        chapter_num = 2
        step = "Testing Display"
        self.service.total_tokens_generated_this_run = 200 # Set a value for the test

        self.service._update_rich_display(chapter_num=chapter_num, step=step)

        mock_display_update.assert_called_once_with(
            plot_outline=expected_plot_outline,
            chapter_num=chapter_num,
            step=step,
            total_tokens=200,
            run_start_time=123.45
        )
        self.mock_orchestrator_ref.state_manager.get_plot_outline.assert_called_once()


    def test_get_total_tokens(self):
        self.service.total_tokens_generated_this_run = 500
        self.assertEqual(self.service.get_total_tokens_generated_this_run(), 500)

    @patch.object(TokenManagementService, '_update_rich_display')
    def test_reset_tokens_for_new_run(self, mock_update_display):
        self.service.total_tokens_generated_this_run = 1000
        self.service.token_accountant.record_usage("some_stage", {"total_tokens": 1000}) # Simulate usage

        self.service.reset_tokens_for_new_run()

        self.assertEqual(self.service.total_tokens_generated_this_run, 0)
        # Check if a new TokenAccountant instance was created
        # This is a bit white-boxy, but important for reset logic
        self.assertIsInstance(self.service.token_accountant, TokenAccountant)
        # Ensure the new accountant is also reset (e.g. its total is 0)
        self.assertEqual(self.service.token_accountant.total, 0)
        mock_update_display.assert_called_once_with(step="New Run Initialized - Tokens Reset")

    def test_get_set_run_start_time(self):
        self.service.set_run_start_time(500.0)
        self.assertEqual(self.mock_orchestrator_ref.run_start_time, 500.0)
        self.assertEqual(self.service.get_run_start_time(), 500.0)


if __name__ == '__main__':
    unittest.main()

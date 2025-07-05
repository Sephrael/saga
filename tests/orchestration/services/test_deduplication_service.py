# tests/orchestration/services/test_deduplication_service.py

import asyncio
import unittest
from unittest.mock import patch, AsyncMock

from orchestration.services.deduplication_service import DeduplicationService
from processing.text_deduplicator import TextDeduplicator # To mock its behavior
from config import settings # To potentially access settings if needed

class TestDeduplicationService(unittest.TestCase):

    def setUp(self):
        self.service = DeduplicationService()

    @patch('orchestration.services.deduplication_service.TextDeduplicator')
    def test_perform_deduplication_empty_text(self, MockTextDeduplicator):
        """Test that empty or whitespace-only text returns immediately without calling TextDeduplicator."""
        mock_deduper_instance = MockTextDeduplicator.return_value

        loop = asyncio.get_event_loop()

        # Test with empty string
        text_empty = ""
        result_text, chars_removed = loop.run_until_complete(
            self.service.perform_deduplication(text_empty, 1, "test_empty")
        )
        self.assertEqual(result_text, text_empty)
        self.assertEqual(chars_removed, 0)
        mock_deduper_instance.deduplicate.assert_not_called()

        # Test with whitespace only string
        text_whitespace = "   \n\t  "
        result_text, chars_removed = loop.run_until_complete(
            self.service.perform_deduplication(text_whitespace, 1, "test_whitespace")
        )
        self.assertEqual(result_text, text_whitespace)
        self.assertEqual(chars_removed, 0)
        mock_deduper_instance.deduplicate.assert_not_called()

    @patch('orchestration.services.deduplication_service.TextDeduplicator')
    async def test_perform_deduplication_calls_deduplicator(self, MockTextDeduplicator):
        """Test that TextDeduplicator is called for non-empty text."""
        mock_deduper_instance = MockTextDeduplicator.return_value
        mock_deduper_instance.deduplicate = AsyncMock(return_value=("deduplicated text", 10))

        text_to_dedup = "This is some sample text with repetition repetition."
        chapter_number = 1
        context = "test_normal_call"

        result_text, chars_removed = await self.service.perform_deduplication(
            text_to_dedup, chapter_number, context
        )

        MockTextDeduplicator.assert_called_once_with(
            similarity_threshold=settings.DEDUPLICATION_SEMANTIC_THRESHOLD,
            use_semantic_comparison=settings.DEDUPLICATION_USE_SEMANTIC,
            min_segment_length_chars=settings.DEDUPLICATION_MIN_SEGMENT_LENGTH,
        )
        mock_deduper_instance.deduplicate.assert_called_once_with(
            text_to_dedup, segment_level="sentence"
        )
        self.assertEqual(result_text, "deduplicated text")
        self.assertEqual(chars_removed, 10)

    @patch('orchestration.services.deduplication_service.TextDeduplicator')
    async def test_perform_deduplication_no_chars_removed(self, MockTextDeduplicator):
        """Test scenario where deduplicator removes no characters."""
        mock_deduper_instance = MockTextDeduplicator.return_value
        original_text = "This text has no duplicates."
        mock_deduper_instance.deduplicate = AsyncMock(return_value=(original_text, 0))

        chapter_number = 2
        context = "test_no_removal"

        result_text, chars_removed = await self.service.perform_deduplication(
            original_text, chapter_number, context
        )
        self.assertEqual(result_text, original_text)
        self.assertEqual(chars_removed, 0)

    @patch('orchestration.services.deduplication_service.TextDeduplicator')
    async def test_perform_deduplication_exception_in_deduplicator(self, MockTextDeduplicator):
        """Test that if TextDeduplicator raises an exception, the original text is returned."""
        mock_deduper_instance = MockTextDeduplicator.return_value
        mock_deduper_instance.deduplicate = AsyncMock(side_effect=Exception("Dedupe error"))

        original_text = "Some problematic text."
        chapter_number = 3
        context = "test_exception"

        # Suppress logger errors during this specific test
        with patch('orchestration.services.deduplication_service.logger') as mock_logger:
            result_text, chars_removed = await self.service.perform_deduplication(
                original_text, chapter_number, context
            )
            self.assertEqual(result_text, original_text)
            self.assertEqual(chars_removed, 0)
            mock_logger.error.assert_called_once()

if __name__ == '__main__':
    unittest.main()

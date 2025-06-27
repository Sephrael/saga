import pytest
from chapter_generation.drafting_service import DraftingService, DraftResult


class DummyFileManager:
    pass


class DummyOrchestrator:
    async def _draft_initial_chapter_text(self, *args):
        return DraftResult("draft", "raw")


@pytest.mark.asyncio
async def test_draft_initial_text_returns_dataclass():
    service = DraftingService(DummyOrchestrator(), DummyFileManager())
    result = await service.draft_initial_text(1, "focus", "ctx", None)
    assert isinstance(result, DraftResult)
    assert result.text == "draft"
    assert result.raw_llm_output == "raw"


@pytest.mark.asyncio
async def test_draft_result_is_unpackable():
    service = DraftingService(DummyOrchestrator(), DummyFileManager())
    result = await service.draft_initial_text(1, "focus", "ctx", None)
    text, raw = result
    assert text == "draft"
    assert raw == "raw"

# tests/test_drafting_service.py
import pytest
from chapter_generation.drafting_service import DraftResult


class DummyOrchestrator:
    async def _draft_initial_chapter_text(self, *args):
        return DraftResult("draft", "raw", {"total_tokens": 1})


@pytest.mark.asyncio
async def test_draft_initial_text_returns_dataclass():
    orch = DummyOrchestrator()
    result = await orch._draft_initial_chapter_text(1, "focus", "ctx", None)
    assert isinstance(result, DraftResult)
    assert result.text == "draft"
    assert result.raw_llm_output == "raw"


@pytest.mark.asyncio
async def test_draft_result_is_unpackable():
    result = DraftResult("draft", "raw", {"total_tokens": 1})
    text, raw, usage = result
    assert text == "draft"
    assert raw == "raw"
    assert usage == {"total_tokens": 1}

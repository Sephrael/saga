import numpy as np
import pytest

import utils
from chapter_revision_logic import _apply_patches_to_text
from llm_interface import llm_service
import config
from patch_validation_agent import PatchValidationAgent
import chapter_revision_logic


@pytest.mark.asyncio
async def test_patch_skipped_when_high_similarity(monkeypatch):
    original = "Hello world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hello",
            "reason_for_change": "same",
        }
    ]

    async def fake_embed(_text: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result, _ = await _apply_patches_to_text(original, patches, None, None)
    assert result == original


@pytest.mark.asyncio
async def test_patch_applied_when_low_similarity(monkeypatch):
    original = "Hello world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greeting",
        }
    ]

    embeddings = {
        "Hello": np.array([1.0, 0.0]),
        "Hi": np.array([0.0, 1.0]),
    }

    async def fake_embed(text: str) -> np.ndarray:
        return embeddings[text]

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result, _ = await _apply_patches_to_text(original, patches, None, None)
    assert result == "Hi world!"


@pytest.mark.asyncio
async def test_dedup_prefer_newer(monkeypatch):
    text = "First\n\nSecond\n\nFirst"

    async def fake_embed(_text: str) -> np.ndarray:
        return np.array([0.5, 0.5])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    dedup, _ = await utils.deduplicate_text_segments(
        text,
        segment_level="sentence",
        use_semantic_comparison=False,
        min_segment_length_chars=0,
        prefer_newer=True,
    )

    assert isinstance(dedup, str)


@pytest.mark.asyncio
async def test_skip_repatch_same_segment(monkeypatch):
    text = "Hello world!"
    first_patch = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greet",
        }
    ]

    second_patch = [
        {
            "original_problem_quote_text": "Hi",
            "target_char_start": 0,
            "target_char_end": 2,
            "replace_with": "Hey",
            "reason_for_change": "again",
        }
    ]

    async def fake_embed(_text: str) -> np.ndarray:
        return np.array([0.5, 0.5])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    patched1, spans1 = await _apply_patches_to_text(text, first_patch, None, None)
    patched2, _ = await _apply_patches_to_text(patched1, second_patch, spans1, None)

    assert patched2 == patched1
    dedup, _ = await utils.deduplicate_text_segments(
        "First\n\nSecond\n\nFirst",
        segment_level="sentence",
        use_semantic_comparison=False,
        min_segment_length_chars=0,
        prefer_newer=True,
    )
    assert isinstance(dedup, str)


@pytest.mark.asyncio
async def test_multiple_patches_applied(monkeypatch):
    original = "Hello world! Bye world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greeting",
        },
        {
            "original_problem_quote_text": "Bye",
            "target_char_start": 13,
            "target_char_end": 16,
            "replace_with": "See ya",
            "reason_for_change": "farewell",
        },
    ]

    embeddings = {
        "Hello": np.array([1.0, 0.0]),
        "Hi": np.array([0.0, 1.0]),
        "Bye": np.array([1.0, 0.0]),
        "See ya": np.array([0.0, 1.0]),
    }

    async def fake_embed(text: str) -> np.ndarray:
        return embeddings[text]

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result, _ = await _apply_patches_to_text(original, patches, None, None)
    assert result == "Hi world! See ya world!"


@pytest.mark.asyncio
async def test_duplicate_patch_skipped(monkeypatch):
    original = "Hello world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greeting",
        },
        {
            "original_problem_quote_text": "Hello again",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hey",
            "reason_for_change": "greeting2",
        },
    ]

    embeddings = {
        "Hello": np.array([1.0, 0.0]),
        "Hi": np.array([0.0, 1.0]),
        "Hello again": np.array([1.0, 0.0]),
        "Hey": np.array([0.0, 1.0]),
    }

    async def fake_embed(text: str) -> np.ndarray:
        return embeddings[text]

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result, _ = await _apply_patches_to_text(original, patches, None, None)
    # Only the first patch should be applied because the second overlaps exactly
    assert result == "Hi world!"


@pytest.mark.asyncio
async def test_patch_validation_toggle(monkeypatch):
    config.settings.AGENT_ENABLE_PATCH_VALIDATION = False
    config.AGENT_ENABLE_PATCH_VALIDATION = False

    called = False

    async def fake_validate(*_args, **_kwargs):
        nonlocal called
        called = True
        return True, None

    monkeypatch.setattr(PatchValidationAgent, "validate_patch", fake_validate)

    async def fake_generate(*_args, **_kwargs):
        return (
            {
                "original_problem_quote_text": "Hello",
                "target_char_start": 0,
                "target_char_end": 5,
                "replace_with": "Hi",
                "reason_for_change": "greet",
                "quote_from_original_text": "Hello",
                "sentence_char_start": 0,
                "sentence_char_end": 5,
                "quote_char_start": 0,
                "quote_char_end": 5,
                "issue_category": "cat",
                "problem_description": "desc",
                "suggested_fix_focus": "fix",
            },
            None,
        )

    monkeypatch.setattr(
        chapter_revision_logic,
        "_generate_single_patch_instruction_llm",
        fake_generate,
    )

    problems = [
        {
            "issue_category": "cat",
            "problem_description": "desc",
            "quote_from_original_text": "Hello",
            "sentence_char_start": 0,
            "sentence_char_end": 5,
            "suggested_fix_focus": "fix",
        }
    ]

    validator = PatchValidationAgent()
    result, _ = await chapter_revision_logic._generate_patch_instructions_logic(
        {},
        "Hello world",
        problems,
        1,
        "",
        None,
        validator,
    )

    assert result
    assert not called

    config.settings.AGENT_ENABLE_PATCH_VALIDATION = True
    config.AGENT_ENABLE_PATCH_VALIDATION = True

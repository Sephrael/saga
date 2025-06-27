from unittest.mock import AsyncMock

import processing.patch as patch_generator
import pytest
from agents.patch_validation_agent import PatchValidationAgent
from core.llm_interface import llm_service, truncate_text_by_tokens
from processing.revision_manager import RevisionManager


@pytest.mark.asyncio
async def test_patch_generator_successful_application(monkeypatch):
    async def fake_generate(*_args, **_kwargs):
        return [
            {
                "original_problem_quote_text": "Hello",
                "target_char_start": 0,
                "target_char_end": 5,
                "replace_with": "Hi",
                "reason_for_change": "greet",
            }
        ], None

    async def fake_apply(orig, instr, *_args, **_kwargs):
        assert instr[0]["replace_with"] == "Hi"
        return "Hi world", [(0, 2)]

    monkeypatch.setattr(
        patch_generator, "_generate_patch_instructions_logic", fake_generate
    )
    monkeypatch.setattr(patch_generator, "_apply_patches_to_text", fake_apply)

    patcher = patch_generator.PatchGenerator()
    result, spans = await patcher.generate_and_apply(
        {"plot_points": ["a"]},
        "Hello world",
        [
            {
                "issue_category": "c",
                "problem_description": "d",
                "quote_from_original_text": "Hello",
                "sentence_char_start": 0,
                "sentence_char_end": 5,
                "suggested_fix_focus": "f",
            }
        ],
        1,
        "",
        None,
        None,
        PatchValidationAgent(),
    )

    assert result == "Hi world"
    assert spans == [(0, 2)]


@pytest.mark.asyncio
async def test_patch_generator_failed_validation(monkeypatch):
    async def fake_generate(plot, original, problems, chap, ctx, plan, validator):
        patch = {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greet",
        }
        ok, _ = await validator.validate_patch("", patch, problems)
        return ([patch] if ok else []), None

    async def fake_apply(orig, instr, *_args, **_kwargs):
        assert instr == []
        return orig, []

    class FakeValidator:
        async def validate_patch(self, *_a, **_k):
            return False, None

    monkeypatch.setattr(
        patch_generator, "_generate_patch_instructions_logic", fake_generate
    )
    monkeypatch.setattr(patch_generator, "_apply_patches_to_text", fake_apply)
    monkeypatch.setattr(
        patch_generator, "_get_sentence_embeddings", AsyncMock(return_value=[])
    )

    patcher = patch_generator.PatchGenerator()
    result, spans = await patcher.generate_and_apply(
        {"plot_points": ["a"]},
        "Hello world",
        [
            {
                "issue_category": "c",
                "problem_description": "d",
                "quote_from_original_text": "Hello",
                "sentence_char_start": 0,
                "sentence_char_end": 5,
                "suggested_fix_focus": "f",
            }
        ],
        1,
        "",
        None,
        None,
        FakeValidator(),
    )

    assert result == "Hello world"
    assert spans == []


@pytest.mark.asyncio
async def test_revision_manager_full_rewrite(monkeypatch):
    async def fake_generate_and_apply(*_args, **_kwargs):
        return "Hello world", []

    async def fake_call_llm(*_args, **_kwargs):
        return "Rewrite done", None

    monkeypatch.setattr(
        patch_generator.PatchGenerator, "generate_and_apply", fake_generate_and_apply
    )
    monkeypatch.setattr(llm_service, "async_call_llm", fake_call_llm)
    monkeypatch.setattr(llm_service, "clean_model_response", lambda t: t)
    monkeypatch.setattr(
        truncate_text_by_tokens,
        "__call__",
        lambda text, *_args, **_k: text,
        raising=False,
    )

    manager = RevisionManager()
    eval_result = {
        "needs_revision": True,
        "reasons": ["bad"],
        "problems_found": [
            {
                "issue_category": "style",
                "problem_description": "d",
                "quote_from_original_text": "Hello",
                "sentence_char_start": 0,
                "sentence_char_end": 5,
                "suggested_fix_focus": "fix",
            }
        ],
    }

    res, _ = await manager.revise_chapter(
        {"plot_points": ["a"]},
        {},
        {},
        "Hello world",
        1,
        eval_result,
        "ctx",
        None,
    )

    assert res[0] == "Rewrite done"
    assert res[2] == []

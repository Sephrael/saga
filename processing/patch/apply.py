# processing/patch/apply.py
"""Apply generated patch instructions to text."""

import asyncio
import hashlib
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Any

import structlog
import utils
from config import settings
from core.llm_interface import llm_service

from models import PatchInstruction


class LRUDict(OrderedDict[str, list[tuple[int, int, Any]]]):
    """Simple LRU cache based on ``OrderedDict``."""

    def __init__(self, maxsize: int) -> None:
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key: str) -> list[tuple[int, int, Any]]:  # type: ignore[override]
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key: str, value: list[tuple[int, int, Any]]) -> None:  # type: ignore[override]
        if key in self:
            super().__delitem__(key)
        elif len(self) >= self.maxsize:
            super().popitem(last=False)
        super().__setitem__(key, value)

    def cache_clear(self) -> None:
        super().clear()


logger = structlog.get_logger(__name__)

_sentence_embedding_cache: LRUDict = LRUDict(settings.SENTENCE_EMBEDDING_CACHE_SIZE)


async def _get_sentence_embeddings(
    text: str,
    cache: MutableMapping[str, list[tuple[int, int, Any]]] | None = None,
) -> list[tuple[int, int, Any]]:
    """Return (start, end, embedding) for each sentence."""
    if cache is None:
        cache = _sentence_embedding_cache
    utils.load_spacy_model_if_needed()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_hash in cache:
        return cache[text_hash]

    segments = utils.get_text_segments(text, "sentence")
    if not segments:
        return []
    tasks = [llm_service.async_get_embedding(seg[0]) for seg in segments]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    embeddings: list[tuple[int, int, Any]] = []
    for (_seg_text, start, end), res in zip(segments, results, strict=False):
        if isinstance(res, Exception) or res is None:
            continue
        embeddings.append((start, end, res))
    cache[text_hash] = embeddings
    if (
        not isinstance(cache, LRUDict)
        and len(cache) > settings.SENTENCE_EMBEDDING_CACHE_SIZE
    ):
        for _ in range(len(cache) - settings.SENTENCE_EMBEDDING_CACHE_SIZE):
            oldest_key = next(iter(cache))
            del cache[oldest_key]
    return embeddings


async def _find_sentence_via_embeddings(
    quote_text: str, embeddings: list[tuple[int, int, Any]]
) -> tuple[int, int] | None:
    if not embeddings or not quote_text.strip():
        return None
    q_emb = await llm_service.async_get_embedding(quote_text)
    if q_emb is None:
        return None
    best_sim = -1.0
    best_span: tuple[int, int] | None = None
    for start, end, emb in embeddings:
        sim = utils.numpy_cosine_similarity(q_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_span = (start, end)
    return best_span


async def locate_patch_targets(
    original_text: str,
    patch: PatchInstruction,
    embeddings: list[tuple[int, int, Any]] | None = None,
) -> tuple[int, int] | None:
    """Resolve and return the target offsets for a patch instruction."""
    start = patch.get("sentence_char_start")
    end = patch.get("sentence_char_end")
    if start is not None and end is not None:
        return start, end

    start = patch.get("quote_char_start")
    end = patch.get("quote_char_end")
    if start is not None and end is not None:
        return start, end

    start = patch.get("target_char_start")
    end = patch.get("target_char_end")
    if start is not None and end is not None:
        return start, end

    quote_text = patch.get("original_problem_quote_text", "")
    if not quote_text or quote_text == "N/A - General Issue":
        logger.warning("Patch %s lacks offsets and quote text.", patch)
        return None

    logger.info("Locating patch target for '%s...'", quote_text[:50])
    if embeddings:
        found = await _find_sentence_via_embeddings(quote_text, embeddings)
        if found:
            return found

    match = await utils.find_semantically_closest_segment(
        original_text, quote_text, "sentence"
    )
    if match:
        start, end, _ = match
        return start, end

    logger.warning("Failed to locate patch target for '%s...'.", quote_text[:50])
    return None


async def _apply_patches_to_text(
    original_text: str,
    patch_instructions: list[PatchInstruction],
    already_patched_spans: list[tuple[int, int]] | None = None,
    sentence_embeddings: list[tuple[int, int, Any]] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Apply patch instructions to text and return new text with spans."""
    if already_patched_spans is None:
        already_patched_spans = []

    if not patch_instructions:
        return original_text, already_patched_spans

    replacements: list[tuple[int, int, str]] = []
    for patch_idx, patch in enumerate(patch_instructions):
        replacement_text = patch.get("replace_with", "") or ""

        target_span = await locate_patch_targets(
            original_text, patch, sentence_embeddings
        )
        if not target_span:
            logger.warning("Patch %s: Failed to locate target span.", patch_idx + 1)
            continue

        segment_start, segment_end = target_span

        candidate_spans = [(segment_start, segment_end)]
        for s_key, e_key in [
            ("sentence_char_start", "sentence_char_end"),
            ("quote_char_start", "quote_char_end"),
            ("target_char_start", "target_char_end"),
        ]:
            span_start = patch.get(s_key)
            span_end = patch.get(e_key)
            if span_start is not None and span_end is not None:
                candidate_spans.append((span_start, span_end))

        is_overlapping = any(
            max(c_start, old_start) < min(c_end, old_end)
            for c_start, c_end in candidate_spans
            for old_start, old_end in already_patched_spans
        ) or any(
            max(c_start, r_start) < min(c_end, r_end)
            for c_start, c_end in candidate_spans
            for r_start, r_end, _ in replacements
        )

        if is_overlapping:
            logger.warning(
                "Patch %s for segment %s-%s overlaps with a previously patched area or another new patch. Skipping.",
                patch_idx + 1,
                segment_start,
                segment_end,
            )
            continue

        original_segment = original_text[segment_start:segment_end]
        if replacement_text.strip() == original_segment.strip():
            logger.info(
                "Patch %s: replacement identical to original segment %s-%s. Skipping.",
                patch_idx + 1,
                segment_start,
                segment_end,
            )
            continue
        if replacement_text.strip() and original_segment.strip():
            orig_emb, repl_emb = await asyncio.gather(
                llm_service.async_get_embedding(original_segment),
                llm_service.async_get_embedding(replacement_text),
            )
            if (
                orig_emb is not None
                and repl_emb is not None
                and utils.numpy_cosine_similarity(orig_emb, repl_emb)
                >= settings.REVISION_SIMILARITY_ACCEPTANCE
            ):
                logger.info(
                    "Patch %s: replacement highly similar to original segment %s-%s. Skipping.",
                    patch_idx + 1,
                    segment_start,
                    segment_end,
                )
                continue

        replacements.append((segment_start, segment_end, replacement_text))
        log_action = "DELETION" if not replacement_text.strip() else "REPLACEMENT"
        logger.info(
            "Patch %s: Queued %s for %s-%s.",
            patch_idx + 1,
            log_action,
            segment_start,
            segment_end,
        )

    if not replacements:
        logger.info("No non-overlapping patches to apply in this cycle.")
        return original_text, already_patched_spans

    all_ops: list[dict[str, Any]] = []
    for start, end in already_patched_spans:
        all_ops.append(
            {
                "type": "old",
                "start": start,
                "end": end,
                "text": original_text[start:end],
            }
        )
    for start, end, text in replacements:
        all_ops.append({"type": "new", "start": start, "end": end, "text": text})

    all_ops.sort(key=lambda x: x["start"])

    result_parts = []
    all_spans_in_new_text = []
    last_original_end = 0

    for op in all_ops:
        result_parts.append(original_text[last_original_end : op["start"]])
        new_span_start = len("".join(result_parts))
        result_parts.append(op["text"])
        new_span_end = len("".join(result_parts))
        if new_span_end > new_span_start:
            all_spans_in_new_text.append((new_span_start, new_span_end))
        last_original_end = op["end"]

    result_parts.append(original_text[last_original_end:])

    patched_text = "".join(result_parts)
    final_spans = sorted(all_spans_in_new_text)

    num_deletions = sum(1 for _, _, txt in replacements if not txt.strip())
    num_replacements = len(replacements) - num_deletions
    logger.info(
        "Applied %s replacements and %s deletions. Total protected spans in new text: %s.",
        num_replacements,
        num_deletions,
        len(final_spans),
    )

    return patched_text, final_spans

"""Apply generated patch instructions to text."""

import asyncio
import hashlib
from typing import Any

import structlog
import utils
from config import settings
from core.llm_interface import llm_service

from models import PatchInstruction

logger = structlog.get_logger(__name__)

_sentence_embedding_cache: dict[str, list[tuple[int, int, Any]]] = {}


async def _get_sentence_embeddings(
    text: str, cache: dict[str, list[tuple[int, int, Any]]] | None = None
) -> list[tuple[int, int, Any]]:
    """Return (start, end, embedding) for each sentence."""
    if cache is None:
        cache = _sentence_embedding_cache
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
        replacement_text = patch.get("replace_with", "")
        if replacement_text is None:
            replacement_text = ""

        segment_start: int | None = patch.get("target_char_start")
        segment_end: int | None = patch.get("target_char_end")
        method_used = "direct offsets"

        if segment_start is None or segment_end is None:
            quote_text = patch["original_problem_quote_text"]
            if quote_text != "N/A - General Issue" and quote_text.strip():
                logger.info(
                    "Patch %s: Missing direct offsets for '%s...'. Using semantic search.",
                    patch_idx + 1,
                    quote_text[:50],
                )
                method_used = "semantic search"
                if sentence_embeddings:
                    found = await _find_sentence_via_embeddings(
                        quote_text, sentence_embeddings
                    )
                    if found:
                        segment_start, segment_end = found
                if segment_start is None or segment_end is None:
                    match = await utils.find_semantically_closest_segment(
                        original_text, quote_text, "sentence"
                    )
                    if match:
                        segment_start, segment_end, _ = match
            else:
                logger.warning(
                    "Patch %s: Cannot apply, no quote text for search and no offsets.",
                    patch_idx + 1,
                )
                continue

        if segment_start is None or segment_end is None:
            logger.warning(
                "Patch %s: Failed to find target segment via %s.",
                patch_idx + 1,
                method_used,
            )
            continue

        is_overlapping = any(
            max(segment_start, old_start) < min(segment_end, old_end)
            for old_start, old_end in already_patched_spans
        ) or any(
            max(segment_start, r_start) < min(segment_end, r_end)
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
            "Patch %s: Queued %s for %s-%s via %s.",
            patch_idx + 1,
            log_action,
            segment_start,
            segment_end,
            method_used,
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

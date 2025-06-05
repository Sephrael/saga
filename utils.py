# utils.py
"""
General utility functions for the Saga Novel Generation system.
MODIFIED: Enhanced find_quote_and_sentence_offsets_with_spacy for more robust quote matching.
ADDED: deduplicate_text_segments for removing near-identical text.
ADDED: _is_fill_in helper function.
"""

import numpy as np
import logging
import re
import asyncio
from typing import Optional, Tuple, List, Union, Set, Any  # Added Set, Any
from type import SceneDetail

# Local application imports - ensure these paths are correct for your project
from llm_interface import llm_service, count_tokens
import spacy
import config  # For MARKDOWN_FILL_IN_PLACEHOLDER

logger = logging.getLogger(__name__)


class SpaCyModelManager:
    """Lazily loads and stores the spaCy model used across the project."""

    def __init__(self) -> None:
        self._nlp: Optional[spacy.language.Language] = None

    @property
    def nlp(self) -> Optional[spacy.language.Language]:
        return self._nlp

    def load(self) -> None:
        """Load the spaCy model if it hasn't been loaded yet."""
        if self._nlp is not None:
            return
        try:
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Please run: python -m spacy download en_core_web_sm. "
                "spaCy dependent features will be disabled."
            )
            self._nlp = None
        except ImportError:
            logger.error(
                "spaCy library not installed. Please install it: pip install spacy. "
                "spaCy dependent features will be disabled."
            )
            self._nlp = None


# Singleton-like instance accessible to other modules
spacy_manager = SpaCyModelManager()


def _is_fill_in(value: Any) -> bool:
    """Checks if a value is the [Fill-in] placeholder."""
    return isinstance(value, str) and value == config.MARKDOWN_FILL_IN_PLACEHOLDER


def load_spacy_model_if_needed() -> None:
    """Load the spaCy model using the shared manager if needed."""
    spacy_manager.load()


def _normalize_text_for_matching(text: str) -> str:
    """Normalizes text for more robust matching (lowercase, remove punctuation, normalize whitespace)."""
    if not text:
        return ""
    text = text.lower()
    # Remove leading/trailing quotes and ellipses that might be added by the LLM for the quote itself
    text = re.sub(
        r"^[ '\"\(]*(\.\.\.)?[ '\"\(]*|[ '\"\(]*(\.\.\.)?[ '\"\(]*$", "", text
    )
    # General punctuation removal for broader matching
    text = re.sub(r"[^\w\s]", "", text)  # Remove non-alphanumeric, non-whitespace
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


async def find_quote_and_sentence_offsets_with_spacy(
    doc_text: str, quote_text_from_llm: str
) -> Optional[
    Tuple[int, int, int, int]
]:  # (quote_start, quote_end, sentence_start, sentence_end)
    """
    Finds character offsets for a quote (potentially non-verbatim) within a document
    and its containing sentence using spaCy and fallbacks.
    Returns None if spaCy isn't loaded or the quote isn't reasonably found.
    """
    load_spacy_model_if_needed()
    if (
        spacy_manager.nlp is None
        or not quote_text_from_llm.strip()
        or not doc_text.strip()
    ):
        if spacy_manager.nlp is None:
            logger.debug("find_quote_offsets: spaCy model not loaded.")
        else:
            logger.debug("find_quote_offsets: Empty quote_text or doc_text.")
        return None

    if "N/A - General Issue" in quote_text_from_llm:
        logger.debug(
            f"Quote is '{quote_text_from_llm}', treating as general issue. No offsets."
        )
        return None

    cleaned_llm_quote_for_direct_search = quote_text_from_llm.strip(
        " \"'."
    )  # Light clean for direct search
    if not cleaned_llm_quote_for_direct_search:
        logger.debug(
            "LLM quote became empty after basic stripping for direct search, cannot match."
        )
        return None

    spacy_doc = spacy_manager.nlp(doc_text) if spacy_manager.nlp else None
    if spacy_doc is None:
        return None
    # best_direct_match_offsets: Optional[Tuple[int, int, int, int]] = None # Not used directly

    # Attempt 1: Direct Substring Match (case-insensitive)
    current_pos = 0
    while current_pos < len(doc_text):
        match_start = doc_text.lower().find(
            cleaned_llm_quote_for_direct_search.lower(), current_pos
        )
        if match_start == -1:
            break

        match_end = match_start + len(cleaned_llm_quote_for_direct_search)
        found_sentence_span = None
        for sent in spacy_doc.sents:
            if (
                sent.start_char <= match_start < sent.end_char
                and sent.start_char < match_end <= sent.end_char
            ):
                found_sentence_span = sent
                break

        if found_sentence_span:
            logger.info(
                f"Direct Substring Match: Found LLM quote (approx) '{cleaned_llm_quote_for_direct_search[:30]}...' at {match_start}-{match_end} in sentence {found_sentence_span.start_char}-{found_sentence_span.end_char}"
            )
            # Return the first good direct match found
            return (
                match_start,
                match_end,
                found_sentence_span.start_char,
                found_sentence_span.end_char,
            )

        current_pos = match_end  # Continue search after this non-ideal match

    # Attempt 2: Semantic Sentence Search (if direct match failed)
    logger.warning(
        f"Direct substring match failed for LLM quote '{quote_text_from_llm[:50]}...'. Falling back to semantic sentence search."
    )
    semantic_sentence_match = await find_semantically_closest_segment(
        original_doc=doc_text,
        query_text=quote_text_from_llm,  # Use original LLM quote for richer semantics
        segment_type="sentence",
        min_similarity_threshold=0.75,  # Higher threshold for sentence precision
    )

    if semantic_sentence_match:
        s_start, s_end, similarity = semantic_sentence_match
        logger.info(
            f"Semantic Match: Found sentence for LLM quote '{quote_text_from_llm[:30]}...' from {s_start}-{s_end} (Similarity: {similarity:.2f}). Using whole sentence as target."
        )
        # For semantic matches at sentence level, we consider the whole sentence as the "quote" area for revision.
        return s_start, s_end, s_start, s_end

    logger.warning(
        f"Could not confidently locate quote TEXT from LLM: '{quote_text_from_llm[:50]}...' in document using direct or semantic search."
    )
    return None


def numpy_cosine_similarity(
    vec1: Optional[np.ndarray], vec2: Optional[np.ndarray]
) -> float:
    """
    Calculates cosine similarity between two numpy vectors.
    Handles None inputs, shape mismatches, and zero vectors gracefully.
    """
    if vec1 is None or vec2 is None:
        logger.debug("Cosine similarity: one or both vectors are None. Returning 0.0.")
        return 0.0
    try:
        v1 = np.asarray(vec1, dtype=np.float32).flatten()
        v2 = np.asarray(vec2, dtype=np.float32).flatten()
    except ValueError as e:
        logger.warning(
            f"Cosine similarity: Could not convert input to numpy array: {e}. Returning 0.0."
        )
        return 0.0
    if v1.shape != v2.shape:
        logger.warning(
            f"Cosine similarity: shape mismatch {v1.shape} vs {v2.shape}. Returning 0.0."
        )
        return 0.0
    if v1.size == 0:
        logger.debug("Cosine similarity: input vector(s) are empty. Returning 0.0.")
        return 0.0
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        logger.debug("Cosine similarity: at least one zero-norm vector. Returning 0.0.")
        return 0.0
    dot_product = np.dot(v1, v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return float(np.clip(similarity, -1.0, 1.0))


async def find_semantically_closest_segment(
    original_doc: str,
    query_text: str,
    segment_type: str = "paragraph",
    min_similarity_threshold: float = 0.65,
) -> Optional[Tuple[int, int, float]]:
    """
    Finds the segment (paragraph or sentence) in original_doc most semantically
    similar to query_text.
    """
    if not original_doc or not query_text:
        logger.debug(
            "find_semantically_closest_segment: original_doc or query_text is empty."
        )
        return None

    query_embedding = await llm_service.async_get_embedding(query_text)
    if query_embedding is None:
        logger.warning(
            f"Could not get embedding for semantic search query: '{query_text[:60].replace(chr(10), ' ')}...'"
        )
        return None

    segments_with_indices: List[Tuple[str, int, int]] = get_text_segments(
        original_doc, segment_type
    )  # Use helper

    if not segments_with_indices:
        logger.warning(
            f"No segments found in original document for semantic search (Type: {segment_type})."
        )
        return None

    best_match_info: Optional[Tuple[int, int, float]] = None
    highest_similarity = -2.0

    segment_texts = [s[0] for s in segments_with_indices]  # s[0] is stripped text

    segment_embeddings_tasks = [
        llm_service.async_get_embedding(seg_text) for seg_text in segment_texts
    ]
    segment_embeddings_results = await asyncio.gather(
        *segment_embeddings_tasks, return_exceptions=True
    )

    for i, seg_embedding_or_exc in enumerate(segment_embeddings_results):
        if isinstance(seg_embedding_or_exc, Exception) or seg_embedding_or_exc is None:
            logger.warning(
                f"Could not get embedding for segment: '{segment_texts[i][:60].replace(chr(10), ' ')}...' Error: {seg_embedding_or_exc}"
            )
            continue

        seg_embedding = seg_embedding_or_exc
        similarity = numpy_cosine_similarity(query_embedding, seg_embedding)

        if similarity > highest_similarity:
            highest_similarity = similarity
            _, start_char, end_char = segments_with_indices[
                i
            ]  # Use offsets from helper
            best_match_info = (start_char, end_char, highest_similarity)

    if best_match_info and best_match_info[2] < min_similarity_threshold:
        logger.info(
            f"Semantic match found for query '{query_text[:60].replace(chr(10), ' ')}...', "
            f"but similarity ({best_match_info[2]:.2f}) is below threshold ({min_similarity_threshold})."
        )
        return None

    if best_match_info:
        logger.info(
            f"Best semantic match for query '{query_text[:60].replace(chr(10), ' ')}...' "
            f"has similarity {best_match_info[2]:.2f} at span {best_match_info[0]}-{best_match_info[1]}."
        )
    else:
        logger.info(
            f"No suitable semantic match found (above threshold {min_similarity_threshold}) "
            f"for query '{query_text[:60].replace(chr(10), ' ')}...'. Highest sim was {highest_similarity:.2f}."
        )

    return best_match_info


def format_scene_plan_for_prompt(
    chapter_plan: List[SceneDetail],
    model_name_for_tokens: str,
    max_tokens_budget: int,
) -> str:
    """Formats a chapter plan into plain text for LLM prompts respecting token limits."""
    if not chapter_plan:
        return "No detailed scene plan available."

    plan_lines = ["**Detailed Scene Plan (MUST BE FOLLOWED CLOSELY):**"]
    current_plan_parts = [plan_lines[0]]

    for scene_idx, scene in enumerate(chapter_plan):
        scene_lines = [
            f"Scene Number: {scene.get('scene_number', 'N/A')}",
            f"  Summary: {scene.get('summary', 'N/A')}",
            f"  Characters Involved: {', '.join(scene.get('characters_involved', [])) if scene.get('characters_involved') else 'None'}",
            "  Key Dialogue Points:",
        ]
        for point in scene.get("key_dialogue_points", []):
            scene_lines.append(f"    - {point}")
        scene_lines.append(f"  Setting Details: {scene.get('setting_details', 'N/A')}")
        scene_lines.append("  Scene Focus Elements:")
        for focus_el in scene.get("scene_focus_elements", []):
            scene_lines.append(f"    - {focus_el}")
        scene_lines.append(f"  Contribution: {scene.get('contribution', 'N/A')}")

        if scene_idx < len(chapter_plan) - 1:
            scene_lines.append("-" * 20)

        scene_segment = "\n".join(scene_lines)
        prospective_plan = "\n".join(current_plan_parts + [scene_segment])

        if count_tokens(prospective_plan, model_name_for_tokens) > max_tokens_budget:
            current_plan_parts.append(
                "... (plan truncated in prompt due to token limit)"
            )
            logger.warning(
                f"Chapter plan was token-truncated for the prompt. Max tokens for plan: {max_tokens_budget}. "
                f"Stopped before scene {scene.get('scene_number', 'N/A')}."
            )
            break

        current_plan_parts.append(scene_segment)

    if len(current_plan_parts) <= 1:
        return "No detailed scene plan available or plan was too long to include any scenes."

    return "\n".join(current_plan_parts)


# --- De-duplication Logic ---


def get_text_segments(
    text: str, segment_level: str = "paragraph"
) -> List[Tuple[str, int, int]]:
    """
    Segments text into paragraphs or sentences with their original character offsets.
    The returned segment text is stripped of leading/trailing whitespace.
    """
    load_spacy_model_if_needed()
    segments: List[
        Tuple[str, int, int]
    ] = []  # (stripped_text, start_char_original, end_char_original)

    if not text.strip():
        return segments

    if segment_level == "paragraph":
        # Iterate over non-empty lines, grouping them into paragraphs.
        # A paragraph ends when one or more blank lines are encountered.
        current_paragraph_lines = []
        current_paragraph_start_char = -1

        for line_match in re.finditer(
            r"([^\r\n]*(?:\r\n|\r|\n)?)", text
        ):  # Iterate line by line with original newlines
            line_text = line_match.group(0)
            line_text_stripped = line_text.strip()

            if line_text_stripped:  # Non-empty line
                if not current_paragraph_lines:  # Start of a new paragraph
                    current_paragraph_start_char = line_match.start()
                current_paragraph_lines.append(line_text)
            else:  # Empty line (or line with only whitespace)
                if current_paragraph_lines:  # End of current paragraph
                    full_para_text = "".join(current_paragraph_lines)
                    segments.append(
                        (
                            full_para_text.strip(),
                            current_paragraph_start_char,
                            current_paragraph_start_char + len(full_para_text),
                        )
                    )
                    current_paragraph_lines = []
                    current_paragraph_start_char = -1

        if current_paragraph_lines:  # Append any trailing paragraph
            full_para_text = "".join(current_paragraph_lines)
            segments.append(
                (
                    full_para_text.strip(),
                    current_paragraph_start_char,
                    current_paragraph_start_char + len(full_para_text),
                )
            )

        # Fallback for text without explicit paragraph breaks (e.g. single block)
        if not segments and text.strip():
            segments.append((text.strip(), 0, len(text)))

    elif segment_level == "sentence":
        if spacy_manager.nlp:
            doc = spacy_manager.nlp(text)
            for sent in doc.sents:
                sent_text_stripped = sent.text.strip()
                if sent_text_stripped:
                    segments.append(
                        (sent_text_stripped, sent.start_char, sent.end_char)
                    )
        else:
            logger.warning(
                "get_text_segments: spaCy model not loaded. Falling back to basic sentence segmentation (less accurate)."
            )
            # Basic regex fallback for sentences (less accurate)
            for match in re.finditer(
                r"([^\.!?]+(?:[\.!?]|$))", text
            ):  # Simple sentence split
                sent_text_stripped = match.group(1).strip()
                if sent_text_stripped:
                    segments.append((sent_text_stripped, match.start(), match.end()))
            if (
                not segments and text.strip()
            ):  # If regex found nothing, treat as one sentence
                segments.append((text.strip(), 0, len(text)))
    else:
        raise ValueError(
            f"Unsupported segment_level for get_text_segments: {segment_level}"
        )

    return segments


async def deduplicate_text_segments(
    original_text: str,
    segment_level: str = "paragraph",
    similarity_threshold: float = config.DEDUPLICATION_SEMANTIC_THRESHOLD,  # Use config
    use_semantic_comparison: bool = config.DEDUPLICATION_USE_SEMANTIC,  # Use config
    min_segment_length_chars: int = config.DEDUPLICATION_MIN_SEGMENT_LENGTH,  # Use config
) -> Tuple[str, int]:
    """
    Removes near-duplicate segments from text.
    Supports normalized string comparison or semantic comparison (async).
    """
    if not original_text.strip():
        return original_text, 0

    segments_with_offsets = get_text_segments(original_text, segment_level)
    if not segments_with_offsets:
        return original_text, 0

    kept_segment_info_for_semantic: List[Tuple[int, int, np.ndarray]] = []
    seen_normalized_texts_for_string: Set[str] = set()
    segments_to_build_final_text: List[Tuple[int, int]] = []

    embeddings: List[Optional[np.ndarray]]
    if use_semantic_comparison:
        tasks = [
            llm_service.async_get_embedding(seg_text)
            for seg_text, _, _ in segments_with_offsets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        embeddings = [
            res if not isinstance(res, Exception) else None for res in results
        ]
    else:
        embeddings = [None] * len(segments_with_offsets)

    for i, (seg_text, seg_start, seg_end) in enumerate(segments_with_offsets):
        seg_embedding = embeddings[i]
        if len(seg_text) < min_segment_length_chars:
            segments_to_build_final_text.append((seg_start, seg_end))
            if use_semantic_comparison and seg_embedding is not None:
                kept_segment_info_for_semantic.append(
                    (seg_start, seg_end, seg_embedding)
                )
            continue

        is_duplicate = False
        if use_semantic_comparison:
            current_seg_embedding = seg_embedding
            if current_seg_embedding is None:
                logger.warning(
                    f"De-duplication: Could not get embedding for segment (idx {i}, chars {seg_start}-{seg_end}). Keeping it."
                )
                segments_to_build_final_text.append((seg_start, seg_end))
                continue  # Skip to next segment

            for _, _, kept_embedding in kept_segment_info_for_semantic:
                similarity = numpy_cosine_similarity(
                    current_seg_embedding, kept_embedding
                )
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_segment_info_for_semantic.append(
                    (seg_start, seg_end, current_seg_embedding)
                )
                segments_to_build_final_text.append((seg_start, seg_end))
        else:  # Normalized string comparison
            normalized_current_seg = _normalize_text_for_matching(seg_text)
            if normalized_current_seg in seen_normalized_texts_for_string:
                is_duplicate = True
            else:
                seen_normalized_texts_for_string.add(normalized_current_seg)
                segments_to_build_final_text.append((seg_start, seg_end))

        if is_duplicate:
            method_used = "semantic" if use_semantic_comparison else "normalized string"
            logger.info(
                f"De-duplication: Removing segment (idx {i}, chars {seg_start}-{seg_end}, method: {method_used}) starting with: '{seg_text[:60].replace(chr(10), ' ')}...'"
            )

    if len(segments_to_build_final_text) == len(segments_with_offsets):
        return original_text, 0

    segments_to_build_final_text.sort(key=lambda x: x[0])

    reconstructed_parts = []
    last_kept_end = 0
    for seg_start_orig, seg_end_orig in segments_to_build_final_text:
        if seg_start_orig > last_kept_end:
            reconstructed_parts.append(original_text[last_kept_end:seg_start_orig])
        reconstructed_parts.append(original_text[seg_start_orig:seg_end_orig])
        last_kept_end = seg_end_orig

    if last_kept_end < len(original_text):
        reconstructed_parts.append(original_text[last_kept_end:])

    deduplicated_text = "".join(reconstructed_parts)

    deduplicated_text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", deduplicated_text)
    deduplicated_text = re.sub(r"\n{3,}", "\n\n", deduplicated_text).strip()

    characters_removed_count = len(original_text) - len(deduplicated_text)
    return deduplicated_text, characters_removed_count

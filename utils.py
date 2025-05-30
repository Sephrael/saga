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
from typing import Optional, Tuple, List, Union, Set, Any # Added Set, Any

# Local application imports - ensure these paths are correct for your project
import llm_interface
import spacy
import config # For MARKDOWN_FILL_IN_PLACEHOLDER

logger = logging.getLogger(__name__)

# Global spaCy model instance
NLP_SPACY: Optional[spacy.language.Language] = None

def _is_fill_in(value: Any) -> bool:
    """Checks if a value is the [Fill-in] placeholder."""
    return isinstance(value, str) and value == config.MARKDOWN_FILL_IN_PLACEHOLDER

def load_spacy_model_if_needed():
    """Loads the spaCy model if it hasn't been loaded yet."""
    global NLP_SPACY
    if NLP_SPACY is None:
        try:
            NLP_SPACY = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found. "
                "Please run: python -m spacy download en_core_web_sm. "
                "spaCy dependent features will be disabled."
            )
            NLP_SPACY = None
        except ImportError:
            logger.error(
                "spaCy library not installed. Please install it: pip install spacy. "
                "spaCy dependent features will be disabled."
            )
            NLP_SPACY = None

def _normalize_text_for_matching(text: str) -> str:
    """Normalizes text for more robust matching (lowercase, remove punctuation, normalize whitespace)."""
    if not text:
        return ""
    text = text.lower()
    # Remove leading/trailing quotes and ellipses that might be added by the LLM for the quote itself
    text = re.sub(r"^[ '\"\(]*(\.\.\.)?[ '\"\(]*|[ '\"\(]*(\.\.\.)?[ '\"\(]*$", "", text)
    # General punctuation removal for broader matching
    text = re.sub(r'[^\w\s]', '', text) # Remove non-alphanumeric, non-whitespace
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

async def find_quote_and_sentence_offsets_with_spacy(
    doc_text: str,
    quote_text_from_llm: str
) -> Optional[Tuple[int, int, int, int]]: # (quote_start, quote_end, sentence_start, sentence_end)
    """
    Finds character offsets for a quote (potentially non-verbatim) within a document
    and its containing sentence using spaCy and fallbacks.
    Returns None if spaCy isn't loaded or the quote isn't reasonably found.
    """
    load_spacy_model_if_needed()
    if NLP_SPACY is None or not quote_text_from_llm.strip() or not doc_text.strip():
        if NLP_SPACY is None: logger.debug("find_quote_offsets: spaCy model not loaded.")
        else: logger.debug("find_quote_offsets: Empty quote_text or doc_text.")
        return None

    if "N/A - General Issue" in quote_text_from_llm:
        logger.debug(f"Quote is '{quote_text_from_llm}', treating as general issue. No offsets.")
        return None

    cleaned_llm_quote_for_direct_search = quote_text_from_llm.strip(' "\'.') # Light clean for direct search
    if not cleaned_llm_quote_for_direct_search:
        logger.debug("LLM quote became empty after basic stripping for direct search, cannot match.")
        return None

    spacy_doc = NLP_SPACY(doc_text)
    best_direct_match_offsets: Optional[Tuple[int, int, int, int]] = None

    # Attempt 1: Direct Substring Match (case-insensitive)
    current_pos = 0
    while current_pos < len(doc_text):
        match_start = doc_text.lower().find(cleaned_llm_quote_for_direct_search.lower(), current_pos)
        if match_start == -1:
            break
        
        match_end = match_start + len(cleaned_llm_quote_for_direct_search)
        found_sentence_span = None
        for sent in spacy_doc.sents:
            if sent.start_char <= match_start < sent.end_char and sent.start_char < match_end <= sent.end_char:
                found_sentence_span = sent
                break
        
        if found_sentence_span:
            logger.info(f"Direct Substring Match: Found LLM quote (approx) '{cleaned_llm_quote_for_direct_search[:30]}...' at {match_start}-{match_end} in sentence {found_sentence_span.start_char}-{found_sentence_span.end_char}")
            # Return the first good direct match found
            return (match_start, match_end, found_sentence_span.start_char, found_sentence_span.end_char)
        
        current_pos = match_end # Continue search after this non-ideal match

    # Attempt 2: Semantic Sentence Search (if direct match failed)
    logger.warning(f"Direct substring match failed for LLM quote '{quote_text_from_llm[:50]}...'. Falling back to semantic sentence search.")
    semantic_sentence_match = await find_semantically_closest_segment(
        original_doc=doc_text,
        query_text=quote_text_from_llm, # Use original LLM quote for richer semantics
        segment_type="sentence",
        min_similarity_threshold=0.75 # Higher threshold for sentence precision
    )

    if semantic_sentence_match:
        s_start, s_end, similarity = semantic_sentence_match
        logger.info(f"Semantic Match: Found sentence for LLM quote '{quote_text_from_llm[:30]}...' from {s_start}-{s_end} (Similarity: {similarity:.2f}). Using whole sentence as target.")
        # For semantic matches at sentence level, we consider the whole sentence as the "quote" area for revision.
        return s_start, s_end, s_start, s_end

    logger.warning(f"Could not confidently locate quote TEXT from LLM: '{quote_text_from_llm[:50]}...' in document using direct or semantic search.")
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
        logger.warning(f"Cosine similarity: Could not convert input to numpy array: {e}. Returning 0.0.")
        return 0.0
    if v1.shape != v2.shape:
        logger.warning(f"Cosine similarity: shape mismatch {v1.shape} vs {v2.shape}. Returning 0.0.")
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
    min_similarity_threshold: float = 0.65
) -> Optional[Tuple[int, int, float]]:
    """
    Finds the segment (paragraph or sentence) in original_doc most semantically
    similar to query_text.
    """
    if not original_doc or not query_text:
        logger.debug("find_semantically_closest_segment: original_doc or query_text is empty.")
        return None

    query_embedding = await llm_interface.async_get_embedding(query_text)
    if query_embedding is None:
        logger.warning(f"Could not get embedding for semantic search query: '{query_text[:60].replace(chr(10),' ')}...'")
        return None

    segments_with_indices: List[Tuple[str, int, int]] = get_text_segments(original_doc, segment_type) # Use helper
    
    if not segments_with_indices:
        logger.warning(f"No segments found in original document for semantic search (Type: {segment_type}).")
        return None

    best_match_info: Optional[Tuple[int, int, float]] = None
    highest_similarity = -2.0 

    segment_texts = [s[0] for s in segments_with_indices] # s[0] is stripped text
    
    segment_embeddings_tasks = [llm_interface.async_get_embedding(seg_text) for seg_text in segment_texts]
    segment_embeddings_results = await asyncio.gather(*segment_embeddings_tasks, return_exceptions=True)

    for i, seg_embedding_or_exc in enumerate(segment_embeddings_results):
        if isinstance(seg_embedding_or_exc, Exception) or seg_embedding_or_exc is None:
            logger.warning(f"Could not get embedding for segment: '{segment_texts[i][:60].replace(chr(10),' ')}...' Error: {seg_embedding_or_exc}")
            continue
        
        seg_embedding = seg_embedding_or_exc
        similarity = numpy_cosine_similarity(query_embedding, seg_embedding)
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            _, start_char, end_char = segments_with_indices[i] # Use offsets from helper
            best_match_info = (start_char, end_char, highest_similarity)
    
    if best_match_info and best_match_info[2] < min_similarity_threshold:
        logger.info(f"Semantic match found for query '{query_text[:60].replace(chr(10),' ')}...', "
                    f"but similarity ({best_match_info[2]:.2f}) is below threshold ({min_similarity_threshold}).")
        return None 
            
    if best_match_info:
        logger.info(f"Best semantic match for query '{query_text[:60].replace(chr(10),' ')}...' "
                    f"has similarity {best_match_info[2]:.2f} at span {best_match_info[0]}-{best_match_info[1]}.")
    else:
        logger.info(f"No suitable semantic match found (above threshold {min_similarity_threshold}) "
                    f"for query '{query_text[:60].replace(chr(10),' ')}...'. Highest sim was {highest_similarity:.2f}.")
        
    return best_match_info

# --- De-duplication Logic ---

def get_text_segments(text: str, segment_level: str = "paragraph") -> List[Tuple[str, int, int]]:
    """
    Segments text into paragraphs or sentences with their original character offsets.
    The returned segment text is stripped of leading/trailing whitespace.
    """
    load_spacy_model_if_needed()
    segments: List[Tuple[str, int, int]] = [] # (stripped_text, start_char_original, end_char_original)
    
    if not text.strip():
        return segments

    if segment_level == "paragraph":
        # Iterate over non-empty lines, grouping them into paragraphs.
        # A paragraph ends when one or more blank lines are encountered.
        current_paragraph_lines = []
        current_paragraph_start_char = -1
        
        for line_match in re.finditer(r"([^\r\n]*(?:\r\n|\r|\n)?)", text): # Iterate line by line with original newlines
            line_text = line_match.group(0)
            line_text_stripped = line_text.strip()

            if line_text_stripped: # Non-empty line
                if not current_paragraph_lines: # Start of a new paragraph
                    current_paragraph_start_char = line_match.start()
                current_paragraph_lines.append(line_text)
            else: # Empty line (or line with only whitespace)
                if current_paragraph_lines: # End of current paragraph
                    full_para_text = "".join(current_paragraph_lines)
                    segments.append((full_para_text.strip(), current_paragraph_start_char, current_paragraph_start_char + len(full_para_text)))
                    current_paragraph_lines = []
                    current_paragraph_start_char = -1
        
        if current_paragraph_lines: # Append any trailing paragraph
            full_para_text = "".join(current_paragraph_lines)
            segments.append((full_para_text.strip(), current_paragraph_start_char, current_paragraph_start_char + len(full_para_text)))

        # Fallback for text without explicit paragraph breaks (e.g. single block)
        if not segments and text.strip():
            segments.append((text.strip(), 0, len(text)))

    elif segment_level == "sentence":
        if NLP_SPACY:
            doc = NLP_SPACY(text)
            for sent in doc.sents:
                sent_text_stripped = sent.text.strip()
                if sent_text_stripped:
                    segments.append((sent_text_stripped, sent.start_char, sent.end_char))
        else:
            logger.warning("get_text_segments: spaCy model not loaded. Falling back to basic sentence segmentation (less accurate).")
            # Basic regex fallback for sentences (less accurate)
            for match in re.finditer(r"([^\.!?]+(?:[\.!?]|$))", text): # Simple sentence split
                sent_text_stripped = match.group(1).strip()
                if sent_text_stripped:
                    segments.append((sent_text_stripped, match.start(), match.end()))
            if not segments and text.strip(): # If regex found nothing, treat as one sentence
                 segments.append((text.strip(), 0, len(text)))
    else:
        raise ValueError(f"Unsupported segment_level for get_text_segments: {segment_level}")
    
    return segments

async def deduplicate_text_segments(
    original_text: str,
    segment_level: str = "paragraph",
    similarity_threshold: float = 0.95, # Only used if use_semantic_comparison is True
    use_semantic_comparison: bool = False,
    min_segment_length_chars: int = 50
) -> Tuple[str, int]:
    """
    Removes near-duplicate segments from text.
    Default strategy is paragraph-level, normalized string comparison.
    """
    if not original_text.strip():
        return original_text, 0

    # segments_with_offsets contains (stripped_segment_text, original_start_char, original_end_char)
    segments_with_offsets = get_text_segments(original_text, segment_level)
    if not segments_with_offsets:
        return original_text, 0

    # Store info for segments we decide to keep
    # (original_start_char, original_end_char, comparison_key)
    # comparison_key will be normalized text or embedding vector
    kept_segment_info: List[Tuple[int, int, Union[str, np.ndarray]]] = []
    
    # For normalized string comparison, we can use a set of seen normalized texts for O(N) check
    seen_normalized_texts: Set[str] = set()
    # For semantic, we'd need to store embeddings and compare, which is O(N^2)

    segments_to_build_final_text: List[Tuple[int, int]] = [] # (start_char, end_char) of segments to keep from original

    for i, (seg_text, seg_start, seg_end) in enumerate(segments_with_offsets):
        if len(seg_text) < min_segment_length_chars:
            segments_to_build_final_text.append((seg_start, seg_end)) # Keep short segments
            continue

        is_duplicate = False
        if use_semantic_comparison:
            # This part needs async handling for embeddings if we were to implement it fully here.
            # For now, sticking to the recommendation of normalized string comparison.
            # If semantic were used, we'd compare current_seg_embedding with embeddings of segments in kept_segment_info.
            # For simplicity in this direct implementation, we'll show the string comparison path.
            # To make semantic work here, this whole function would need to be async,
            # and embedding generation would be interleaved or batched.
            logger.warning("Semantic de-duplication path in deduplicate_text_segments is a placeholder and will use normalized text.")
            # Fallthrough to normalized string comparison if use_semantic_comparison is True but not fully implemented async
            normalized_current_seg = _normalize_text_for_matching(seg_text)
            for _, _, kept_key_comp in kept_segment_info: # Assuming kept_key is normalized string here
                if isinstance(kept_key_comp, str) and normalized_current_seg == kept_key_comp: # Ensure type for comparison
                    is_duplicate = True
                    break
            if not is_duplicate:
                 kept_segment_info.append((seg_start, seg_end, normalized_current_seg))
                 segments_to_build_final_text.append((seg_start, seg_end))

        else: # Normalized string comparison
            normalized_current_seg = _normalize_text_for_matching(seg_text)
            if normalized_current_seg in seen_normalized_texts:
                is_duplicate = True
            else:
                seen_normalized_texts.add(normalized_current_seg)
                # For normalized string, we only need to store the fact that we've seen it.
                # We don't need to store the normalized key in kept_segment_info for future comparisons
                # if we are using the `seen_normalized_texts` set.
                # However, `kept_segment_info` was more for the semantic path.
                # For string path, `segments_to_build_final_text` is primary.
                segments_to_build_final_text.append((seg_start, seg_end))
        
        if is_duplicate:
            logger.info(f"De-duplication: Removing segment (idx {i}, chars {seg_start}-{seg_end}) starting with: '{seg_text[:60].replace(chr(10),' ')}...'")

    if len(segments_to_build_final_text) == len(segments_with_offsets): # No duplicates found
        return original_text, 0

    # Reconstruct text from the segments_to_build_final_text, preserving original whitespace
    # Sort by start offset just in case (though get_text_segments should return them in order)
    segments_to_build_final_text.sort(key=lambda x: x[0])
    
    final_parts = []
    # Add text from original_text based on the start/end of the segments TO KEEP.
    # This implicitly handles the whitespace between kept segments correctly.
    for start_char, end_char in segments_to_build_final_text:
        final_parts.append(original_text[start_char:end_char])

    # The joining character depends on the segment_level
    # If paragraphs, they already contain their trailing newlines (or should).
    # If sentences, they need a space.
    # However, `get_text_segments` returns stripped text but original offsets.
    # So, `original_text[start_char:end_char]` will have the original segment with its surrounding context.
    # The problem is, if we remove a paragraph, we need to ensure the newline structure is maintained.

    # Simpler reconstruction: build based on slices.
    # This was the more complex part of the previous attempt.
    # If we have a list of (start, end) of parts to KEEP, we can build it.
    # We need to manage the "glue" (whitespace) between them carefully.

    # Let's try a slightly different reconstruction:
    # Create a "mask" of what to keep.
    # This is still tricky. The original slice-based removal was probably better.

    # Reverting to a simpler reconstruction based on keeping entire original slices.
    # The key is that `segments_to_build_final_text` now contains the *original* start/end
    # of the *stripped* segments that were kept. We need to use these to slice original_text.
    
    reconstructed_text_parts = []
    current_doc_pointer = 0
    for seg_start, seg_end in segments_to_build_final_text: # These are segments to KEEP
        # Append text from end of last kept segment (or doc start) up to start of current kept segment
        # This captures inter-segment text/whitespace.
        if seg_start > current_doc_pointer:
            reconstructed_text_parts.append(original_text[current_doc_pointer:seg_start])
        
        # Append the kept segment itself
        reconstructed_text_parts.append(original_text[seg_start:seg_end])
        current_doc_pointer = seg_end
    
    # Append any trailing text after the last kept segment
    if current_doc_pointer < len(original_text):
        reconstructed_text_parts.append(original_text[current_doc_pointer:])

    deduplicated_text = "".join(reconstructed_text_parts)
    
    # Final cleanup of excessive newlines that might form from stitching
    deduplicated_text = re.sub(r'\n\s*\n(\s*\n)+', '\n\n', deduplicated_text) # Multiple blank lines to one
    deduplicated_text = re.sub(r'\n{3,}', '\n\n', deduplicated_text) # 3+ newlines to 2
    deduplicated_text = deduplicated_text.strip() # Remove leading/trailing whitespace from the whole result

    characters_removed_count = len(original_text) - len(deduplicated_text)
    
    return deduplicated_text, characters_removed_count
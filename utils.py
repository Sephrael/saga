# utils.py
"""
General utility functions for the Saga Novel Generation system.
MODIFIED: Enhanced find_quote_and_sentence_offsets_with_spacy for more robust quote matching.
"""

import numpy as np
import logging
import re
import asyncio
from typing import Optional, Tuple, List, Union

# Local application imports - ensure these paths are correct for your project
import llm_interface
import spacy

logger = logging.getLogger(__name__)

# Global spaCy model instance
NLP_SPACY: Optional[spacy.language.Language] = None

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
    """Normalizes text for more robust matching."""
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove leading/trailing quotes and ellipses that might be added by the LLM
    text = re.sub(r"^[ '\"\(]*(\.\.\.)?[ '\"\(]*|[ '\"\(]*(\.\.\.)?[ '\"\(]*$", "", text)
    # Normalize whitespace (replace multiple spaces/newlines with a single space)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove punctuation that might differ
    text = re.sub(r'[^\w\s]', '', text)
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

    # Handle "N/A - General Issue" directly
    if "N/A - General Issue" in quote_text_from_llm: # Check with "in" to catch variations like "N/A - General Issue..."
        logger.debug(f"Quote is '{quote_text_from_llm}', treating as general issue. No offsets.")
        return None

    # 1. Attempt Direct (Case-Insensitive) Substring Match of the core quote text
    # Clean the LLM quote slightly to remove common LLM artifacts like leading/trailing quotes or ellipses.
    # This is NOT the full normalization yet.
    cleaned_llm_quote = quote_text_from_llm.strip(' "\'.')
    if not cleaned_llm_quote: # if stripping leaves nothing
        logger.debug("LLM quote became empty after basic stripping, cannot match.")
        return None

    spacy_doc = NLP_SPACY(doc_text)
    
    # Try finding the cleaned_llm_quote as a substring first.
    # This handles cases where the LLM gives a mostly verbatim quote but maybe with different surrounding punctuation.
    best_direct_match_offsets: Optional[Tuple[int, int, int, int]] = None
    try:
        # Iterate to find the best match (e.g., one that fits well within a sentence)
        # We will make this simpler for now and take the first good one.
        current_pos = 0
        while current_pos < len(doc_text):
            # Case-insensitive search for the cleaned quote
            match_start = doc_text.lower().find(cleaned_llm_quote.lower(), current_pos)
            if match_start == -1:
                break
            
            match_end = match_start + len(cleaned_llm_quote)
            
            # Check if this match falls within a sentence
            found_sentence = None
            for sent in spacy_doc.sents:
                if sent.start_char <= match_start < sent.end_char and \
                   sent.start_char < match_end <= sent.end_char:
                    found_sentence = sent
                    break
            
            if found_sentence:
                logger.info(f"Direct Substring Match: Found LLM quote (approx) '{cleaned_llm_quote[:30]}...' at {match_start}-{match_end} in sentence {found_sentence.start_char}-{found_sentence.end_char}")
                best_direct_match_offsets = (match_start, match_end, found_sentence.start_char, found_sentence.end_char)
                return best_direct_match_offsets # Take the first good direct match

            current_pos = match_end # Continue search after this match
            
    except Exception as e_direct:
        logger.warning(f"Error during direct substring match attempt: {e_direct}")

    # 2. If direct match fails, fallback to semantic matching for the *sentence*
    #    This is more robust if the LLM heavily paraphrased.
    #    We use the original LLM quote for semantic search as it has more context.
    logger.warning(f"Direct substring match failed for LLM quote '{quote_text_from_llm[:50]}...'. Falling back to semantic sentence search.")
    
    # Find the semantically closest sentence in doc_text to quote_text_from_llm
    semantic_sentence_match = await find_semantically_closest_segment(
        original_doc=doc_text,
        query_text=quote_text_from_llm, # Use the original LLM quote for richer semantics
        segment_type="sentence",
        min_similarity_threshold=0.75 # Higher threshold for sentence-level precision
    )

    if semantic_sentence_match:
        s_start, s_end, similarity = semantic_sentence_match
        # For the "quote" offsets within this sentence, we can either:
        #   a) Try to find a smaller, more specific match of the LLM quote *within* this sentence.
        #   b) Or, consider the whole sentence as the "effective quote" area.
        # Let's try (a) with normalization.
        sentence_text = doc_text[s_start:s_end]
        normalized_llm_quote_for_subsearch = _normalize_text_for_matching(cleaned_llm_quote)
        normalized_sentence_text = _normalize_text_for_matching(sentence_text)

        q_match_in_sentence_start = normalized_sentence_text.find(normalized_llm_quote_for_subsearch)
        
        quote_start_abs, quote_end_abs = s_start, s_end # Default to whole sentence
        if q_match_in_sentence_start != -1:
            # This is tricky because normalized_sentence_text has different indexing.
            # We can't directly map these back to original doc_text character offsets easily without a more complex alignment.
            # For simplicity now, if semantic search gives us a sentence, we'll consider the whole sentence the target.
            # The patch LLM will be given this sentence as context.
            logger.info(f"Semantic Match: Found sentence for LLM quote '{quote_text_from_llm[:30]}...' from {s_start}-{s_end} (Similarity: {similarity:.2f}). Using whole sentence as target.")
            # We mark quote_start/end as the sentence start/end because the semantic match was at sentence level.
            return s_start, s_end, s_start, s_end
        else:
            logger.info(f"Semantic Match: Found sentence for LLM quote '{quote_text_from_llm[:30]}...' from {s_start}-{s_end} (Similarity: {similarity:.2f}). Sub-quote not found in normalized sentence. Using whole sentence.")
            return s_start, s_end, s_start, s_end # Return sentence boundaries for both

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
    segment_type: str = "paragraph", # "paragraph" or "sentence"
    min_similarity_threshold: float = 0.65 # Adjust as needed
) -> Optional[Tuple[int, int, float]]: # (start_char, end_char, similarity_score)
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

    segments_with_indices: List[Tuple[str, int, int]] = []
    if segment_type == "paragraph":
        # Simple paragraph split by double newlines (or more)
        # More robust would be regex accounting for mixed whitespace lines
        # For now, this is a common convention.
        current_pos = 0
        for para_text in re.split(r'\n\s*\n', original_doc): # Split by one or more blank lines
            para_text_stripped = para_text.strip()
            if para_text_stripped:
                # Find the actual start of this paragraph in the original to get correct offsets
                try:
                    start_char = original_doc.index(para_text, current_pos) # Search from last position
                    end_char = start_char + len(para_text)
                    segments_with_indices.append((para_text_stripped, start_char, end_char))
                    current_pos = end_char
                except ValueError: # Should not happen if split correctly from original
                    logger.warning(f"Paragraph segment '{para_text_stripped[:30]}...' not found in original doc for offset calc. Skipping.")
                    current_pos += len(para_text) + 2 # Estimate advance

    elif segment_type == "sentence":
        load_spacy_model_if_needed() # Ensure spaCy model is loaded
        if NLP_SPACY:
            spacy_doc = NLP_SPACY(original_doc)
            for sent in spacy_doc.sents:
                segment_text = sent.text.strip()
                if segment_text:
                    segments_with_indices.append((segment_text, sent.start_char, sent.end_char))
        else: 
            logger.warning("find_semantically_closest_segment (sentence): spaCy model not loaded, using regex.")
            for sent_match in re.finditer(r'[^.!?\n]+(?:[.!?](?![\'"])|\n|$)', original_doc): 
                segment_text = sent_match.group(0).strip()
                if segment_text:
                     segments_with_indices.append((segment_text, sent_match.start(), sent_match.end()))
        
        if not segments_with_indices and original_doc.strip(): # Fallback for sentence mode
            segments_with_indices.append((original_doc.strip(), 0, len(original_doc.strip())))
    else:
        logger.error(f"Unsupported segment_type for semantic search: {segment_type}")
        return None
    
    if not segments_with_indices:
        logger.warning(f"No segments found in original document for semantic search (Type: {segment_type}).")
        return None

    best_match_info: Optional[Tuple[int, int, float]] = None
    highest_similarity = -2.0 

    segment_texts = [s[0] for s in segments_with_indices]
    
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
            _, start_char, end_char = segments_with_indices[i]
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
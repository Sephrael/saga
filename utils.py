# utils.py
"""
General utility functions for the Saga Novel Generation system.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2025 Dennis Lewis
"""

import numpy as np
import logging
import re
import asyncio
from typing import Optional, Tuple, List

# Local application imports - ensure these paths are correct for your project
import llm_interface 

logger = logging.getLogger(__name__)

# To use NLTK for sentence tokenization:
# import nltk
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')
#     logger.info("NLTK 'punkt' tokenizer downloaded.")


def numpy_cosine_similarity(
    vec1: Optional[np.ndarray], vec2: Optional[np.ndarray]
) -> float:
    """
    Calculates cosine similarity between two numpy vectors.
    Handles None inputs, shape mismatches, and zero vectors gracefully.

    Args:
        vec1: The first numpy array (or None).
        vec2: The second numpy array (or None).

    Returns:
        The cosine similarity as a float between -1.0 and 1.0 (or 0.0 if inputs are invalid).
    """
    if vec1 is None or vec2 is None:
        logger.debug("Cosine similarity: one or both vectors are None. Returning 0.0.")
        return 0.0

    try:
        # Ensure inputs are numpy arrays and flatten them to 1D
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

    if v1.size == 0:  # Handles empty vectors
        logger.debug("Cosine similarity: input vector(s) are empty. Returning 0.0.")
        return 0.0

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0.0 or norm_v2 == 0.0:
        # If one or both vectors are zero vectors, similarity is 0
        # (unless both are zero, then it's undefined, but 0 is a safe return)
        logger.debug("Cosine similarity: at least one zero-norm vector. Returning 0.0.")
        return 0.0
    
    dot_product = np.dot(v1, v2)
    similarity = dot_product / (norm_v1 * norm_v2)

    # Clamp to [-1.0, 1.0] to handle potential floating point inaccuracies
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

    Returns:
        A tuple (start_char_index, end_char_index, similarity_score) or None.
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
        # This regex attempts to capture blocks of text separated by one or more blank lines.
        # It captures the text itself (group 0), its start, and its end in the original_doc.
        # It's a bit more robust than simple `split('\n\n')` as it handles varying numbers of newlines.
        for match in re.finditer(r"((?:[^\n]+\n?)+)", original_doc): # Simpler paragraph splitter: find non-empty lines
            segment_text = match.group(0).strip()
            if segment_text: # Ensure we don't process segments that are only whitespace
                 segments_with_indices.append((segment_text, match.start(), match.end()))
    elif segment_type == "sentence":
        # Using a robust sentence tokenizer is recommended (e.g., nltk.sent_tokenize)
        # For a simpler regex-based approach (less robust for complex sentences):
        # This regex tries to match up to a sentence-ending punctuation mark,
        # ensuring it's not part of an abbreviation or inside quotes if possible.
        # nltk.sent_tokenize(original_doc) would be better here.
        # Example NLTK usage:
        # sentences = nltk.sent_tokenize(original_doc)
        # current_pos = 0
        # for sentence in sentences:
        #     start_char = original_doc.find(sentence, current_pos)
        #     if start_char != -1:
        #         end_char = start_char + len(sentence)
        #         segments_with_indices.append((sentence.strip(), start_char, end_char))
        #         current_pos = end_char
        # For now, using a simpler regex for demonstration without adding NLTK dependency yet:
        for sent_match in re.finditer(r'[^.!?\n]+(?:[.!?](?![\'"])|\n|$)', original_doc): # Basic sentence splitter
            segment_text = sent_match.group(0).strip()
            if segment_text:
                 segments_with_indices.append((segment_text, sent_match.start(), sent_match.end()))
        if not segments_with_indices and original_doc.strip(): # Fallback: treat whole doc as one segment
            segments_with_indices.append((original_doc.strip(), 0, len(original_doc.strip())))
    else:
        logger.error(f"Unsupported segment_type for semantic search: {segment_type}")
        return None
    
    if not segments_with_indices:
        logger.warning(f"No segments found in original document for semantic search (Type: {segment_type}).")
        return None

    best_match_info: Optional[Tuple[int, int, float]] = None
    highest_similarity = -2.0 # Initialize below possible cosine similarity range

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
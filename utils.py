# utils.py
"""
General utility functions for the Saga Novel Generation system.
"""

import numpy as np
import logging
from typing import Optional

# Initialize logger for this module
# Use __name__ to get the module name ("utils") for the logger
logger = logging.getLogger(__name__)

def numpy_cosine_similarity(vec1: Optional[np.ndarray], vec2: Optional[np.ndarray]) -> float:
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
        logger.debug("Cosine similarity calculation skipped: one or both vectors are None.")
        return 0.0

    # Ensure inputs are numpy arrays and flatten them
    try:
        v1 = np.asarray(vec1, dtype=np.float32).flatten() # Use a consistent dtype
        v2 = np.asarray(vec2, dtype=np.float32).flatten()
    except ValueError as e:
        logger.warning(f"Could not convert input to numpy array for cosine similarity: {e}. Returning 0.0")
        return 0.0

    # Check for shape mismatch
    if v1.shape != v2.shape:
        logger.warning(f"Cosine similarity shape mismatch: {v1.shape} vs {v2.shape}. Returning 0.0.")
        return 0.0

    # Check for zero-sized vectors (although flatten should prevent shape mismatch)
    if v1.size == 0:
        logger.debug("Cosine similarity calculated for zero-sized vector: 0.0")
        return 0.0

    # Calculate norms
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Handle cases where norm is zero (zero vector)
    if norm_v1 == 0 or norm_v2 == 0:
        similarity = 0.0
        logger.debug("Cosine similarity calculated with zero norm vector: 0.0")
    else:
        # Calculate dot product and similarity
        dot_product = np.dot(v1, v2)
        similarity = dot_product / (norm_v1 * norm_v2)
        # Clamp values to handle potential floating point inaccuracies near +/- 1
        similarity = np.clip(similarity, -1.0, 1.0)

    # Ensure the return value is a standard Python float
    final_similarity = float(similarity)
    # logger.debug(f"Cosine similarity calculated: {final_similarity:.4f}") # Can be verbose
    return final_similarity

# Add other general utility functions here if needed in the future.
# For example, functions for text chunking, advanced JSON merging, etc.


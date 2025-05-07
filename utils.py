# utils.py
"""
General utility functions for the Saga Novel Generation system.
"""

import numpy as np
import logging
from typing import Optional

# Initialize logger for this module
logger = logging.getLogger(__name__)


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
        logger.debug(
            "Cosine similarity calculation skipped: one or both vectors are None."
        )
        return 0.0

    try:
        v1 = np.asarray(vec1, dtype=np.float32).flatten()
        v2 = np.asarray(vec2, dtype=np.float32).flatten()
    except ValueError as e:
        logger.warning(
            f"Could not convert input to numpy array for cosine similarity: {e}. Returning 0.0"
        )
        return 0.0

    if v1.shape != v2.shape:
        logger.warning(
            f"Cosine similarity shape mismatch: {v1.shape} vs {v2.shape}. Returning 0.0."
        )
        return 0.0

    if v1.size == 0:  # Or v2.size == 0, since shapes match
        logger.debug("Cosine similarity calculated for zero-sized vector(s): 0.0")
        return 0.0

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        similarity = 0.0
        logger.debug(
            "Cosine similarity calculated with at least one zero norm vector: 0.0"
        )
    else:
        dot_product = np.dot(v1, v2)
        similarity = dot_product / (norm_v1 * norm_v2)
        similarity = np.clip(similarity, -1.0, 1.0)  # Clamp for float precision issues

    return float(similarity)

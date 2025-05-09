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
from typing import Optional

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
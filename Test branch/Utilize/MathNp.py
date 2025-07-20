""" Math function of numpy
created by hsy on 2025-07-20
"""
import numpy as np
import numpy.typing as npt

def compress_vectors(vectors:npt.NDArray, threshold:float)->npt.NDArray:
    """
    Args:
        vectors (np.ndarray): An N x 2 array of vectors.
        threshold (float): The threshold value for the vector norms.

    Returns:
        np.ndarray: An N x 2 array of vectors after applying the compression.
    """
    # Compute the Euclidean norms for each row vector (shape: N x 1).
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scales = np.where(norms > threshold, threshold / norms, 1.0)
    
    return vectors * scales
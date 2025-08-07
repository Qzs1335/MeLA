import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    # Clip distances to avoid extreme values
    clipped_distances = np.clip(distance_matrix, epsilon, None)
    # Compute inverse probabilities
    inv_probs = 1 / clipped_distances
    # Normalize to get valid probability distribution
    normalized_probs = inv_probs / np.sum(inv_probs, axis=1, keepdims=True)
    # Ensure all probabilities are finite and positive
    normalized_probs = np.clip(normalized_probs, 1e-10, 1.0)
    # Renormalize after clipping
    normalized_probs = normalized_probs / np.sum(normalized_probs, axis=1, keepdims=True)
    return normalized_probs
    #EVOLVE-END
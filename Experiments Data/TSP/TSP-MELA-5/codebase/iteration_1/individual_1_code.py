import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    # Ensure all distances are positive and non-zero
    safe_distances = np.maximum(distance_matrix, epsilon)
    # Compute weights (inverse of distance)
    weights = 1 / safe_distances
    # Normalize to get valid probabilities
    probabilities = weights / np.sum(weights, axis=1, keepdims=True)
    return probabilities
    #EVOLVE-END
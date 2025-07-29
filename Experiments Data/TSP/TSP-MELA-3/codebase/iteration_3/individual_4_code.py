import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # Small constant for numerical stability
    # Clip very small distances to avoid division by zero
    safe_distances = np.maximum(distance_matrix, epsilon)
    # Calculate inverse distances
    inv_dist = 1 / safe_distances
    # Normalize so everything sums to 1
    if len(inv_dist) > 0:
        inv_dist = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
    return inv_dist
    #EVOLVE-END
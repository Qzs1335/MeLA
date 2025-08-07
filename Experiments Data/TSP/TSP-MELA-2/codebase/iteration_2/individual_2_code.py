import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = np.mean(distance_matrix) * 1e-6 + 1e-10  # Small epsilon to avoid division by zero
    log_scaled = np.log(1 + distance_matrix)/(1 + np.exp(-distance_matrix/np.mean(distance_matrix)))
    k_nearest = np.partition(distance_matrix, 3, axis=1)[:,:4]  # Find 4 nearest neighbors
    neighbor_weights = np.mean(k_nearest, axis=1, keepdims=True)  # Fixed variable name
    enhanced_weights = (1 / (distance_matrix + eps)) * neighbor_weights / log_scaled  # Correct var naming
    return enhanced_weights
    #EVOLVE-END
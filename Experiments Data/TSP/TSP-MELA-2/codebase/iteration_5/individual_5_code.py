import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    sqrt_dist = np.sqrt(distance_matrix + eps)  # Smoother scaling than log
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:5]  # Skip self-edge
    neigh_weights = np.median(nearest_neigh, axis=1, keepdims=True)  # Avoid broadcasting issues
    exp_decay = np.exp(-distance_matrix / neigh_weights) 
    return (exp_decay / sqrt_dist) * (neigh_weights.T + neigh_weights) / (2 * np.sqrt(sqrt_dist))  # Smoother denominator
    #EVOLVE-END
    return 1 / distance_matrix
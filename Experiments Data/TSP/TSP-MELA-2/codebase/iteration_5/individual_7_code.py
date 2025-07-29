import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8  # Numerical stability threshold
    sqrt_dist = np.sqrt(distance_matrix + eps)
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:5]
    neigh_weights = np.mean(nearest_neigh, axis=1)
    exp_decay = np.exp(-distance_matrix/neigh_weights[:,None])
    return (exp_decay + eps) / sqrt_dist
    #EVOLVE-END
    return 1 / distance_matrix
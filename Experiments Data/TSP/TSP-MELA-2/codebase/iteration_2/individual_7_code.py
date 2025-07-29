import numpy as np
import numpy as np
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    adaptive_factor = 0.5 * (1 + np.exp(-distance_matrix.mean()))
    inv_dist = 1/(distance_matrix + eps)
    exp_term = np.exp(-distance_matrix/distance_matrix.mean())
    neighbor_weights = np.sort(distance_matrix, axis=1)[:,1:4].mean(axis=1, keepdims=True)
    return adaptive_factor*(inv_dist + exp_term)/(1 + neighbor_weights.T)
    #EVOLVE-END
    return 1 / distance_matrix
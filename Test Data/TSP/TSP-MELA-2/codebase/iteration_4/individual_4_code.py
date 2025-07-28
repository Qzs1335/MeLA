import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-10
    neigh_weights = np.median(np.sort(distance_matrix, axis=1)[:,1:5], axis=1)
    exp_decay = np.exp(-distance_matrix/(neigh_weights[:,None] + eps))
    weights = exp_decay / (distance_matrix + eps)
    return np.exp(weights) / np.sum(np.exp(weights), axis=1, keepdims=True)
    #EVOLVE-END
    return 1 / distance_matrix
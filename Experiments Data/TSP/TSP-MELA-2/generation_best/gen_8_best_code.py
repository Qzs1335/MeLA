import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    neigh_mean = np.mean(np.sort(distance_matrix, axis=1)[:,1:5], axis=1)
    norm_term = (neigh_mean[:,None] + neigh_mean)/np.maximum(distance_matrix, eps)**2
    return np.reciprocal(distance_matrix + eps) * norm_term
    #EVOLVE-END       
    return 1 / distance_matrix
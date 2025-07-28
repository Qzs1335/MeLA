import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    neigh_dist = np.median(np.sort(distance_matrix, axis=1)[:,1:5], axis=1)
    scaled_dist = distance_matrix / (neigh_dist[:,None] + eps)
    return np.exp(-scaled_dist) + 1/(1 + np.log1p(distance_matrix))
    #EVOLVE-END       
    return 1 / distance_matrix
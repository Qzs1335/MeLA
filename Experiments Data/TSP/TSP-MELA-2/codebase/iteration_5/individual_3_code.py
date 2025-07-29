import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:5]
    neigh_weights = np.median(nearest_neigh, axis=1)
    weighted_dist = distance_matrix/(neigh_weights[:,None] + eps)
    return np.exp(-weighted_dist)/(distance_matrix + eps)
    #EVOLVE-END
    return 1 / distance_matrix
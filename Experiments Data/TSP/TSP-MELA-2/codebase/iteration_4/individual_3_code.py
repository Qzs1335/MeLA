import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    neigh_weights = np.median(np.sort(distance_matrix, axis=1)[:,1:5], axis=1)
    scaled_dist = np.log1p(distance_matrix) / (distance_matrix + eps)
    return np.exp(-distance_matrix/(neigh_weights[:,None]+eps)) * scaled_dist
    #EVOLVE-END        
    return 1 / distance_matrix
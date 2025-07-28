import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    neigh_dist = np.partition(distance_matrix, 1)[:,1]
    safe_exp = np.exp(-np.clip(distance_matrix,0,100)/neigh_dist[:,None])
    norm_factor = np.sqrt(distance_matrix + eps) + neigh_dist[:,None]
    return safe_exp / norm_factor
    #EVOLVE-END
    return 1 / distance_matrix
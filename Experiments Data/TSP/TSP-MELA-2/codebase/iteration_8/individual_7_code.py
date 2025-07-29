import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    neigh_dist = np.partition(distance_matrix, 1)[:,1]
    local_scale = neigh_dist[:,None]
    exp_term = np.exp(-np.sqrt(distance_matrix)/local_scale)
    return exp_term/(distance_matrix + eps*local_scale)
    #EVOLVE-END
    return 1 / distance_matrix
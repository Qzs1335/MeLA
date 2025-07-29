import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-12
    log_scaled = np.log1p(distance_matrix)
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:3]
    neigh_weights = np.median(nearest_neigh, axis=1)
    exp_decay = np.exp(-distance_matrix/(neigh_weights[:,None]+eps))
    return exp_decay / (log_scaled + eps)
    #EVOLVE-END
    return 1 / distance_matrix
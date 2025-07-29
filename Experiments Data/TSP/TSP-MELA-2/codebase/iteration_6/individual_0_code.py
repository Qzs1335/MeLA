import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    log_scaled = np.log1p(distance_matrix)
    neigh_weights = np.mean(np.sort(distance_matrix, axis=1)[:,1:5], axis=1)
    exp_decay = np.exp(-distance_matrix/neigh_weights[:,None])
    return exp_decay / (distance_matrix + eps + 0.5*log_scaled)
    #EVOLVE-END
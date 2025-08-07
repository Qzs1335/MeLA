import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    log_term = np.sqrt(np.log1p(distance_matrix + eps))
    nearest = np.partition(distance_matrix, 4, axis=1)[:,:4]
    neigh_weights = np.mean(np.sort(nearest,axis=1)[:,1:], axis=1) 
    asym_decay = np.exp(-np.abs(distance_matrix-neigh_weights[:,None])/neigh_weights[:,None])
    return (asym_decay + 1/(distance_matrix + eps)) / (2 * log_term)
    #EVOLVE-END
    return 1 / distance_matrix
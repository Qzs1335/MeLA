import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-12
    k = min(3, distance_matrix.shape[1]-1)
    nearest_neigh = np.partition(distance_matrix, k, axis=1)[:,1:k+1]
    neigh_weights = np.mean(nearest_neigh, axis=1)
    decay = np.exp(-distance_matrix/neigh_weights[:,None])
    log_scale = np.log1p(distance_matrix)
    return decay / (distance_matrix + eps + log_scale)
    #EVOLVE-END       
    return 1 / distance_matrix
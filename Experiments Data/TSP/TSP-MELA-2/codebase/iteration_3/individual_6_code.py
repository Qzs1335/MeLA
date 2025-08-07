import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    neigh_weights = np.median(np.sort(distance_matrix, axis=1)[:,1:5], axis=1)
    safe_exp = np.exp(-np.minimum(distance_matrix, 5)/neigh_weights[:,None])
    return (safe_exp + 1)/(distance_matrix + eps)
    #EVOLVE-END       
    return 1 / distance_matrix
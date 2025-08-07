import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START  
    eps = 1e-8
    log_scaled = np.log(1 + distance_matrix + eps)
    neigh_weights = np.median(np.sort(distance_matrix, axis=1)[:,1:5], axis=1)
    return 1 / (distance_matrix + eps * (distance_matrix < eps)) + neigh_weights[:,None] / log_scaled  
    #EVOLVE-END       
    return 1 / distance_matrix
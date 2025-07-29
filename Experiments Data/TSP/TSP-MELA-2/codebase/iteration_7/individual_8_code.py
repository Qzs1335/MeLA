import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    log_scaled = np.log1p(distance_matrix)  # More stable than sqrt(log)
    median_neigh = np.median(np.sort(distance_matrix, axis=1)[:,1:5], axis=1)
    return np.exp(-distance_matrix / median_neigh[:,None]) / (distance_matrix + eps)
    #EVOLVE-END
    return 1 / distance_matrix
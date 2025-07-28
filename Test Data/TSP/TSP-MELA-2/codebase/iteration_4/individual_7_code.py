import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    eps = 1e-8
    log_scaled = np.sqrt(np.log(1 + distance_matrix))
    nearest_neigh = np.sort(distance_matrix, axis=1)[:,1:5]
    neigh_weights = np.median(nearest_neigh, axis=1)
    return 1 / (distance_matrix**0.5 * log_scaled * (distance_matrix + neigh_weights[:,None] + eps))
    #EVOLVE-END
    return 1 / distance_matrix
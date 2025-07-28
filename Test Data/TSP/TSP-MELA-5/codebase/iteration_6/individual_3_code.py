import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    inv_dist_sq = 1 / (distance_matrix**2 + epsilon)
    log_dist = np.log1p(distance_matrix)
    return 0.7 * inv_dist_sq + 0.3 / log_dist
    #EVOLVE-END       
    return 1 / distance_matrix
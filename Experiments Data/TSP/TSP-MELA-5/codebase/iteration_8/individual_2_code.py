import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    adaptive_eps = np.finfo(float).eps * np.max(distance_matrix)
    norm_dist = distance_matrix / np.max(distance_matrix)
    inv_weights = 1 / (norm_dist**2 + adaptive_eps)
    log_weights = 1 / np.log(1 + norm_dist + adaptive_eps)
    return (inv_weights * log_weights) / np.sum(inv_weights * log_weights)
    #EVOLVE-END       
    return 1 / distance_matrix
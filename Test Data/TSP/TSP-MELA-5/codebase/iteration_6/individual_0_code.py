import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    log_weights = np.log1p(distance_matrix)
    sqrt_dist = np.sqrt(distance_matrix)
    return (1 / (sqrt_dist + epsilon)) * (1 / log_weights)
    #EVOLVE-END       
    return 1 / distance_matrix
import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    sqrt_dist = np.sqrt(distance_matrix + epsilon)
    log_term = np.log(epsilon + distance_matrix)
    return (1 / sqrt_dist) * (1 / log_term)
    #EVOLVE-END       
    return 1 / distance_matrix
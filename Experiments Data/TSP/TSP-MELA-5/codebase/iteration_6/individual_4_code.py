import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    inv_square = 1 / (distance_matrix**2 + epsilon)
    log_scaled = 1 / np.log1p(distance_matrix)
    return np.sqrt(inv_square * log_scaled)
    #EVOLVE-END       
    return 1 / distance_matrix
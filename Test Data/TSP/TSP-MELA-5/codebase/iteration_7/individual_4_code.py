import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    log_term = np.log(distance_matrix + epsilon + 1)
    weights = (1 / (distance_matrix + epsilon)) * (1 / log_term)
    return weights / np.sum(weights, axis=1, keepdims=True)
    #EVOLVE-END       
    return 1 / distance_matrix
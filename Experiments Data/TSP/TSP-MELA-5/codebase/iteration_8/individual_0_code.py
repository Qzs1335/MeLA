import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    inv_dist = 1 / (distance_matrix**2 + epsilon)
    log_weights = np.log(1 + distance_matrix + epsilon)
    weights = inv_dist / (log_weights + epsilon)
    return weights / np.sum(weights, axis=1, keepdims=True)
    #EVOLVE-END       
    return 1 / distance_matrix
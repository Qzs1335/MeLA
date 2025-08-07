import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    median_dist = np.median(distance_matrix)
    epsilon = np.maximum(1e-10, 0.01 * median_dist)
    return np.log1p(1/(distance_matrix + epsilon))  # log1p for numerical stability
    #EVOLVE-END       
    return np.exp(1/(distance_matrix + 1e-10))
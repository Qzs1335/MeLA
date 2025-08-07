import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    safe_dist = np.maximum(distance_matrix, epsilon)
    return np.exp(-safe_dist/5) + np.log1p(1/safe_dist)
    #EVOLVE-END       
    return 1 / distance_matrix
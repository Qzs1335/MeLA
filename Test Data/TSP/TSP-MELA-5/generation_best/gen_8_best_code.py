import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    scale = np.median(distance_matrix)
    epsilon = max(1e-15, scale * 1e-6)
    weights = (1 / (distance_matrix**2 + epsilon)) * (1 / np.log1p(distance_matrix))
    return weights / np.sum(weights, axis=1, keepdims=True)
    #EVOLVE-END       
    return 1 / distance_matrix
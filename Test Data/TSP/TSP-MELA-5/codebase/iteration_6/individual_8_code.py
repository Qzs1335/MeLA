import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    weights = 1 / (distance_matrix**2 + epsilon)
    return np.clip(weights, 1e-10, 1e10)
    #EVOLVE-END       
    return 1 / distance_matrix
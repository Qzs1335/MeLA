import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-20
    safe_dist = np.maximum(distance_matrix, epsilon)
    inv_dist = 1 / safe_dist
    exp_dist = np.exp(inv_dist)
    return exp_dist / np.sum(exp_dist, axis=1, keepdims=True)
    #EVOLVE-END       
    return distance_matrix
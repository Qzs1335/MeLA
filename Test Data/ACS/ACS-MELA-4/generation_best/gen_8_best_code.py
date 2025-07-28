import numpy as np
import numpy as np 

def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    safe_dist = np.maximum(distance_matrix, epsilon)
    log_dist = -np.log(safe_dist)
    scaled_dist = log_dist / np.mean(safe_dist)
    return np.exp(scaled_dist) / np.sum(np.exp(scaled_dist), axis=1, keepdims=True)
    #EVOLVE-END       
    return distance_matrix
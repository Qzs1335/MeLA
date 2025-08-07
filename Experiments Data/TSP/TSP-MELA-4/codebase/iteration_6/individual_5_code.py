import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    normalized_dist = distance_matrix/(np.median(distance_matrix) + epsilon)
    dynamic_beta = 1.5 + 0.5*np.sin(np.mean(normalized_dist))
    decay = np.exp(-normalized_dist)
    inv_dist = (1/(normalized_dist + epsilon))**dynamic_beta
    return np.tanh(1/(normalized_dist + epsilon)) + decay * inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix
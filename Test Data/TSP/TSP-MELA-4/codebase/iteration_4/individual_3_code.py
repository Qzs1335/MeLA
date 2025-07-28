import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    adaptive_scale = np.mean(distance_matrix)
    beta = 2.0 - (np.min(distance_matrix)/(np.max(distance_matrix) + epsilon))  # dynamic β
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**beta
    return (decay * inv_dist)/np.max(decay * inv_dist)  # normalized
    #EVOLVE-END       
    return 1 / distance_matrix
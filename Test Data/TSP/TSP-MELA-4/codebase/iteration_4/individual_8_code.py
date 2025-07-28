import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    adaptive_scale = np.mean(distance_matrix)
    dynamic_beta = 1.5 + (distance_matrix/np.max(distance_matrix))  # adaptive exponent
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**dynamic_beta
    return (decay * inv_dist)/np.max(decay * inv_dist)  # normalized
    #EVOLVE-END       
    return 1 / distance_matrix
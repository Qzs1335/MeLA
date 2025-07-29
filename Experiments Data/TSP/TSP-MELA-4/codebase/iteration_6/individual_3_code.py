import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.mean(distance_matrix)
    dynamic_beta = 1.0 + 0.3*np.sin(adaptive_scale)  # simpler adaptation
    inv_dist = (1/(distance_matrix + epsilon))**dynamic_beta
    return inv_dist * np.log1p(1/(distance_matrix + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix
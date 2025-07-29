import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8  # improved numerical stability
    adaptive_scale = np.median(distance_matrix)  # more robust scaling
    dynamic_beta = 1.5 + np.log1p(adaptive_scale)  # adaptive exponent
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**dynamic_beta
    return decay * inv_dist * np.log1p(1/(distance_matrix + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix
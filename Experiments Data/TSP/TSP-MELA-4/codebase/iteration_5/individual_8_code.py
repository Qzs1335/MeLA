import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # tighter numerical stability
    adaptive_scale = np.median(distance_matrix[distance_matrix > 0])  # exclude zeros
    dynamic_beta = 1.2 + 0.8*np.log1p(adaptive_scale)  # optimized coefficient
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**dynamic_beta
    return decay * inv_dist * (1 + np.log1p(1/(distance_matrix + epsilon)))
    #EVOLVE-END       
    return 1 / distance_matrix
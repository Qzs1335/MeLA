import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-7  # tighter numerical stability
    adaptive_scale = np.percentile(distance_matrix, 75)  # robust scaling
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**1.5  # optimized exponent
    return (decay * inv_dist) / np.sum(decay * inv_dist)  # normalized
    #EVOLVE-END       
    return 1 / distance_matrix
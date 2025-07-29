import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-6  # increased stability margin
    harmonic_scale = len(distance_matrix)/np.sum(1/(distance_matrix + epsilon))  # harmonic mean
    dynamic_beta = 1.2 + np.log1p(harmonic_scale)  
    decay = np.exp(-distance_matrix/(harmonic_scale + epsilon))
    inv_dist = np.clip(1/(distance_matrix + epsilon), 0.1, 10)**dynamic_beta  
    return decay * inv_dist * np.log1p(np.clip(1/(distance_matrix + epsilon), 0.1, 10))
    #EVOLVE-END       
    return 1 / distance_matrix
import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    dynamic_beta = 1.2 + 0.3*np.log1p(adaptive_scale)  # tighter beta range
    decay = np.exp(-distance_matrix/(0.5*adaptive_scale + epsilon))  # faster decay
    inv_dist = (1/(distance_matrix + epsilon))**dynamic_beta
    norm_factor = 1/(np.max(inv_dist) + epsilon)  # adaptive normalization
    return norm_factor * (0.9*decay*inv_dist + 0.1*np.random.rand(*distance_matrix.shape))  # balanced exploration
    #EVOLVE-END       
    return 1 / distance_matrix
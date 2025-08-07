import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    dynamic_beta = 1.0 + distance_matrix.mean()/adaptive_scale  # linear adaptation
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**dynamic_beta
    return np.tanh(adaptive_scale/distance_matrix) * decay * inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix
import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    dynamic_beta = 1.0 + 0.3*np.tanh(adaptive_scale/10)  # smoother adaptation
    norm_dist = distance_matrix/(adaptive_scale + epsilon)
    sigmoid_weight = 1/(1+np.exp(-norm_dist))
    decay = np.exp(-norm_dist)
    inv_dist = (1/(distance_matrix + epsilon))**dynamic_beta
    return sigmoid_weight * decay * inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix
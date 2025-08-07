import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    dynamic_beta = 1.5 + 0.5*np.sin(adaptive_scale)  # oscillating adaptation
    sigmoid_weight = 1/(1+np.exp(-distance_matrix/adaptive_scale))  # smooth transition
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**dynamic_beta
    return sigmoid_weight * decay * inv_dist * np.log1p(1/(distance_matrix + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix
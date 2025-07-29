import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    dynamic_beta = 1.0 + distance_matrix/adaptive_scale  # linear adaptation
    scaled_dist = distance_matrix/(adaptive_scale + epsilon)
    return np.tanh(1/scaled_dist) * (1/(distance_matrix + epsilon))**dynamic_beta
    #EVOLVE-END
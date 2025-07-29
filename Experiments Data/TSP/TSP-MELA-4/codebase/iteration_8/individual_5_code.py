import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    sigmoid_weight = 1/(1+np.exp(-distance_matrix/adaptive_scale))
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**1.5
    return sigmoid_weight * decay * inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix
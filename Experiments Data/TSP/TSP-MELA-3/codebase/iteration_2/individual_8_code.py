import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    local_scale = np.min(distance_matrix, axis=1, keepdims=True)
    global_scale = np.mean(distance_matrix, axis=1, keepdims=True)
    return np.exp(-distance_matrix/(0.5*(local_scale + global_scale) + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix
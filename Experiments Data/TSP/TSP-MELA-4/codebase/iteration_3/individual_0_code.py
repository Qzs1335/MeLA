import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    beta = 2.0 - np.random.rand()*0.4  # dynamic exponent [1.6,2.0]
    adaptive_scale = np.mean(distance_matrix)
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**beta
    result = decay * inv_dist
    return result/np.max(result)  # normalized
    #EVOLVE-END       
    return 1 / distance_matrix
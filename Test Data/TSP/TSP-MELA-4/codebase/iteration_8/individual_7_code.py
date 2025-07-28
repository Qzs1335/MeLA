import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    beta = 1.5  # stable coefficient
    ratio = distance_matrix/(adaptive_scale + epsilon)
    sigmoid_weight = 1/(1 + np.exp(-ratio))
    decay = np.exp(-ratio)
    inv_dist = (1/(distance_matrix + epsilon))**beta
    return sigmoid_weight * decay * inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix
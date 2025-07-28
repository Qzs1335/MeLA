import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    inv_dist = 1 / (distance_matrix + epsilon)
    return np.exp(inv_dist) / np.sum(np.exp(inv_dist))  # softmax normalization
    #EVOLVE-END       
    return 1 / distance_matrix
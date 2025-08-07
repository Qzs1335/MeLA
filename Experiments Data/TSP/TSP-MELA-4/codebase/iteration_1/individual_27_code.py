import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # prevents division by zero
    alpha = 0.5      # controls decay rate
    normalized_dist = distance_matrix / np.max(distance_matrix)
    return np.exp(-alpha * normalized_dist) + epsilon
    #EVOLVE-END       
    return 1 / distance_matrix
import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8  # Prevent division by zero
    normalized_dist = distance_matrix / (np.max(distance_matrix) + epsilon)
    log_term = np.log(1 + normalized_dist)
    exp_term = np.exp(-normalized_dist)
    return (exp_term + 1/log_term) / 2
    #EVOLVE-END       
    return 1 / distance_matrix
import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # Prevent division by zero
    scaled_dist = distance_matrix + epsilon
    log_term = np.log1p(distance_matrix)
    return (1/scaled_dist) * np.exp(-log_term)
    #EVOLVE-END       
    return 1 / distance_matrix
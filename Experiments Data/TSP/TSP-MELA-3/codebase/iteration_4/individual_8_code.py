import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    lam = 0.5  # Balancing parameter
    smooth_dist = distance_matrix + epsilon
    return lam*(1/smooth_dist) + (1-lam)*np.exp(1/smooth_dist)
    #EVOLVE-END       
    return 1 / distance_matrix
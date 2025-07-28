import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    weighted_dist = distance_matrix * 0.8  # reduces over-penalization
    return np.exp(1/(weighted_dist + epsilon)) + (1/(distance_matrix + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix
import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    log_dist = np.log1p(distance_matrix)
    return 1 / (distance_matrix * log_dist + epsilon)
    #EVOLVE-END       
    return 1 / distance_matrix
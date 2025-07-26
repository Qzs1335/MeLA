import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    base = 1 / (distance_matrix + epsilon)
    log_mod = 1 / (np.log1p(distance_matrix) + epsilon)
    return base * log_mod
    #EVOLVE-END       
    return 1 / distance_matrix

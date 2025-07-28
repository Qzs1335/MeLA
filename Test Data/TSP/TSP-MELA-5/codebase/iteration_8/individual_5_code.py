import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    soft_inv = 1/(distance_matrix + 1)
    log_weights = np.log(distance_matrix + 1 + epsilon)
    return soft_inv * (1/log_weights)
    #EVOLVE-END       
    return 1 / distance_matrix
import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    inv_dist = 1/(np.sqrt(distance_matrix) + epsilon)
    log_dist = 1/np.log(1 + distance_matrix)
    return 1/(1 + np.exp(-(inv_dist * log_dist)))
    #EVOLVE-END       
    return 1 / distance_matrix
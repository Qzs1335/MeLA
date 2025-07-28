import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    log_weights = np.log(1 + distance_matrix)
    return (1 / (distance_matrix**2 + epsilon)) * (1 / log_weights)
    #EVOLVE-END       
    return 1 / distance_matrix
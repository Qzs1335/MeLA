import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    log_dist = np.log(distance_matrix + epsilon)
    return 1 / (log_dist**2 + epsilon)
    #EVOLVE-END       
    return 1 / distance_matrix
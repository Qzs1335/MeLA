import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    log_scaled = np.log(1 + distance_matrix)
    return 1 / (log_scaled**2 + epsilon)
    #EVOLVE-END       
    return 1 / distance_matrix
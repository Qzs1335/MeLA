import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-15
    log_transformed = -np.log(distance_matrix + epsilon)
    return np.exp(log_transformed) / np.sum(np.exp(log_transformed))
    #EVOLVE-END       
    return 1 / distance_matrix
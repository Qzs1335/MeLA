import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    return np.exp(-0.5 * distance_matrix) / (np.log(distance_matrix + epsilon) + 1)
    #EVOLVE-END       
    return 1 / distance_matrix
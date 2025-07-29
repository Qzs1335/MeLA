import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    return np.exp(-0.1 * distance_matrix) / (distance_matrix + epsilon)
    #EVOLVE-END       
    return 1 / distance_matrix
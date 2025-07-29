import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-5
    scaled_dist = np.log(distance_matrix + epsilon)
    return np.exp(-0.5 * scaled_dist)
    #EVOLVE-END       
    return 1 / distance_matrix
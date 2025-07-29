import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10
    scaled_matrix = np.log1p(distance_matrix) + 1
    return 1 / (scaled_matrix + epsilon)
    #EVOLVE-END       
    return 1 / distance_matrix
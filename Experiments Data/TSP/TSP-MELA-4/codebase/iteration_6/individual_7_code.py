import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    return np.log1p(1/(distance_matrix + epsilon)) * (1/(distance_matrix + epsilon))**0.5
    #EVOLVE-END       
    return 1 / distance_matrix
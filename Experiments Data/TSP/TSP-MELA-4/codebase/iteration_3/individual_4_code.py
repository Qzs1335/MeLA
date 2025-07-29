import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-10  # improved precision
    adaptive_scale = np.median(distance_matrix)  # more robust scaling
    decay = np.exp(-distance_matrix/(adaptive_scale + epsilon))
    inv_dist = (1/(distance_matrix + epsilon))**2.0
    return decay * inv_dist
    #EVOLVE-END       
    return 1 / distance_matrix
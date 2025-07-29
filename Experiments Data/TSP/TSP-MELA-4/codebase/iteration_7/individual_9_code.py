import numpy as np
import numpy as np 
def heuristics_v2(distance_matrix):
    #EVOLVE-START
    epsilon = 1e-8
    adaptive_scale = np.median(distance_matrix)
    decay = np.exp(-distance_matrix/(2*adaptive_scale + epsilon))
    inv_dist = 1/(distance_matrix + epsilon)
    return decay * inv_dist * np.log1p(1/(distance_matrix + epsilon))
    #EVOLVE-END       
    return 1 / distance_matrix